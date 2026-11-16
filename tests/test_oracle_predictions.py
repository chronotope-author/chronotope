"""U2: the harness must reproduce KNOWN outcomes for a scripted oracle with declared delays.

The prediction below is computed from the spec parameters alone, with no reference to the
harness. If the harness disagrees, the harness is wrong — not the prediction.
"""
from __future__ import annotations

import math

import pytest

from chronotope.__main__ import run_triplet
from chronotope.spec import TrialSpec
from conftest import base_spec_dict


def predict(d: dict) -> dict:
    s, c, o, dl = d["scenario"], d["controller"], d["observation"], d["delivery"]
    t_e, W, P = s["event_time_ns"], s["response_window_ns"], s["frame_period_ns"]
    t_close = t_e + W
    pub, comp, deliv = o["publication_delay_ns"], c["compute_ns"], dl["delivery_delay_ns"]
    semantic_ok = abs(c["coordinate_offset_px"]) < s["target_size_px"] // 2 + 1 and c["coordinate_offset_px"] == 0
    # first frame depicting the target
    k1 = math.ceil(t_e / P)
    t1 = k1 * P
    has_target_frame = t1 < t_close
    if not has_target_frame:
        return {"paused": False, "replay": False, "live": False, "live_labels": {"missed_evidence:target_appears"},
                "action": False}
    O1 = t1 + pub
    D = O1 + comp
    A = D + deliv
    # first frame depicting the closed window
    t2 = math.ceil(t_close / P) * P
    O2 = t2 + pub
    live_ok = semantic_ok and A < t_close
    labels: set[str] = set()
    if not semantic_ok:
        labels.add("semantic_error")
    if O1 > t_close:
        labels.add("missed_evidence:target_appears")
    if A >= t_close:
        labels.add("late")
        if semantic_ok:
            if D < t_close:
                labels.add("invalidated_after_commitment")
            elif O2 < D:
                labels.add("outdated_at_commitment")
            else:
                labels.add("missed_evidence:window_closes")
    return {"paused": semantic_ok, "replay": semantic_ok, "live": live_ok, "live_labels": labels, "action": True,
            "E_O": O1 - t_e if O1 <= t_close else None, "O_D": comp if O1 <= t_close else None,
            "D_A": deliv, "E_A": A - t_e, "S_A": t_close - A}


GRID = [
    dict(controller__compute_ns=c, delivery__delivery_delay_ns=d, scenario__response_window_ns=w,
         scenario__irrelevant_motion=im)
    for c in (0, 100_000_000, 250_000_000, 340_000_000, 355_000_000, 400_000_000, 600_000_000)
    for d in (0, 20_000_000, 60_000_000, 200_000_000)
    for w in (400_000_000, 100_000_000)
    for im in (False, True)
]
GRID += [
    dict(controller__coordinate_offset_px=10),                         # semantic error everywhere
    dict(observation__publication_delay_ns=450_000_000),              # evidence arrives after the window closed
    dict(scenario__frame_period_ns=450_000_000, scenario__event_time_ns=500_000_000,
         scenario__response_window_ns=400_000_000),                   # no frame ever depicts the target
]


@pytest.mark.parametrize("over", GRID)
def test_oracle_matches_closed_form(ws, over):
    d = base_spec_dict(**over)
    pred = predict(d)
    out = run_triplet(TrialSpec.from_dict(d), ws)
    res, pair = out["results"], out["pair"]
    assert pair.valid, pair.reason
    for p in ("paused", "replay", "live"):
        r = res[p]
        assert r.valid_trial and not r.contract_violations, (p, r)
        assert r.action_committed == pred["action"], (p, r)
        assert bool(r.valid_at_receipt) == pred[p], (p, r.to_dict(), pred)
    live = res["live"]
    assert set(live.labels) == pred["live_labels"], (live.labels, pred["live_labels"])
    if pred["action"]:
        for k in ("E_O", "O_D", "D_A", "E_A", "S_A"):
            assert live.intervals_ns[k] == pred[k], (k, live.intervals_ns, pred)
        # Replay's committed action and evidence are Live's by construction.
        assert res["replay"].semantic_correct == live.semantic_correct
        assert pair.same_evidence_at_decision is True
    # Paused and Replay never carry timing labels.
    for p in ("paused", "replay"):
        assert not ({"late", "invalidated_after_commitment", "outdated_at_commitment",
                     "missed_evidence:window_closes"} & set(res[p].labels)), res[p].labels


def test_live_failure_classes_are_exhaustive_and_exclusive(ws):
    """For the oracle, every Live failure carries exactly one of the three 'where did the closing
    fall' labels, plus 'late'. This is the reader table for WI-001."""
    seen = set()
    for c in range(0, 700_000_000, 25_000_000):
        for dl in (0, 40_000_000, 120_000_000):
            d = base_spec_dict(controller__compute_ns=c, delivery__delivery_delay_ns=dl)
            live = run_triplet(TrialSpec.from_dict(d), ws / f"{c}_{dl}")["results"]["live"]
            if live.valid_at_receipt:
                assert live.labels == []
                continue
            where = {"invalidated_after_commitment", "outdated_at_commitment", "missed_evidence:window_closes"}
            hit = where & set(live.labels)
            assert len(hit) == 1 and "late" in live.labels, live.labels
            seen |= hit
    assert seen == {"invalidated_after_commitment", "outdated_at_commitment", "missed_evidence:window_closes"}


def test_determinism_same_spec_same_trace(ws):
    d = base_spec_dict()
    a = run_triplet(TrialSpec.from_dict(d), ws / "a")
    b = run_triplet(TrialSpec.from_dict(d), ws / "b")
    for p in ("paused", "replay", "live"):
        assert (ws / "a" / f"{p}.jsonl").read_bytes() == (ws / "b" / f"{p}.jsonl").read_bytes()


@pytest.mark.parametrize("skin", ["notification", "balloon"])
@pytest.mark.parametrize("over", [
    dict(controller__compute_ns=100_000_000, delivery__delivery_delay_ns=20_000_000),
    dict(controller__compute_ns=355_000_000, delivery__delivery_delay_ns=60_000_000),
    dict(controller__compute_ns=400_000_000, delivery__delivery_delay_ns=0),
    dict(controller__compute_ns=600_000_000, delivery__delivery_delay_ns=0),
])
def test_skins_are_render_only(ws, skin, over):
    """U10: a skin changes pixels, never outcomes — the same closed form must hold."""
    d = base_spec_dict(scenario__skin=skin, scenario__canvas_w_px=320, scenario__canvas_h_px=240,
                       scenario__target_size_px=24, **over)
    pred = predict(d)
    out = run_triplet(TrialSpec.from_dict(d), ws)
    assert out["pair"].valid
    for p in ("paused", "replay", "live"):
        assert bool(out["results"][p].valid_at_receipt) == pred[p], (skin, p, out["results"][p].to_dict())
    assert set(out["results"]["live"].labels) == pred["live_labels"]
    for k in ("E_O", "O_D", "D_A", "E_A", "S_A"):
        assert out["results"]["live"].intervals_ns[k] == pred[k]
