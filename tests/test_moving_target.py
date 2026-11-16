"""U7–U9: moving target. Closed-form post-commit delay sweep (S3.1), the four bindings with
matched B, D, A (S3.2), predictor perturbation, and the coordinate-correction trace contract."""
from __future__ import annotations

import json
import math

import pytest

from chronotope.__main__ import run_triplet
from chronotope.evaluate import evaluate_trace
from chronotope.hashing import canonical_json
from chronotope.runner import PayloadStore, run_trial
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict

PUB, PERIOD = 30_000_000, 16_666_667


def mt_spec(vx=120, vy=0, compute=100_000_000, delivery=60_000_000, binding="state_at_B", seed=7,
            exp_obs=None, exp_del=None, window=400_000_000) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "moving_target", "scenario_seed": seed, "canvas_w_px": 96, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": 500_000_000, "response_window_ns": window,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": 2_000_000_000, "irrelevant_motion": False, "skin": "plain",
                     "velocity_x_px_per_s": vx, "velocity_y_px_per_s": vy}
    d["controller"].update({"compute_ns": compute, "binding": binding,
                            "expected_observation_delay_ns": exp_obs, "expected_delivery_delay_ns": exp_del})
    d["delivery"]["delivery_delay_ns"] = delivery
    return TrialSpec.from_dict(d)


def closed_form_state_at_B(spec: TrialSpec) -> dict:
    """Timing in closed form; geometry via the manifest's pure state function."""
    m = spec.manifest()
    t1 = math.ceil(m.event_time_ns / PERIOD) * PERIOD           # first frame depicting the target
    O1 = t1 + PUB
    D = O1 + spec.raw["controller"]["compute_ns"]
    A = D + spec.raw["delivery"]["delivery_delay_ns"]
    click = dict(zip(("x", "y"), m.target_box_at(t1).center()))
    act = {"kind": "click", **click}
    return {"valid": A < m.window_close_ns() and m.target_box_at(A).contains(click["x"], click["y"]),
            "error": m.spatial_error_px(A, act) if A < m.window_close_ns() else None, "A": A, "D": D}


# ---- S3.1: post-commit delay sweep on a predictably moving target ---------------------------
@pytest.mark.parametrize("delivery", [0, 20_000_000, 40_000_000, 60_000_000, 100_000_000, 200_000_000])
@pytest.mark.parametrize("vx", [0, 60, 120, -90])
def test_s31_delay_sweep_matches_closed_form(ws, delivery, vx):
    spec = mt_spec(vx=vx, delivery=delivery)
    pred = closed_form_state_at_B(spec)
    out = run_triplet(spec, ws)
    live = out["results"]["live"]
    assert out["pair"].valid and live.valid_trial and not live.contract_violations
    assert live.valid_at_receipt == pred["valid"], (live.to_dict(), pred)
    assert live.spatial_error_px == pytest.approx(pred["error"])
    # the static target is a negative control: delay never changes the result
    if vx == 0:
        assert live.valid_at_receipt is True and live.spatial_error_px == 0.0
    # Paused/Replay: validity frozen at the triggering state -> always valid, zero error
    for p in ("paused", "replay"):
        assert out["results"][p].valid_at_receipt is True and out["results"][p].spatial_error_px == 0.0


# ---- S3.2: four bindings, matched B, D, A ----------------------------------------------------
def _run_bindings(ws, vx, exp_del=60_000_000, seeds=(7, 8, 9)):
    rows = {}
    for b in ("state_at_B", "newest_at_D", "predicted_at_A", "true_at_A"):
        rows[b] = []
        for seed in seeds:
            spec = mt_spec(vx=vx, binding=b, seed=seed,
                           exp_obs=PUB if b == "predicted_at_A" else None,
                           exp_del=exp_del if b == "predicted_at_A" else None)
            p = run_trial(spec, ws / f"{b}_{seed}.jsonl", PayloadStore(ws / "p"))
            t = Trace.load(p)
            r = evaluate_trace(t)
            assert r.valid_trial and not r.contract_violations, r
            rows[b].append((r, t))
    return rows


def test_s32_bindings_share_B_D_A_and_order_as_predicted(ws):
    rows = _run_bindings(ws, vx=120)
    # matched B, D, A across bindings (same seed)
    for i in range(3):
        keys = [(t.committing_decision()["begin_time_ns"], t.first("commit")["commitment_time_ns"],
                 t.first("receipt")["receipt_time_ns"]) for (_, t) in (rows[b][i] for b in rows)]
        assert len(set(keys)) == 1, keys
    err = {b: [r.spatial_error_px for r, _ in rows[b]] for b in rows}
    for i in range(3):
        assert err["true_at_A"][i] <= err["predicted_at_A"][i] <= err["newest_at_D"][i] <= err["state_at_B"][i], err
    assert all(e == 0.0 for e in err["true_at_A"])
    assert all(e <= 1.5 for e in err["predicted_at_A"])       # velocity from integer positions: <=~1 px
    assert all(e > 3.0 for e in err["state_at_B"])            # 120 px/s over ~190 ms
    # predicted/true succeed, state_at_B fails at this speed
    assert all(r.valid_at_receipt for r, _ in rows["true_at_A"] + rows["predicted_at_A"])
    assert not any(r.valid_at_receipt for r, _ in rows["state_at_B"])
    # trace contract: the correcting bindings are recorded as coordinate corrections, not updates
    for b in ("newest_at_D", "predicted_at_A", "true_at_A"):
        for r, t in rows[b]:
            c = t.first("coordinate_correction")
            assert c and c["binding"] == b and c["corrected_action"] == t.first("commit")["action"]
            assert t.first("action_proposed")["action"] != c["corrected_action"]
            assert r.binding == b
    for r, t in rows["state_at_B"]:
        assert t.first("coordinate_correction") is None


def test_s32_static_target_all_bindings_tie(ws):
    rows = _run_bindings(ws, vx=0, seeds=(7, 8))
    for i in range(2):
        outcomes = {b: (rows[b][i][0].valid_at_receipt, rows[b][i][0].spatial_error_px) for b in rows}
        assert set(outcomes.values()) == {(True, 0.0)}, outcomes


@pytest.mark.parametrize("delta", [-60_000_000, -40_000_000, -20_000_000, 0, 20_000_000, 40_000_000, 60_000_000])
def test_s32_predictor_degrades_linearly_with_delivery_estimate_error(ws, delta):
    vx, true_del = 120, 60_000_000
    spec = mt_spec(vx=vx, binding="predicted_at_A", exp_obs=PUB, exp_del=true_del + delta)
    r = evaluate_trace(Trace.load(run_trial(spec, ws / "p.jsonl", PayloadStore(ws / "p"))))
    expected = abs(vx * delta / 1e9)
    assert r.spatial_error_px == pytest.approx(expected, abs=1.5), (delta, r.spatial_error_px, expected)


# ---- contract: corrections must be recorded; kind changes rejected; spec strictness ----------
def test_uncorrected_action_change_is_flagged(ws):
    spec = mt_spec(binding="newest_at_D")
    p = run_trial(spec, ws / "c.jsonl", PayloadStore(ws / "p"))
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    recs = [r for r in recs if r["record"] != "coordinate_correction"]
    for n, r in enumerate(recs, 1):
        r["seq"] = n
    (ws / "bad.jsonl").write_text("".join(canonical_json(r) + "\n" for r in recs))
    assert "uncorrected_action_change" in evaluate_trace(Trace.load(ws / "bad.jsonl")).contract_violations


def test_spec_rejects_bad_binding_config():
    with pytest.raises(SpecError, match="required for binding predicted_at_A"):
        mt_spec(binding="predicted_at_A")
    with pytest.raises(SpecError, match="must be null unless"):
        mt_spec(binding="state_at_B", exp_del=5)
    with pytest.raises(SpecError, match="not in"):
        d = base_spec_dict(); d["controller"]["binding"] = "same_decision_update"; TrialSpec.from_dict(d)
    with pytest.raises(SpecError, match="unknown"):
        d = base_spec_dict(); d["scenario"]["velocity_x_px_per_s"] = 5; TrialSpec.from_dict(d)  # not on revealed_target
    with pytest.raises(ValueError, match="leave the canvas"):
        mt_spec(vx=5000)
