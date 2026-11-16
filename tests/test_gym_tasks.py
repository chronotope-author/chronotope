"""Gym completion cards (gym-outcome-contract, A1 sponsor skip, B1 live trading).

U22-U24: (Y_g, G_g) defined for every valid terminal trace; success-by-deadline curve; seeded
trial-level D->A jitter. U25-U27: A1 closed form on the player skin. U28-U30: B1 reversal
regimes, terminal utilities, armed-colour discipline on the chart skin.
"""
from __future__ import annotations

import math

import pytest

from chronotope.__main__ import run_triplet
from chronotope.controller import _find_target_box
from chronotope.evaluate import evaluate_trace, summarize
from chronotope.runner import PayloadStore, run_trial
from chronotope.scenario import TARGET_COLOR, manifest_from_dict
from chronotope.spec import SpecError, TrialSpec
from chronotope.stats import success_by_deadline
from chronotope.trace import Trace
from conftest import base_spec_dict
from test_moving_target import mt_spec

PUB, PERIOD = 30_000_000, 16_666_667
E = 500_000_000


def first_frame_at_or_after(t: int) -> int:
    return math.ceil(t / PERIOD) * PERIOD


T1 = first_frame_at_or_after(E)   # first frame depicting the triggering event


# ---- spec builders ---------------------------------------------------------------------------

def sponsor_spec(compute=100_000_000, delivery=20_000_000, seek=1_000_000_000, window=5_000_000_000,
                 seed=7, **delivery_over) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "sponsor_skip", "scenario_seed": seed, "canvas_w_px": 96, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": E, "response_window_ns": window,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": E + window + 500_000_000,
                     "irrelevant_motion": False, "skin": "player", "seek_delay_ns": seek}
    d["controller"]["compute_ns"] = compute
    d["delivery"]["delivery_delay_ns"] = delivery
    d["delivery"].update(delivery_over)
    return TrialSpec.from_dict(d)


def trading_spec(compute=200_000_000, delivery=60_000_000, reversal=0, window=2_000_000_000,
                 seed=7, u=(1.0, -1.0, 0.0), **delivery_over) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "live_trading", "scenario_seed": seed, "canvas_w_px": 128, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": E, "response_window_ns": window,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": E + window + 500_000_000,
                     "irrelevant_motion": False, "skin": "chart", "reversal_delay_ns": reversal,
                     "u_valid_order": u[0], "u_invalid_order": u[1], "u_no_order": u[2]}
    d["controller"]["compute_ns"] = compute
    d["delivery"]["delivery_delay_ns"] = delivery
    d["delivery"].update(delivery_over)
    return TrialSpec.from_dict(d)


# ---- U25: A1 closed form -----------------------------------------------------------------------

def a1_closed_form(spec: TrialSpec) -> dict:
    m = spec.manifest()
    A = T1 + PUB + spec.raw["controller"]["compute_ns"] + spec.raw["delivery"]["delivery_delay_ns"]
    close = m.window_close_ns()
    valid = A < close
    g_live = -(min(A + m.seek_delay_ns, close) - E) / 1e9 if valid else -m.response_window_ns / 1e9
    g_pinned = -(min(T1 + m.seek_delay_ns, close) - E) / 1e9   # paused/replay: t_recv pinned at T1
    return {"A": A, "valid": valid, "g_live": g_live, "g_pinned": g_pinned}


@pytest.mark.parametrize("compute", [100_000_000, 2_000_000_000, 4_800_000_000, 5_500_000_000])
@pytest.mark.parametrize("seek", [0, 1_000_000_000])
def test_a1_closed_form_sweep(ws, compute, seek):
    spec = sponsor_spec(compute=compute, seek=seek)
    pred = a1_closed_form(spec)
    out = run_triplet(spec, ws)
    assert out["pair"].valid
    live = out["results"]["live"]
    assert live.valid_trial and not live.contract_violations
    assert live.valid_at_receipt == pred["valid"]
    assert live.return_G == pytest.approx(pred["g_live"])
    if not pred["valid"]:
        assert "late" in live.labels and live.return_G == pytest.approx(-5.0)
    for p in ("paused", "replay"):
        r = out["results"][p]
        assert r.valid_at_receipt is True and r.return_G == pytest.approx(pred["g_pinned"])


def test_a1_graded_return_orders_equal_success_trials(ws):
    """Two timely skips, same Y=1, different G: the graded exposure separates them (Tom's ask)."""
    fast = evaluate_trace(Trace.load(run_trial(sponsor_spec(compute=100_000_000), ws / "f.jsonl", PayloadStore(ws / "p"))))
    slow = evaluate_trace(Trace.load(run_trial(sponsor_spec(compute=3_000_000_000), ws / "s.jsonl", PayloadStore(ws / "p"))))
    assert fast.valid_at_receipt and slow.valid_at_receipt
    assert fast.return_G > slow.return_G
    assert fast.return_G - slow.return_G == pytest.approx(2.9, abs=0.01)   # the extra compute time


def test_a1_seek_slower_than_window_costs_full_sponsor(ws):
    spec = sponsor_spec(compute=100_000_000, seek=6_000_000_000)   # seek > window: effect never lands in time
    r = evaluate_trace(Trace.load(run_trial(spec, ws / "t.jsonl", PayloadStore(ws / "p"))))
    assert r.valid_at_receipt is True and r.return_G == pytest.approx(-5.0)


def test_a1_dropped_delivery_full_sponsor(ws):
    spec = sponsor_spec(**{"route": "simulated_drop_first"})
    r = evaluate_trace(Trace.load(run_trial(spec, ws / "d.jsonl", PayloadStore(ws / "p"))))
    assert r.valid_trial and not r.valid_at_receipt and "dropped" in r.labels
    assert r.return_G == pytest.approx(-5.0)


def test_a1_player_skin_target_discipline():
    """U27: on the player skin the oracle's colour box equals the seeded target box, and no
    target-coloured pixel exists outside the sponsor segment."""
    m = sponsor_spec().manifest()
    b = m.target_box()
    box = _find_target_box(m.render(E + 1_000_000_000), m.canvas_w_px, m.canvas_h_px, TARGET_COLOR["player"])
    assert box == (b.x0, b.y0, b.x1, b.y1)
    for t in (0, E - PERIOD, m.window_close_ns() + PERIOD):
        assert _find_target_box(m.render(t), m.canvas_w_px, m.canvas_h_px, TARGET_COLOR["player"]) is None


# ---- U28-U30: B1 reversal regimes -----------------------------------------------------------

def b1_times(spec: TrialSpec) -> tuple[int, int]:
    d = T1 + PUB + spec.raw["controller"]["compute_ns"]
    return d, d + spec.raw["delivery"]["delivery_delay_ns"]


def test_b1_no_reversal_valid_order(ws):
    out = run_triplet(trading_spec(), ws)
    assert out["pair"].valid
    for p in ("paused", "replay", "live"):
        r = out["results"][p]
        assert r.valid_trial and r.valid_at_receipt is True and r.return_G == 1.0, (p, r.to_dict())
    assert out["results"]["live"].labels == []


def test_b1_reversal_before_D_outdated_at_commitment(ws):
    spec = trading_spec(reversal=100_000_000)          # reversal at 600 ms, observed before D at 730 ms
    d, a = b1_times(spec)
    m = spec.manifest()
    assert m.reversal_time_ns() < d
    out = run_triplet(spec, ws)
    live = out["results"]["live"]
    assert live.valid_at_receipt is False and live.return_G == -1.0
    assert "outdated_at_commitment" in live.labels and "late" not in live.labels
    for p in ("paused", "replay"):                      # pinned at the trigger state: still valid
        assert out["results"][p].valid_at_receipt is True and out["results"][p].return_G == 1.0


def test_b1_reversal_between_D_and_A_invalidated(ws):
    spec = trading_spec(reversal=250_000_000)          # reversal at 750 ms, inside (D=730, A=790]
    d, a = b1_times(spec)
    assert d < spec.manifest().reversal_time_ns() <= a
    live = run_triplet(spec, ws)["results"]["live"]
    assert live.valid_at_receipt is False and live.return_G == -1.0
    assert "invalidated_after_commitment" in live.labels and "outdated_at_commitment" not in live.labels
    assert "late" not in live.labels


def test_b1_reversal_after_A_harmless(ws):
    spec = trading_spec(reversal=400_000_000)          # reversal at 900 ms, after A at 790 ms
    d, a = b1_times(spec)
    assert spec.manifest().reversal_time_ns() > a
    live = run_triplet(spec, ws)["results"]["live"]
    assert live.valid_at_receipt is True and live.return_G == 1.0 and live.labels == []


def test_b1_order_after_expiry_u_invalid(ws):
    spec = trading_spec(compute=2_500_000_000)         # D past the 2.5 s close
    live = run_triplet(spec, ws)["results"]["live"]
    assert live.valid_at_receipt is False and live.return_G == -1.0 and "late" in live.labels


def test_b1_no_order_terminals_u_no_order(ws):
    r = evaluate_trace(Trace.load(run_trial(trading_spec(**{"route": "simulated_drop_first"}),
                                            ws / "d.jsonl", PayloadStore(ws / "p"))))
    assert r.valid_trial and not r.valid_at_receipt and "dropped" in r.labels and r.return_G == 0.0


def test_b1_armed_colour_is_signal_validity():
    """U30: the button carries the armed colour exactly while the signal is valid, so the oracle's
    evidence respects validity by construction."""
    m = trading_spec(reversal=250_000_000).manifest()
    b = m.target_box()
    armed = lambda t: _find_target_box(m.render(t), m.canvas_w_px, m.canvas_h_px, TARGET_COLOR["chart"])  # noqa: E731
    assert armed(E - PERIOD) is None
    assert armed(E + 100_000_000) == (b.x0, b.y0, b.x1, b.y1)
    assert armed(m.reversal_time_ns() + PERIOD) is None
    assert armed(m.window_close_ns() + PERIOD) is None


def test_b1_oracle_first_evidence_at_or_after_signal(ws):
    t = Trace.load(run_trial(trading_spec(), ws / "t.jsonl", PayloadStore(ws / "p")))
    d0 = t.committing_decision()
    ob = next(o for o in t.all("observation") if o["observation_id"] == d0["observation_id"])
    assert ob["source_time_ns"] >= E


# ---- U22: the return contract on Task C terminals --------------------------------------------

def test_g_contract_late_click_charges_e_max(ws):
    """A late click on a static target: Y = 0 (window closed) and G = -e_max — the target is absent in the
    state at receipt, so there is nothing to be near (2026-08-24 rule; O1 card, paper return matrix). A
    click received while the target is present is graded by its distance (test_g_contract_moving_target)."""
    d = base_spec_dict(controller__compute_ns=600_000_000)
    m = manifest_from_dict(d["scenario"])
    r = evaluate_trace(Trace.load(run_trial(TrialSpec.from_dict(d), ws / "l.jsonl", PayloadStore(ws / "p"))))
    assert r.valid_trial and r.valid_at_receipt is False and "late" in r.labels
    assert r.return_G == pytest.approx(-m.e_max_px())


def test_g_contract_no_receipt_is_display_diagonal(ws):
    d = base_spec_dict(delivery__route="simulated_drop_first")
    m = manifest_from_dict(d["scenario"])
    r = evaluate_trace(Trace.load(run_trial(TrialSpec.from_dict(d), ws / "n.jsonl", PayloadStore(ws / "p"))))
    assert r.return_G == pytest.approx(-m.e_max_px())


def test_g_contract_moving_target_matches_error(ws):
    spec = mt_spec(vx=120, delivery=100_000_000)
    r = evaluate_trace(Trace.load(run_trial(spec, ws / "m.jsonl", PayloadStore(ws / "p"))))
    assert r.return_G == pytest.approx(-min(r.spatial_error_px, spec.manifest().e_max_px()))


def test_summarize_reports_mean_G(ws):
    rs = [evaluate_trace(Trace.load(run_trial(sponsor_spec(seed=s), ws / f"{s}.jsonl", PayloadStore(ws / "p"))))
          for s in (7, 8)]
    s = summarize(rs)["live"]
    assert s["mean_G"] == pytest.approx(sum(r.return_G for r in rs) / 2)


# ---- U23: success-by-deadline curve -----------------------------------------------------------

def test_deadline_curve_unit():
    out = success_by_deadline([(True, 100), (True, 300), (False, 200), (True, None)], [50, 101, 200, 301, 400])
    assert out == [0.0, 0.25, 0.25, 0.5, 0.5]
    assert out == sorted(out)   # nondecreasing
    with pytest.raises(ValueError):
        success_by_deadline([], [1])


@pytest.mark.parametrize("w", [200_000_000, 400_000_000, 800_000_000])
def test_deadline_curve_equals_window_sweep_for_oracle(ws, w):
    """U23: F(w) from one generous-window run equals the direct run at window w, per compute cell
    (the oracle never reads the rendered countdown)."""
    computes = (100_000_000, 300_000_000, 700_000_000)
    outcomes = []
    for c in computes:
        d = base_spec_dict(controller__compute_ns=c, scenario__response_window_ns=1_000_000_000)
        r = evaluate_trace(Trace.load(run_trial(TrialSpec.from_dict(d), ws / f"g{c}.jsonl", PayloadStore(ws / "p"))))
        assert r.valid_trial
        outcomes.append((bool(r.valid_at_receipt), r.intervals_ns["E_A"]))
    curve = success_by_deadline(outcomes, [w])[0]
    direct = []
    for c in computes:
        d = base_spec_dict(controller__compute_ns=c, scenario__response_window_ns=w)
        r = evaluate_trace(Trace.load(run_trial(TrialSpec.from_dict(d), ws / f"w{w}_{c}.jsonl", PayloadStore(ws / "p"))))
        direct.append(bool(r.valid_at_receipt))
    assert curve == sum(direct) / len(direct)


# ---- U24: seeded jitter route ------------------------------------------------------------------

def jitter_spec(seed=3, lo=10_000_000, hi=50_000_000, **over):
    d = base_spec_dict(**over)
    d["delivery"] = {"route": "simulated_seeded_jitter", "delivery_delay_min_ns": lo,
                     "delivery_delay_max_ns": hi, "delivery_seed": seed, "max_attempts": 1}
    return TrialSpec.from_dict(d)


def test_jitter_route_deterministic_and_bounded(ws):
    das = []
    for tag in ("a", "b"):
        r = evaluate_trace(Trace.load(run_trial(jitter_spec(), ws / f"{tag}.jsonl", PayloadStore(ws / "p"))))
        assert r.valid_trial and 10_000_000 <= r.intervals_ns["D_A"] <= 50_000_000
        das.append(r.intervals_ns["D_A"])
    assert das[0] == das[1]   # same delivery_seed -> same D->A
    others = [evaluate_trace(Trace.load(run_trial(jitter_spec(seed=s), ws / f"s{s}.jsonl", PayloadStore(ws / "p"))))
              .intervals_ns["D_A"] for s in (4, 5, 6)]
    assert any(x != das[0] for x in others)   # the delay really varies trial to trial


def test_jitter_harmless_on_static_target_linear_on_moving(ws):
    r = evaluate_trace(Trace.load(run_trial(jitter_spec(), ws / "st.jsonl", PayloadStore(ws / "p"))))
    assert r.valid_at_receipt is True and r.return_G == pytest.approx(0.0)   # negative control
    for seed in (3, 4, 5):
        d = jitter_spec(seed=seed).raw
        d["scenario"] = dict(mt_spec(vx=120).raw["scenario"])
        rm = evaluate_trace(Trace.load(run_trial(TrialSpec.from_dict(d), ws / f"mv{seed}.jsonl", PayloadStore(ws / "p"))))
        drift = 120 * (rm.intervals_ns["I_A"]) / 1e9   # error ~ v * (age of the used evidence at A)
        assert rm.spatial_error_px == pytest.approx(drift, abs=1.6), (seed, rm.spatial_error_px, drift)


# ---- spec strictness ---------------------------------------------------------------------------

def test_spec_rejects_bad_gym_configs():
    with pytest.raises(SpecError, match="not in"):
        d = sponsor_spec().raw | {}
        d2 = {**d, "scenario": {**d["scenario"], "skin": "plain"}}
        TrialSpec.from_dict(d2)
    with pytest.raises(SpecError, match="non-negative"):
        d = sponsor_spec().raw
        TrialSpec.from_dict({**d, "scenario": {**d["scenario"], "seek_delay_ns": -1}})
    with pytest.raises(SpecError, match="expected float"):
        trading_spec(u=(1, -1.0, 0.0))
    with pytest.raises(ValueError, match="at or after expiry"):
        trading_spec(reversal=2_000_000_000)
    with pytest.raises(SpecError, match="delivery_delay_min_ns must be <="):
        jitter_spec(lo=50_000_000, hi=10_000_000)
    with pytest.raises(SpecError, match="unknown"):
        d = jitter_spec().raw
        TrialSpec.from_dict({**d, "delivery": {**d["delivery"], "delivery_delay_ns": 5}})
    with pytest.raises(SpecError, match="missing"):
        d = trading_spec().raw
        sc = {k: v for k, v in d["scenario"].items() if k != "u_no_order"}
        TrialSpec.from_dict({**d, "scenario": sc})
