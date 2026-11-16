"""WI-006: pending_effect task, multi-action runner, PendingMemory, evaluator (simulated clock,
scripted controllers; no model)."""
from __future__ import annotations

import json

import pytest

from chronotope.evaluate import evaluate_pending_trace
from chronotope.runner import PayloadStore, run_trial
from chronotope.scenario import EVENT_EFFECT_VISIBLE
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict

PERIOD, PUB, DEL = 16_666_667, 30_000_000, 20_000_000


def pe_spec(memory="none", effect_delay=2_000_000_000, compute=300_000_000, timeout=10_000_000_000,
            route="simulated_fixed_delay", seed=5, horizon=8_000_000_000, irrelevant=False) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "pending_effect", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 500_000_000, "response_window_ns": 6_000_000_000,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": horizon, "irrelevant_motion": irrelevant,
                     "skin": "notification", "effect_delay_ns": effect_delay}
    d["controller"] = {"kind": "scripted_pending", "compute_ns": compute, "memory": memory,
                       "pending_timeout_ns": timeout, "pending_radius_px": 10}
    d["delivery"] = {"route": route, "delivery_delay_ns": DEL, "max_attempts": 1}
    return TrialSpec.from_dict(d)


def run(spec, ws, name="t"):
    p = run_trial(spec, ws / f"{name}.jsonl", PayloadStore(ws / "p"))
    t = Trace.load(p)
    return t, evaluate_pending_trace(t)


# (i) env: first click accepted; effect visible exactly effect_delay after receipt; frames show it
def test_env_accepts_first_click_and_shows_effect_after_delay(ws):
    t, r = run(pe_spec("harness_pending"), ws)
    recs = t.all("receipt")
    acc = [x for x in recs if x["outcome"] == "accepted"]
    assert len(acc) == 1 and acc[0]["action_id"] == 1
    ev = next(e for e in t.all("event") if e["event_id"] == EVENT_EFFECT_VISIBLE)
    assert ev["task_time_ns"] == acc[0]["task_time_at_receipt_ns"] + 2_000_000_000
    # the first observation whose source event is the effect depicts a time >= effect time
    o = next(o for o in t.all("observation") if o["source_event_id"] == EVENT_EFFECT_VISIBLE)
    assert o["source_time_ns"] >= ev["task_time_ns"]
    assert t.first("trial_end")["reason"] == "effect_done"
    assert r.accepted and r.exact_success and r.contract_violations == []


# (ii) memoryless repeater: one click per changed frame (the bar drains every frame) until the effect shows
def test_memoryless_repeats_until_effect_visible(ws):
    t, r = run(pe_spec("none", effect_delay=2_000_000_000, compute=300_000_000), ws)
    # after the first click (committed ~0.83 s), the effect is visible at ~0.85+2.0 s; decisions every
    # ~0.3 s (compute) + frame -> ~6 more clicks, all duplicates
    assert r.accepted and r.duplicates >= 4 and not r.exact_success
    assert r.n_clicks_committed == 1 + r.duplicates + r.invalid_clicks
    assert r.contract_violations == []
    # duplicates stop once the effect is visible: every receipt after the effect is 'invalid' or there is none
    eff = next(e for e in t.all("event") if e["event_id"] == EVENT_EFFECT_VISIBLE)["task_time_ns"]
    for x in t.all("receipt"):
        if x["task_time_at_receipt_ns"] >= eff:
            assert x["outcome"] == "invalid"


# (iii) harness_pending: duplicates suppressed, first action's D and A unchanged vs memoryless
def test_harness_pending_suppresses_without_delaying_first_action(ws):
    t0, r0 = run(pe_spec("none"), ws, "none")
    t1, r1 = run(pe_spec("harness_pending"), ws, "hp")
    assert r1.duplicates == 0 and r1.exact_success and r1.n_suppressed >= 4
    assert r0.duplicates >= 4
    assert t0.first("commit")["commitment_time_ns"] == t1.first("commit")["commitment_time_ns"]
    assert r0.first_valid_E_A_ns == r1.first_valid_E_A_ns
    assert r1.n_decisions == r0.n_decisions          # same trigger, same number of invocations
    assert r1.pending_events.get("pending_created") == 1 and r1.pending_events.get("pending_resolved") == 1
    assert r1.contract_violations == [] and r0.contract_violations == []


# (iv) dropped first click: memoryless retries at once; harness_pending only after the timeout
def test_dropped_first_click_retry_cost(ws):
    t0, r0 = run(pe_spec("none", route="simulated_drop_first"), ws, "none")
    # timeout must exceed the effect delay, otherwise the record expires before the effect shows and a
    # duplicate follows (checked below as its own fact)
    t1, r1 = run(pe_spec("harness_pending", route="simulated_drop_first", timeout=3_000_000_000), ws, "hp")
    assert r0.dropped == 1 and r1.dropped == 1
    assert r0.accepted and r1.accepted
    assert r1.first_valid_E_A_ns > r0.first_valid_E_A_ns        # the record delays the retry
    assert r1.first_valid_E_A_ns - r0.first_valid_E_A_ns >= 2_000_000_000   # by roughly the timeout
    assert r1.pending_events.get("pending_expired") == 1
    t2, r2 = run(pe_spec("harness_pending", route="simulated_fixed_delay", timeout=1_500_000_000), ws, "short")
    assert r2.duplicates >= 1 and r2.pending_events.get("pending_expired", 0) >= 1   # too-short timeout -> duplicate
    assert r1.duplicates == 0 and r1.contract_violations == []


# (v) model_informed: no suppression by the harness (scripted controller ignores the note -> same as none)
def test_model_informed_does_not_suppress_in_harness(ws):
    t, r = run(pe_spec("model_informed"), ws)
    assert r.n_suppressed == 0 and r.duplicates >= 4


# negative contract tests
def _rewrite(path, fn):
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in fn(recs)))


def test_evaluator_flags_unflagged_duplicate(ws):
    run(pe_spec("none"), ws)

    def forge(recs):
        seen = 0
        for x in recs:
            if x["record"] == "receipt" and x["outcome"] == "duplicate":
                x["outcome"] = "accepted"; x["valid_at_receipt"] = True
                break
        return recs
    _rewrite(ws / "t.jsonl", forge)
    r = evaluate_pending_trace(Trace.load(ws / "t.jsonl"))
    assert "receipt_outcome_inconsistent_with_manifest" in r.contract_violations


def test_evaluator_flags_suppressed_decision_that_commits(ws):
    run(pe_spec("harness_pending"), ws)

    def forge(recs):
        sup = next(x["decision_id"] for x in recs if x["record"] == "decision_invocation" and x.get("outcome") == "suppressed_pending")
        for x in recs:
            if x["record"] == "commit":
                x["decision_id"] = sup
                break
        return recs
    _rewrite(ws / "t.jsonl", forge)
    r = evaluate_pending_trace(Trace.load(ws / "t.jsonl"))
    assert "suppressed_decision_committed" in r.contract_violations


def test_pending_effect_is_live_only_and_strict(ws):
    d = json.loads(json.dumps(pe_spec().raw))
    d["task_clock_policy"] = "paused"
    with pytest.raises(ValueError, match="Live only"):
        run_trial(TrialSpec.from_dict(d), ws / "x.jsonl", PayloadStore(ws / "p"))
    with pytest.raises(SpecError):
        d2 = json.loads(json.dumps(pe_spec().raw)); d2["controller"]["memory"] = "perfect_recall"
        TrialSpec.from_dict(d2)
    with pytest.raises(ValueError):
        d3 = json.loads(json.dumps(pe_spec().raw)); d3["scenario"]["skin"] = "plain"
        TrialSpec.from_dict(d3)


# ---- WI-010: calendar skin (baked assets) as a real stimulus --------------------------------------
def cal_spec(memory="none", effect_delay=2_000_000_000, policy="live", seed=400,
             compute=300_000_000) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "pending_effect", "scenario_seed": seed, "canvas_w_px": 640,
                     "canvas_h_px": 480, "target_size_px": 48, "event_time_ns": 500_000_000,
                     "response_window_ns": 16_000_000_000, "frame_period_ns": PERIOD,
                     "episode_horizon_ns": 17_000_000_000, "irrelevant_motion": False,
                     "skin": "calendar", "effect_delay_ns": effect_delay}
    d["controller"] = {"kind": "scripted_pending", "compute_ns": compute, "memory": memory,
                       "pending_timeout_ns": 10_000_000_000, "pending_radius_px": 10}
    d["delivery"] = {"route": "simulated_fixed_delay", "delivery_delay_ns": DEL, "max_attempts": 1}
    d["task_clock_policy"] = policy
    return TrialSpec.from_dict(d)


def test_calendar_assets_match_target_box():
    """Every baked seed: the button-blue pixels are exactly the manifest's target box; the done and
    plain states contain no button-blue at all."""
    for seed in (100, 105, 109, 400, 405, 409):
        m = cal_spec(seed=seed).manifest()
        b, key, w = m.target_box(), bytes((37, 99, 235)), m.canvas_w_px
        data = m.render_env(6_000_000_000, None)
        pts = [(i % w, i // w) for i in range(w * m.canvas_h_px) if data[i * 3:i * 3 + 3] == key]
        assert pts, f"seed {seed}: no button pixels"
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        assert (min(xs), min(ys), max(xs) + 1, max(ys) + 1) == (b.x0, b.y0, b.x1, b.y1), f"seed {seed}"
        for state_bytes in (m.render_env(14_000_000_000, 5_800_000_000),  # done
                            m.render_env(16_900_000_000, None)):          # window closed, no toast
            assert key not in state_bytes or all(
                state_bytes[i:i+3] != key for i in range(0, len(state_bytes) - 2, 3)), f"seed {seed}: stray blue"


def test_calendar_scripted_repeater_duplicates_like_notification(ws):
    """Same mechanism on the new pixels: memoryless scripted repeater duplicates while the effect is
    pending; harness_pending suppresses them with the first action unchanged."""
    t0, r0 = run(cal_spec("none"), ws, "cal_none")
    t1, r1 = run(cal_spec("harness_pending"), ws, "cal_pend")
    assert r0.valid_trial and r1.valid_trial and not r0.contract_violations and not r1.contract_violations
    assert r0.accepted and r1.accepted
    assert r0.duplicates > 0 and r1.duplicates == 0 and r1.n_suppressed > 0
    assert r0.first_valid_E_A_ns == r1.first_valid_E_A_ns


def test_calendar_paused_gate_uses_single_response_task(ws):
    """Gate cell (WI-010): multi-action tasks are Live-only (enforced by the runner), so the Paused
    competence gate runs the single-response revealed_target task on the same baked calendar pixels.
    A scripted oracle clicks the frozen frame's button; the trial is valid and the click accepted."""
    with pytest.raises(ValueError, match="Live only"):
        run(cal_spec("none", policy="paused"), ws, "cal_paused_rejected")
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "revealed_target", "scenario_seed": 400, "canvas_w_px": 640,
                     "canvas_h_px": 480, "target_size_px": 48, "event_time_ns": 500_000_000,
                     "response_window_ns": 16_000_000_000, "frame_period_ns": PERIOD,
                     "episode_horizon_ns": 17_000_000_000, "irrelevant_motion": False, "skin": "calendar"}
    d["task_clock_policy"] = "paused"
    spec = TrialSpec.from_dict(d)
    p = run_trial(spec, ws / "cal_gate.jsonl", PayloadStore(ws / "p"))
    t = Trace.load(p)
    r = t.first("receipt")
    assert r is not None and r["valid_at_receipt"], "oracle click on frozen calendar frame must be valid"
