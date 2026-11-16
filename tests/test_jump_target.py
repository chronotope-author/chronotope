"""WI-009: jump_target (pre-/post-commit discrete change). Simulated clock, oracle bindings, closed form;
plus the mock late-binding agent re-finding a jumped balloon."""
from __future__ import annotations

import time

import pytest

from chronotope.__main__ import run_triplet
from chronotope.controller import CallBudget
from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.spec import TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict
from test_late_binding import SlowMock, answer
from test_vlm_mock import MockAdapter  # noqa: F401

PUB, PERIOD, DEL = 30_000_000, 16_666_667, 60_000_000


def jt_spec(jump_delay=300_000_000, compute=500_000_000, delivery=DEL, binding="state_at_B", seed=7,
            window=1_500_000_000, dx=40, dy=0) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "jump_target", "scenario_seed": seed, "canvas_w_px": 96, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": 500_000_000, "response_window_ns": window,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": 2_500_000_000, "irrelevant_motion": False,
                     "skin": "plain", "jump_delay_ns": jump_delay, "jump_dx_px": dx, "jump_dy_px": dy}
    d["controller"].update({"compute_ns": compute, "binding": binding,
                            "expected_observation_delay_ns": PUB if binding == "predicted_at_A" else None,
                            "expected_delivery_delay_ns": delivery if binding == "predicted_at_A" else None})
    d["delivery"]["delivery_delay_ns"] = delivery
    return TrialSpec.from_dict(d)


def run(spec, ws, name="t"):
    p = run_trial(spec, ws / f"{name}.jsonl", PayloadStore(ws / "p"))
    t = Trace.load(p)
    return t, evaluate_trace(t)


def test_jump_is_an_event_and_moves_the_box():
    m = jt_spec().manifest()
    assert [e.event_id for e in m.events()] == [1, 4, 2]
    b0, b1 = m.target_box_at(600_000_000), m.target_box_at(900_000_000)
    assert (b1.x0 - b0.x0, b1.y0 - b0.y0) == (40, 0)
    assert m.state_at(900_000_000).latest_event_id == 4 and m.state_at(600_000_000).latest_event_id == 1


# U18 (a): jump observed before D, action bound at B -> outdated_at_commitment, invalid at A
def test_jump_before_D_with_state_at_B_is_outdated(ws):
    # B ≈ 0.53 s (first frame + 30 ms), D = B + 0.5 s ≈ 1.03 s, jump at 0.8 s -> observed before D
    t, r = run(jt_spec(jump_delay=300_000_000, compute=500_000_000, binding="state_at_B"), ws)
    assert r.policy == "live" and r.valid_at_receipt is False
    assert "outdated_at_commitment" in r.labels and "invalidated_after_commitment" not in r.labels
    assert r.semantic_correct is True          # right for the state it was bound to
    assert r.contract_violations == []


# U18 (b): jump in (D, A] -> invalidated_after_commitment, not outdated
def test_jump_between_D_and_A_is_invalidated_after_commitment(ws):
    # D ≈ 0.53 + 0.1 = 0.63 s; delivery 300 ms -> A ≈ 0.93 s; jump at 0.8 s
    t, r = run(jt_spec(jump_delay=300_000_000, compute=100_000_000, delivery=300_000_000, binding="state_at_B"), ws)
    assert r.valid_at_receipt is False
    assert "invalidated_after_commitment" in r.labels and "outdated_at_commitment" not in r.labels


# U18 (c): jump after A -> nothing happens
def test_jump_after_A_is_inert(ws):
    t, r = run(jt_spec(jump_delay=1_400_000_000, compute=100_000_000, binding="state_at_B"), ws)
    assert r.valid_at_receipt is True and r.labels == []


# U19: bindings with matched B, D, A: newest_at_D and true_at_A succeed; state_at_B fails; predictor does not help
def test_binding_ordering_on_a_pre_commit_jump(ws):
    res = {}
    for b in ("state_at_B", "newest_at_D", "predicted_at_A", "true_at_A"):
        t, r = run(jt_spec(binding=b), ws, b)
        res[b] = r
        c = t.first("commit"); rcp = t.first("receipt")
        res[b + "_D"], res[b + "_A"] = c["commitment_time_ns"], rcp["receipt_time_ns"]
    assert res["state_at_B_D"] == res["newest_at_D_D"] == res["predicted_at_A_D"] == res["true_at_A_D"]
    assert res["state_at_B_A"] == res["newest_at_D_A"] == res["predicted_at_A_A"] == res["true_at_A_A"]
    assert res["state_at_B"].valid_at_receipt is False
    assert res["newest_at_D"].valid_at_receipt is True and res["true_at_A"].valid_at_receipt is True
    # the velocity extrapolator sees one frame of 'motion' (the jump) and overshoots: no help
    assert res["predicted_at_A"].valid_at_receipt is False
    assert res["newest_at_D"].binding == "newest_at_D" and res["predicted_at_A"].binding == "predicted_at_A"


def test_bindings_tie_when_jump_is_after_A(ws):
    outs = {b: run(jt_spec(binding=b, jump_delay=1_400_000_000, compute=100_000_000), ws, b)[1].valid_at_receipt
            for b in ("state_at_B", "newest_at_D", "true_at_A")}
    assert all(outs.values())


def test_triplet_replay_matches_live_evidence_with_a_jump(ws):
    out = run_triplet(jt_spec(), ws)
    assert out["pair"].valid
    assert out["results"]["paused"].valid_at_receipt is True     # the triggering state is held valid
    assert out["results"]["replay"].valid_at_receipt is True     # same frames, frozen validity
    assert out["results"]["live"].valid_at_receipt is False and "outdated_at_commitment" in out["results"]["live"].labels


# U20: the mock late-binding agent re-finds the balloon after a 96-px jump during B->D
def test_late_bind_refinds_jumped_balloon(ws):
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "jump_target", "scenario_seed": 3, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 100_000_000, "response_window_ns": 1_500_000_000,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": 2_000_000_000, "irrelevant_motion": False,
                     "skin": "balloon", "jump_delay_ns": 250_000_000, "jump_dx_px": 96, "jump_dy_px": 0}
    d["controller"] = {"kind": "vlm_late_binding", "server": "openai_compatible_local",
                       "base_url": "http://127.0.0.1:8000/v1", "model_id": "mock-model", "prompt_version": 3,
                       "image_scale": 2, "max_tokens": 64, "temperature": 0.0, "enable_thinking": False,
                       "reasoning_effort": None, "invoke_on": "frame_change", "at_commit_policy": "late_bind",
                       "template_radius_px": 12, "min_match_score": 0.8, "restart_threshold_px": None}
    d["timestamp_source"] = "monotonic_ns"
    d["delivery"]["delivery_delay_ns"] = 20_000_000
    spec = TrialSpec.from_dict(d)
    cx, cy = spec.manifest().target_box().center()
    ad = SlowMock(lambda n: answer({"action": "click", "x": 2 * (cx + 2), "y": 2 * (cy - 3)}), 0.5)
    p = run_trial(spec, ws / "lb.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(3)))
    t = Trace.load(p); r = evaluate_trace(t)
    corr = t.first("coordinate_correction")
    assert corr["matched"] and abs(corr["moved_px"] - 96) < 3
    assert r.valid_at_receipt is True and r.proposed_valid_at_receipt is False
    assert r.contract_violations == []


def test_spec_rejects_small_or_offcanvas_jump():
    with pytest.raises(ValueError):
        jt_spec(dx=3)
    with pytest.raises(ValueError):
        jt_spec(dx=200)
