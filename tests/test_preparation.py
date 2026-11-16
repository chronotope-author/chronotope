"""A8-PREP (cards/A8-preparation.md): the fixed-layout option, the scripted `prepared` and `restart_if_changed`
bindings (simulated clock), the typed-value scripted oracle on rotating_code, and VLMPreparedController against a
mock (one pre-event call, release at first detection; 'not available' when the model waits)."""
from __future__ import annotations

import pytest

from chronotope.controller import CallBudget
from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, prepared_xy, run_trial
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from test_kind_validation import SlowMock, W, H, TS, MS, run
from test_vlm_mock import answer

S = 1_000_000_000


def notif(seed: int, window_ns: int, *, fixed: bool = False, event_ns: int = 500 * MS, skin: str = "notification",
          scenario_id: str = "revealed_target", **extra) -> dict:
    sc = {"scenario_id": scenario_id, "scenario_seed": seed, "canvas_w_px": W, "canvas_h_px": H, "target_size_px": TS,
          "event_time_ns": event_ns, "response_window_ns": window_ns, "frame_period_ns": 16_666_667,
          "episode_horizon_ns": event_ns + window_ns + 500 * MS, "irrelevant_motion": False, "skin": skin, **extra}
    if fixed:
        sc["fixed_layout"] = True
    return sc


def scripted(sc: dict, binding: str, compute_ns: int = 5 * S) -> TrialSpec:
    pred = binding == "predicted_at_A"
    return TrialSpec.from_dict({
        "schema_version": 1, "scenario": sc, "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": {"kind": "scripted_oracle", "compute_ns": compute_ns, "coordinate_offset_px": 0, "binding": binding,
                       "expected_observation_delay_ns": 30 * MS if pred else None,
                       "expected_delivery_delay_ns": 20 * MS if pred else None},
        "observation": {"boundary_version": 1, "publication_delay_ns": 30 * MS},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "simulated",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": 20 * MS, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision"})


def go(spec: TrialSpec, ws, name="t"):
    p = run_trial(spec, ws / f"{name}.jsonl", PayloadStore(ws / "p"))
    t = Trace.load(p)
    return t, evaluate_trace(t)


# ---- the fixed layout: one box for every seed; the seeded layout differs; hashes of old manifests unchanged
def test_fixed_layout_is_seed_independent_and_hash_stable():
    fixed = [scripted(notif(s, S, fixed=True), "state_at_B").manifest().target_box() for s in (1, 2, 3)]
    seeded = [scripted(notif(s, S), "state_at_B").manifest().target_box() for s in (1, 2, 3)]
    assert fixed[0] == fixed[1] == fixed[2] and len(set(seeded)) == 3 and fixed[0] not in seeded
    a = scripted(notif(1, S), "state_at_B").manifest()
    d = notif(1, S); d["fixed_layout"] = False
    assert scripted(d, "state_at_B").manifest().manifest_hash() == a.manifest_hash()
    assert scripted(notif(1, S, fixed=True), "state_at_B").manifest().manifest_hash() != a.manifest_hash()
    assert prepared_xy(a) == fixed[0].center()                       # the builder's prior = the skin's usual place


# ---- prepared succeeds at a 100-ms window where every 5-s deliberating binding fails
@pytest.mark.parametrize("binding", ["prepared", "state_at_B", "newest_at_D", "restart_if_changed", "predicted_at_A"])
def test_prepared_vs_deliberating_at_a_short_window(ws, binding):
    t, r = go(scripted(notif(7, 100 * MS, fixed=True), binding), ws, binding)
    c, d0 = t.first("commit"), t.committing_decision()
    if binding == "prepared":
        assert r.valid_at_receipt is True and r.labels == []
        assert c["commitment_time_ns"] == d0["begin_time_ns"]        # compute 0: D = B (the detecting frame's admission)
        assert t.first("decision_invocation")["invocation_id"] == "prepared"
        assert c["task_time_at_commit_ns"] - 500 * MS < 40 * MS      # E -> D within one frame + publication
    elif binding == "restart_if_changed":
        # the box at D (target gone) differs from the box at B: the proposal is discarded, nothing stale is committed
        rs = t.all("decision_restarted")
        assert c is None and len(rs) == 1 and rs[0]["reason"] == "box_changed_since_B" and r.valid_at_receipt is False
    else:
        assert r.valid_at_receipt is False and "late" in r.labels
        assert c["commitment_time_ns"] - d0["begin_time_ns"] == 5 * S
    assert r.valid_trial and r.contract_violations == []


def test_deliberating_bindings_succeed_at_a_generous_window(ws):
    for binding in ("state_at_B", "newest_at_D", "restart_if_changed", "predicted_at_A"):
        t, r = go(scripted(notif(7, 8 * S, fixed=True), binding), ws, binding)
        assert r.valid_at_receipt is True and r.n_restarts == 0 and r.contract_violations == []


# ---- prepared has the wrong content on a seeded layout and nothing to prepare on a revealed value
def test_prepared_fails_on_seeded_layout_and_late_bind_succeeds(ws):
    t, r = go(scripted(notif(far_seed(), 8 * S), "prepared"), ws, "p")
    assert r.valid_at_receipt is False and "semantic_error" in r.labels and r.action_committed
    t, r = go(scripted(notif(7, 8 * S), "newest_at_D"), ws, "lb")
    assert r.valid_at_receipt is True


def test_prepared_on_moving_balloon_fails(ws):
    sc = notif(7, 8 * S, skin="balloon", scenario_id="moving_target", velocity_x_px_per_s=8, velocity_y_px_per_s=0)
    t, r = go(scripted(sc, "prepared"), ws, "b")
    assert r.valid_at_receipt is False and r.action_committed


def rotating(seed: int, change_ns: int = 2 * S) -> dict:
    return notif(seed, 16 * S, change_delay_ns=change_ns, code_digits=4, scenario_id="rotating_code")


def test_scripted_value_oracle_on_rotating_code(ws):
    t, r = go(scripted(rotating(1000), "prepared"), ws, "prep")
    assert t.first("commit") is None and r.n_decisions == 0 and r.valid_at_receipt is False and r.return_G == -16.0
    t, r = go(scripted(rotating(1000), "newest_at_D"), ws, "lb")     # the code changes at 2.5 s, during B->D
    c, corr = t.first("commit"), t.first("coordinate_correction")
    assert c["action"] == {"kind": "type", "value": t.header()["manifest"] and scripted(rotating(1000), "newest_at_D").manifest().codes()[1]}
    assert corr["binding"] == "decoded_at_D" and r.valid_at_receipt is True and r.binding == "decoded_at_D"
    t, r = go(scripted(rotating(1000), "state_at_B"), ws, "fr")
    assert r.valid_at_receipt is False and "outdated_at_commitment" in r.labels


# ---- restart_if_changed: restarts once on a pre-commit jump, never on a static target
def test_scripted_restart_if_changed(ws):
    sc = {"scenario_id": "jump_target", "scenario_seed": 3, "canvas_w_px": 96, "canvas_h_px": 64, "target_size_px": 6,
          "event_time_ns": 500 * MS, "response_window_ns": 3 * S, "frame_period_ns": 16_666_667, "episode_horizon_ns": 4 * S,
          "irrelevant_motion": False, "skin": "plain", "jump_delay_ns": 300 * MS, "jump_dx_px": 40, "jump_dy_px": 0}
    t, r = go(scripted(sc, "restart_if_changed", compute_ns=500 * MS), ws, "j")
    rs = t.all("decision_restarted")
    assert len(rs) == 1 and rs[0]["reason"] == "box_changed_since_B" and t.first("commit")["decision_id"] == 2
    assert r.valid_at_receipt is True and r.n_restarts == 1
    t, r = go(scripted(notif(7, 8 * S), "restart_if_changed"), ws, "s")
    assert t.all("decision_restarted") == [] and r.valid_at_receipt is True


# ---- the model version: one pre-event call, release at first detection
def prepared_spec(sc: dict) -> TrialSpec:
    d = {"schema_version": 1, "scenario": sc, "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
         "controller": {"kind": "vlm_prepared", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
                        "model_id": "mock-model", "prompt_version": 14, "image_scale": 2, "max_tokens": 64, "temperature": 0.0,
                        "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change"},
         "observation": {"boundary_version": 1, "publication_delay_ns": 30 * MS},
         "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
         "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": 20 * MS, "max_attempts": 1},
         "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision"}
    return TrialSpec.from_dict(d)


def usual_place(spec: TrialSpec) -> dict:
    x, y = prepared_xy(spec.manifest())
    return {"action": "click", "x": x * 1000 // W + 1, "y": y * 1000 // H + 1}


def test_vlm_prepared_one_call_then_release_at_detection(ws):
    spec = prepared_spec(notif(4, 300 * MS, fixed=True, event_ns=300 * MS))
    ad = SlowMock(lambda n: answer(usual_place(spec)))
    t, r = run(spec, ws, ad)
    inv = t.all("decision_invocation")
    assert len(ad.calls) == 1 and "usual" not in ad.calls[0][1] and "always appears at the same place" in ad.calls[0][1]
    assert [i["outcome"] for i in inv] == ["prepared", "release_prepared"] and inv[1]["invocation_id"] == "prepared"
    assert t.all("decision_no_action")[0]["decision_id"] == 1 and t.first("commit")["decision_id"] == 2
    ob = next(o for o in t.all("observation") if o["observation_id"] == t.committing_decision()["observation_id"])
    assert ob["source_event_id"] == 1                                   # released on a frame showing the target
    assert t.first("commit")["commitment_time_ns"] - t.committing_decision()["begin_time_ns"] < 50 * MS
    assert r.valid_trial and r.valid_at_receipt is True and r.n_decisions == 2 and r.contract_violations == []


def far_seed() -> int:
    """A seed whose seeded box does not contain the skin's usual place (most do not; seed 4 happens to)."""
    for seed in range(1, 50):
        m = prepared_spec(notif(seed, 2 * S)).manifest()
        if not m.target_box().contains(*prepared_xy(m)):
            return seed
    raise AssertionError("no seed found")


def test_vlm_prepared_wrong_on_seeded_layout(ws):
    spec = prepared_spec(notif(far_seed(), 2 * S))
    t, r = run(spec, ws, SlowMock(lambda n: answer(usual_place(spec))))
    assert r.action_committed and r.valid_at_receipt is False and "semantic_error" in r.labels


def test_vlm_prepared_not_available_when_the_model_waits(ws):
    spec = prepared_spec(notif(4, 300 * MS, fixed=True, event_ns=300 * MS))
    ad = SlowMock(lambda n: answer({"action": "wait"}))
    t, r = run(spec, ws, ad)
    assert len(ad.calls) == 1 and t.first("commit") is None
    assert t.first("decision_invocation")["outcome"] == "prepare_unavailable" and t.first("trial_end")["reason"] == "horizon"
    assert r.valid_trial and r.valid_at_receipt is False


def test_vlm_prepared_requires_prompt_14():
    d = prepared_spec(notif(4, S, fixed=True)).raw
    d = {**d, "controller": {**d["controller"], "prompt_version": 4}}
    with pytest.raises(SpecError):
        TrialSpec.from_dict(d)
