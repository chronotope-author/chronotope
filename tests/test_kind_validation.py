"""V1 (cards/V1-action-kind-validation.md): the action-kind validator on the binding controllers, against a mock VLM
on the wall clock. Precision: on the static and no-motion controls the validator never discards. Recall: a wait whose
decision ended after a frame showing the target was published is discarded and a new decision starts from that frame;
a type(value) into a field that vanished by D (rotating_code kind_change) is discarded and the re-decision clicks the
dismiss button, where late_bind and restart_if_changed commit the stale value (labelled outdated_at_commitment)."""
from __future__ import annotations

import time

import pytest

from chronotope.controller import CallBudget, RawInvocation
from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict
from test_vlm_mock import answer

W, H, TS = 320, 240, 24
MS = 1_000_000


class SlowMock:
    """Answers after `delay_s`; accepts the schema keyword the value binder passes."""

    def __init__(self, responder, delay_s: float = 0.0) -> None:
        self.calls, self.responder, self.delay_s = [], responder, delay_s

    def spec_fields(self):
        return {"server": "mock", "model_id": "mock-model"}

    def invoke(self, png, system, user_text, **kw) -> RawInvocation:
        self.calls.append((png, system, user_text, kw))
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.responder(len(self.calls))


def click_spec(scenario: dict, *, validate: bool, policy: str = "frozen", prompt: int = 9) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = scenario
    d["controller"] = {"kind": "vlm_late_binding", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
                       "model_id": "mock-model", "prompt_version": prompt, "image_scale": 2, "max_tokens": 64,
                       "temperature": 0.0, "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change",
                       "at_commit_policy": policy, "template_radius_px": 12, "min_match_score": 0.8,
                       "restart_threshold_px": None}
    if validate:
        d["controller"]["validate_kind"] = True
    d["timestamp_source"] = "monotonic_ns"
    return TrialSpec.from_dict(d)


def value_spec(seed: int, *, validate: bool, policy: str, kind_change: bool = True, change_ms: int = 200) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "rotating_code", "scenario_seed": seed, "canvas_w_px": W, "canvas_h_px": H,
                     "target_size_px": TS, "event_time_ns": 100 * MS, "response_window_ns": 2_000 * MS,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 2_600 * MS, "irrelevant_motion": False,
                     "skin": "notification", "change_delay_ns": change_ms * MS, "code_digits": 4}
    if kind_change:
        d["scenario"]["kind_change"] = True
    d["controller"] = {"kind": "vlm_value_binding", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
                       "model_id": "mock-model", "prompt_version": 13, "image_scale": 2, "max_tokens": 64,
                       "temperature": 0.0, "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change",
                       "at_commit_policy": policy}
    if validate:
        d["controller"]["validate_kind"] = True
    d["timestamp_source"] = "monotonic_ns"
    return TrialSpec.from_dict(d)


def player(seed=5, event_ms=100, window_ms=1000, skin="player") -> dict:
    return {"scenario_id": "sponsor_skip", "scenario_seed": seed, "canvas_w_px": W, "canvas_h_px": H, "target_size_px": TS,
            "event_time_ns": event_ms * MS, "response_window_ns": window_ms * MS, "frame_period_ns": 16_666_667,
            "episode_horizon_ns": (event_ms + window_ms + 300) * MS, "irrelevant_motion": False, "skin": skin,
            "seek_delay_ns": 0}


def notification(seed=3, event_ms=100, window_ms=1000, irrelevant=False) -> dict:
    return {"scenario_id": "revealed_target", "scenario_seed": seed, "canvas_w_px": W, "canvas_h_px": H, "target_size_px": TS,
            "event_time_ns": event_ms * MS, "response_window_ns": window_ms * MS, "frame_period_ns": 16_666_667,
            "episode_horizon_ns": (event_ms + window_ms + 300) * MS, "irrelevant_motion": irrelevant, "skin": "notification"}


def norm(spec: TrialSpec, t_ns: int | None = None) -> dict:
    m = spec.manifest()
    cx, cy = m.target_box_at(t_ns if t_ns is not None else m.event_time_ns).center()
    return {"action": "click", "x": cx * 1000 // W + 1, "y": cy * 1000 // H + 1}


def run(spec, ws, adapter, name="t", budget=8):
    p = run_trial(spec, ws / f"{name}.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=adapter, budget=CallBudget(budget)))
    t = Trace.load(p)
    return t, evaluate_trace(t)


def discards(t: Trace) -> list[dict]:
    return [e for e in t.all("pending_event") if e["event"] == "proposal_discarded"]


# ---- recall: an outdated wait on the animated player is discarded; the next decision starts from the newer frame
def test_outdated_wait_is_discarded_and_redecided(ws):
    spec = click_spec(player(), validate=True)
    click = norm(spec)
    ad = SlowMock(lambda n: answer({"action": "wait"} if n == 1 else click), 0.3)   # the sponsor starts during call 1
    t, r = run(spec, ws, ad)
    d = discards(t)
    assert len(d) == 1 and d[0]["reason"] == "kind" and d[0]["proposal_kind"] == "wait" and d[0]["expected_kind"] == "click"
    ob = next(o for o in t.all("observation") if o["observation_id"] == d[0]["observation_id"])
    end1 = next(i for i in t.all("decision_invocation") if i["decision_id"] == 1)["end_time_ns"]
    assert ob["publication_time_ns"] <= end1                       # the contradicting frame was available at that D
    assert t.first("commit")["decision_id"] == 2 and t.committing_decision()["observation_id"] == d[0]["observation_id"]
    assert r.valid_trial and r.valid_at_receipt is True and r.contract_violations == []


def test_recorded_controller_takes_the_same_path_without_the_record(ws):
    """The sequential runner already starts the next decision on the newest frame after a wait: the validator
    only makes the discard explicit (LEDGER, V1 finding)."""
    spec = click_spec(player(), validate=False)
    click = norm(spec)
    t, r = run(spec, ws, SlowMock(lambda n: answer({"action": "wait"} if n == 1 else click), 0.3))
    assert discards(t) == [] and t.first("commit")["decision_id"] == 2 and r.valid_at_receipt is True


# ---- recall: a click whose target is gone by D is discarded (restart with reason kind), never committed stale
def test_click_on_vanished_target_is_discarded(ws):
    spec = click_spec(notification(window_ms=200), validate=True)   # window closes at 0.3 s, the call takes 0.5 s
    click = norm(spec)
    t, r = run(spec, ws, SlowMock(lambda n: answer(click if n == 1 else {"action": "wait"}), 0.5))
    rs = t.all("decision_restarted")
    assert len(rs) == 1 and rs[0]["reason"] == "kind" and rs[0]["proposal_kind"] == "click" and rs[0]["expected_kind"] == "wait"
    assert len(discards(t)) == 1 and t.first("commit") is None and r.valid_trial and r.valid_at_receipt is False


# ---- precision: static target, no motion, static player -> never a discard
@pytest.mark.parametrize("case", ["static_click", "no_motion_balloon", "player_static"])
def test_validator_never_discards_on_controls(ws, case):
    if case == "static_click":
        spec = click_spec(notification(), validate=True, prompt=4)
    elif case == "no_motion_balloon":
        sc = {**notification(), "scenario_id": "moving_target", "skin": "balloon", "velocity_x_px_per_s": 0, "velocity_y_px_per_s": 0}
        spec = click_spec(sc, validate=True, policy="late_bind", prompt=5)
    else:
        spec = click_spec(player(skin="player_static"), validate=True)
    t, r = run(spec, ws, SlowMock(lambda n: answer(norm(spec)), 0.2), budget=50)
    assert discards(t) == [] and t.all("decision_restarted") == []
    assert r.valid_trial and r.valid_at_receipt is True and r.contract_violations == []


# ---- identity with the evaluator's rule: a wait is discarded iff a frame showing the target was published before
# that decision ended. On a stream with pre-event motion (the blinking corner) the deliberating mock is invoked
# back-to-back, so exactly one wait straddles the appearance; competent waits before it are never discarded.
def test_discards_equal_outdated_waits_on_a_moving_pre_event_stream(ws):
    spec = click_spec(notification(event_ms=600, irrelevant=True), validate=True, prompt=4)
    t0 = [None]

    def resp(n):   # wait on frames before E, click once the call begins after E (a competent model's answers)
        t0[0] = t0[0] or time.monotonic()
        return answer(norm(spec) if time.monotonic() - t0[0] > 0.62 else {"action": "wait"})
    t, r = run(spec, ws, SlowMock(resp, 0.2), budget=50)
    obs = t.all("observation")
    straddled = []
    for inv in t.all("decision_invocation"):
        if inv.get("outcome") != "wait":
            continue
        if any(o["source_event_id"] == 1 and o["publication_time_ns"] <= inv["end_time_ns"] for o in obs):
            straddled.append(inv["decision_id"])
    d = discards(t)
    assert len(straddled) == 1 and len(d) == 1 and d[0]["proposal_kind"] == "wait"
    assert t.first("commit") is not None and r.valid_at_receipt is True and r.contract_violations == []


# ---- rotating_code kind_change: type -> click; the validator re-decides, the binders commit the stale value
def typed_then_click(spec: TrialSpec):
    m = spec.manifest()
    a, _ = m.codes()
    cx, cy = m.dismiss_box().center()
    return lambda n: answer({"action": "type", "value": a} if n == 1 else
                            {"action": "click", "x": cx * 1000 // W + 1, "y": cy * 1000 // H + 1})


def test_kind_change_validator_redecides_to_click(ws):
    spec = value_spec(11, validate=True, policy="late_bind")
    t, r = run(spec, ws, SlowMock(typed_then_click(spec), 0.4))    # the code expires at 0.3 s, during call 1
    d = discards(t); rs = t.all("decision_restarted")
    assert len(d) == 1 and d[0]["proposal_kind"] == "type" and d[0]["expected_kind"] == "click"
    assert len(rs) == 1 and rs[0]["reason"] == "kind" and rs[0]["decision_id"] == 1
    c = t.first("commit")
    assert c["decision_id"] == 2 and c["action"]["kind"] == "click"
    assert r.valid_trial and r.valid_at_receipt is True and r.contract_violations == [] and r.n_restarts == 1


@pytest.mark.parametrize("policy", ["late_bind", "restart_if_changed", "frozen"])
def test_kind_change_binders_commit_the_stale_value(ws, policy):
    spec = value_spec(11, validate=False, policy=policy)
    t, r = run(spec, ws, SlowMock(typed_then_click(spec), 0.4))
    c = t.first("commit")
    assert c["decision_id"] == 1 and c["action"]["kind"] == "type" and discards(t) == []
    corr = t.first("coordinate_correction")
    if policy == "late_bind":
        assert corr is not None and corr["matched"] is False and corr["corrected_action"] == corr["proposed_action"]
    assert r.valid_trial and r.valid_at_receipt is False
    assert "outdated_at_commitment" in r.labels and "semantic_error" not in r.labels   # the field vanished before D
    assert r.contract_violations == []


def test_kind_change_manifest_contract():
    m = value_spec(11, validate=False, policy="frozen").manifest()
    a, _ = m.codes()
    cx, cy = m.dismiss_box().center()
    t_before, t_after = m.event_time_ns + 50 * MS, m.change_time_ns() + 50 * MS
    assert m.action_valid_at(t_before, {"kind": "type", "value": a}) and not m.action_valid_at(t_after, {"kind": "type", "value": a})
    assert m.action_valid_at(t_after, {"kind": "click", "x": cx, "y": cy}) and not m.action_valid_at(t_before, {"kind": "click", "x": cx, "y": cy})
    assert m.validate_action({"kind": "click", "x": cx, "y": cy}) is None
    assert m.code_at(t_after) is None and [e.kind for e in m.events()] == ["target_appears", "code_expires", "window_closes"]
    plain = value_spec(11, validate=False, policy="frozen", kind_change=False).manifest()
    assert plain.validate_action({"kind": "click", "x": cx, "y": cy}) is not None   # not in the plain action space


# ---- spec contract: the options are optional (absent = off), unknown fields are still rejected
def test_spec_options_are_optional_and_hash_stable():
    on = value_spec(11, validate=True, policy="late_bind")
    off = value_spec(11, validate=False, policy="late_bind", kind_change=False)
    assert on.manifest().kind_change and not off.manifest().kind_change
    d = dict(off.raw); d["scenario"] = {**d["scenario"], "kind_change": False}
    assert TrialSpec.from_dict(d).manifest().manifest_hash() == off.manifest().manifest_hash()
    with pytest.raises(SpecError):
        d2 = dict(on.raw); d2["controller"] = {**d2["controller"], "validate_kinds": True}
        TrialSpec.from_dict(d2)
