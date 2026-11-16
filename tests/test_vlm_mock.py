"""VLM direct controller against a mock adapter (no network, no model): invocation policy,
coordinate mapping, wait/unparseable handling, API-failure handling, and the per-kind schema."""
from __future__ import annotations

import pytest

from chronotope.__main__ import run_triplet
from chronotope.controller import ACTION_JSON_SCHEMA, CallBudget, RawInvocation
from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict


def vlm_spec(**over) -> dict:
    d = base_spec_dict(**over)
    d["controller"] = {"kind": "vlm_direct", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
                       "model_id": "mock-model", "prompt_version": 1, "image_scale": 10, "max_tokens": 64,
                       "temperature": 0.0, "enable_thinking": False, "reasoning_effort": None,
                       "invoke_on": "frame_change"}
    d["timestamp_source"] = "monotonic_ns"
    d["scenario"]["event_time_ns"] = 100_000_000
    d["scenario"]["response_window_ns"] = 1_000_000_000
    d["scenario"]["episode_horizon_ns"] = 1_500_000_000
    return d


class MockAdapter:
    """Stands in for LocalOpenAIAdapter: records every invocation; answers via `responder`."""

    def __init__(self, responder):
        self.calls = []
        self.responder = responder

    def spec_fields(self):
        return {"server": "mock", "model_id": "mock-model", "schema": ACTION_JSON_SCHEMA}

    def invoke(self, png, system, user_text):
        self.calls.append((png, system, user_text))
        return self.responder(len(self.calls))


def answer(obj):
    return RawInvocation(parsed=obj, invocation_id="req_mock",
                         detail={"model_reported": "mock-model", "finish_reason": "stop", "input_tokens": 10,
                                 "output_tokens": 5, "reasoning_tokens": None, "raw_text": str(obj)})


def test_vlm_clicks_center_once_on_frame_change(ws):
    spec = TrialSpec.from_dict(vlm_spec())
    cx, cy = spec.manifest().target_box().center()
    ad = MockAdapter(lambda n: answer({"action": "click", "x": cx * 10 + 5, "y": cy * 10 + 5}))
    p = run_trial(spec, ws / "v.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(5)))
    t = Trace.load(p)
    r = evaluate_trace(t)
    assert len(ad.calls) == 1                                  # invoked once: on the first CHANGED frame
    assert ad.calls[0][0][:8] == b"\x89PNG\r\n\x1a\n"           # a real PNG went to the model
    assert "640x480" in ad.calls[0][1]                          # prompt states the upscaled image size
    assert r.valid_trial and r.valid_at_receipt and r.semantic_correct and r.n_decisions == 1
    inv = t.first("decision_invocation")
    assert inv["invocation_id"] == "req_mock" and inv["outcome"] == "click" and inv["input_tokens"] == 10
    assert t.committing_decision()["observation_id"] == inv["observation_id"]


def test_vlm_wait_then_click_is_two_decisions(ws):
    d = vlm_spec(scenario__irrelevant_motion=True)  # blinking distractor: first change is NOT the target
    spec = TrialSpec.from_dict(d)
    cx, cy = spec.manifest().target_box().center()
    ad = MockAdapter(lambda n: answer({"action": "wait"}) if n < 3 else answer({"action": "click", "x": cx * 10, "y": cy * 10}))
    p = run_trial(spec, ws / "w.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(50)))
    t = Trace.load(p)
    r = evaluate_trace(t)
    assert len(t.all("decision_no_action")) == 2 and t.first("commit")["decision_id"] == 3
    assert r.n_decisions == 3 and r.valid_at_receipt is True
    assert [x["decision_id"] for x in t.all("decision_begin")] == [1, 2, 3] and not r.contract_violations


def test_vlm_unparseable_or_incomplete_answers_are_no_action_decisions(ws):
    spec = TrialSpec.from_dict(vlm_spec())
    cx, cy = spec.manifest().target_box().center()
    seq = [answer(None), answer({"action": "click"}), answer({"action": "click", "x": cx * 10, "y": cy * 10})]
    d = vlm_spec(scenario__irrelevant_motion=True)
    spec = TrialSpec.from_dict(d)
    cx, cy = spec.manifest().target_box().center()
    seq = [answer(None), answer({"action": "click"}), answer({"action": "click", "x": cx * 10, "y": cy * 10})]
    ad = MockAdapter(lambda n: seq[min(n, len(seq)) - 1])
    t = Trace.load(run_trial(spec, ws / "u.jsonl", PayloadStore(ws / "p"),
                             controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(50))))
    outcomes = [i["outcome"] for i in t.all("decision_invocation")]
    assert outcomes[:2] == ["unparseable", "click_without_coordinates"] and outcomes[-1] == "click"
    assert evaluate_trace(t).valid_at_receipt is True


def test_vlm_server_error_is_invalid_trial_not_agent_failure(ws):
    spec = TrialSpec.from_dict(vlm_spec())

    def boom(n):
        raise ConnectionError("simulated server down")
    p = run_trial(spec, ws / "e.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=MockAdapter(boom), budget=CallBudget(5)))
    r = evaluate_trace(Trace.load(p))
    assert not r.valid_trial and r.invalid_reason.startswith("decision_failed:ConnectionError")


def test_vlm_budget_exhaustion_is_invalid_trial(ws):
    spec = TrialSpec.from_dict(vlm_spec())
    ad = MockAdapter(lambda n: answer({"action": "click", "x": 0, "y": 0}))
    p = run_trial(spec, ws / "b.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(0)))
    r = evaluate_trace(Trace.load(p))
    assert not r.valid_trial and "budget" in r.invalid_reason and ad.calls == []


def test_vlm_triplet_replay_is_a_rerun_with_same_evidence(ws):
    spec = TrialSpec.from_dict(vlm_spec())
    cx, cy = spec.manifest().target_box().center()
    ad = MockAdapter(lambda n: answer({"action": "click", "x": cx * 10, "y": cy * 10}))
    out = run_triplet(spec, ws, controller_factory=lambda s: build_controller(s, vlm_adapter=ad, budget=CallBudget(10)))
    assert out["pair"].valid and out["pair"].same_evidence_at_decision
    assert len(ad.calls) == 3                                  # one real invocation per condition
    assert all(out["results"][p].valid_at_receipt for p in ("paused", "replay", "live"))


def test_vlm_spec_schema_is_strict():
    with pytest.raises(SpecError, match="monotonic"):
        d = vlm_spec(); d["timestamp_source"] = "simulated"; TrialSpec.from_dict(d)
    with pytest.raises(SpecError, match="unknown"):
        d = vlm_spec(); d["controller"]["top_p"] = 0.9; TrialSpec.from_dict(d)
    with pytest.raises(SpecError, match="missing"):
        d = vlm_spec(); del d["controller"]["enable_thinking"]; TrialSpec.from_dict(d)
    with pytest.raises(SpecError, match="expected float"):
        d = vlm_spec(); d["controller"]["temperature"] = 0; TrialSpec.from_dict(d)   # int is not float
    with pytest.raises(SpecError, match="local server"):
        d = vlm_spec(); d["controller"]["base_url"] = "https://api.example.com/v1"; TrialSpec.from_dict(d)
    with pytest.raises(SpecError, match="not in"):
        d = vlm_spec(); d["controller"]["kind"] = "vlm_streaming"; TrialSpec.from_dict(d)
