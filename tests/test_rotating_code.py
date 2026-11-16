"""C4 (cards/C4-late-bound-text.md): a NON-SPATIAL late-bound field. The rotating_code manifest, the value decoder,
and the value-binding controller against a slow mock VLM: frozen types the code it saw at B (outdated at
commitment once the code changed), late binding fills the value from the newest frame at D under the same decision
id, restart_if_changed discards and decides again."""
from __future__ import annotations

import time

import pytest

from chronotope.controller import CallBudget, RawInvocation, VALUE_JSON_SCHEMA
from chronotope.decoder import decode
from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.scenario import manifest_from_dict
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict

W, H = 320, 240
SC = {"scenario_id": "rotating_code", "scenario_seed": 1000, "canvas_w_px": W, "canvas_h_px": H, "target_size_px": 24,
      "event_time_ns": 100_000_000, "response_window_ns": 3_000_000_000, "frame_period_ns": 16_666_667,
      "episode_horizon_ns": 3_600_000_000, "irrelevant_motion": False, "skin": "notification",
      "change_delay_ns": 400_000_000, "code_digits": 4}


def vb_spec(policy="late_bind", change_ns=400_000_000, seed=1000) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {**SC, "change_delay_ns": change_ns, "scenario_seed": seed}
    d["controller"] = {"kind": "vlm_value_binding", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
                       "model_id": "mock-model", "prompt_version": 11, "image_scale": 2, "max_tokens": 64, "temperature": 0.0,
                       "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change", "at_commit_policy": policy}
    d["timestamp_source"] = "monotonic_ns"
    return TrialSpec.from_dict(d)


class ReadingMock:
    """Answers with the code it can read in the frame it was given (the ideal model), after `delay_s`."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s, self.calls = delay_s, []

    def spec_fields(self):
        return {"server": "mock", "model_id": "mock-model", "schema": VALUE_JSON_SCHEMA}

    def invoke(self, png, system, user_text, schema=None):
        import io
        from PIL import Image
        self.calls.append((system, schema))
        im = Image.open(io.BytesIO(png)).convert("RGB")
        time.sleep(self.delay_s)
        im = im.resize((W, H), Image.NEAREST)          # the PNG is the 2x pixel-replicated frame
        v = decode(im.tobytes(), W, H).value
        obj = {"action": "type", "value": v} if v else {"action": "wait"}
        return RawInvocation(parsed=obj, invocation_id="req_mock",
                             detail={"model_reported": "mock-model", "finish_reason": "stop", "input_tokens": 1,
                                     "output_tokens": 1, "reasoning_tokens": None, "raw_text": str(obj)})


def test_manifest_codes_events_validity_and_return():
    m = manifest_from_dict(SC)
    a, b = m.codes()
    assert a != b and len(a) == len(b) == 4 and a.isdigit()
    assert m.code_at(50_000_000) is None and m.code_at(200_000_000) == a and m.code_at(600_000_000) == b
    assert [e.kind for e in m.events()] == ["target_appears", "code_changes", "window_closes"]
    assert m.action_valid_at(200_000_000, {"kind": "type", "value": a}) and not m.action_valid_at(600_000_000, {"kind": "type", "value": a})
    assert m.validate_action({"kind": "click", "x": 1, "y": 1}) is not None
    assert m.validate_action({"kind": "type", "value": "12a4"}) is not None
    assert m.return_on_receipt(600_000_000, {"kind": "type", "value": b}) == pytest.approx(-0.5)
    assert m.return_on_receipt(600_000_000, {"kind": "type", "value": a}) == pytest.approx(-3.0)
    assert m.return_no_receipt() == pytest.approx(-3.0)
    # never changes when the delay exceeds the window
    m2 = manifest_from_dict({**SC, "change_delay_ns": 10_000_000_000})
    assert [e.kind for e in m2.events()] == ["target_appears", "window_closes"] and m2.code_at(2_900_000_000) == m2.codes()[0]


def test_decoder_reads_every_seed_and_nothing_before_appearance():
    for seed in range(1000, 1040):
        m = manifest_from_dict({**SC, "scenario_seed": seed})
        for t in (200_000_000, 600_000_000):
            d = decode(m.render(t), W, H)
            assert d.value == m.code_at(t) and d.hamming == 0, (seed, t, d)
        assert decode(m.render(0), W, H).value is None


def test_spec_rejects_bad_policy_and_skin():
    with pytest.raises(SpecError):
        d = base_spec_dict(); d["scenario"] = {**SC, "skin": "balloon"}; TrialSpec.from_dict(d)
    d = base_spec_dict(); d["scenario"] = dict(SC)
    d["controller"] = {**vb_spec().raw["controller"], "at_commit_policy": "restart_if_moved"}; d["timestamp_source"] = "monotonic_ns"
    with pytest.raises(SpecError):
        TrialSpec.from_dict(d)


def run(policy, ws, delay_s=0.6, change_ns=400_000_000):
    spec = vb_spec(policy, change_ns)
    ad = ReadingMock(delay_s)
    p = run_trial(spec, ws / f"{policy}.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(5)))
    return evaluate_trace(Trace.load(p)), Trace.load(p), ad


def test_frozen_types_the_old_code_and_is_outdated_at_commitment(ws):
    r, t, ad = run("frozen", ws)
    a, b = vb_spec().manifest().codes()
    assert ad.calls[0][1] is VALUE_JSON_SCHEMA and "type" in ad.calls[0][0]
    assert t.first("commit")["action"] == {"kind": "type", "value": a}
    assert r.valid_trial and r.valid_at_receipt is False and r.semantic_correct is True
    assert "outdated_at_commitment" in r.labels and not r.contract_violations
    assert r.return_G == pytest.approx(-3.0)


def test_late_bind_fills_the_new_code_under_the_same_decision(ws):
    r, t, _ = run("late_bind", ws)
    a, b = vb_spec().manifest().codes()
    c = t.first("coordinate_correction")
    assert c["proposed_action"] == {"kind": "type", "value": a} and c["corrected_action"] == {"kind": "type", "value": b}
    assert c["binding"] == "decoded_at_D" and c["decoded_at_B"] == a and c["matched"] is True
    assert t.first("commit")["decision_id"] == c["decision_id"] and len(t.all("decision_begin")) == 1
    assert r.valid_at_receipt is True and r.binding == "decoded_at_D" and not r.contract_violations
    assert r.proposed_valid_at_receipt is False          # the frozen counterfactual at the same A
    assert r.return_G > -1.5 and r.n_restarts == 0


def test_restart_if_changed_decides_again_and_succeeds_later(ws):
    r, t, ad = run("restart_if_changed", ws)
    assert len(ad.calls) == 2 and r.n_restarts == 1 and r.valid_at_receipt is True
    assert t.first("decision_restarted")["reason"] == "value_changed_since_B"
    assert r.first_b_to_d_ns > r.b_to_d_ns


def test_no_change_all_policies_agree(ws):
    for pol in ("frozen", "late_bind", "restart_if_changed"):
        r, t, ad = run(pol, ws, change_ns=10_000_000_000)
        assert r.valid_at_receipt is True and len(ad.calls) == 1 and "outdated_at_commitment" not in r.labels, pol
        if pol == "late_bind":
            assert t.first("coordinate_correction")["moved_px"] == 0.0
