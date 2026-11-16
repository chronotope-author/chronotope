"""Prompt 7: UI-TARS native grammar (WI-007 U21) — parsing, coordinate mapping, controller round trip."""
from chronotope.controller import (ACTION_JSON_SCHEMA, CallBudget, RawInvocation, SYSTEM_PROMPTS, UITARS_INSTRUCTION,
                                   parse_uitars, smart_resize)
from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.spec import TrialSpec
from chronotope.trace import Trace
from test_vlm_mock import vlm_spec


def test_parse_uitars_variants():
    assert parse_uitars("Thought: the button is here.\nAction: click(point='<point>265 393</point>')") == {"action": "click", "x": 265, "y": 393}
    assert parse_uitars("Action: click(start_box='(197,525)')") == {"action": "click", "x": 197, "y": 525}
    assert parse_uitars('Action: click(point="<point>10 20</point>")') == {"action": "click", "x": 10, "y": 20}
    assert parse_uitars("Thought: nothing yet.\nAction: wait()") == {"action": "wait"}
    assert parse_uitars("Action: finished(content='done')") is None
    assert parse_uitars("") is None and parse_uitars(None) is None


def test_smart_resize_matches_uitars_readme():
    assert smart_resize(480, 640) == (476, 644)      # our 320x240 canvas shown at scale 2
    assert smart_resize(1080, 1920) == (1092, 1932)
    h, w = smart_resize(3000, 3000, max_pixels=12845056)
    assert h * w <= 12845056 and h % 28 == 0 and w % 28 == 0


class TextMock:
    """Adapter double that returns UI-TARS text (parsed by the same function the real adapter uses)."""

    def __init__(self, text):
        self.text, self.calls = text, []

    def spec_fields(self):
        return {"server": "mock", "model_id": "mock-model", "schema": ACTION_JSON_SCHEMA}

    def invoke(self, png, system, user_text, structured=True):
        self.calls.append((system, structured))
        return RawInvocation(parsed=parse_uitars(self.text), invocation_id="req_mock",
                             detail={"model_reported": "mock-model", "finish_reason": "stop", "input_tokens": 1,
                                     "output_tokens": 1, "reasoning_tokens": None, "raw_text": self.text})


def test_prompt7_round_trip_on_notification(ws):
    d = vlm_spec()
    d["scenario"].update({"canvas_w_px": 320, "canvas_h_px": 240, "target_size_px": 24, "skin": "notification"})
    d["controller"].update({"prompt_version": 7, "image_scale": 2})
    spec = TrialSpec.from_dict(d)
    cx, cy = spec.manifest().target_box().center()
    # the model answers in the 644x476 smart-resized frame: x_r = x_img * 644/640
    xr, yr = round(cx * 2 * 644 / 640), round(cy * 2 * 476 / 480)
    ad = TextMock(f"Thought: click the blue button.\nAction: click(point='<point>{xr} {yr}</point>')")
    p = run_trial(spec, ws / "t.jsonl", PayloadStore(ws / "p"), controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(3)))
    r = evaluate_trace(Trace.load(p))
    assert ad.calls and ad.calls[0][1] is False                 # no structured output for the native grammar
    assert UITARS_INSTRUCTION["notification"] in ad.calls[0][0] and "click(point=" in ad.calls[0][0]
    c = Trace.load(p).first("commit")["action"]
    assert abs(c["x"] - cx) <= 1 and abs(c["y"] - cy) <= 1 and r.valid_at_receipt is True


def test_prompt7_wait_is_a_no_action_decision(ws):
    d = vlm_spec(); d["controller"].update({"prompt_version": 7})
    spec = TrialSpec.from_dict(d)
    ad = TextMock("Thought: no target yet.\nAction: wait()")
    p = run_trial(spec, ws / "t.jsonl", PayloadStore(ws / "p"), controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(3)))
    t = Trace.load(p)
    assert t.first("commit") is None and t.all("decision_no_action")
