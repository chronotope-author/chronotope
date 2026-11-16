"""G1 (cards/G1-realistic-reflow.md): the Chromium-rendered reflow page through the same harness — the button geometry
comes from the DOM, the frames from the baked assets; frozen click misses by the reflow, late binding (template
tracker on the rendered button) hits under the same decision id."""
from __future__ import annotations

import time

import pytest

from chronotope.controller import CallBudget, _find_target_box
from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.scenario import TARGET_COLOR, manifest_from_dict
from chronotope.spec import TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict
from test_vlm_mock import MockAdapter, answer

W, H = 640, 480
SC = {"scenario_id": "reflow_page", "scenario_seed": 1003, "canvas_w_px": W, "canvas_h_px": H, "target_size_px": 37,
      "event_time_ns": 100_000_000, "response_window_ns": 3_000_000_000, "frame_period_ns": 16_666_667,
      "episode_horizon_ns": 3_600_000_000, "irrelevant_motion": False, "skin": "browser", "change_delay_ns": 400_000_000}


def lb_spec(policy="late_bind", change_ns=400_000_000, seed=1003) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {**SC, "change_delay_ns": change_ns, "scenario_seed": seed}
    d["controller"] = {"kind": "vlm_late_binding", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
                       "model_id": "mock-model", "prompt_version": 12, "image_scale": 1, "max_tokens": 64, "temperature": 0.0,
                       "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change", "at_commit_policy": policy,
                       "template_radius_px": 12, "min_match_score": 0.8, "restart_threshold_px": 6 if policy == "restart_if_moved" else None}
    d["timestamp_source"] = "monotonic_ns"
    return TrialSpec.from_dict(d)


def test_manifest_geometry_from_dom_and_frames_from_assets():
    m = manifest_from_dict(SC)
    b0, b1 = m.target_box_at(200_000_000), m.target_box_at(600_000_000)
    assert b1.y0 - b0.y0 == m.reflow_dy_px() and 72 <= m.reflow_dy_px() <= 212 and b1.y1 <= H
    assert _find_target_box(m.render(200_000_000), W, H, TARGET_COLOR["browser"]) == (b0.x0, b0.y0, b0.x1 - 1, b0.y1 - 1) or \
        _find_target_box(m.render(200_000_000), W, H, TARGET_COLOR["browser"])[:2] == (b0.x0, b0.y0)
    assert _find_target_box(m.render(0), W, H, TARGET_COLOR["browser"]) is None       # no button before the page loads
    assert [e.kind for e in m.events()] == ["target_appears", "target_jumps", "window_closes"]
    cx, cy = b0.center()
    assert m.action_valid_at(200_000_000, {"kind": "click", "x": cx, "y": cy}) and not m.action_valid_at(600_000_000, {"kind": "click", "x": cx, "y": cy})


def test_every_baked_seed_loads():
    for seed in list(range(100, 110)) + list(range(1000, 1100)):
        m = manifest_from_dict({**SC, "scenario_seed": seed})
        assert m.target_box_at(m.window_close_ns() - 1).y1 <= H, seed


class SlowCenterMock(MockAdapter):
    """Clicks the button centre of the frame it was given (0-1000 grid), after `delay_s`."""

    def __init__(self, delay_s):
        super().__init__(None); self.delay_s = delay_s

    def invoke(self, png, system, user_text, schema=None):
        import io
        from PIL import Image
        self.calls.append((png, system, user_text))
        im = Image.open(io.BytesIO(png)).convert("RGB")
        box = _find_target_box(im.tobytes(), im.width, im.height, TARGET_COLOR["browser"])
        time.sleep(self.delay_s)
        if box is None:
            return answer({"action": "wait"})
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        return answer({"action": "click", "x": cx * 1000 // im.width, "y": cy * 1000 // im.height})


def run(policy, ws, delay_s=0.6, change_ns=400_000_000):
    spec = lb_spec(policy, change_ns)
    ad = SlowCenterMock(delay_s)
    p = run_trial(spec, ws / f"{policy}.jsonl", PayloadStore(ws / "p"), controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(5)))
    return evaluate_trace(Trace.load(p)), Trace.load(p), ad


def test_frozen_misses_by_the_reflow_and_late_bind_hits(ws):
    rf, tf, _ = run("frozen", ws)
    assert rf.valid_at_receipt is False and "outdated_at_commitment" in rf.labels and rf.semantic_correct is True
    m = lb_spec().manifest()
    assert abs(rf.spatial_error_px - m.reflow_dy_px()) < 3
    rl, tl, ad = run("late_bind", ws)
    c = tl.first("coordinate_correction")
    assert rl.valid_at_receipt is True and c["matched"] and abs(c["translation_y"] - m.reflow_dy_px()) <= 2 and c["translation_x"] == 0
    assert rl.proposed_valid_at_receipt is False and len(ad.calls) == 1 and not rl.contract_violations


def test_restart_if_moved_decides_again(ws):
    r, t, ad = run("restart_if_moved", ws)
    assert r.n_restarts == 1 and len(ad.calls) == 2 and r.valid_at_receipt is True


def test_no_reflow_controllers_agree(ws):
    for pol in ("frozen", "late_bind"):
        r, t, _ = run(pol, ws, change_ns=10_000_000_000)
        assert r.valid_at_receipt is True and "outdated_at_commitment" not in r.labels
