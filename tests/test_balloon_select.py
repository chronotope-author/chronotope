"""C3 balloon select: the model chooses WHICH balloon, the binder only keeps WHERE current."""
from chronotope.controller import CallBudget
from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.scenario import BALLOON_ORDER
from chronotope.spec import TrialSpec
from chronotope.trace import Trace
from chronotope.tracker import cut_template, track
from conftest import base_spec_dict
from test_late_binding import SlowMock, answer

W, H = 320, 240


def bs_spec(policy="late_bind", vx=40, color="yellow", seed=3) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "balloon_select", "scenario_seed": seed, "canvas_w_px": W, "canvas_h_px": H,
                     "target_size_px": 24, "event_time_ns": 100_000_000, "response_window_ns": 1_000_000_000,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 1_500_000_000, "irrelevant_motion": False,
                     "skin": "balloon", "named_color": color, "velocity_x_px_per_s": vx}
    d["controller"] = {"kind": "vlm_late_binding", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
                       "model_id": "mock-model", "prompt_version": 8, "image_scale": 2, "max_tokens": 64, "temperature": 0.0,
                       "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change",
                       "at_commit_policy": policy, "template_radius_px": 12, "min_match_score": 0.8,
                       "restart_threshold_px": 6 if policy == "restart_if_moved" else None}
    d["timestamp_source"] = "monotonic_ns"
    return TrialSpec.from_dict(d)


def grid(spec, x, y):  # canvas -> 0-1000 grid answer that maps back to (x, y)
    return {"action": "click", "x": (x * 1000 + 999) // W, "y": (y * 1000 + 999) // H}


def test_layout_and_scoring():
    m = bs_spec().manifest()
    bal = dict(m.balloons_at(200_000_000))
    assert set(bal) == set(BALLOON_ORDER)
    t = 600_000_000
    for c, b in m.balloons_at(t):
        cx, cy = b.center()
        assert m.action_valid_at(t, {"kind": "click", "x": cx, "y": cy}) == (c == "yellow")
        assert m.clicked_balloon(t, {"kind": "click", "x": cx, "y": cy}) == c
    assert m.target_color() == (234, 179, 8) and m.instruction_color() == "yellow"


def test_rgb_tracker_follows_the_chosen_colour_not_the_nearest_balloon():
    m = bs_spec(vx=40).manifest()
    f0 = m.render(200_000_000); cx, cy = m.target_box_at(200_000_000).center()
    tpl = cut_template(f0, W, H, cx + 2, cy - 1, 12, rgb=True)
    for t in (200_000_000, 600_000_000, 1_000_000_000):
        mt = track(tpl, m.render(t), W, H); bx, by = m.target_box_at(t).center()
        assert (mt.x, mt.y) == (bx + 2, by - 1) and mt.score > 0.99 and tpl.channels == "rgb"


def test_prompt_names_the_colour(ws):
    spec = bs_spec()
    ad = SlowMock(lambda n: answer({"action": "wait"}), 0.05)
    run_trial(spec, ws / "t.jsonl", PayloadStore(ws / "p"), controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(3)))
    assert "the yellow balloon" in ad.calls[0][1] and "Do not click any other balloon" in ad.calls[0][1]


def test_right_pick_is_recovered_and_wrong_pick_stays_wrong(ws):
    spec = bs_spec("late_bind", vx=40)
    m = spec.manifest(); t0 = 100_000_000 + 16_666_667 * 2
    yx, yy = m.target_box_at(t0).center()
    # right pick: the model points at the yellow balloon where it WAS; 0.5 s later it has moved 20 px
    ad = SlowMock(lambda n: answer(grid(spec, yx + 1, yy)), 0.5)
    p = run_trial(spec, ws / "right.jsonl", PayloadStore(ws / "p"), controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(3)))
    t = Trace.load(p); r = evaluate_trace(t); rc = t.first("receipt")
    assert r.valid_at_receipt is True and r.proposed_valid_at_receipt is False
    assert m.clicked_balloon(rc["task_time_at_receipt_ns"], t.first("commit")["action"]) == "yellow"
    assert t.first("coordinate_correction")["binder_channels"] == "rgb"
    # wrong pick: the model points at the RED balloon (a distractor) — the binder must track red, and the
    # click must still be wrong at A (it is never converted into the named balloon)
    rx, ry = dict(m.balloons_at(t0))["red"].center()
    ad = SlowMock(lambda n: answer(grid(spec, rx, ry)), 0.5)
    p = run_trial(spec, ws / "wrong.jsonl", PayloadStore(ws / "p"), controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(3)))
    t = Trace.load(p); r = evaluate_trace(t); rc = t.first("receipt")
    assert r.valid_at_receipt is False and "semantic_error" in r.labels
    assert m.clicked_balloon(rc["task_time_at_receipt_ns"], t.first("commit")["action"]) == "red"   # tracked the wrong balloon faithfully
