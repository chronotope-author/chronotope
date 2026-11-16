"""Cards A1-sponsor-skip-agent, A2-sponsor-skip-pending, B1-live-trading-agent, C-prl-rows (2026-08-24).

sponsor_skip_pending contract (scripted pending controller, simulated clock); the outcome contract on multi-action
terminals (return_G on sponsor_skip_pending and pending_effect); prompts 9/10; the vlm_chronotope controller with a
mock adapter: suppression + same-decision binding, an inert binder on a static button, a live binder on a moving
target; spec strictness for the new kinds."""
from __future__ import annotations

import time

import pytest

from chronotope.controller import ACTION_JSON_SCHEMA, PROMPT_COORDS, SYSTEM_PROMPTS, CallBudget, RawInvocation, _find_target_box
from chronotope.evaluate import evaluate_pending_trace, evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.scenario import EVENT_EFFECT_VISIBLE, TARGET_COLOR, manifest_from_dict
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict

PERIOD, PUB, DEL = 16_666_667, 30_000_000, 20_000_000
E = 500_000_000


def ssp_scenario(seed=5, seek=2_000_000_000, window=6_000_000_000, horizon=9_000_000_000) -> dict:
    return {"scenario_id": "sponsor_skip_pending", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
            "target_size_px": 24, "event_time_ns": E, "response_window_ns": window, "frame_period_ns": PERIOD,
            "episode_horizon_ns": horizon, "irrelevant_motion": False, "skin": "player", "seek_delay_ns": seek}


def ssp_spec(memory="none", compute=300_000_000, route="simulated_fixed_delay", **sc) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = ssp_scenario(**sc)
    d["controller"] = {"kind": "scripted_pending", "compute_ns": compute, "memory": memory,
                       "pending_timeout_ns": 10_000_000_000, "pending_radius_px": 10}
    d["delivery"] = {"route": route, "delivery_delay_ns": DEL, "max_attempts": 1}
    return TrialSpec.from_dict(d)


def run(spec, ws, name="t"):
    p = run_trial(spec, ws / f"{name}.jsonl", PayloadStore(ws / "p"))
    t = Trace.load(p)
    return t, evaluate_pending_trace(t)


# ---- sponsor_skip_pending: the seek delay is a pending effect --------------------------------------------

def test_ssp_memoryless_repeats_skip_until_seek_completes(ws):
    t, r = run(ssp_spec("none"), ws)
    acc = [x for x in t.all("receipt") if x["outcome"] == "accepted"]
    assert len(acc) == 1 and acc[0]["action_id"] == 1
    ev = next(e for e in t.all("event") if e["event_id"] == EVENT_EFFECT_VISIBLE)
    assert ev["task_time_ns"] == acc[0]["task_time_at_receipt_ns"] + 2_000_000_000
    assert r.accepted and r.duplicates >= 3 and not r.exact_success and r.contract_violations == []
    # the panning content changes every frame, so the memoryless controller re-decides until the sponsor is gone
    for x in t.all("receipt"):
        if x["task_time_at_receipt_ns"] >= ev["task_time_ns"]:
            assert x["outcome"] == "invalid"
    m = manifest_from_dict(t.header()["manifest"])
    assert r.return_G == pytest.approx(-(min(acc[0]["task_time_at_receipt_ns"] + 2_000_000_000, m.window_close_ns()) - E) / 1e9)
    assert t.first("trial_end")["reason"] == "effect_done"


def test_ssp_record_removes_duplicates_at_same_first_receipt_and_same_return(ws):
    t0, r0 = run(ssp_spec("none"), ws, "none")
    t1, r1 = run(ssp_spec("harness_pending"), ws, "rec")
    assert r0.duplicates > 0 and r1.duplicates == 0 and r1.n_suppressed > 0 and r1.exact_success
    assert r0.first_valid_E_A_ns == r1.first_valid_E_A_ns          # the record never delays the first skip
    assert r0.return_G == r1.return_G                              # duplicates are not charged in G_A
    assert r1.contract_violations == []


def test_ssp_never_accepted_charges_the_full_sponsor(ws):
    t, r = run(ssp_spec("none", compute=300_000_000, window=200_000_000, horizon=2_000_000_000), ws)
    assert not r.accepted and r.return_G == pytest.approx(-0.2)


def test_ssp_render_matches_sponsor_skip_before_acceptance_and_hides_sponsor_after_effect():
    sc = ssp_scenario()
    mp = manifest_from_dict(sc)
    ms = manifest_from_dict({**sc, "scenario_id": "sponsor_skip"})
    for t in (0, E - PERIOD, E + 1_000_000_000, E + 5_000_000_000):
        assert mp.render_env(t, None) == ms.render(t)               # byte-identical to the one-response task
    acc = E + 1_000_000_000
    assert mp.render_env(acc + 1_000_000_000, acc) == ms.render(acc + 1_000_000_000)   # before the effect: unchanged
    after = mp.render_env(acc + 2_000_000_000, acc)
    assert _find_target_box(after, 320, 240, TARGET_COLOR["player"]) is None            # sponsor gone at the effect
    assert mp.receipt_outcome(acc + 2_000_000_000, {"kind": "click", **dict(zip("xy", mp.target_box().center()))}, acc) == "invalid"
    assert mp.receipt_outcome(acc + 1_000_000_000, {"kind": "click", **dict(zip("xy", mp.target_box().center()))}, acc) == "duplicate"
    with pytest.raises(ValueError, match="positive seek"):
        manifest_from_dict({**sc, "seek_delay_ns": 0})


# ---- pending_effect: a time-based return of the same shape as G_A ----------------------------------------

def test_pending_effect_return_is_time_to_effect(ws):
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "pending_effect", "scenario_seed": 5, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": E, "response_window_ns": 6_000_000_000,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": 8_000_000_000, "irrelevant_motion": False,
                     "skin": "notification", "effect_delay_ns": 2_000_000_000}
    d["controller"] = {"kind": "scripted_pending", "compute_ns": 300_000_000, "memory": "harness_pending",
                       "pending_timeout_ns": 10_000_000_000, "pending_radius_px": 10}
    d["delivery"] = {"route": "simulated_fixed_delay", "delivery_delay_ns": DEL, "max_attempts": 1}
    t, r = run(TrialSpec.from_dict(d), ws)
    acc = next(x for x in t.all("receipt") if x["outcome"] == "accepted")["task_time_at_receipt_ns"]
    assert r.return_G == pytest.approx(-(acc + 2_000_000_000 - E) / 1e9)
    m = manifest_from_dict(d["scenario"])
    assert m.return_pending(None) == pytest.approx(-6.0)


# ---- prompts 9 / 10 ------------------------------------------------------------------------------------

def test_prompts_9_and_10_exist_on_the_grid():
    assert "orange button" in SYSTEM_PROMPTS[9] and "green" in SYSTEM_PROMPTS[10]
    assert PROMPT_COORDS[9] == PROMPT_COORDS[10] == "norm1000"
    for v in (9, 10):
        SYSTEM_PROMPTS[v].format(w=640, h=480)   # no stray braces


# ---- vlm_chronotope: one agent = binder + record --------------------------------------------------------

def decode_png_rgb(png: bytes) -> tuple[int, int, bytes]:
    """Decode a PNG written by chronotope.png.encode_png (8-bit RGB, filter type 0 on every row)."""
    import struct
    import zlib
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(png):
        (ln,), kind = struct.unpack(">I", png[pos:pos + 4]), png[pos + 4:pos + 8]
        data = png[pos + 8:pos + 8 + ln]
        if kind == b"IHDR":
            w, h = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            idat += data
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = 1 + 3 * w
    return w, h, b"".join(raw[y * stride + 1:(y + 1) * stride] for y in range(h))


def competent(color: tuple[int, int, int], scale: int = 2):
    """A responder that clicks the centre of the colour's bounding box (on the 0-1000 grid) when the colour is
    visible in the screenshot, and waits otherwise — what a competent model does with prompt 9 / 5."""
    def respond(png: bytes):
        W, H, rgb = decode_png_rgb(png)
        box = _find_target_box(rgb, W, H, color)
        if box is None:
            return answer({"action": "wait"})
        x0, y0, x1, y1 = box
        return answer({"action": "click", "x": ((x0 + x1) // 2 * 1000 + 500) // W, "y": ((y0 + y1) // 2 * 1000 + 500) // H})
    return respond


class MockAdapter:
    def __init__(self, responder, sleep_s: float = 0.0, sees_png: bool = False):
        self.calls, self.responder, self.sleep_s, self.sees_png = [], responder, sleep_s, sees_png

    def spec_fields(self):
        return {"server": "mock", "model_id": "mock-model", "schema": ACTION_JSON_SCHEMA}

    def invoke(self, png, system, user_text):
        self.calls.append((png, system, user_text))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return self.responder(png) if self.sees_png else self.responder(len(self.calls))


def answer(obj):
    return RawInvocation(parsed=obj, invocation_id="req_mock",
                         detail={"model_reported": "mock-model", "finish_reason": "stop", "input_tokens": 10,
                                 "output_tokens": 5, "reasoning_tokens": None, "raw_text": str(obj)})


def grid(cx: int, cy: int, w: int = 320, h: int = 240) -> dict:
    return {"action": "click", "x": (cx * 1000 + 500) // w, "y": (cy * 1000 + 500) // h}


def chrono_controller(memory="harness_pending", prompt_version=9, image_scale=2):
    return {"kind": "vlm_chronotope", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
            "model_id": "mock-model", "prompt_version": prompt_version, "image_scale": image_scale, "max_tokens": 64,
            "temperature": 0.0, "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change",
            "memory": memory, "pending_timeout_ns": 10_000_000_000, "pending_radius_px": 10,
            "template_radius_px": 12, "min_match_score": 0.8}


def chrono_ssp_spec(memory="harness_pending") -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = ssp_scenario(seek=600_000_000, window=2_000_000_000, horizon=3_000_000_000)
    d["scenario"]["event_time_ns"] = 100_000_000
    d["controller"] = chrono_controller(memory)
    d["timestamp_source"] = "monotonic_ns"
    d["delivery"] = {"route": "simulated_fixed_delay", "delivery_delay_ns": DEL, "max_attempts": 1}
    return TrialSpec.from_dict(d)


@pytest.mark.parametrize("memory,expect_dup", [("harness_pending", False), ("none", True)])
def test_chronotope_on_static_button_binder_inert_record_suppresses(ws, memory, expect_dup):
    spec = chrono_ssp_spec(memory)
    ad = MockAdapter(competent(TARGET_COLOR["player"]), sees_png=True)   # waits until the orange button is visible
    p = run_trial(spec, ws / f"{memory}.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(2000)))
    t = Trace.load(p)
    r = evaluate_pending_trace(t)
    assert r.valid_trial and r.accepted and r.contract_violations == []
    assert (r.duplicates > 0) == expect_dup and r.exact_success == (not expect_dup)
    assert (r.n_suppressed > 0) == (memory == "harness_pending")
    corr = t.all("coordinate_correction")
    assert corr and all(c["binding"] == "tracked_at_D" and c["translation_x"] == 0 and c["translation_y"] == 0 for c in corr)
    # a 60x24 uniform button is not unique horizontally: binder v2 reports the binding ambiguous and keeps the
    # proposal (translation 0) instead of sliding the click to the button's edge as v1's argmax would have
    assert all(c["binder_version"] == 2 and (c["matched"] or c["ambiguous"]) for c in corr)
    assert all(c["ambiguous"] and c["plateau"] > 4 for c in corr)   # a 60x24 uniform button is a plateau, never a peak
    assert ad.calls and t.all("decision_no_action")                         # it waited before the sponsor appeared
    assert all(c["decision_id"] == cm["decision_id"] for c, cm in zip(corr, t.all("commit")))   # same decision id
    assert r.return_G is not None and r.return_G < 0
    assert t.header()["controller_spec_hash"] != ""


def test_chronotope_on_moving_target_binds_the_same_decision(ws):
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "moving_target", "scenario_seed": 11, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 100_000_000, "response_window_ns": 1_500_000_000,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": 2_000_000_000, "irrelevant_motion": False,
                     "skin": "balloon", "velocity_x_px_per_s": 120, "velocity_y_px_per_s": 0}
    d["controller"] = chrono_controller("harness_pending", prompt_version=5)
    d["timestamp_source"] = "monotonic_ns"
    d["delivery"] = {"route": "simulated_fixed_delay", "delivery_delay_ns": DEL, "max_attempts": 1}
    spec = TrialSpec.from_dict(d)
    cx, cy = spec.manifest().target_box().center()          # the balloon where it appeared
    ad = MockAdapter(lambda n: answer(grid(cx, cy)), sleep_s=0.25)   # "thinks" for 250 ms: ~30 px of drift
    p = run_trial(spec, ws / "mv.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(5)))
    t = Trace.load(p)
    r = evaluate_trace(t)
    assert r.valid_trial and r.n_decisions == 1 and r.binding == "tracked_at_D" and not r.contract_violations
    c = t.first("coordinate_correction")
    assert c["decision_id"] == t.first("commit")["decision_id"] and c["matched"] and c["translation_x"] >= 20
    assert r.valid_at_receipt is True and r.proposed_valid_at_receipt is False   # the unbound proposal would have missed
    assert r.return_G == pytest.approx(-r.spatial_error_px)


# ---- an outdated wait is labelled (the A1 w=1 terminal) -------------------------------------------------

def test_wait_only_terminal_is_outdated_at_commitment_when_target_appeared_during_the_wait(ws):
    """Player skin (pre-event animation): a mock that waits on every frame; the sponsor appears and expires during
    the waits. Live: the wait committed after the sponsor frame reached O is outdated at commitment; Paused: no
    timing label (frozen window)."""
    from chronotope.spec import TrialSpec as TS
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "sponsor_skip", "scenario_seed": 5, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 200_000_000, "response_window_ns": 400_000_000,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": 1_200_000_000, "irrelevant_motion": False,
                     "skin": "player", "seek_delay_ns": 0}
    d["controller"] = {"kind": "vlm_direct", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1",
                       "model_id": "mock-model", "prompt_version": 9, "image_scale": 2, "max_tokens": 64, "temperature": 0.0,
                       "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change"}
    d["timestamp_source"] = "monotonic_ns"
    for policy, want in (("live", True), ("paused", False)):
        spec = TS.from_dict({**d, "task_clock_policy": policy})
        ad = MockAdapter(lambda n: answer({"action": "wait"}), sleep_s=0.3)   # each wait outlasts several frames
        p = run_trial(spec, ws / f"{policy}.jsonl", PayloadStore(ws / "p"), controller=build_controller(spec, vlm_adapter=ad, budget=CallBudget(50)))
        r = evaluate_trace(Trace.load(p))
        assert r.valid_trial and not r.action_committed and r.valid_at_receipt is False and r.return_G == pytest.approx(-0.4)
        assert ("outdated_at_commitment" in r.labels) is want, (policy, r.labels)


# ---- Q3: player_static is the player with nothing moving before the sponsor ----------------------------

def test_player_static_has_no_pre_event_motion_and_same_sponsor():
    sc = {**ssp_scenario(), "scenario_id": "sponsor_skip"}
    anim = manifest_from_dict(sc); stat = manifest_from_dict({**sc, "skin": "player_static"})
    assert anim.target_box() == stat.target_box()
    assert stat.render(0) == stat.render(E - PERIOD)                      # nothing changes before the sponsor
    assert anim.render(0) != anim.render(E - PERIOD)                      # the animated skin does change
    assert _find_target_box(stat.render(E + PERIOD), 320, 240, TARGET_COLOR["player_static"]) == \
        _find_target_box(anim.render(E + PERIOD), 320, 240, TARGET_COLOR["player"])   # same button, same place
    assert stat.render(E + PERIOD) != stat.render(E + 1_000_000_000)      # the sponsor countdown still drains (1-px steps)


# ---- spec strictness for the new kinds -----------------------------------------------------------------

def test_spec_rejects_bad_chronotope_and_pending_configs():
    good = chrono_ssp_spec().raw
    with pytest.raises(SpecError, match="template_radius_px"):
        TrialSpec.from_dict({**good, "controller": {**good["controller"], "template_radius_px": 0}})
    with pytest.raises(SpecError, match="monotonic_ns"):
        TrialSpec.from_dict({**good, "timestamp_source": "simulated"})
    with pytest.raises(SpecError, match="unknown"):
        TrialSpec.from_dict({**good, "controller": {**good["controller"], "at_commit_policy": "frozen"}})
    with pytest.raises(SpecError, match="not in"):
        TrialSpec.from_dict({**good, "controller": {**good["controller"], "prompt_version": 99}})
    with pytest.raises(ValueError, match="positive seek"):
        TrialSpec.from_dict({**good, "scenario": {**good["scenario"], "seek_delay_ns": 0}})
    with pytest.raises(SpecError, match="not in"):
        TrialSpec.from_dict({**good, "scenario": {**good["scenario"], "skin": "notification"}})
