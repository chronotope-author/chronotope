"""G2 (cards/G2-live-page.md): the live Chromium page. Pure parts (clock mapping, manifest from the page log, the
evaluator's labels on a synthetic trace of a live trial) run everywhere; the browser parts skip when Chromium or
Playwright is unavailable."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from chronotope.evaluate import evaluate_trace
from chronotope.hashing import hash_obj, sha256_hex
from chronotope.live_page import (CHROMIUM, EVENT_DIALOG_SHOWN, EVENT_TASK_COMPLETE, ClockMap, LivePageManifest,
                                  evaluate_live_two_step, measured_from_log, sync_clock)
from chronotope.scenario import EVENT_TARGET_APPEARS, EVENT_TARGET_JUMPS, manifest_from_dict
from chronotope.trace import Trace

S = 1_000_000_000
W, H = 640, 480
SC = {"scenario_id": "live_page", "scenario_seed": 7, "canvas_w_px": W, "canvas_h_px": H, "target_size_px": 37,
      "event_time_ns": 500_000_000, "response_window_ns": 16 * S, "frame_period_ns": 66_666_667, "episode_horizon_ns": 17 * S,
      "irrelevant_motion": False, "skin": "browser", "reveal_delay_ns": 500_000_000, "banner_delay_ns": 2 * S, "banner_h_px": 100,
      "dialog_dx_px": 40, "dialog_dy_px": -120, "slide_dx_px": 60, "slide_dy_px": 80, "slide_ns": 2 * S, "two_step": False}
BEFORE = {"x0": 200, "y0": 160, "x1": 304, "y1": 197}
AFTER = {"x0": 200, "y0": 272, "x1": 304, "y1": 309}


def synthetic_log(clicks=(), two_step=False):
    log = [{"kind": "load", "t_ns": 20_000_000, "time_origin": 1.7e12},
           {"kind": "reveal", "t_ns": 520_000_000, "cta": BEFORE},
           {"kind": "banner", "t_ns": 2_150_000_000, "cta": AFTER, "banner_h": 100}]
    for c in clicks:
        log.append({"kind": "pointerdown", **c})
    if two_step:
        log.append({"kind": "step", "t_ns": 6 * S, "t0_ns": 6 * S + 1_000_000, "step": 2, "slide_ms": 2000,
                    "confirm_start": {"x0": 300, "y0": 100, "x1": 393, "y1": 133}, "confirm_end": {"x0": 360, "y0": 180, "x1": 453, "y1": 213}})
    return log


# ---- clock mapping --------------------------------------------------------------------------------------------------

def test_clock_mapping_uses_the_shortest_round_trip_and_inverts():
    samples = [(1_000_000, 10.0, 1_600_000), (2_000_000, 11.0, 2_100_000), (3_000_000, 12.0, 4_000_000)]
    cm = sync_clock(samples, time_origin_ms=1.7e12)
    assert cm.roundtrip_ns == 100_000 and cm.offset_ns == 2_050_000 - 11_000_000
    assert cm.harness_ns(11.0) == 2_050_000 and cm.harness_ns(12.5) == 2_050_000 + 1_500_000
    assert abs(cm.page_ms(cm.harness_ns(123.456)) - 123.456) < 1e-6
    assert ClockMap(**cm.to_dict()) == cm
    with pytest.raises(ValueError):
        sync_clock([], 0.0)


# ---- manifest from the page log ---------------------------------------------------------------------------------------

def test_manifest_from_log_geometry_events_and_state():
    me = measured_from_log(synthetic_log())
    m = manifest_from_dict(SC).with_trailer({"measured": me})
    assert m.event_time_ns == 520_000_000 and m.t_reveal() == 520_000_000 and m.window_close_ns() == 520_000_000 + 16 * S
    assert [e.kind for e in m.events()] == ["target_appears", "target_jumps", "window_closes"]
    assert m.reflow_dy_px() == 112 and m.target_box().y0 == 160 and m.target_box_at(3 * S).y0 == 272
    assert m.state_at(100_000_000).latest_event_id == 0 and not m.state_at(100_000_000).target_visible
    assert m.state_at(1 * S).latest_event_id == EVENT_TARGET_APPEARS and m.state_at(2_150_000_000).latest_event_id == EVENT_TARGET_JUMPS
    assert not m.state_at(17 * S).target_visible
    click = {"kind": "click", "x": 252, "y": 178}
    assert m.action_valid_at(1 * S, click) and not m.action_valid_at(3 * S, click)   # geometry: the button moved down
    assert m.displacement_at(3 * S) == (0, 112)
    # identity: the seeded page, not what was measured on it (Replay must match Live)
    assert m.manifest_hash() == manifest_from_dict(SC).manifest_hash()
    assert manifest_from_dict({**SC, "banner_h_px": 101}).manifest_hash() != m.manifest_hash()


def test_page_verdict_overrides_geometry_for_the_recorded_click():
    me = measured_from_log(synthetic_log(clicks=[{"t_ns": 5 * S, "x": 252, "y": 178, "target": "main", "step": 1, "hit": False},
                                                  {"t_ns": 9 * S, "x": 252, "y": 290, "target": "cta", "step": 1, "hit": True}]))
    m = manifest_from_dict(SC).with_trailer({"measured": me})
    assert m.click_record(5 * S, {"x": 252, "y": 178})["hit"] is False and m.click_record(5 * S, {"x": 1, "y": 1}) is None
    assert not m.action_valid_at(5 * S, {"kind": "click", "x": 252, "y": 178})
    assert m.action_valid_at(9 * S, {"kind": "click", "x": 252, "y": 290})
    # a page verdict that disagrees with the geometry is the page's call (an edge pixel the DOM hit-test rejects)
    me2 = measured_from_log(synthetic_log(clicks=[{"t_ns": 9 * S, "x": 252, "y": 290, "target": "main", "step": 1, "hit": False}]))
    m2 = manifest_from_dict(SC).with_trailer({"measured": me2})
    assert m2.geometry_valid_at(9 * S, {"kind": "click", "x": 252, "y": 290}) and not m2.action_valid_at(9 * S, {"kind": "click", "x": 252, "y": 290})


def test_two_step_dialog_slides_from_the_click_and_the_page_ends_the_episode():
    me = measured_from_log(synthetic_log(two_step=True))
    m = manifest_from_dict({**SC, "two_step": True}).with_trailer({"measured": me})
    assert m.multi_action and [e.kind for e in m.events()] == ["target_appears", "target_jumps", "dialog_shown", "window_closes"]
    assert m.state_at(6 * S).latest_event_id == EVENT_TARGET_JUMPS            # the click's own instant: still step 1
    assert m.state_at(6 * S + 2).latest_event_id == EVENT_DIALOG_SHOWN
    assert m.target_box_at(6 * S + 1_000_000).x0 == 300 and m.target_box_at(7 * S + 1_000_000).x0 == 330 and m.target_box_at(9 * S).x0 == 360
    assert m.effect_visible_ns(6 * S) == 6 * S and m.POST_EFFECT_NS == m.episode_horizon_ns   # not complete: runs to the horizon
    assert m.return_pending(None) == -16.0
    me["complete_ns"] = 9 * S
    m = m.with_trailer({"measured": me})
    assert m.POST_EFFECT_NS == 0 and m.effect_visible_ns(9 * S) == 9 * S and m.return_pending(9 * S) == pytest.approx(-(9 - 0.52))
    assert m.state_at(9 * S + 1).latest_event_id == EVENT_TASK_COMPLETE and not m.state_at(9 * S + 1).target_visible


# ---- the evaluator's labels on a synthetic trace of a live trial ---------------------------------------------------------------

def write_trace(path: Path, *, corrected: bool, click, proposed, hit: bool, receipt_ns: int, two_step=False, extra_records=()):
    """A live trial: reveal frame at 0.53 s, banner frame at 2.2 s (published before D at 5.6 s), receipt at 5.7 s."""
    h0 = 1_000_000_000_000
    sc = {**SC, "two_step": two_step}
    m = manifest_from_dict(sc)
    ab = json.dumps(click, sort_keys=True, separators=(",", ":")).encode()
    log = synthetic_log(clicks=[{"t_ns": receipt_ns, "x": click["x"], "y": click["y"], "target": "cta" if hit else "main", "step": 1, "hit": hit}],
                        two_step=two_step)
    me = measured_from_log(log)
    recs = [dict(record="trial_header", spec={"controller": {"model_id": "scripted-page-oracle"}, "delivery": {"max_attempts": 1}},
                 spec_hash="s", matching_key_hash="k", manifest=sc, manifest_hash=m.manifest_hash(), condition="live",
                 task_clock_policy="live", trial_class="agent_decision", timestamp_source="monotonic_ns", controller_spec_hash="c",
                 action_space_version=1, scorer_version=1, observation_boundary_version=1, replay_source_live_trace_sha256=None,
                 harness_origin_ns=h0),
            dict(record="observation", observation_id=1, source_event_id=0, source_time_ns=100_000_000, publication_time_ns=h0 + 130_000_000, payload_hash="a", observation_version=1),
            dict(record="admission", observation_id=1, control_state_version_before=0, control_state_version_after=1, relevance="action", controller_state_hash="x", latest_source_time_ns=100_000_000, harness_time_ns=h0 + 130_000_000),
            dict(record="event", event_id=1, kind="target_appears", task_time_ns=520_000_000, harness_time_ns=h0 + 520_000_000, source="environment"),
            dict(record="observation", observation_id=2, source_event_id=1, source_time_ns=530_000_000, publication_time_ns=h0 + 560_000_000, payload_hash="b", observation_version=2),
            dict(record="admission", observation_id=2, control_state_version_before=1, control_state_version_after=2, relevance="action", controller_state_hash="y", latest_source_time_ns=530_000_000, harness_time_ns=h0 + 560_000_000),
            dict(record="decision_begin", decision_id=1, begin_time_ns=h0 + 600_000_000, observation_id=2, supporting_control_state_version=2, invocation_kind="fixed_input"),
            dict(record="event", event_id=4, kind="target_jumps", task_time_ns=2_150_000_000, harness_time_ns=h0 + 2_150_000_000, source="environment"),
            dict(record="observation", observation_id=3, source_event_id=4, source_time_ns=2_200_000_000, publication_time_ns=h0 + 2_230_000_000, payload_hash="c", observation_version=3),
            dict(record="decision_invocation", decision_id=1, invocation_id="i", invocation_kind="fixed_input", end_time_ns=h0 + 5_600_000_000, outcome="click", model_reported="scripted-page-oracle"),
            dict(record="action_proposed", decision_id=1, action=proposed, coordinate_status="provisional", proposal_time_ns=h0 + 5_600_000_000),
            dict(record="admission", observation_id=3, control_state_version_before=2, control_state_version_after=3, relevance="action", controller_state_hash="z", latest_source_time_ns=2_200_000_000, harness_time_ns=h0 + 5_600_000_000)]
    if corrected:
        recs.append(dict(record="coordinate_correction", decision_id=1, binding="tracked_at_D", observation_id=3, proposed_action=proposed,
                         corrected_action=click, harness_time_ns=h0 + 5_650_000_000, proposal_time_ns=h0 + 5_600_000_000,
                         binder_start_ns=h0 + 5_600_000_000, binder_end_ns=h0 + 5_650_000_000, translation_x=click["x"] - proposed["x"],
                         translation_y=click["y"] - proposed["y"], coordinate_status="resolved", match_score=1.0, matched=True, moved_px=112.0, track_ns=50_000_000))
    D = h0 + (5_650_000_000 if corrected else 5_600_000_000)
    recs += [dict(record="commit", action_id=1, decision_id=1, action=click, serialized_action_hash=hash_obj(click), supporting_control_state_version=2,
                  control_state_version_at_commit=3, commitment_time_ns=D, task_time_at_commit_ns=D - h0),
             dict(record="delivery", action_id=1, attempt=1, submit_time_ns=D, route="playwright_page"),
             dict(record="receipt", action_id=1, attempt=1, receipt_time_ns=h0 + receipt_ns, task_time_at_receipt_ns=receipt_ns, validity_policy="advancing",
                  received_action_hash=sha256_hex(ab), target_visible=True, window_open=True, click_inside_target=hit, valid_at_receipt=hit,
                  **({"outcome": "accepted" if hit else "invalid", "effect_visible_at_receipt": False} if two_step else {})),
             *extra_records,
             dict(record="scenario_trailer", measured=me, clock={"offset_ns": 0, "roundtrip_ns": 500_000, "time_origin_ms": 1.7e12}),
             dict(record="trial_end", reason="receipt", harness_time_ns=h0 + receipt_ns, task_time_ns=receipt_ns)]
    with path.open("w") as f:
        for i, r in enumerate(recs, start=1):
            f.write(json.dumps({"seq": i, **r}) + "\n")
    return path


def test_frozen_click_is_outdated_at_commitment_and_late_bound_click_is_valid(tmp_path):
    frozen = {"kind": "click", "x": 252, "y": 178}
    r = evaluate_trace(Trace.load(write_trace(tmp_path / "f.jsonl", corrected=False, click=frozen, proposed=frozen, hit=False, receipt_ns=5_700_000_000)))
    assert r.valid_trial and r.valid_at_receipt is False and r.semantic_correct is True
    assert "outdated_at_commitment" in r.labels and "invalidated_after_commitment" not in r.labels and not r.contract_violations
    assert r.spatial_error_px == pytest.approx(112.0) and r.return_G == pytest.approx(-112.0)
    assert r.intervals_ns["E_O"] == 40_000_000 and r.intervals_ns["D_A"] == 100_000_000   # E from the page log, A from the page
    bound = {"kind": "click", "x": 252, "y": 290}
    r2 = evaluate_trace(Trace.load(write_trace(tmp_path / "l.jsonl", corrected=True, click=bound, proposed=frozen, hit=True, receipt_ns=5_700_000_000)))
    assert r2.valid_at_receipt is True and r2.binding == "tracked_at_D" and r2.proposed_valid_at_receipt is False
    assert r2.labels == [] and not r2.contract_violations and r2.return_G == pytest.approx(0.0)


def test_two_step_evaluation_reads_completion_from_the_page(tmp_path):
    hit = {"kind": "click", "x": 252, "y": 290}
    t = Trace.load(write_trace(tmp_path / "t.jsonl", corrected=True, click=hit, proposed={"kind": "click", "x": 252, "y": 178}, hit=True,
                               receipt_ns=6 * S, two_step=True))
    r = evaluate_live_two_step(t)
    assert r.valid_trial and not r.completed and r.return_G == -16.0 and r.n_clicks == 1 and r.n_hits == 1
    assert r.clicks[0]["step"] == 1 and r.clicks[0]["labels"] == [] and r.step2_b_to_d_ns is None


# ---- the browser ----------------------------------------------------------------------------------------------------------

def _browser():
    if not Path(CHROMIUM).exists():
        pytest.skip(f"{CHROMIUM} not installed")
    pytest.importorskip("playwright")
    from chronotope.live_page import LiveBrowser
    try:
        return LiveBrowser(CHROMIUM)
    except Exception as e:  # pragma: no cover - environment
        pytest.skip(f"Chromium could not launch: {e}")


def test_live_page_captures_clicks_and_logs(tmp_path):
    from chronotope.live_page import LivePageSource, PlaywrightRoute, live_source_factory, page_url
    b = _browser()
    try:
        m = manifest_from_dict({**SC, "reveal_delay_ns": 100_000_000, "banner_delay_ns": 400_000_000, "episode_horizon_ns": 17 * S})
        s = b.open(page_url(m), W, H)
        m.attach(s)
        h0 = time.monotonic_ns()
        src = live_source_factory(s, m, 66_666_667)(h0, None)
        assert s.clock is not None and s.clock.roundtrip_ns < 50_000_000 and s.clock.time_origin_ms > 1e12
        assert s.wait_for(lambda e: e.get("kind") == "banner", 5.0) is not None
        time.sleep(0.5)
        frames = []
        while (f := src.peek_due(time.monotonic_ns())) is not None:
            frames.append(src.pop())
        assert len(frames) >= 8, {k: v for k, v in s.capture.items() if k != "capture_ms"}
        assert all(f.payload[:8] == b"\x89PNG\r\n\x1a\n" for f in frames)
        assert all(a.source_time_ns < b_.source_time_ns and a.due_harness_ns < b_.due_harness_ns for a, b_ in zip(frames, frames[1:]))
        assert len({sha256_hex(f.payload) for f in frames}) >= 2                    # blank -> revealed -> banner
        assert {f.source_event_id for f in frames} >= {EVENT_TARGET_APPEARS, EVENT_TARGET_JUMPS}
        me = measured_from_log(s.log())
        assert me["reveal_ns"] is not None and me["banner_ns"] is not None and me["button_after"][1] - me["button_before"][1] == 112
        assert me["reveal_ns"] > 0 and abs(me["reveal_ns"] - 100_000_000 - me["load_ns"]) < 100_000_000
        route = PlaywrightRoute(s, 2 * S)
        x, y = m.target_box_at(me["banner_ns"] + 1).center()
        submit = time.monotonic_ns()
        out = route.deliver(json.dumps({"kind": "click", "x": x, "y": y}).encode(), 1, submit)
        assert out is not None
        a_ns, _ = out
        assert 0 <= a_ns - submit < 500_000_000 and not route.lost
        me = measured_from_log(s.log())
        assert me["clicks"][-1]["hit"] and me["clicks"][-1]["t_ns"] == a_ns - h0 and me["complete_ns"] is not None
        assert m.action_valid_at(a_ns - h0, {"kind": "click", "x": x, "y": y})
        assert s.achieved_hz() is not None and s.achieved_hz() > 10
        s.close()
    finally:
        b.close()


def test_scripted_controllers_through_the_live_page(tmp_path):
    from chronotope.controller import CallBudget
    from chronotope.live_page import PlaywrightRoute, ScriptedPageOracle, live_source_factory, page_url
    from chronotope.runner import PayloadStore, build_controller, run_trial
    from chronotope.spec import TrialSpec
    b = _browser()
    try:
        out = {}
        for pol in ("frozen", "late_bind"):
            sc = {**SC, "banner_delay_ns": 900_000_000, "response_window_ns": 4 * S, "episode_horizon_ns": 5 * S}
            spec = TrialSpec.from_dict({
                "schema_version": 1, "scenario": sc, "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
                "controller": {"kind": "vlm_late_binding", "server": "openai_compatible_local", "base_url": "http://127.0.0.1:0/v1",
                               "model_id": "scripted-page-oracle", "prompt_version": 12, "image_scale": 1, "max_tokens": 64, "temperature": 0.0,
                               "enable_thinking": False, "reasoning_effort": None, "invoke_on": "frame_change", "at_commit_policy": pol,
                               "template_radius_px": 12, "min_match_score": 0.8, "restart_threshold_px": None},
                "observation": {"boundary_version": 1, "publication_delay_ns": 0}, "action_space_version": 1, "scorer_version": 1,
                "timestamp_source": "monotonic_ns", "delivery": {"route": "playwright_page", "receipt_timeout_ns": 2 * S, "max_attempts": 1},
                "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision"})
            m = spec.manifest()
            s = b.open(page_url(m), W, H)
            m.attach(s)
            ctrl = build_controller(spec, vlm_adapter=ScriptedPageOracle(s, 1.0), budget=CallBudget(3))
            p = run_trial(spec, tmp_path / f"{pol}.jsonl", PayloadStore(tmp_path / "p"), controller=ctrl, route=PlaywrightRoute(s, 2 * S),
                          source=live_source_factory(s, m, 66_666_667), manifest=m)
            s.close()
            t = Trace.load(p)
            out[pol] = (evaluate_trace(t), t)
        rf, tf = out["frozen"]
        assert rf.valid_trial and rf.valid_at_receipt is False and "outdated_at_commitment" in rf.labels and not rf.contract_violations
        assert abs(rf.spatial_error_px - 112) < 4
        rl, tl = out["late_bind"]
        c = tl.first("coordinate_correction")
        assert rl.valid_at_receipt is True and c["matched"] and abs(c["translation_y"] - 112) <= 2 and rl.proposed_valid_at_receipt is False
        tr = tl.first("scenario_trailer")
        assert tr["capture"]["achieved_hz"] > 10 and tr["clock"]["roundtrip_ns"] > 0 and tr["measured"]["clicks"][-1]["hit"]
        assert all(o["source_event_id"] in (0, 1, 4, 8) for o in tl.all("observation"))
    finally:
        b.close()
