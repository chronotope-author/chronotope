"""WI-004 (Gate 5): late binding with a separately identified tracker, against a mock VLM that
answers slowly enough for the balloon to drift during B->D. Wall clock (monotonic_ns), real frames."""
from __future__ import annotations

import json
import time

import pytest

from chronotope.evaluate import evaluate_trace
from chronotope.runner import PayloadStore, build_controller, run_trial
from chronotope.controller import CallBudget
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from chronotope.tracker import cut_template, track
from conftest import base_spec_dict
from test_vlm_mock import MockAdapter, answer

W, H, TS = 320, 240, 24


def lb_spec(policy="late_bind", vx=40, irrelevant=False, threshold=None, seed=3) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "moving_target", "scenario_seed": seed, "canvas_w_px": W, "canvas_h_px": H,
                     "target_size_px": TS, "event_time_ns": 100_000_000, "response_window_ns": 1_000_000_000,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 1_500_000_000,
                     "irrelevant_motion": irrelevant, "skin": "balloon",
                     "velocity_x_px_per_s": vx, "velocity_y_px_per_s": 0}
    d["controller"] = {"kind": "vlm_late_binding", "server": "openai_compatible_local",
                       "base_url": "http://127.0.0.1:8000/v1", "model_id": "mock-model", "prompt_version": 3,
                       "image_scale": 2, "max_tokens": 64, "temperature": 0.0, "enable_thinking": False,
                       "reasoning_effort": None, "invoke_on": "frame_change", "at_commit_policy": policy,
                       "template_radius_px": 12, "min_match_score": 0.8, "restart_threshold_px": threshold}
    d["timestamp_source"] = "monotonic_ns"
    return TrialSpec.from_dict(d)


class SlowMock(MockAdapter):
    """Answers after `delay_s` so that B->D is long enough for the target to move."""

    def __init__(self, responder, delay_s: float) -> None:
        super().__init__(responder)
        self.delay_s = delay_s

    def invoke(self, png, system, user_text):
        time.sleep(self.delay_s)
        return super().invoke(png, system, user_text)


def proposal_at_appearance(spec: TrialSpec, off=(2, -3)):
    """The mock 'sees' the balloon where it is when it appears (prompt 3: image pixels, scale 2)."""
    cx, cy = spec.manifest().target_box().center()
    return cx + off[0], cy + off[1]


def run(spec, ws, adapter, name="t"):
    p = run_trial(spec, ws / f"{name}.jsonl", PayloadStore(ws / "p"),
                  controller=build_controller(spec, vlm_adapter=adapter, budget=CallBudget(8)))
    t = Trace.load(p)
    return t, evaluate_trace(t)


# (i) the tracker recovers a known shift on rendered frames, preserving the click's offset in the patch
def test_tracker_recovers_shift_and_keeps_offset():
    m = lb_spec(vx=16, irrelevant=True).manifest()
    f0 = m.render(200_000_000)
    cx, cy = m.target_box_at(200_000_000).center()
    tpl = cut_template(f0, W, H, cx + 4, cy - 3, 12)
    for t_ns in (200_000_000, 600_000_000, 1_000_000_000):
        bx, by = m.target_box_at(t_ns).center()
        mt = track(tpl, m.render(t_ns), W, H)
        assert (mt.x, mt.y) == (bx + 4, by - 3) and mt.score > 0.99


# (ii) late_bind: same decision id, coordinate_correction, frozen counterfactual misses at the same A
def test_late_bind_commits_same_decision_with_tracked_coordinate(ws):
    spec = lb_spec("late_bind")
    x, y = proposal_at_appearance(spec)
    t, r = run(spec, ws, SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.5))
    c = t.first("commit")
    corr = t.first("coordinate_correction")
    assert c["decision_id"] == t.first("action_proposed")["decision_id"] == 1
    assert corr is not None and corr["binding"] == "tracked_at_D" and corr["matched"]
    assert corr["moved_px"] > TS / 2                      # 40 px/s x 0.5 s = 20 px > half a target
    assert corr["proposed_action"] == {"kind": "click", "x": x, "y": y}
    assert c["action"] == corr["corrected_action"] != corr["proposed_action"]
    assert r.valid_trial and r.valid_at_receipt is True and r.binding == "tracked_at_D"
    assert r.proposed_valid_at_receipt is False             # frozen counterfactual, scored at the SAME A
    assert r.proposed_spatial_error_px > r.spatial_error_px
    assert r.n_restarts == 0 and r.contract_violations == [] and "late" not in r.labels
    assert t.all("decision_restarted") == []


# (iii) a real frozen run on the same answer agrees with the counterfactual
def test_frozen_run_matches_counterfactual(ws):
    spec = lb_spec("frozen")
    x, y = proposal_at_appearance(spec)
    t, r = run(spec, ws, SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.5))
    assert t.first("coordinate_correction") is None
    assert t.first("commit")["action"] == {"kind": "click", "x": x, "y": y}
    assert r.valid_at_receipt is False and r.binding == "state_at_B" and r.proposed_valid_at_receipt is None
    assert r.contract_violations == []


# (iv) restart_if_moved: the proposal is discarded, a NEW decision id commits, D is later, cost is visible
def test_restart_if_moved_starts_a_new_decision(ws):
    spec = lb_spec("restart_if_moved", threshold=6)
    x, y = proposal_at_appearance(spec)
    ad = SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.2)
    t, r = run(spec, ws, ad)
    rs = t.all("decision_restarted")   # the mock keeps answering the stale position, so it restarts again
    assert len(rs) >= 1 and rs[0]["decision_id"] == 1 and rs[0]["reason"] == "moved_since_B" and rs[0]["moved_px"] > 6
    assert [x["decision_id"] for x in rs] == list(range(1, len(rs) + 1))
    assert t.first("commit")["decision_id"] == len(rs) + 1 and len(ad.calls) == len(rs) + 1
    assert r.n_restarts == len(rs) and r.first_b_to_d_ns > r.b_to_d_ns >= 200_000_000
    assert r.contract_violations == []


def test_restart_policy_commits_when_nothing_moved(ws):
    spec = lb_spec("restart_if_moved", vx=0, threshold=6)
    x, y = proposal_at_appearance(spec)
    t, r = run(spec, ws, SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.2))
    assert t.all("decision_restarted") == [] and t.first("commit")["decision_id"] == 1
    assert r.valid_at_receipt is True and r.n_restarts == 0


# (v) negative controls: static target and irrelevant motion leave the action unchanged
@pytest.mark.parametrize("irrelevant", [False, True])
def test_late_bind_is_inert_when_target_is_static(ws, irrelevant):
    spec = lb_spec("late_bind", vx=0, irrelevant=irrelevant)
    x, y = proposal_at_appearance(spec)
    click = {"action": "click", "x": 2 * x, "y": 2 * y}
    # with irrelevant motion the blinking corner triggers an invocation before the balloon appears;
    # a real model answers "wait" on that empty frame, and so does the mock
    t, r = run(spec, ws, SlowMock(lambda n: answer({"action": "wait"} if (irrelevant and n == 1) else click), 0.2))
    corr = t.first("coordinate_correction")
    assert r.n_decisions == (2 if irrelevant else 1)
    assert corr["matched"] and corr["moved_px"] == 0 and corr["corrected_action"] == corr["proposed_action"]
    assert r.valid_at_receipt is True and r.proposed_valid_at_receipt is True
    assert r.spatial_error_px == r.proposed_spatial_error_px


# negative contract tests: a tracker record without its fields; a restarted id that commits
def _rewrite(path, fn):
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    recs = fn(recs)
    path.write_text("".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in recs))


def test_evaluator_flags_incomplete_tracker_record(ws):
    spec = lb_spec("late_bind")
    x, y = proposal_at_appearance(spec)
    run(spec, ws, SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.5))

    def strip(recs):
        for rr in recs:
            if rr["record"] == "coordinate_correction":
                for k in ("match_score", "track_ns"):
                    rr.pop(k)
        return recs
    _rewrite(ws / "t.jsonl", strip)
    r = evaluate_trace(Trace.load(ws / "t.jsonl"))
    assert "correction_record_incomplete" in r.contract_violations


def test_evaluator_flags_restarted_decision_that_commits(ws):
    spec = lb_spec("restart_if_moved", threshold=6)
    x, y = proposal_at_appearance(spec)
    run(spec, ws, SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.2))

    def forge(recs):  # pretend the restarted decision (1) is the one that committed
        for rr in recs:
            if rr["record"] in ("commit", "decision_restarted"):
                rr["decision_id"] = 1
        return recs
    _rewrite(ws / "t.jsonl", forge)
    r = evaluate_trace(Trace.load(ws / "t.jsonl"))
    assert "decision_id_reused" in r.contract_violations


def test_spec_rejects_inconsistent_restart_threshold():
    with pytest.raises(SpecError):
        lb_spec("late_bind", threshold=6)
    with pytest.raises(SpecError):
        lb_spec("restart_if_moved", threshold=None)
    with pytest.raises(SpecError):
        lb_spec("rebind_please")


def test_restart_with_a_different_second_proposal_is_not_an_uncorrected_change(ws):
    """Regression (WI-009 agent run): the evaluator must compare the committed action with the proposal of
    the COMMITTING decision, not with the first proposal of the trial (which belonged to a restarted one)."""
    spec = lb_spec("restart_if_moved", threshold=6)          # moving target -> the stale proposal restarts
    x, y = proposal_at_appearance(spec)
    ad = SlowMock(lambda n: answer({"action": "click", "x": 2 * (x + n), "y": 2 * y}), 0.2)   # each proposal distinct
    t, r = run(spec, ws, ad)
    c = t.first("commit")
    assert c["decision_id"] >= 2 and len(t.all("action_proposed")) == c["decision_id"]
    assert c["action"] != t.first("action_proposed")["action"]
    assert "uncorrected_action_change" not in r.contract_violations and r.contract_violations == []


# ---- reviewer-requested controls (C1): duplicate frame, post-D immutability, action-kind change ---------------

def test_duplicate_frame_gives_zero_translation_and_typed_fields(ws):
    """Static target, no distractor: the newest frame at D is byte-identical to the frame at B -> the binder's
    translation is exactly (0, 0) and the provisional/resolved records carry the typed fields."""
    spec = lb_spec("late_bind", vx=0)
    x, y = proposal_at_appearance(spec)
    t, r = run(spec, ws, SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.2))
    prop = t.first("action_proposed"); corr = t.first("coordinate_correction")
    by_id = {o["observation_id"]: o for o in t.all("observation")}
    assert by_id[corr["observation_id"]]["payload_hash"] == by_id[t.committing_decision()["observation_id"]]["payload_hash"]
    assert (corr["translation_x"], corr["translation_y"]) == (0, 0) and corr["corrected_action"] == prop["action"]
    assert prop["coordinate_status"] == "provisional" and corr["coordinate_status"] == "resolved"
    assert corr["binder_end_ns"] >= corr["binder_start_ns"] == corr["proposal_time_ns"] == prop["proposal_time_ns"]
    assert t.first("commit")["commitment_time_ns"] == corr["binder_end_ns"]     # D is stamped AFTER binding


def test_frames_after_D_do_not_change_the_committed_click(ws):
    """Post-D motion: the balloon keeps moving after commitment; what the application receives is byte-identical
    to what was committed, and no later record touches the action."""
    spec = lb_spec("late_bind")
    x, y = proposal_at_appearance(spec)
    t, r = run(spec, ws, SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.5))
    c = t.first("commit"); rc = t.first("receipt")
    later = [o for o in t.all("observation") if o["publication_time_ns"] > c["commitment_time_ns"]]
    assert later, "frames did arrive after D"
    assert rc["received_action_hash"] == c["serialized_action_hash"] and "actuation_error" not in r.labels
    assert [x_["record"] for x_ in t.records if x_.get("decision_id") == c["decision_id"] and x_["seq"] > c["seq"]] == []


def test_action_kind_change_before_D_is_never_a_rebind(ws):
    """The target vanishes before D (window closes while the model thinks), so the right response at D is
    `wait`. late_bind must not turn the click into a wait under the same decision id: it commits the stale
    click (labelled late) — and late_bind_or_restart must instead discard the proposal and begin a NEW decision,
    which then answers wait."""
    # window 1.0 s from 0.1 s; the mock takes 1.2 s -> the balloon is gone at D
    for policy in ("late_bind", "late_bind_or_restart"):
        spec = lb_spec(policy, vx=0)
        x, y = proposal_at_appearance(spec)
        ad = SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y} if n == 1 else {"action": "wait"}), 1.2)
        t, r = run(spec, ws, ad, name=policy)
        if policy == "late_bind":
            c = t.first("commit"); corr = t.first("coordinate_correction")
            assert c and c["action"]["kind"] == "click" and corr["matched"] is False and corr["corrected_action"] == corr["proposed_action"]
            assert "late" in r.labels and t.all("decision_restarted") == []
        else:
            rs = t.all("decision_restarted")
            assert len(rs) == 1 and rs[0]["reason"] == "target_not_found_at_D" and rs[0]["decision_id"] == 1
            assert t.first("commit") is None                       # the new decision answered wait; nothing committed
            assert [i.get("outcome") for i in t.all("decision_invocation")][-1] == "wait"
