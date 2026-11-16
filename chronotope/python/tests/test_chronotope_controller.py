"""KGUAController on the software track: the resident binder against tracker.py, run_trial with the detector
and the vlm planners (mock model), the frozen counterfactual, restart_if_moved, the tracker_py negative
control, and the spec registration. Needs the built library (chronotope/build/libchronotope.so) and a GPU."""
from __future__ import annotations

import os
import pathlib
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]          # chronotope/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "chronotope" / "python"))

import chronotope_core  # noqa: E402
from chronotope_core import glue  # noqa: E402
from chronotope_core.controller import build  # noqa: E402

from chronotope.controller import CallBudget  # noqa: E402
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, run_trial  # noqa: E402
from chronotope.spec import SpecError, TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402
from chronotope.tracker import cut_template, track  # noqa: E402
from conftest import base_spec_dict  # noqa: E402
from test_late_binding import SlowMock, proposal_at_appearance  # noqa: E402
from test_vlm_mock import answer  # noqa: E402

W, H, TS = 320, 240, 24
glue.register()


def k_spec(policy="late_bind", planner="detector", binder="cuda", vx=40, irrelevant=False, threshold=None, seed=3,
           compute_ns=500_000_000) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "moving_target", "scenario_seed": seed, "canvas_w_px": W, "canvas_h_px": H,
                     "target_size_px": TS, "event_time_ns": 100_000_000, "response_window_ns": 1_000_000_000,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 1_500_000_000,
                     "irrelevant_motion": irrelevant, "skin": "balloon", "velocity_x_px_per_s": vx, "velocity_y_px_per_s": 0}
    vlm = None
    if planner == "vlm":
        vlm = {"server": "openai_compatible_local", "base_url": "http://127.0.0.1:8000/v1", "model_id": "mock-model",
               "prompt_version": 3, "image_scale": 2, "max_tokens": 64, "temperature": 0.0, "enable_thinking": False,
               "reasoning_effort": None, "invoke_on": "frame_change"}
    d["controller"] = glue.controller_dict(planner=planner, binder=binder, at_commit_policy=policy, restart_threshold_px=threshold,
                                           compute_ns=compute_ns if planner == "detector" else 0, vlm=vlm)
    d["timestamp_source"] = "monotonic_ns"
    return TrialSpec.from_dict(d)


def run(spec, ws, adapter=None, name="t"):
    ctrl = build(spec, adapter=adapter, budget=CallBudget(8))
    try:
        p = run_trial(spec, ws / f"{name}.jsonl", PayloadStore(ws / "p"), controller=ctrl)
    finally:
        ctrl.close()
    t = Trace.load(p)
    return t, evaluate_trace(t), ctrl


def test_library_loads():
    assert chronotope_core.version().startswith("chronotope-core")


def test_resident_binder_agrees_with_tracker_py():
    m = k_spec(vx=16, irrelevant=True).manifest()
    f0 = m.render(200_000_000)
    cx, cy = m.target_box_at(200_000_000).center()
    with chronotope_core.Core(width=W, height=H, target_rgb=m.target_color()) as core:
        for t_ns in (200_000_000, 600_000_000, 1_000_000_000):
            fd = m.render(t_ns)
            py = track(cut_template(f0, W, H, cx + 4, cy - 3, 12), fd, W, H)
            cu = core.track(f0, fd, cx + 4, cy - 3, 12)
            assert (cu["x"], cu["y"]) == (py.x, py.y) == (m.target_box_at(t_ns).center()[0] + 4, m.target_box_at(t_ns).center()[1] - 3)
            # scores agree to the card's 1e-4 (the residue is the float32 template mean: numpy sums pairwise, the kernel in double)
            assert abs(cu["score"] - py.score) < 1e-4 and abs(cu["second"] - py.second) < 1e-4 and cu["plateau"] == py.plateau
            assert cu["matched"] and not cu["ambiguous"]


def test_detector_late_bind_commits_same_decision(ws):
    spec = k_spec("late_bind")
    t, r, ctrl = run(spec, ws)
    c, corr = t.first("commit"), t.first("coordinate_correction")
    assert c["decision_id"] == t.first("action_proposed")["decision_id"] == 1
    assert corr is not None and corr["binding"] == "tracked_at_D" and corr["matched"] and corr["binder"] == "cuda"
    assert corr["moved_px"] > TS / 2 and c["action"] == corr["corrected_action"] != corr["proposed_action"]
    assert corr["residency_ok"] is True and corr["audit"]["d2h_state_in_window"] == 0 and corr["audit"]["host_action_computations"] == 0
    assert corr["device_outcome"] == "committed" and corr["device_action_id"] == 1 and corr["track_device_ns"] > 0
    assert r.valid_trial and r.valid_at_receipt is True and r.binding == "tracked_at_D"
    assert r.proposed_valid_at_receipt is False and r.proposed_spatial_error_px > r.spatial_error_px
    assert r.n_restarts == 0 and r.contract_violations == [] and "late" not in r.labels
    assert r.b_to_d_ns >= 500_000_000                         # the emulated deliberation is real time


def test_detector_frozen_misses_by_the_travel(ws):
    spec = k_spec("frozen")
    t, r, ctrl = run(spec, ws)
    corr = t.first("coordinate_correction")
    assert corr is not None and corr["binding"] == "state_at_B" and corr["corrected_action"] == corr["proposed_action"]
    assert t.first("commit")["action"] == corr["proposed_action"]
    assert r.valid_at_receipt is False and r.binding == "state_at_B" and "outdated_at_commitment" in r.labels
    assert r.contract_violations == [] and corr["residency_ok"] is True
    # the error law: the unchanged coordinate misses by v (t_A - tau)
    assert abs(r.spatial_error_px - 40 * (r.intervals_ns["I_A"] / 1e9)) <= 2.5


def test_vlm_planner_with_slow_mock(ws):
    spec = k_spec("late_bind", planner="vlm")
    x, y = proposal_at_appearance(spec)
    ad = SlowMock(lambda n: answer({"action": "click", "x": 2 * x, "y": 2 * y}), 0.5)
    t, r, ctrl = run(spec, ws, ad)
    corr = t.first("coordinate_correction")
    assert len(ad.calls) == 1 and corr is not None and corr["binding"] == "tracked_at_D" and corr["matched"]
    assert corr["proposed_action"] == {"kind": "click", "x": x, "y": y} and corr["residency_ok"] is True
    assert r.valid_at_receipt is True and r.proposed_valid_at_receipt is False and r.contract_violations == []
    inv = t.first("decision_invocation")
    assert inv["invocation_id"] == "req_mock" and inv["planner"] == "vlm"


def test_restart_if_moved_starts_a_new_decision(ws):
    spec = k_spec("restart_if_moved", threshold=6, compute_ns=200_000_000)
    t, r, ctrl = run(spec, ws)
    rs = t.all("decision_restarted")
    assert len(rs) >= 1 and rs[0]["reason"] == "moved_since_B" and rs[0]["moved_px"] > 6 and rs[0]["device_outcome"] == "restart_moved"
    assert t.first("commit")["decision_id"] == len(rs) + 1 and r.n_restarts == len(rs)
    assert r.contract_violations == []


def test_tracker_py_binder_is_a_recorded_violation(ws):
    spec = k_spec("late_bind", binder="tracker_py")
    t, r, ctrl = run(spec, ws)
    corr = t.first("coordinate_correction")
    assert corr["binder"] == "tracker_py" and corr["binding"] == "tracked_at_D" and corr["matched"]
    assert corr["residency_ok"] is False and corr["audit"]["d2h_state_in_window"] >= 1 and corr["audit"]["host_action_computations"] == 1
    assert r.valid_at_receipt is True and r.contract_violations == []   # it binds correctly; it is just not resident
    assert corr["track_ns"] > 1_000_000                                 # the host binder's cost is visible (2.5 ms on an idle GPU, 2026-08-27)


def test_spec_registration_and_unknown_field():
    spec = k_spec()
    assert spec.raw["controller"]["kind"] == "chronotope"
    d = base_spec_dict()
    d["controller"] = {**glue.controller_dict(planner="detector", binder="cuda", at_commit_policy="frozen"), "bogus": 1}
    d["timestamp_source"] = "monotonic_ns"
    with pytest.raises(SpecError):
        TrialSpec.from_dict(d)


def test_grounder_binder_with_mock_re_grounds_at_D(ws):
    """P2 H-B1: a grounding model asked on the newest frame at D fills the open field; the mock answers the true centre
    of the frame it is shown (the oracle upper bound), so the same-decision result must match the NCC binder's."""
    spec = k_spec("late_bind", binder="grounder")
    m = spec.manifest()
    from test_vlm_mock import MockAdapter

    class TruthGrounder(MockAdapter):
        ctrl = None

        def invoke(self, png, system, user_text, structured=True, schema=None, parser=None):
            self.calls.append((png, system, user_text))
            import numpy as np                            # the truth of the frame it was shown: the target-colour centroid
            fr = np.frombuffer(self.ctrl._newest.payload, dtype=np.uint8).reshape(m.canvas_h_px, m.canvas_w_px, 3)
            ys, xs = np.nonzero(np.all(fr == np.array(m.target_color(), dtype=np.uint8), axis=2))
            cx, cy = int(round(xs.mean())), int(round(ys.mean()))
            return answer({"action": "click", "x": round(1000 * cx / m.canvas_w_px), "y": round(1000 * cy / m.canvas_h_px)})   # prompt 5: norm1000

    g = TruthGrounder(lambda n: None)
    ctrl = build(spec, budget=CallBudget(8), grounder_adapter=g)
    g.ctrl = ctrl
    try:
        p = run_trial(spec, ws / "g.jsonl", PayloadStore(ws / "p"), controller=ctrl)
    finally:
        ctrl.close()
    t = Trace.load(p)
    r = evaluate_trace(t)
    corr = t.first("coordinate_correction")
    assert len(g.calls) == 1 and corr["binder"] == "grounder" and corr["binding"] == "tracked_at_D" and corr["matched"] and corr["binder_host_steps"] == 1
    assert corr["evidence_age_ns"] >= 0 and corr["ledger_row"] >= 0
    assert r.valid_at_receipt is True and r.proposed_valid_at_receipt is False and r.contract_violations == []


def test_detector_binder_at_d(ws):
    """2026-08-27: the detector as binder. Rule 1 (binder "detector"): the colour box on the newest frame at D is the
    committed location, binding detected_at_D; rule 2 ("ncc_then_detector"): the tracked template wins when it matches."""
    for binder, want in (("detector", "detected_at_D"), ("ncc_then_detector", "tracked_at_D")):
        spec = k_spec("late_bind", binder=binder, vx=40)
        t, r, ctrl = run(spec, ws, name=f"det_{binder}")
        corr, c = t.first("coordinate_correction"), t.first("commit")
        assert corr["binder"] == binder and corr["device_binding"] == want and corr["matched"], (binder, corr["device_binding"])
        assert r.valid_at_receipt is True and r.contract_violations == [] and corr["residency_ok"] is True
        m = spec.manifest()
        cx, cy = m.target_box_at(c["task_time_at_commit_ns"]).center()
        assert abs(c["action"]["x"] - cx) <= 2 and abs(c["action"]["y"] - cy) <= 2, (binder, c["action"], (cx, cy))


def test_cpp_agent_process_matches_ctypes_path(ws):
    """K4b: the same trial through the C++ agent process (chronotope --serve, SocketCore) and through the ctypes Core commits
    the same bound action with the same binding; the trace names the agent process."""
    import json
    spec_dict = json.loads(k_spec("late_bind", binder="detector", vx=40).to_json()) if hasattr(k_spec("late_bind"), "to_json") else None
    spec_a = k_spec("late_bind", binder="detector", vx=40)
    t_a, r_a, _ = run(spec_a, ws, name="ctypes")
    d = json.loads(json.dumps(spec_a.raw)); d["controller"]["server"] = "cpp_socket"
    spec_b = TrialSpec.from_dict(d)
    t_b, r_b, ctrl_b = run(spec_b, ws, name="cpp")
    ca, cb = t_a.first("commit"), t_b.first("commit")
    corr_a, corr_b = t_a.first("coordinate_correction"), t_b.first("coordinate_correction")
    assert cb["action"] == ca["action"] and corr_b["device_binding"] == corr_a["device_binding"] == "detected_at_D"
    assert r_b.valid_at_receipt is True and r_b.contract_violations == [] and corr_b["residency_ok"] is True
    inv = t_b.first("decision_invocation")
    assert "chronotope --serve" in (inv.get("agent_process") or corr_b.get("agent_process") or ctrl_b.core.agent_process)


@pytest.mark.skipif(not os.environ.get("CHRONOTOPE_TRT_ENGINES"), reason="CHRONOTOPE_TRT_ENGINES not set (the 0.8B engines)")
def test_library_trt_planner_matches_process_path(ws):
    """K4d: the model inside libchronotope.so (ctypes, no socket) and inside the `chronotope --serve` process commit the
    same action on the same episode; the audit shows no host copy of the frame on either path."""
    import json
    E = os.environ["CHRONOTOPE_TRT_ENGINES"]
    base = k_spec("late_bind", binder="detector", vx=0)
    d = json.loads(json.dumps(base.raw))
    d["controller"].update({"planner": "trt", "server": "in_process", "base_url": E, "model_id": "Qwen3.5-0.8B", "prompt_version": 5,
                            "image_scale": 2, "max_tokens": 64, "temperature": 0.0, "enable_thinking": False, "reasoning_effort": None,
                            "invoke_on": "frame_change", "compute_ns": 0})
    t_lib, r_lib, _ = run(TrialSpec.from_dict(d), ws, name="lib_trt")
    d["controller"]["server"] = "cpp_socket"
    t_proc, r_proc, _ = run(TrialSpec.from_dict(d), ws, name="proc_trt")
    for t, r, tag in ((t_lib, r_lib, "lib"), (t_proc, r_proc, "proc")):
        inv, corr = t.first("decision_invocation"), t.first("coordinate_correction")
        assert inv["planner"] == "tensorrt_edgellm" and inv["outcome"] == "click", (tag, inv.get("outcome"), inv.get("raw_text"))
        assert corr["audit"]["planner_d2h_pixels"] == 0 and corr["residency_ok"] is True, (tag, corr["audit"])
        assert r.valid_at_receipt is True and r.contract_violations == [], (tag, r.contract_violations)
    assert t_lib.first("commit")["action"] == t_proc.first("commit")["action"]
    assert "libchronotope" in t_lib.first("decision_invocation")["agent_process"] and "--serve" in t_proc.first("decision_invocation")["agent_process"]
