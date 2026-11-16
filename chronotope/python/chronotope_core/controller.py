"""KGUAController — the resident control core as a chronotope controller (software track).

Two planners: `detector` (the resident colour detector clicks the target's centre in the frame at B, then
an emulated deliberation of `compute_ns` wall-clock, as the scripted oracle does) and `vlm` (the paper's
own fixed-input model call through LocalOpenAIAdapter; its answer enters the core as the typed pre-commit
state). Two binders: `cuda` (resident: the newest frame never leaves the device between O and D) and
`tracker_py` (the host binder, chronotope.tracker, the NEGATIVE control: the frame is read to the host
inside the window and the binding is pushed back; the audit records it).

The runner's protocol is host-driven (decide -> correct_at_commit -> host commit record); on this
controller the device performs the atomic commit inside correct_at_commit, and the runner's commit record
mirrors it. The device record, the binder's report and the audit counters travel in the correction
record's detail. The kGUA writes a coordinate_correction record for EVERY commit so that the device record
enters the trace; binding "state_at_B" marks a frozen commit (the evaluator judges it as frozen).
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from chronotope.controller import (CallBudget, ControlUpdate, Correction, Decision, LocalOpenAIAdapter, Observation, Restart,
                                   VLMDirectController)
from chronotope.hashing import hash_obj
from chronotope.png import to_rgb

from . import BINDING, Core, version

RESTART_REASONS = {"restart_moved": "moved_since_B", "restart_not_found": "target_not_found_at_D", "restart_kind": "kind",
                   "rejected_duplicate": "rejected_duplicate", "rejected_repeated": "rejected_repeated",
                   "failed_state_changed": "control_state_changed_during_validation", "failed_schema": "schema"}


class _VLMPlanner(VLMDirectController):
    """The paper's fixed-input model call, restartable (a discarded proposal lets the next changed frame start a decision)."""

    kind = "chronotope_vlm_planner"

    def restart(self) -> None:
        self._done, self._baseline_hash = False, "<restart>"


class KGUAController:
    kind = "chronotope"

    def __init__(self, *, canvas_w_px: int, canvas_h_px: int, planner: str, binder: str, at_commit_policy: str,
                 template_radius_px: int, min_match_score: float, restart_threshold_px: int | None, target_color,
                 coordinate_offset_px: int = 0, compute_ns: int = 0, validate_kind: bool = False, movement_threshold_px: int = 6,
                 rgb_template: bool = False, vlm: _VLMPlanner | None = None, wall_clock: bool = True,
                 grounder: _VLMPlanner | None = None, max_evidence_age_ns: int = 0, multi_action: bool = False,
                 memory: str = "none", pending_timeout_ns: int = 0, pending_radius_px: int = 0,
                 cpp_socket: dict | None = None, ack_aware_pending: bool = False, receipt_timeout_ns: int = 0,
                 trt: dict | None = None) -> None:
        if planner not in ("detector", "vlm", "trt", "api") or binder not in ("cuda", "tracker_py", "grounder", "detector", "ncc_then_detector"):
            raise ValueError((planner, binder))
        if planner == "api" and cpp_socket is None:
            raise ValueError('Hosted legacy inference requires the C++ socket process')
        if planner == "vlm" and vlm is None:
            raise ValueError("planner vlm needs the model planner")
        if planner == "detector" and target_color is None:
            raise ValueError("the detector planner needs the target colour")
        if binder == "grounder" and grounder is None:
            raise ValueError("binder grounder needs a grounding model (an adapter)")
        self.grounder, self.max_evidence_age_ns = grounder, max_evidence_age_ns
        self.multi_action, self.memory = multi_action, memory
        if memory not in ("none", "harness_pending"):
            raise ValueError(memory)
        self.w, self.h = canvas_w_px, canvas_h_px
        self.planner, self.binder, self.policy = planner, binder, at_commit_policy
        self.radius, self.min_score, self.restart_px = template_radius_px, min_match_score, restart_threshold_px
        self.target_color = tuple(target_color) if target_color is not None else None
        self.offset, self.compute_ns, self.validate_kind, self.wall_clock = coordinate_offset_px, compute_ns, validate_kind, wall_clock
        self.rgb_template = rgb_template
        self.vlm = vlm
        core_cls = Core
        if cpp_socket is not None:   # K4b: the agent is a C++ process; this object is only the referee's client
            from .remote import SocketCore
            core_cls = lambda **kw: SocketCore(**kw, **cpp_socket)   # noqa: E731
        self.core = core_cls(width=canvas_w_px, height=canvas_h_px, template_radius_px=template_radius_px, min_match_score=min_match_score,
                         policy=at_commit_policy, restart_threshold_px=restart_threshold_px, target_rgb=self.target_color,
                         coordinate_offset_px=coordinate_offset_px, movement_threshold_px=movement_threshold_px,
                         validate_kind=validate_kind, rgb_template=rgb_template, track_every_frame=True,
                         max_evidence_age_ns=max_evidence_age_ns,
                         pending_radius_px=pending_radius_px if memory == "harness_pending" else 0,
                         pending_timeout_ns=pending_timeout_ns if memory == "harness_pending" else 0,
                         detector_at_d={"detector": 1, "ncc_then_detector": 2}.get(binder, 0),
                         ack_aware_pending=bool(ack_aware_pending), receipt_timeout_ns=int(receipt_timeout_ns))
        self.model_id = (trt or cpp_socket or {}).get("model_id")   # the spec's checkpoint name, reported on every invocation
        if planner == "trt" and cpp_socket is None:   # K4d: the model inside libchronotope.so (ctypes path, no socket)
            if not trt:
                raise ValueError("planner trt needs trt={engines_root, prompt_version, image_scale, max_tokens, temperature}")
            self.core.planner_trt_open(**{k: v for k, v in trt.items() if k != "model_id"})
        self.clock_ns = None
        self._newest: Observation | None = None
        self._box_present = False
        self._decided = False
        self._last_decided_hash: str | None = None   # multi-action: decide on every changed frame (ScriptedPendingController)
        self._decision_id = 0
        self._proposal: dict | None = None
        self._frame_b: Observation | None = None
        self._events: list[dict[str, Any]] = []
        self.commits: list[dict[str, Any]] = []      # every device commit result (for the study's report)

    # ---- protocol ----------------------------------------------------------------------------
    def spec_hash(self) -> str:
        d = {"kind": self.kind, "planner": self.planner, "binder": self.binder, "at_commit_policy": self.policy,
             "template_radius_px": self.radius, "min_match_score": self.min_score, "restart_threshold_px": self.restart_px,
             "coordinate_offset_px": self.offset, "compute_ns": self.compute_ns, "validate_kind": self.validate_kind,
             "rgb_template": self.rgb_template, "binder_version": "chronotope-1", "core_version": version(),
             "max_evidence_age_ns": self.max_evidence_age_ns, "multi_action": self.multi_action, "memory": self.memory,
             **({"grounder": self.grounder.spec_hash()} if self.grounder is not None else {})}
        if self.vlm is not None:
            d["vlm"] = self.vlm.spec_hash()
        return hash_obj(d)

    def bind_clock(self, clock_ns) -> None:
        self.clock_ns = clock_ns
        if self.vlm is not None:
            self.vlm.bind_clock(clock_ns)

    def drain_events(self) -> list[dict[str, Any]]:
        ev, self._events = self._events, []
        if self.vlm is not None:
            ev += self.vlm.drain_events()
        return ev

    def _rgb(self, obs: Observation) -> bytes:
        p = to_rgb(obs.payload, self.w, self.h)
        if len(p) == self.w * self.h:            # grey payload: the core wants RGB24
            p = np.repeat(np.frombuffer(p, dtype=np.uint8), 3).tobytes()
        if len(p) != self.w * self.h * 3:
            raise ValueError(f"payload of {len(p)} bytes is not {self.w}x{self.h} RGB")
        return p

    def admit(self, obs: Observation) -> ControlUpdate:
        a = self.core.admit(self._rgb(obs), observation_id=obs.observation_id, source_event_id=obs.source_event_id,
                            source_time_ns=obs.source_time_ns, publication_time_ns=obs.publication_time_ns)
        if a["stale"]:
            raise RuntimeError(f"observation {obs.observation_id} is not newer than the newest admitted (§7.2)")
        if self.vlm is not None:
            self.vlm.admit(obs)
        self._newest = obs
        self._box_present = bool(a["box_present"])
        if self.planner in ("trt", "api") and self._last_decided_hash is None:
            self._last_decided_hash = obs.payload_hash   # like the vlm planner: the first frame is the baseline; decide on the first CHANGE
        return ControlUpdate(relevance=a["relevance_name"], controller_state_hash=f"{a['controller_state_hash']:016x}")

    def ready_to_decide(self) -> bool:
        if self.planner == "vlm":
            return self.vlm.ready_to_decide()
        if self.multi_action:   # the scripted-pending convention: a new decision on every frame that differs from the last decided one
            return self._newest is not None and self._box_present and self._newest.payload_hash != self._last_decided_hash
        if self.planner in ("trt", "api"):
            # the in-process model on a single-response task: like the vlm planner, one decision per CHANGED frame (a "wait"
            # on a frozen frame is not re-asked; under Paused the clock is frozen, so re-asking would never end)
            return not self._decided and self._newest is not None and self._newest.payload_hash != self._last_decided_hash
        return not self._decided and self._newest is not None and self._box_present

    def decide(self) -> Decision:
        assert self._newest is not None
        obs = self._newest
        self._decision_id += 1
        b = self.core.begin_decision(obs.observation_id)
        self._frame_b = obs
        self._last_decided_hash = obs.payload_hash
        if self.planner in ("trt", "api"):
            # K4b: the model runs inside the C++ agent on the frame at B (device pointer); this call blocks for the answer
            r = self.core.decide_api() if self.planner == 'api' else self.core.decide_trt()
            detail = {"observation_id": obs.observation_id, "device_b_ns": b["b_device_ns"], "b_host_ns": b["b_host_ns"], "planner": "hosted_api" if self.planner == 'api' else "tensorrt_edgellm",
                      "model_reported": self.model_id or r.get("model_reported"), "raw_text": r.get("raw_text"), "input_tokens": r.get("input_tokens"),
                      "output_tokens": r.get("output_tokens"), "deliberation_ms": r.get("deliberation_ms"), "prefill_ms": r.get("prefill_ms"),
                      "decode_ms": r.get("decode_ms"), "planner_host_steps": r.get("planner_host_steps"), "image_x": r.get("image_x"), "image_y": r.get("image_y"),
                      "agent_process": getattr(self.core, "agent_process", None)}
            if self.planner == 'api':
                detail.update(model_reported=r.get('model_reported'), cpu_image_staging=True,
                              planner_d2h_pixels=r.get('planner_d2h_pixels'),
                              inference_runtime='remote API', precision='provider undisclosed')
            if not (r.get("answer_ok") and r.get("click") and r.get("proposal")):
                return Decision(action=None, declared_compute_ns=None, invocation_id=str(r.get("invocation_id")), invocation_detail={**detail, "outcome": "wait" if r.get("answer_ok") else "unparseable"})
            self._decided = True
            self._proposal = {"kind": "click", "x": int(r["x"]), "y": int(r["y"])}
            return Decision(action=dict(self._proposal), declared_compute_ns=None, invocation_id=str(r.get("invocation_id")), invocation_detail={**detail, "outcome": "click"})
        if self.planner == "detector":
            p = self.core.decide_detector()            # the proposal exists at B, from the frame at B
            if self.wall_clock and self.compute_ns > 0:
                time.sleep(self.compute_ns / 1e9)      # the emulated deliberation (the scripted oracle's convention)
            detail = {"observation_id": obs.observation_id, "device_b_ns": b["b_device_ns"], "b_host_ns": b["b_host_ns"],
                      "planner": "detector"}
            if p["status"] != 1:
                return Decision(action=None, declared_compute_ns=self.compute_ns, invocation_id="chronotope-detector",
                                invocation_detail={**detail, "outcome": "wait"})
            self._decided = True
            self._proposal = {"kind": "click", "x": p["x"], "y": p["y"]}
            return Decision(action=dict(self._proposal), declared_compute_ns=self.compute_ns, invocation_id="chronotope-detector",
                            invocation_detail={**detail, "outcome": "click"})
        d = self.vlm.decide()
        if d.action is None:
            self.core.abort_decision()
            return d
        p = self.core.propose("click", d.action["x"], d.action["y"])
        self._decided = True
        self._proposal = dict(d.action)
        return Decision(action=d.action, declared_compute_ns=None, invocation_id=d.invocation_id,
                        invocation_detail={**d.invocation_detail, "planner": "vlm", "device_b_ns": b["b_device_ns"],
                                           "chronotope_proposal_status": p["status"]})

    def restart(self) -> None:
        if self.core.decision_open():
            self.core.restart()
        self._decided, self._proposal, self._frame_b = False, None, None
        if self.multi_action:
            self._decided = False   # the next changed frame starts a new decision (a rejected duplicate is one such restart)
        if self.vlm is not None:
            self.vlm.restart()

    def correct_at_commit(self, commit_time_ns: int) -> Correction | Restart | None:
        if self._proposal is None or self._newest is None:
            return None
        px, py = self._proposal["x"], self._proposal["y"]
        host_track_ns = None
        if self.binder == "tracker_py" and self.policy != "frozen":
            # the negative control: the host binder (chronotope.tracker) reads the newest frame inside O->D and
            # returns a revised proposal to the device; the audit counts both
            from chronotope.tracker import AMBIGUITY_MARGIN, PLATEAU_MAX, cut_template, track
            assert self._frame_b is not None
            tpl = cut_template(self._frame_b.payload, self.w, self.h, px, py, self.radius, rgb=self.rgb_template)
            frame = self.core.read_frame()
            t0 = time.perf_counter_ns()
            m = track(tpl, frame, self.w, self.h)
            host_track_ns = time.perf_counter_ns() - t0
            ambiguous = m.second >= m.score - AMBIGUITY_MARGIN or m.plateau > PLATEAU_MAX
            matched = m.score >= self.min_score and not ambiguous
            x, y = (m.x, m.y) if matched else (px, py)
            x, y = min(max(0, x), self.w - 1), min(max(0, y), self.h - 1)
            self.core.override_binding(x, y)
            r = self.core.commit_only()
            r.update({"match_score": m.score, "second_best": m.second, "plateau": m.plateau, "matched": int(matched),
                      "ambiguous": int(ambiguous), "moved_px": math.hypot(m.x - px, m.y - py) if matched else 0.0,
                      "bind_valid": 1, "binding_name": "tracked_at_D" if r["outcome_name"] == "committed" else r["binding_name"]})
            r["track_ns"] = host_track_ns + r["track_ns"]
        elif self.binder == "grounder" and self.policy != "frozen":
            # P2 H-B1/H-B2: a small model re-grounds the NAMED target on the newest frame at D (Agent S3's planner/grounder
            # split applied at commitment). Its answer crosses host memory: counted as binder_host_steps, not as a host algorithm.
            from chronotope.png import encode_png
            t0 = time.perf_counter_ns()
            png = encode_png(self._newest.payload, self.w, self.h, self.grounder.scale)
            ans = self.grounder._invoke(png, "Screenshot attached. Click the target's center or wait.")
            host_track_ns = time.perf_counter_ns() - t0
            p = ans.parsed
            if isinstance(p, dict) and p.get("action") == "click" and type(p.get("x")) is int and type(p.get("y")) is int:
                g = self.grounder._to_canvas(p["x"], p["y"])
                self.core.bind_external(g["x"], g["y"], "grounder")
                matched = True
            else:
                self.core.bind_external(px, py, "grounder")   # no grounding: the proposal stands (recorded matched = False)
                matched = False
            r = self.core.commit_only()
            r.update({"match_score": 1.0 if matched else 0.0, "second_best": -1.0, "plateau": 1, "matched": int(matched), "ambiguous": 0,
                      "bind_valid": 1, "binding_name": "tracked_at_D" if r["outcome_name"] == "committed" else r["binding_name"],
                      "grounder_raw": ans.detail.get("raw_text", "")[:200], "grounder_invocation_id": ans.invocation_id})
            r["track_ns"] = host_track_ns + r["track_ns"]
        else:
            r = self.core.resolve_and_commit()
        self.commits.append(r)
        au = self.core.audit()
        detail = {"binder": self.binder, "binder_version": "chronotope-1" if self.binder == "cuda" else "tracker.py-v2-on-host",
                  "match_score": round(float(r["match_score"]), 4), "matched": bool(r["matched"]),
                  "moved_px": round(float(r["moved_px"]), 2), "track_ns": int(r["track_ns"]), "track_device_ns": int(r["track_device_ns"]),
                  "template_radius_px": self.radius, "min_match_score": self.min_score,
                  "second_best_score": round(float(r["second_best"]), 4), "plateau": int(r["plateau"]), "ambiguous": bool(r["ambiguous"]),
                  "proposal_observation_id": int(r["proposal_observation_id"]), "binder_channels": "rgb" if self.rgb_template else "grey",
                  "device_outcome": r["outcome_name"], "device_action_id": int(r["action_id"]), "device_decision_id": int(r["decision_id"]),
                  "device_commitment_time_ns": int(r["d_device_mapped_ns"]), "device_commit_globaltimer_ns": int(r["d_device_ns"]),
                  "device_commit_host_ns": int(r["d_host_ns"]), "device_serialized_action_hash": f"{r['serialized_action_hash']:016x}",
                  "device_control_state_version_at_commit": int(r["control_state_version_at_commit"]),
                  "device_supporting_control_state_version": int(r["supporting_control_state_version"]),
                  "device_binding": r["binding_name"], "audit": {k: au[k] for k in ("d2h_state_in_window", "d2h_log_in_window",
                                                                                "host_syncs_log_in_window", "host_syncs_action_in_window",
                                                                                "host_action_computations", "mallocs_after_warmup",
                                                                                "declared_commit_waits", "windows_opened", "h2d_publications",
                                                                                "planner_host_steps", "planner_d2h_pixels", "planner_requests")},
                  "residency_ok": bool(au["residency_ok"]), "clock_calibration_latency_ns": int(au["clock_calibration_latency_ns"]),
                  "evidence_age_ns": int(r["evidence_age_ns"]), "evidence_bracket_ns": int(r["evidence_bracket_ns"]), "ledger_row": int(r["ledger_row"]),
                  "binder_host_steps": int(au["binder_host_steps"]),
                  **({k: r[k] for k in ("grounder_raw", "grounder_invocation_id") if k in r})}
        newest_id = self._newest.observation_id
        out = r["outcome_name"]
        if out == "committed":
            if self.multi_action:
                self._decided = False
            if r["binding_name"] in ("tracked_at_D", "host_override", "grounder", "oracle"):
                binding = "tracked_at_D"
            else:
                binding = "state_at_B"
            self._decided = True
            return Correction({"kind": "click", "x": int(r["x"]), "y": int(r["y"])}, binding,
                              newest_id if binding == "tracked_at_D" else self._proposal_observation_id(), detail)
        reason = RESTART_REASONS.get(out, out)   # restart_stale_evidence passes through by name
        rs = Restart(reason, newest_id, {**detail, **({"restart_threshold_px": self.restart_px} if out == "restart_moved" else {})})
        return rs

    def _proposal_observation_id(self) -> int:
        return self._frame_b.observation_id if self._frame_b is not None else self._newest.observation_id

    def audit(self) -> dict:
        return self.core.audit()

    def close(self) -> None:
        self.core.close()


def build(spec, *, adapter=None, budget: CallBudget | None = None, grounder_adapter=None) -> KGUAController:
    """A KGUAController from a spec whose controller kind is "chronotope" (see glue.controller_dict)."""
    c, m = spec.raw["controller"], spec.manifest()
    if c["kind"] != "chronotope":
        raise ValueError(c["kind"])
    if spec.raw["timestamp_source"] != "monotonic_ns":
        raise ValueError("the kGUA core consumes real time; timestamp_source must be monotonic_ns")
    cpp_socket = None
    trt = None
    if c["planner"] == "trt" and c.get("server") != "cpp_socket":   # K4d: in-library planner (server "in_process" or unset)
        trt = {"engines_root": c.get("base_url"), "prompt_version": c.get("prompt_version") or 5, "image_scale": c.get("image_scale") or 2,
               "max_tokens": c.get("max_tokens") or 64, "temperature": c.get("temperature") or 0.0, "model_id": c.get("model_id")}
    if c.get("server") == "cpp_socket":
        cpp_socket = {"planner": c["planner"], "engines_root": c.get("base_url"), "model_id": c.get("model_id"), "prompt_version": c.get("prompt_version") or 5,
                      "image_scale": c.get("image_scale") or 2, "max_tokens": c.get("max_tokens") or 64, "temperature": c.get("temperature") or 0.0}
        if c['planner'] == 'api':
            cpp_socket.update(api_endpoint=c['base_url'], reasoning_effort=c['reasoning_effort'])
    vlm = None
    if c["planner"] == "vlm":
        if budget is None:
            raise ValueError("the vlm planner needs a call budget")
        if adapter is None and c.get("server") == "anthropic_api":   # 2026-08-28: a Claude model plans; the CUDA core binds and commits
            from chronotope.anthropic_adapter import AnthropicAdapter
            adapter = AnthropicAdapter(model_id=c["model_id"], max_tokens=c["max_tokens"] or 256, temperature=c["temperature"] or 0.0,
                                       effort=(c.get("reasoning_effort") or "low"))
        if adapter is None and c.get("server") == "edge_llm":
            from .edgellm import EdgeLLMAdapter   # in-process TensorRT Edge-LLM; base_url = the engines root
            adapter = EdgeLLMAdapter(engines_root=c["base_url"], model_id=c["model_id"], max_tokens=c["max_tokens"],
                                     temperature=c["temperature"], enable_thinking=bool(c["enable_thinking"]),
                                     reasoning_effort=c["reasoning_effort"])
        if adapter is None:
            adapter = LocalOpenAIAdapter(base_url=c["base_url"], model_id=c["model_id"], max_tokens=c["max_tokens"],
                                         temperature=c["temperature"], enable_thinking=c["enable_thinking"],
                                         reasoning_effort=c["reasoning_effort"])
        vlm = _VLMPlanner(adapter=adapter, budget=budget, prompt_version=c["prompt_version"], image_scale=c["image_scale"],
                          invoke_on=c["invoke_on"], canvas_w_px=m.canvas_w_px, canvas_h_px=m.canvas_h_px, skin=m.skin,
                          instruction_color=getattr(m, "instruction_color", lambda: None)(), target_color=m.target_color())
    grounder = None
    if c["binder"] == "grounder":
        if grounder_adapter is None and c.get("server") == "edge_llm" and budget is not None:
            from .edgellm import EdgeLLMAdapter   # the same in-process engine answers the grounding question at D
            grounder_adapter = EdgeLLMAdapter(engines_root=c["base_url"], model_id=c["model_id"], max_tokens=c["max_tokens"] or 64,
                                              temperature=c["temperature"] or 0.0)
        if grounder_adapter is None or budget is None:
            raise ValueError("binder grounder needs grounder_adapter and a budget")
        grounder = _VLMPlanner(adapter=grounder_adapter, budget=budget, prompt_version=c["prompt_version"] or 5, image_scale=c["image_scale"] or 2,
                               invoke_on="frame_change", canvas_w_px=m.canvas_w_px, canvas_h_px=m.canvas_h_px, skin=m.skin,
                               instruction_color=getattr(m, "instruction_color", lambda: None)(), target_color=m.target_color())
    return KGUAController(canvas_w_px=m.canvas_w_px, canvas_h_px=m.canvas_h_px, planner=c["planner"], binder=c["binder"],
                          at_commit_policy=c["at_commit_policy"], template_radius_px=c["template_radius_px"],
                          min_match_score=c["min_match_score"], restart_threshold_px=c["restart_threshold_px"],
                          target_color=m.target_color(), coordinate_offset_px=c["coordinate_offset_px"], compute_ns=c["compute_ns"],
                          validate_kind=c.get("validate_kind", False), movement_threshold_px=c["movement_threshold_px"],
                          rgb_template=bool(getattr(m, "instruction_color", lambda: None)()), vlm=vlm, grounder=grounder,
                          max_evidence_age_ns=c.get("max_evidence_age_ns", 0), multi_action=bool(getattr(m, "multi_action", False)),
                          memory=c.get("memory", "none"), pending_timeout_ns=c.get("pending_timeout_ns", 0), pending_radius_px=c.get("pending_radius_px", 0),
                          cpp_socket=cpp_socket, ack_aware_pending=bool(c.get("ack_aware_pending") or False), receipt_timeout_ns=int(c.get("receipt_timeout_ns") or 0),
                          trt=trt)
