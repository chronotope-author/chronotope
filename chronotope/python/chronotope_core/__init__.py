"""ctypes binding of libchronotope.so, the kGUA control core (chronotope/src/chronotope_api.h).

    core = chronotope_core.Core(width=320, height=240, policy="late_bind", target_rgb=(220, 38, 38))
    a = core.admit(rgb_bytes, observation_id=1, source_event_id=1, source_time_ns=..., publication_time_ns=...)
    core.begin_decision(1); p = core.decide_detector()          # or core.propose("click", x, y)
    r = core.resolve_and_commit()                                # D, on the device
    core.audit()                                                 # residency counters

Results are plain dicts with the C struct's field names. The library is found through CHRONOTOPE_LIB or
chronotope/build/libchronotope.so.
"""
from __future__ import annotations

import ctypes as C
import os
import pathlib
from typing import Any

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2]   # chronotope/

PIXFMT = {"rgb24": 0, "grey8": 1, "nv12": 2}
ACTION = {"none": 0, "click": 1, "type": 2, "key": 3, "wait": 4}
ACTION_NAME = {v: k for k, v in ACTION.items()}
POLICY = {"frozen": 0, "late_bind": 1, "restart_if_moved": 2, "late_bind_or_restart": 3}
RELEVANCE = {0: "none", 1: "action", 2: "validity", 3: "pending"}
OUTCOME = {0: "not_attempted", 1: "committed", 2: "failed_schema", 3: "failed_state_changed", 4: "failed_immutable",
           5: "restart_moved", 6: "restart_not_found", 7: "restart_kind", 8: "rejected_duplicate", 9: "rejected_repeated",
           10: "no_proposal", 11: "restart_stale_evidence"}
BINDING = {0: "state_at_B", 1: "tracked_at_D", 2: "host_override", 3: "grounder", 4: "oracle", 5: "detected_at_D"}
BINDING_ID = {v: k for k, v in BINDING.items()}
PENDING_EVENT = {0: "none", 1: "created", 2: "resolved", 3: "expired"}
OK, ERR, STALE, NO_DECISION = 0, -1, 1, 2


def _lib_path() -> str:
    env = os.environ.get("CHRONOTOPE_LIB")
    cands = [pathlib.Path(env)] if env else []
    cands += [ROOT / "build" / "libchronotope.so", ROOT / "build-release" / "libchronotope.so"]
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError("libchronotope.so not found: build chronotope/ (cmake -S chronotope -B chronotope/build && make -C chronotope/build) or set CHRONOTOPE_LIB")


class Config(C.Structure):
    _fields_ = [("width", C.c_uint32), ("height", C.c_uint32), ("pixfmt", C.c_uint32), ("template_radius_px", C.c_uint32),
                ("min_match_score", C.c_double), ("policy", C.c_uint32), ("restart_threshold_px", C.c_int32),
                ("target_rgb", C.c_uint8 * 3), ("has_target_color", C.c_uint8), ("coordinate_offset_px", C.c_int32),
                ("movement_threshold_px", C.c_uint32), ("validate_kind", C.c_uint8), ("rgb_template", C.c_uint8),
                ("pending_radius_px", C.c_uint32), ("pending_timeout_ns", C.c_int64), ("max_repeat_without_effect", C.c_uint32),
                ("trace_entries", C.c_uint32), ("committed_slots", C.c_uint32), ("track_every_frame", C.c_uint8),
                ("planner_image_scale", C.c_uint32), ("ledger_rows", C.c_uint32), ("max_evidence_age_ns", C.c_int64),
                ("ack_aware_pending", C.c_uint8), ("receipt_timeout_ns", C.c_int64), ("detector_at_d", C.c_uint32)]


class Admission(C.Structure):
    _fields_ = [("status", C.c_uint32), ("relevance", C.c_uint32), ("control_state_version", C.c_uint64),
                ("observation_version", C.c_uint64), ("payload_hash", C.c_uint64), ("controller_state_hash", C.c_uint64),
                ("pending_actions_hash", C.c_uint64), ("change_mean", C.c_double), ("box", C.c_int32 * 4),
                ("box_present", C.c_uint8), ("tracked_x", C.c_int32), ("tracked_y", C.c_int32), ("tracked_score", C.c_double),
                ("tracked_valid", C.c_uint8), ("pending_active", C.c_uint8), ("pending_event", C.c_uint32),
                ("admit_device_ns", C.c_int64), ("admit_host_ns", C.c_int64)]


class Proposal(C.Structure):
    _fields_ = [("status", C.c_uint32), ("kind", C.c_uint32), ("x", C.c_int32), ("y", C.c_int32), ("value", C.c_int32),
                ("observation_id", C.c_uint64), ("decision_id", C.c_uint64), ("supporting_control_state_version", C.c_uint64),
                ("b_device_ns", C.c_int64), ("b_host_ns", C.c_int64)]


class CommitResult(C.Structure):
    _fields_ = [("outcome", C.c_uint32), ("kind", C.c_uint32), ("x", C.c_int32), ("y", C.c_int32), ("value", C.c_int32),
                ("proposed_x", C.c_int32), ("proposed_y", C.c_int32), ("action_id", C.c_uint64), ("decision_id", C.c_uint64),
                ("serialized_action_hash", C.c_uint64), ("supporting_control_state_version", C.c_uint64),
                ("control_state_version_at_commit", C.c_uint64), ("observation_id_at_d", C.c_uint64),
                ("proposal_observation_id", C.c_uint64), ("d_device_ns", C.c_int64), ("d_device_mapped_ns", C.c_int64),
                ("d_host_ns", C.c_int64), ("match_score", C.c_double), ("second_best", C.c_double), ("moved_px", C.c_double),
                ("plateau", C.c_int32), ("matched", C.c_uint8), ("ambiguous", C.c_uint8), ("bind_valid", C.c_uint8),
                ("binding", C.c_uint32), ("track_ns", C.c_int64), ("track_device_ns", C.c_int64), ("reject_reason", C.c_uint32),
                ("repetition_count", C.c_uint32), ("bind_x", C.c_int32), ("bind_y", C.c_int32), ("pending_created", C.c_uint8),
                ("evidence_age_ns", C.c_int64), ("evidence_bracket_ns", C.c_int64), ("ledger_row", C.c_uint32)]


class AuditReport(C.Structure):
    _fields_ = [("h2d_publications", C.c_uint64), ("h2d_proposals", C.c_uint64), ("d2h_log_in_window", C.c_uint64),
                ("d2h_state_in_window", C.c_uint64), ("host_syncs_log_in_window", C.c_uint64),
                ("host_syncs_action_in_window", C.c_uint64), ("declared_commit_waits", C.c_uint64),
                ("host_action_computations", C.c_uint64), ("mallocs_after_warmup", C.c_uint64),
                ("kernel_launches_in_window", C.c_uint64), ("windows_opened", C.c_uint64), ("window_open", C.c_uint8),
                ("residency_ok", C.c_uint8), ("clock_offset_ns", C.c_int64), ("clock_calibration_latency_ns", C.c_int64),
                ("device_bytes_allocated", C.c_size_t), ("planner_host_steps", C.c_uint64), ("planner_d2h_pixels", C.c_uint64),
                ("planner_requests", C.c_uint64), ("binder_host_steps", C.c_uint64)]


class TraceEntry(C.Structure):
    _fields_ = [("observation_id", C.c_uint64), ("decision_id", C.c_uint64), ("kind", C.c_uint32), ("x", C.c_int32),
                ("y", C.c_int32), ("accepted", C.c_uint8), ("reason", C.c_uint32), ("screen_change", C.c_float),
                ("repetition_count", C.c_uint32), ("device_ns", C.c_int64), ("device_mapped_ns", C.c_int64),
                ("control_state_version", C.c_uint64)]


class LedgerRow(C.Structure):
    _fields_ = [("observation_id", C.c_uint64), ("o_ns", C.c_int64), ("o_device_ns", C.c_int64), ("source_time_ns", C.c_int64),
                ("e_bracket_lo_ns", C.c_int64), ("frames_since_prev", C.c_uint32), ("relevance", C.c_uint32),
                ("control_state_version", C.c_uint64), ("decision_proposed", C.c_uint64), ("decision_bound", C.c_uint64),
                ("d_device_ns", C.c_int64), ("effect_observed_ns", C.c_int64), ("receipt_observed_ns", C.c_int64)]


class TrtConfig(C.Structure):
    _fields_ = [("engine_dir", C.c_char_p), ("multimodal_engine_dir", C.c_char_p), ("prompt_version", C.c_int32), ("image_scale", C.c_int32),
                ("max_tokens", C.c_int32), ("temperature", C.c_double), ("enable_thinking", C.c_uint8), ("capture_cuda_graph", C.c_uint8)]


class TrtAnswer(C.Structure):
    _fields_ = [("answer_ok", C.c_uint8), ("click", C.c_uint8), ("x", C.c_int32), ("y", C.c_int32), ("image_x", C.c_int32), ("image_y", C.c_int32),
                ("t_start_ns", C.c_int64), ("t_end_ns", C.c_int64), ("prefill_ms", C.c_double), ("decode_ms", C.c_double),
                ("host_steps", C.c_uint64), ("requests", C.c_uint64), ("input_tokens", C.c_int32), ("output_tokens", C.c_int32),
                ("propose_rc", C.c_int32), ("raw_text", C.c_char * 512), ("error", C.c_char * 256), ("proposal", Proposal)]


class Match(C.Structure):
    _fields_ = [("x", C.c_int32), ("y", C.c_int32), ("score", C.c_double), ("second", C.c_double), ("plateau", C.c_int32),
                ("matched", C.c_uint8), ("ambiguous", C.c_uint8), ("flat", C.c_uint8), ("moved_px", C.c_double),
                ("tpl_x0", C.c_int32), ("tpl_y0", C.c_int32), ("tpl_dx", C.c_int32), ("tpl_dy", C.c_int32), ("device_ns", C.c_int64)]


def _asdict(s: C.Structure) -> dict[str, Any]:
    out = {}
    for name, typ in s._fields_:
        v = getattr(s, name)
        if hasattr(v, "__len__") and not isinstance(v, (bytes, str)):
            v = list(v)
        out[name] = v
    return out


_lib = None


def lib() -> C.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    L = C.CDLL(_lib_path())
    P = C.POINTER
    L.chronotope_version.restype = C.c_char_p
    L.chronotope_create.argtypes = [P(Config), C.c_char_p, C.c_size_t]; L.chronotope_create.restype = C.c_void_p
    L.chronotope_destroy.argtypes = [C.c_void_p]
    L.chronotope_last_error.argtypes = [C.c_void_p]; L.chronotope_last_error.restype = C.c_char_p
    L.chronotope_admit.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_uint64, C.c_uint64, C.c_int64, C.c_int64, P(Admission)]
    L.chronotope_begin_decision.argtypes = [C.c_void_p, C.c_uint64, P(Proposal)]
    L.chronotope_decide_detector.argtypes = [C.c_void_p, P(Proposal)]
    L.chronotope_propose.argtypes = [C.c_void_p, C.c_uint32, C.c_int32, C.c_int32, C.c_int32, P(Proposal)]
    L.chronotope_abort_decision.argtypes = [C.c_void_p]
    L.chronotope_resolve_and_commit.argtypes = [C.c_void_p, P(CommitResult)]
    L.chronotope_resolve.argtypes = [C.c_void_p, P(CommitResult)]
    L.chronotope_commit_only.argtypes = [C.c_void_p, P(CommitResult)]
    L.chronotope_restart.argtypes = [C.c_void_p]
    L.chronotope_decision_open.argtypes = [C.c_void_p]
    L.chronotope_read_frame.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    L.chronotope_override_binding.argtypes = [C.c_void_p, C.c_int32, C.c_int32]
    L.chronotope_audit.argtypes = [C.c_void_p, P(AuditReport)]
    L.chronotope_trace.argtypes = [C.c_void_p, P(TraceEntry), C.c_uint32, P(C.c_uint32)]
    L.chronotope_state_hash.argtypes = [C.c_void_p, P(C.c_uint64), P(C.c_uint64)]
    L.chronotope_track.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t, C.c_int32, C.c_int32, C.c_uint32, C.c_uint8, P(Match)]
    L.chronotope_ledger.argtypes = [C.c_void_p, P(LedgerRow), C.c_uint32, P(C.c_uint32)]
    L.chronotope_bind_external.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_uint32]
    L.chronotope_observe_receipt.argtypes = [C.c_void_p, C.c_uint64, C.c_int64]
    L.chronotope_test_recommit.argtypes = [C.c_void_p, P(CommitResult)]
    L.chronotope_planner_trt_open.argtypes = [C.c_void_p, C.POINTER(TrtConfig)]; L.chronotope_planner_trt_open.restype = C.c_int
    L.chronotope_planner_trt_close.argtypes = [C.c_void_p]; L.chronotope_planner_trt_close.restype = C.c_int
    L.chronotope_decide_trt.argtypes = [C.c_void_p, C.POINTER(TrtAnswer)]; L.chronotope_decide_trt.restype = C.c_int
    L.chronotope_test_alloc.argtypes = [C.c_void_p]
    _lib = L
    return L


def version() -> str:
    return lib().chronotope_version().decode()


class KguaError(RuntimeError):
    pass


class Core:
    """One resident agent. Keyword arguments mirror chronotope_config (chronotope_api.h)."""

    def __init__(self, *, width: int, height: int, pixfmt: str = "rgb24", template_radius_px: int = 12,
                 min_match_score: float = 0.8, policy: str = "late_bind", restart_threshold_px: int | None = None,
                 target_rgb: tuple[int, int, int] | None = None, coordinate_offset_px: int = 0,
                 movement_threshold_px: int = 6, validate_kind: bool = False, rgb_template: bool = False,
                 pending_radius_px: int = 0, pending_timeout_ns: int = 0, max_repeat_without_effect: int = 0,
                 trace_entries: int = 256, committed_slots: int = 64, track_every_frame: bool = True,
                 ledger_rows: int = 256, max_evidence_age_ns: int = 0, ack_aware_pending: bool = False, receipt_timeout_ns: int = 0,
                 detector_at_d: int = 0) -> None:
        cfg = Config()
        cfg.width, cfg.height, cfg.pixfmt = width, height, PIXFMT[pixfmt]
        cfg.template_radius_px, cfg.min_match_score, cfg.policy = template_radius_px, float(min_match_score), POLICY[policy]
        cfg.restart_threshold_px = -1 if restart_threshold_px is None else int(restart_threshold_px)
        if target_rgb is not None:
            cfg.target_rgb[0], cfg.target_rgb[1], cfg.target_rgb[2] = target_rgb
            cfg.has_target_color = 1
        cfg.coordinate_offset_px, cfg.movement_threshold_px = coordinate_offset_px, movement_threshold_px
        cfg.validate_kind, cfg.rgb_template = int(validate_kind), int(rgb_template)
        cfg.pending_radius_px, cfg.pending_timeout_ns = pending_radius_px, pending_timeout_ns
        cfg.max_repeat_without_effect, cfg.trace_entries, cfg.committed_slots = max_repeat_without_effect, trace_entries, committed_slots
        cfg.track_every_frame = int(track_every_frame)
        cfg.ledger_rows, cfg.max_evidence_age_ns = ledger_rows, max_evidence_age_ns
        cfg.ack_aware_pending, cfg.receipt_timeout_ns = int(ack_aware_pending), receipt_timeout_ns
        cfg.detector_at_d = int(detector_at_d)
        self.width, self.height, self.pixfmt = width, height, pixfmt
        self.config = {"width": width, "height": height, "pixfmt": pixfmt, "template_radius_px": template_radius_px,
                       "min_match_score": min_match_score, "policy": policy, "restart_threshold_px": restart_threshold_px,
                       "target_rgb": target_rgb, "coordinate_offset_px": coordinate_offset_px,
                       "movement_threshold_px": movement_threshold_px, "validate_kind": validate_kind, "rgb_template": rgb_template,
                       "pending_radius_px": pending_radius_px, "pending_timeout_ns": pending_timeout_ns,
                       "max_repeat_without_effect": max_repeat_without_effect, "track_every_frame": track_every_frame,
                       "max_evidence_age_ns": max_evidence_age_ns, "ack_aware_pending": ack_aware_pending, "receipt_timeout_ns": receipt_timeout_ns}
        err = C.create_string_buffer(512)
        self._h = lib().chronotope_create(C.byref(cfg), err, 512)
        if not self._h:
            raise KguaError(err.value.decode())

    def close(self) -> None:
        if getattr(self, "_h", None):
            lib().chronotope_destroy(self._h)
            self._h = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "Core":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _check(self, rc: int, what: str, allow: tuple[int, ...] = ()) -> int:
        if rc != OK and rc not in allow:
            raise KguaError(f"{what}: rc={rc}: {lib().chronotope_last_error(self._h).decode()}")
        return rc

    # ---- O
    def admit(self, payload: bytes, *, observation_id: int, source_event_id: int, source_time_ns: int, publication_time_ns: int) -> dict:
        a = Admission()
        rc = lib().chronotope_admit(self._h, payload, len(payload), observation_id, source_event_id, source_time_ns, publication_time_ns, C.byref(a))
        self._check(rc, "admit", allow=(STALE,))
        d = _asdict(a)
        d["relevance_name"] = RELEVANCE[d["relevance"]]
        d["pending_event_name"] = PENDING_EVENT[d["pending_event"]]
        d["stale"] = rc == STALE
        return d

    # ---- B
    def begin_decision(self, observation_id: int) -> dict:
        p = Proposal()
        self._check(lib().chronotope_begin_decision(self._h, observation_id, C.byref(p)), "begin_decision")
        return _asdict(p)

    def decide_detector(self) -> dict:
        p = Proposal()
        self._check(lib().chronotope_decide_detector(self._h, C.byref(p)), "decide_detector")
        return _asdict(p)

    def propose(self, kind: str, x: int = 0, y: int = 0, value: int = 0) -> dict:
        p = Proposal()
        self._check(lib().chronotope_propose(self._h, ACTION[kind], int(x), int(y), int(value), C.byref(p)), "propose")
        return _asdict(p)

    def abort_decision(self) -> None:
        self._check(lib().chronotope_abort_decision(self._h), "abort_decision")

    def restart(self) -> None:
        self._check(lib().chronotope_restart(self._h), "restart")

    def decision_open(self) -> bool:
        return bool(lib().chronotope_decision_open(self._h))

    # ---- D
    def _commit_result(self, r: CommitResult) -> dict:
        d = _asdict(r)
        d["outcome_name"] = OUTCOME.get(d["outcome"], str(d["outcome"]))
        d["binding_name"] = BINDING.get(d["binding"], str(d["binding"]))
        d["kind_name"] = ACTION_NAME.get(d["kind"], str(d["kind"]))
        return d

    def resolve_and_commit(self) -> dict:
        r = CommitResult()
        self._check(lib().chronotope_resolve_and_commit(self._h, C.byref(r)), "resolve_and_commit", allow=(NO_DECISION,))
        return self._commit_result(r)

    def resolve(self) -> dict:
        r = CommitResult()
        self._check(lib().chronotope_resolve(self._h, C.byref(r)), "resolve", allow=(NO_DECISION,))
        return self._commit_result(r)

    def commit_only(self) -> dict:
        r = CommitResult()
        self._check(lib().chronotope_commit_only(self._h, C.byref(r)), "commit_only", allow=(NO_DECISION,))
        return self._commit_result(r)

    # ---- the negative path (host participation, audited)
    def read_frame(self) -> bytes:
        n = self.width * self.height * 3
        buf = C.create_string_buffer(n)
        self._check(lib().chronotope_read_frame(self._h, buf, n), "read_frame")
        return buf.raw

    def override_binding(self, x: int, y: int) -> None:
        self._check(lib().chronotope_override_binding(self._h, int(x), int(y)), "override_binding")

    # ---- after D
    def audit(self) -> dict:
        a = AuditReport()
        self._check(lib().chronotope_audit(self._h, C.byref(a)), "audit")
        d = _asdict(a)
        d["residency_ok"] = bool(d["residency_ok"]); d["window_open"] = bool(d["window_open"])
        return d

    def trace(self, n: int = 256) -> list[dict]:
        buf = (TraceEntry * n)()
        cnt = C.c_uint32(0)
        self._check(lib().chronotope_trace(self._h, buf, n, C.byref(cnt)), "trace")
        out = []
        for i in range(cnt.value):
            d = _asdict(buf[i]); d["reason_name"] = OUTCOME.get(d["reason"], "observation"); out.append(d)
        return out

    def state_hash(self) -> tuple[int, int]:
        h, v = C.c_uint64(0), C.c_uint64(0)
        self._check(lib().chronotope_state_hash(self._h, C.byref(h), C.byref(v)), "state_hash")
        return h.value, v.value

    def track(self, frame_b: bytes, frame_d: bytes, x: int, y: int, radius: int, rgb: bool = False) -> dict:
        m = Match()
        self._check(lib().chronotope_track(self._h, frame_b, frame_d, len(frame_b), int(x), int(y), int(radius), int(rgb), C.byref(m)), "track")
        return _asdict(m)

    def ledger(self, n: int = 256) -> list[dict]:
        buf = (LedgerRow * n)()
        cnt = C.c_uint32(0)
        self._check(lib().chronotope_ledger(self._h, buf, n, C.byref(cnt)), "ledger")
        return [_asdict(buf[i]) for i in range(cnt.value)]

    def bind_external(self, x: int, y: int, binding: str = "grounder") -> None:
        self._check(lib().chronotope_bind_external(self._h, int(x), int(y), BINDING_ID[binding]), "bind_external")

    def observe_receipt(self, decision_id: int, observation_publication_ns: int) -> None:
        self._check(lib().chronotope_observe_receipt(self._h, int(decision_id), int(observation_publication_ns)), "observe_receipt")

    def test_recommit(self) -> dict:
        r = CommitResult()
        self._check(lib().chronotope_test_recommit(self._h, C.byref(r)), "test_recommit")
        return self._commit_result(r)

    # ---- K4d: the in-library TensorRT Edge-LLM planner (no socket, no server) -----------------------------------
    def planner_trt_open(self, *, engines_root: str, prompt_version: int = 5, image_scale: int = 2, max_tokens: int = 64,
                         temperature: float = 0.0, enable_thinking: bool = False, capture_cuda_graph: bool = True) -> None:
        root = os.path.expanduser(engines_root)
        self._trt_cfg = TrtConfig(engine_dir=os.path.join(root, "llm").encode(), multimodal_engine_dir=root.encode(), prompt_version=int(prompt_version),
                                  image_scale=int(image_scale), max_tokens=int(max_tokens), temperature=float(temperature),
                                  enable_thinking=int(enable_thinking), capture_cuda_graph=int(capture_cuda_graph))
        self._check(lib().chronotope_planner_trt_open(self._h, C.byref(self._trt_cfg)), "planner_trt_open")
        self.agent_process = f"libchronotope.so in-process (ctypes) {version()}"

    def decide_trt(self) -> dict:
        """The model inside the library on the frame at B (device pointer); blocks for the answer. Same dict as SocketCore.decide_trt."""
        a = TrtAnswer()
        self._check(lib().chronotope_decide_trt(self._h, C.byref(a)), "decide_trt")
        return {"ok": True, "answer_ok": bool(a.answer_ok), "click": bool(a.click), "x": a.x, "y": a.y, "image_x": a.image_x, "image_y": a.image_y,
                "raw_text": a.raw_text.decode(errors="replace"), "error": a.error.decode(errors="replace"), "invocation_id": f"trt-{a.requests}",
                "model_reported": "tensorrt_edgellm", "input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "t_start_ns": a.t_start_ns,
                "t_end_ns": a.t_end_ns, "deliberation_ms": (a.t_end_ns - a.t_start_ns) / 1e6, "prefill_ms": a.prefill_ms, "decode_ms": a.decode_ms,
                "planner_host_steps": a.host_steps, "planner_requests": a.requests, "propose_rc": a.propose_rc,
                "proposal": _asdict(a.proposal) if (a.click and a.propose_rc == 0) else None}

    def planner_trt_close(self) -> None:
        lib().chronotope_planner_trt_close(self._h)

    def test_alloc(self) -> None:
        self._check(lib().chronotope_test_alloc(self._h), "test_alloc")
