"""The spec schema of the "chronotope" controller kind, registered into chronotope.spec at runtime.

src/chronotope is frozen while studies run (GPU queue convention), so the registration lives here; the
permanent version is the same dict inserted into spec._CONTROLLER_SCHEMAS plus a `kind == "chronotope"` branch
in runner.build_controller that calls chronotope_core.controller.build.
"""
from __future__ import annotations

from chronotope import spec as _spec

NoneT = type(None)

CHRONOTOPE_SCHEMA = {
    "kind": (str, ("chronotope",)),
    "planner": (str, ("detector", "vlm", "trt", "api")),  # api: C++ HTTPS client, remote inference
    "binder": (str, ("cuda", "tracker_py", "grounder", "detector", "ncc_then_detector")),   # detector*: the colour/feature detector at D (2026-08-27)
    "at_commit_policy": (str, ("frozen", "late_bind", "restart_if_moved", "late_bind_or_restart")),
    "template_radius_px": (int, None),
    "min_match_score": (float, None),
    "restart_threshold_px": ((int, NoneT), None),
    "compute_ns": (int, None),                      # detector planner: emulated deliberation (wall-clock sleep after B)
    "coordinate_offset_px": (int, None),
    "validate_kind": (bool, None),
    "movement_threshold_px": (int, None),
    "max_evidence_age_ns": (int, None),        # P2: staleness gate (0 = off)
    "memory": (str, ("none", "harness_pending")),   # the device pending record (P1 arms); multi-action tasks decide on every changed frame
    "pending_timeout_ns": (int, None),
    "ack_aware_pending": ((bool, NoneT), None),   # P2 H-M2 (optional; absent = off)
    "receipt_timeout_ns": ((int, NoneT), None),
    "pending_radius_px": (int, None),
    # the vlm planner (null when planner is detector)
    "server": ((str, NoneT), ("openai_compatible_local", "edge_llm", "cpp_socket", "in_process", None, "anthropic_api")),   # edge_llm: Python binding in-process; cpp_socket: the C++ agent process (K4b)
    "base_url": ((str, NoneT), None),
    "model_id": ((str, NoneT), None),
    "prompt_version": ((int, NoneT), tuple(_spec.PROMPT_VERSIONS) + (None,)),
    "image_scale": ((int, NoneT), None),
    "max_tokens": ((int, NoneT), None),
    "temperature": ((float, NoneT), None),
    "enable_thinking": ((bool, NoneT), None),
    "reasoning_effort": ((str, NoneT), ("low", "medium", "high", "xhigh", None)),
    "invoke_on": ((str, NoneT), ("frame_change", None)),
}


def register() -> None:
    _spec._CONTROLLER_SCHEMAS.setdefault("chronotope", CHRONOTOPE_SCHEMA)


def controller_dict(*, planner: str, binder: str, at_commit_policy: str, template_radius_px: int = 12,
                    min_match_score: float = 0.8, restart_threshold_px: int | None = None, compute_ns: int = 0,
                    coordinate_offset_px: int = 0, validate_kind: bool = False, movement_threshold_px: int = 6,
                    max_evidence_age_ns: int = 0, memory: str = "none", pending_timeout_ns: int = 0, pending_radius_px: int = 0,
                    vlm: dict | None = None, ack_aware_pending: bool = False, receipt_timeout_ns: int = 0) -> dict:
    """A complete controller block for a spec (every vlm field present, null for the detector planner)."""
    d = {"kind": "chronotope", "planner": planner, "binder": binder, "at_commit_policy": at_commit_policy,
         "template_radius_px": template_radius_px, "min_match_score": min_match_score,
         "restart_threshold_px": restart_threshold_px, "compute_ns": compute_ns, "coordinate_offset_px": coordinate_offset_px,
         "validate_kind": validate_kind, "movement_threshold_px": movement_threshold_px, "max_evidence_age_ns": max_evidence_age_ns,
         "memory": memory, "pending_timeout_ns": pending_timeout_ns, "pending_radius_px": pending_radius_px,
         "ack_aware_pending": ack_aware_pending, "receipt_timeout_ns": receipt_timeout_ns,
         "server": None, "base_url": None, "model_id": None, "prompt_version": None, "image_scale": None, "max_tokens": None,
         "temperature": None, "enable_thinking": None, "reasoning_effort": None, "invoke_on": None}
    if vlm:
        d.update(vlm)
    return d
