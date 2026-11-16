"""Strict trial-specification loader (experiments.md §11).

Rejects: unknown fields, missing fields, wrong types (bool is not int), duplicate keys,
non-finite numbers, unversioned schemas, and any time field whose name does not end in `_ns`
(the schema only *defines* `_ns` names, so any other time spelling is an unknown field).
Nothing has a default.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import hash_obj
from .scenario import ACTION_SPACE_VERSION, OPTIONAL_SCENARIO_FIELDS, Manifest, manifest_from_dict

SCHEMA_VERSION = 1
SCORER_VERSION = 1
OBSERVATION_BOUNDARY_VERSION = 1
POLICIES = ("paused", "replay", "live")


class SpecError(ValueError):
    pass


# ---- schema --------------------------------------------------------------------------
# Leaf: (type, allowed_values or None). Nested: dict. All time fields end in _ns.
_SCHEMA: dict[str, Any] = {
    "schema_version": (int, (SCHEMA_VERSION,)),
    "scenario": ("__by_kind__", "scenario_id", "__scenarios__"),
    "task_clock_policy": (str, POLICIES),
    "replay_source_live_trace_sha256": ((str, type(None)), None),
    "controller": ("__by_kind__", "kind", "__controllers__"),
    "observation": {
        "boundary_version": (int, (OBSERVATION_BOUNDARY_VERSION,)),
        "publication_delay_ns": (int, None),
        # PHY2 part 1: where O comes from. Optional (the only optional field): absent means "environment",
        # so every existing spec, its hash and its traces are unchanged. "capture" = the physical path
        # (chronotope.phy.CaptureFrameSource; O is the capture card's kernel timestamp).
        "source": ("__optional__", (str, ("environment", "capture"))),
    },
    "action_space_version": (int, (ACTION_SPACE_VERSION,)),
    "scorer_version": (int, (SCORER_VERSION,)),
    "timestamp_source": (str, ("simulated", "monotonic_ns")),
    "delivery": ("__by_kind__", "route", "__deliveries__"),
    "exclusion_policy": (str, ("report_invalid_separately",)),
    "trial_class": (str, ("agent_decision",)),
}


_REVEALED = {
    "scenario_id": (str, ("revealed_target",)),
    "scenario_seed": (int, None),
    "canvas_w_px": (int, None),
    "canvas_h_px": (int, None),
    "target_size_px": (int, None),
    "event_time_ns": (int, None),
    "response_window_ns": (int, None),
    "frame_period_ns": (int, None),
    "episode_horizon_ns": (int, None),
    "irrelevant_motion": (bool, None),
    "skin": (str, ("plain", "notification", "balloon", "calendar")),
    "fixed_layout": (bool, None),   # optional (OPTIONAL_SCENARIO_FIELDS): absent = False, the pre-2026-08-26 layout
}
_SCENARIO_SCHEMAS: dict[str, dict[str, Any]] = {
    "revealed_target": _REVEALED,
    "moving_target": {**_REVEALED, "scenario_id": (str, ("moving_target",)),
                      "velocity_x_px_per_s": (int, None), "velocity_y_px_per_s": (int, None)},
    "pending_effect": {**_REVEALED, "scenario_id": (str, ("pending_effect",)), "effect_delay_ns": (int, None)},
    "jump_target": {**_REVEALED, "scenario_id": (str, ("jump_target",)), "jump_delay_ns": (int, None),
                    "jump_dx_px": (int, None), "jump_dy_px": (int, None)},
    "balloon_select": {**_REVEALED, "scenario_id": (str, ("balloon_select",)), "named_color": (str, ("red", "blue", "yellow")),
                       "velocity_x_px_per_s": (int, None)},
    "sponsor_skip": {**_REVEALED, "scenario_id": (str, ("sponsor_skip",)), "skin": (str, ("player", "player_static")),
                     "seek_delay_ns": (int, None)},
    "sponsor_skip_pending": {**_REVEALED, "scenario_id": (str, ("sponsor_skip_pending",)), "skin": (str, ("player", "player_static")),
                             "seek_delay_ns": (int, None)},
    "reflow_page": {**_REVEALED, "scenario_id": (str, ("reflow_page",)), "skin": (str, ("browser",)),
                    "change_delay_ns": (int, None)},
    "live_page": {**_REVEALED, "scenario_id": (str, ("live_page",)), "skin": (str, ("browser",)),   # G2: a live Chromium page
                  "reveal_delay_ns": (int, None), "banner_delay_ns": (int, None), "banner_h_px": (int, None),
                  "dialog_dx_px": (int, None), "dialog_dy_px": (int, None), "slide_dx_px": (int, None),
                  "slide_dy_px": (int, None), "slide_ns": (int, None), "two_step": (bool, None)},
    "rotating_code": {**_REVEALED, "scenario_id": (str, ("rotating_code",)), "skin": (str, ("notification",)),
                      "change_delay_ns": (int, None), "code_digits": (int, None),
                      "kind_change": (bool, None)},   # optional: absent = False (V1 card)
    "live_trading": {**_REVEALED, "scenario_id": (str, ("live_trading",)), "skin": (str, ("chart",)),
                     "reversal_delay_ns": (int, None), "u_valid_order": (float, None),
                     "u_invalid_order": (float, None), "u_no_order": (float, None)},
}
# Delivery routes (by kind). The seeded-jitter route exists so D->A can vary at the TRIAL level
# inside one condition (H4.2's software half): with only fixed delays, A = D + c holds by
# construction and "A is a derived clock" could never be false in the software track.
_DELIVERY_SCHEMAS: dict[str, dict[str, Any]] = {
    "simulated_fixed_delay": {"route": (str, ("simulated_fixed_delay",)),
                              "delivery_delay_ns": (int, None), "max_attempts": (int, None)},
    "simulated_drop_first": {"route": (str, ("simulated_drop_first",)),
                             "delivery_delay_ns": (int, None), "max_attempts": (int, None)},
    "simulated_seeded_jitter": {"route": (str, ("simulated_seeded_jitter",)),
                                "delivery_delay_min_ns": (int, None), "delivery_delay_max_ns": (int, None),
                                "delivery_seed": (int, None), "max_attempts": (int, None)},
    # PHY2 part 3: D->A drawn (seeded) from the instrument's measured samples, results/PHY_route_<route_name>.json.
    "simulated_empirical": {"route": (str, ("simulated_empirical",)), "route_name": (str, None),
                            "delivery_seed": (int, None), "max_attempts": (int, None)},
    # PHY3: hardware in the loop (chronotope.phy.PhysicalRoute); A is the target page's reported input time,
    # or the echo strip when the page report is missing; a receipt not observed within receipt_timeout_ns is lost.
    "physical": {"route": (str, ("physical",)), "receipt_timeout_ns": (int, None),
                 "expected_delivery_delay_ns": (int, None), "max_attempts": (int, None)},
    # G2: the browser's own input path (live_page.PlaywrightRoute, passed to run_trial(route=...)); the receipt is the
    # page's pointerdown timestamp, a lost delivery is one the page never reports within the timeout
    "playwright_page": {"route": (str, ("playwright_page",)), "receipt_timeout_ns": (int, None), "max_attempts": (int, None)},
}
# prepared / restart_if_changed: A8-PREP card (content known before E; discard the proposal when the perceived box changed)
BINDINGS = ("state_at_B", "newest_at_D", "predicted_at_A", "true_at_A", "prepared", "restart_if_changed")
# 9 player, 10 chart (2026-08-24); 11 rotating code (C4); 12 browser page (G1); 13 rotating code with a possible
# dismiss button (V1 kind_change, mixed type/click/wait grammar); 14 prepared click from the known layout (A8-PREP)
PROMPT_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)   # 15 live two-step page (G2); 16 EvoCUA native grammar, 17 Agent S3 (M4); 18/19/20 = 16/17/7 + the no-wait sentence (M4c)
_CONTROLLER_SCHEMAS: dict[str, dict[str, Any]] = {
    "scripted_oracle": {
        "kind": (str, ("scripted_oracle",)),
        "compute_ns": (int, None),
        "coordinate_offset_px": (int, None),
        "binding": (str, BINDINGS),
        "expected_observation_delay_ns": ((int, type(None)), None),   # predictor only
        "expected_delivery_delay_ns": ((int, type(None)), None),      # predictor only
    },
    "scripted_pending": {
        "kind": (str, ("scripted_pending",)),
        "compute_ns": (int, None),
        "memory": (str, ("none", "harness_pending", "model_informed")),
        "pending_timeout_ns": (int, None),
        "pending_radius_px": (int, None),
    },
    "vlm_pending": {
        "kind": (str, ("vlm_pending",)),
        "server": (str, ("openai_compatible_local", "agents3_service", "anthropic_api")),
        "base_url": (str, None),
        "model_id": (str, None),
        "prompt_version": (int, PROMPT_VERSIONS),
        "image_scale": (int, None),
        "max_tokens": (int, None),
        "temperature": (float, None),
        "enable_thinking": (bool, None),
        "reasoning_effort": ((str, type(None)), ("low", "medium", "high", "xhigh", None)),
        "invoke_on": (str, ("frame_change",)),
        "memory": (str, ("none", "harness_pending", "model_informed")),
        "pending_timeout_ns": (int, None),
        "pending_radius_px": (int, None),
    },
    "vlm_late_binding": {
        "kind": (str, ("vlm_late_binding",)),
        "server": (str, ("openai_compatible_local", "agents3_service", "anthropic_api")),
        "base_url": (str, None),
        "model_id": (str, None),
        "prompt_version": (int, PROMPT_VERSIONS),
        "image_scale": (int, None),
        "max_tokens": (int, None),
        "temperature": (float, None),
        "enable_thinking": (bool, None),
        "reasoning_effort": ((str, type(None)), ("low", "medium", "high", "xhigh", None)),
        "invoke_on": (str, ("frame_change",)),
        "at_commit_policy": (str, ("frozen", "restart_if_moved", "late_bind", "late_bind_or_restart")),
        "template_radius_px": (int, None),
        "min_match_score": (float, None),
        "restart_threshold_px": ((int, type(None)), None),   # restart_if_moved only
        "validate_kind": (bool, None),   # V1: discard a proposal whose KIND contradicts the newest frame (optional: absent = False)
    },
    "vlm_chronotope": {   # the complete controller as one agent: late binding + pending record (A2 card)
        "kind": (str, ("vlm_chronotope",)),
        "server": (str, ("openai_compatible_local", "agents3_service", "anthropic_api")),
        "base_url": (str, None),
        "model_id": (str, None),
        "prompt_version": (int, PROMPT_VERSIONS),
        "image_scale": (int, None),
        "max_tokens": (int, None),
        "temperature": (float, None),
        "enable_thinking": (bool, None),
        "reasoning_effort": ((str, type(None)), ("low", "medium", "high", "xhigh", None)),
        "invoke_on": (str, ("frame_change",)),
        "memory": (str, ("none", "harness_pending", "model_informed")),
        "pending_timeout_ns": (int, None),
        "pending_radius_px": (int, None),
        "template_radius_px": (int, None),
        "min_match_score": (float, None),
    },
    "vlm_value_binding": {   # C4: the late-bound field is a typed VALUE, resolved by decoder.py at D
        "kind": (str, ("vlm_value_binding",)),
        "server": (str, ("openai_compatible_local", "agents3_service", "anthropic_api")),
        "base_url": (str, None),
        "model_id": (str, None),
        "prompt_version": (int, PROMPT_VERSIONS),
        "image_scale": (int, None),
        "max_tokens": (int, None),
        "temperature": (float, None),
        "enable_thinking": (bool, None),
        "reasoning_effort": ((str, type(None)), ("low", "medium", "high", "xhigh", None)),
        "invoke_on": (str, ("frame_change",)),
        "at_commit_policy": (str, ("frozen", "restart_if_changed", "late_bind")),
        "validate_kind": (bool, None),   # V1 (optional: absent = False)
    },
    "vlm_prepared": {   # A8-PREP: one pre-event call (prompt 14) prepares the click; released at first detection
        "kind": (str, ("vlm_prepared",)),
        "server": (str, ("openai_compatible_local", "agents3_service", "anthropic_api")),
        "base_url": (str, None),
        "model_id": (str, None),
        "prompt_version": (int, PROMPT_VERSIONS),
        "image_scale": (int, None),
        "max_tokens": (int, None),
        "temperature": (float, None),
        "enable_thinking": (bool, None),
        "reasoning_effort": ((str, type(None)), ("low", "medium", "high", "xhigh", None)),
        "invoke_on": (str, ("frame_change",)),
    },
    "vlm_direct": {
        "kind": (str, ("vlm_direct",)),
        "server": (str, ("openai_compatible_local", "agents3_service", "anthropic_api", "agents3_hosted_api")),
        "base_url": (str, None),
        "model_id": (str, None),
        "prompt_version": (int, PROMPT_VERSIONS),
        "image_scale": (int, None),
        "max_tokens": (int, None),
        "temperature": (float, None),
        "enable_thinking": (bool, None),
        "reasoning_effort": ((str, type(None)), ("low", "medium", "high", "xhigh", None)),
        "invoke_on": (str, ("frame_change",)),
    },
}


# The only fields a spec may omit: options added after the first cited runs, each with one documented default
# (scenario: OPTIONAL_SCENARIO_FIELDS; controller: validate_kind = False). Everything else has no default.
OPTIONAL_FIELDS = set(OPTIONAL_SCENARIO_FIELDS) | {"validate_kind"}


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise SpecError(f"duplicate key {k!r}")
        d[k] = v
    return d


def _reject_constant(name: str) -> Any:
    raise SpecError(f"non-finite numeric constant {name!r} is not allowed")


def parse_json_strict(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise SpecError(f"malformed JSON: {e}") from e


def _check(obj: Any, schema: dict[str, Any], path: str) -> None:
    if not isinstance(obj, dict):
        raise SpecError(f"{path or '<root>'}: expected object, got {type(obj).__name__}")
    unknown = set(obj) - set(schema)
    if unknown:
        raise SpecError(f"{path or '<root>'}: unknown field(s) {sorted(unknown)}")
    optional = {k for k, r in schema.items() if isinstance(r, tuple) and r and r[0] == "__optional__"} | OPTIONAL_FIELDS
    missing = set(schema) - set(obj) - optional
    if missing:
        raise SpecError(f"{path or '<root>'}: missing field(s) {sorted(missing)}")
    for key, rule in schema.items():
        if key in optional:
            if key not in obj:
                continue   # an optional field left at its documented default
            if isinstance(rule, tuple) and rule and rule[0] == "__optional__":
                rule = rule[1]
        val = obj[key]
        p = f"{path}.{key}" if path else key
        if isinstance(rule, tuple) and rule and rule[0] == "__by_kind__":
            _, kind_field, which = rule
            table = {"__scenarios__": _SCENARIO_SCHEMAS, "__controllers__": _CONTROLLER_SCHEMAS,
                     "__deliveries__": _DELIVERY_SCHEMAS}[which]
            if not isinstance(val, dict) or kind_field not in val:
                raise SpecError(f"{p}: expected object with {kind_field!r}")
            if val[kind_field] not in table:
                raise SpecError(f"{p}.{kind_field}: {val[kind_field]!r} not in {sorted(table)}")
            _check(val, table[val[kind_field]], p)
            continue
        if isinstance(rule, dict):
            _check(val, rule, p)
            continue
        types, allowed = rule
        types_t = types if isinstance(types, tuple) else (types,)
        # bool is a subclass of int in Python; reject it explicitly where int is expected.
        if type(val) not in types_t:
            raise SpecError(f"{p}: expected {'/'.join(t.__name__ for t in types_t)}, got {type(val).__name__}")
        if isinstance(val, float) and not math.isfinite(val):
            raise SpecError(f"{p}: non-finite value")
        if allowed is not None and val not in allowed:
            raise SpecError(f"{p}: value {val!r} not in {list(allowed)}")
        if key.endswith("_ns") and val is not None and val < 0:
            raise SpecError(f"{p}: time values must be non-negative")


@dataclass(frozen=True)
class TrialSpec:
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrialSpec":
        if not isinstance(d, dict) or "schema_version" not in d:
            raise SpecError("unversioned spec: schema_version is required")
        _check(d, _SCHEMA, "")
        policy = d["task_clock_policy"]
        src = d["replay_source_live_trace_sha256"]
        if policy == "replay" and src is None:
            raise SpecError("replay policy requires replay_source_live_trace_sha256")
        if policy != "replay" and src is not None:
            raise SpecError("replay_source_live_trace_sha256 must be null unless policy is replay")
        if d["delivery"]["max_attempts"] < 1:
            raise SpecError("delivery.max_attempts must be >= 1")
        if (d["delivery"]["route"] == "simulated_seeded_jitter"
                and d["delivery"]["delivery_delay_min_ns"] > d["delivery"]["delivery_delay_max_ns"]):
            raise SpecError("delivery.delivery_delay_min_ns must be <= delivery_delay_max_ns")
        c = d["controller"]
        if c["kind"] == "scripted_oracle":
            pred = c["binding"] == "predicted_at_A"
            for k in ("expected_observation_delay_ns", "expected_delivery_delay_ns"):
                if pred and c[k] is None:
                    raise SpecError(f"controller.{k} is required for binding predicted_at_A")
                if not pred and c[k] is not None:
                    raise SpecError(f"controller.{k} must be null unless binding is predicted_at_A")
        if c["kind"] == "vlm_late_binding":
            restart = c["at_commit_policy"] == "restart_if_moved"
            if restart and c["restart_threshold_px"] is None:
                raise SpecError("controller.restart_threshold_px is required for at_commit_policy restart_if_moved")
            if not restart and c["restart_threshold_px"] is not None:
                raise SpecError("controller.restart_threshold_px must be null unless at_commit_policy is restart_if_moved")
            if c["template_radius_px"] < 1 or not (0.0 <= c["min_match_score"] <= 1.0):
                raise SpecError("controller.template_radius_px >= 1 and 0 <= min_match_score <= 1 required")
        if c["kind"] == "vlm_chronotope":
            if c["template_radius_px"] < 1 or not (0.0 <= c["min_match_score"] <= 1.0):
                raise SpecError("controller.template_radius_px >= 1 and 0 <= min_match_score <= 1 required")
        if c["kind"] == "vlm_prepared" and c["prompt_version"] != 14:
            raise SpecError("vlm_prepared uses the prepared-instruction prompt (prompt_version 14)")
        if c["kind"] in ("vlm_direct", "vlm_late_binding", "vlm_pending", "vlm_chronotope", "vlm_value_binding", "vlm_prepared"):
            if d["timestamp_source"] != "monotonic_ns":
                raise SpecError(f"{c['kind']} consumes real time; timestamp_source must be monotonic_ns")
            if c["image_scale"] < 1 or c["max_tokens"] < 1:
                raise SpecError("controller.image_scale and max_tokens must be >= 1")
            hosted = c.get('server') == 'agents3_hosted_api'
            if hosted:
                endpoints = {'gpt-6.1-sol': 'https://api.openai.com/v1/responses',
                             'claude-fable-5-1': 'https://api.anthropic.com/v1/messages'}
                if c['base_url'] != endpoints.get(c['model_id']) or c['prompt_version'] != 17:
                    raise SpecError('Hosted Agent S3 requires an exact official model/endpoint and upstream prompt')
            elif not c["base_url"].startswith(("http://127.0.0.1", "http://localhost")):
                raise SpecError("controller.base_url must be a local server (no remote APIs, no keys)")
        spec = cls(raw=d)
        spec.manifest()  # validates scenario geometry
        return spec

    @classmethod
    def from_text(cls, text: str) -> "TrialSpec":
        return cls.from_dict(parse_json_strict(text))

    @classmethod
    def from_path(cls, path: Path) -> "TrialSpec":
        return cls.from_text(path.read_text(encoding="utf-8"))

    # --- accessors -------------------------------------------------------------------
    @property
    def policy(self) -> str:
        return self.raw["task_clock_policy"]

    def manifest(self) -> Manifest:
        return manifest_from_dict(self.raw["scenario"])

    def spec_hash(self) -> str:
        return hash_obj(self.raw)

    def matching_key(self) -> dict[str, Any]:
        """Everything that must be identical across a matched pair: the spec minus the
        intended intervention (task-clock policy) and the replay source pointer (§12.5)."""
        return {k: v for k, v in self.raw.items()
                if k not in ("task_clock_policy", "replay_source_live_trace_sha256")}

    def derive(self, policy: str, replay_source_live_trace_sha256: str | None) -> "TrialSpec":
        d = json.loads(json.dumps(self.raw))
        d["task_clock_policy"] = policy
        d["replay_source_live_trace_sha256"] = replay_source_live_trace_sha256
        return TrialSpec.from_dict(d)
