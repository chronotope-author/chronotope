"""Scenario objects for the distributional gym (cards/D1-distributional-gym.md).

Two subclasses of the existing manifests, built directly from a frozen episode row and handed to
run_trial(manifest=...): the existing scenario classes are not modified and their hashes are unchanged.

  DistMovingTargetManifest  constant motion at a real-valued speed along a real-valued direction
                            (the registered moving_target takes integer px/s components, too coarse for a
                            log-uniform speed on [1, 16] px/s).
  DistRotatingCodeManifest  the rotating code with an `update_type`: relevant (the code changes), irrelevant
                            (something else changes), duplicate (the panel is re-issued with the same code),
                            after_D (a relevant change scheduled after the common receipt gate), kind_change
                            (the field is replaced by a dismiss button; the valid action changes kind).

Both are pure functions of task time; nothing here reads a clock.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..hashing import hash_obj
from ..scenario import (EVENT_CODE_CHANGES, EVENT_TARGET_APPEARS, EVENT_WINDOW_CLOSES, Event, Manifest,
                        RotatingCodeManifest, State, _fill)

UPDATE_TYPES = ("none", "relevant", "irrelevant", "duplicate", "after_D", "kind_change")


@dataclass(frozen=True)
class DistMovingTargetManifest(Manifest):
    """p(t) = p(E) + v (t - E) d, d = (cos theta, sin theta); the rendered box floors the displacement to pixels."""
    scenario_id = "dist_moving_target"

    speed_px_s: float = 0.0
    direction_rad: float = 0.0

    def manifest_hash(self) -> str:
        return hash_obj({"scenario_id": self.scenario_id, **asdict(self)})

    def displacement_f(self, task_time_ns: int) -> tuple[float, float]:
        dt = max(0, task_time_ns - self.event_time_ns) / 1e9
        return (self.speed_px_s * dt * math.cos(self.direction_rad), self.speed_px_s * dt * math.sin(self.direction_rad))

    def displacement_at(self, task_time_ns: int) -> tuple[int, int]:
        dx, dy = self.displacement_f(task_time_ns)
        return (math.floor(dx), math.floor(dy))

    def center_f(self, task_time_ns: int) -> tuple[float, float]:
        """The true (real-valued) centre p(t)."""
        cx, cy = self.target_box().center()
        dx, dy = self.displacement_f(task_time_ns)
        return (cx + dx, cy + dy)

    def half_width_px(self) -> float:
        return self.target_size_px / 2


@dataclass(frozen=True)
class DistRotatingCodeManifest(RotatingCodeManifest):
    """RotatingCodeManifest with the update semantics above. change_delay_ns is U - E for every update type; for
    irrelevant and duplicate the code shown never changes, for after_D the change is simply late."""
    scenario_id = "dist_rotating_code"

    update_type: str = "relevant"

    def __post_init__(self) -> None:
        if self.update_type not in UPDATE_TYPES or self.update_type == "none":
            raise ValueError(f"update_type must be one of {UPDATE_TYPES[1:]}")
        if (self.update_type == "kind_change") != self.kind_change:
            raise ValueError("kind_change is exactly the update_type kind_change")
        super().__post_init__()

    def manifest_hash(self) -> str:
        return hash_obj({"scenario_id": self.scenario_id, **asdict(self)})

    def codes(self) -> tuple[str, str]:
        a, b = super().codes()
        return (a, a) if self.update_type == "duplicate" else (a, b)

    def code_at(self, task_time_ns: int) -> str | None:
        if self.update_type == "irrelevant":
            st = Manifest.state_at(self, task_time_ns)
            return self.codes()[0] if st.target_visible else None
        return super().code_at(task_time_ns)

    def events(self) -> tuple[Event, ...]:
        kind = {"relevant": "code_changes", "after_D": "code_changes", "kind_change": "code_expires",
                "irrelevant": "irrelevant_update", "duplicate": "duplicate_update"}[self.update_type]
        base = (Event(EVENT_TARGET_APPEARS, "target_appears", self.event_time_ns),
                Event(EVENT_WINDOW_CLOSES, "window_closes", self.window_close_ns()))
        if self.change_time_ns() < self.window_close_ns():
            return (base[0], Event(EVENT_CODE_CHANGES, kind, self.change_time_ns()), base[1])
        return base

    def state_at(self, task_time_ns: int) -> State:
        return super().state_at(task_time_ns)

    def render(self, task_time_ns: int) -> bytes:
        buf = bytearray(super().render(task_time_ns))
        if self.update_type == "irrelevant" and self.state_at(task_time_ns).target_visible and task_time_ns >= self.change_time_ns():
            # the irrelevant update: the taskbar changes colour; the card, panel and field are untouched
            w, h = self.canvas_w_px, self.canvas_h_px
            _fill(buf, w, h, 0, h - h // 12, w, h, (51, 65, 85))
        return bytes(buf)
