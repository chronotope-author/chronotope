"""Harness clock and task-clock policies.

Two timelines exist and are never confused:
  * harness time  — the controller-side timeline on which O, B, D, A are stamped.
  * task time     — the environment's own time, queried ONLY through TaskClock.

Two harness timestamp sources:
  * "simulated"    — a discrete-event clock the runner advances (Gate 1).
  * "monotonic_ns" — the host monotonic clock. Historically, synthetic frames generated
    after a blocking decision could retain their scheduled publication times. This is a
    controlled simulation convention, not a measurement of physical frame availability.
    New trials can opt into wall-accounted timing in runner.run_trial().
"""
from __future__ import annotations

import time
from typing import Protocol


class SimulatedHarnessClock:
    source = "simulated"

    def __init__(self, start_ns: int = 0) -> None:
        self._now_ns = start_ns

    def now_ns(self) -> int:
        return self._now_ns

    def advance_to(self, t_ns: int) -> None:
        if t_ns < self._now_ns:
            raise ValueError(f"harness clock cannot move backwards: {t_ns} < {self._now_ns}")
        self._now_ns = t_ns


class MonotonicHarnessClock:
    source = "monotonic_ns"

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def advance_to(self, t_ns: int) -> None:
        """Wait until wall time t_ns; a past t_ns is a no-op (lazy publication)."""
        dt = t_ns - time.monotonic_ns()
        if dt > 0:
            time.sleep(dt / 1e9)


HarnessClock = SimulatedHarnessClock | MonotonicHarnessClock


class TaskClock(Protocol):
    """The only clock task logic may consult (experiments.md §6.2)."""

    def now_ns(self) -> int: ...


class LiveTaskClock:
    """Task time advances 1:1 with harness time. Policy: live."""

    def __init__(self, harness: "HarnessClock", harness_origin_ns: int) -> None:
        self._h = harness
        self._h0 = harness_origin_ns

    def now_ns(self) -> int:
        return self.at_ns(self._h.now_ns())

    def at_ns(self, harness_ns: int) -> int:
        """Task time corresponding to a given harness time."""
        return harness_ns - self._h0


class FrozenAfterTriggerTaskClock:
    """Task time advances 1:1 until the response-triggering state has been published,
    then stays at that task time until the trial ends. Policies: paused, replay.

    "The response-triggering state remains valid until the action arrives."
    """

    def __init__(self, harness: "HarnessClock", harness_origin_ns: int) -> None:
        self._h = harness
        self._h0 = harness_origin_ns
        self.frozen_at_ns: int | None = None

    def freeze(self, task_time_ns: int) -> None:
        if self.frozen_at_ns is not None:
            raise RuntimeError("task clock already frozen")
        self.frozen_at_ns = task_time_ns

    def now_ns(self) -> int:
        return self.at_ns(self._h.now_ns())

    def at_ns(self, harness_ns: int) -> int:
        """Task time corresponding to a given harness time (pinned once frozen)."""
        t = harness_ns - self._h0
        return t if self.frozen_at_ns is None else self.frozen_at_ns
