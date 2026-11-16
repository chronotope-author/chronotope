"""Run individual trials and record observation, decision, and receipt events.

The runner supports Live task time, Paused task time, and Replay from a matched
Live trace. The original synthetic benchmark uses scheduled frame times. New
monotonic-clock trials can separately account for frame processing completed
only after inference returns. Physical end-to-end measurements require an
external observation source and independently stamped action receipts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .clock import FrozenAfterTriggerTaskClock, LiveTaskClock, MonotonicHarnessClock, SimulatedHarnessClock
from .controller import (AgentS3Adapter, CallBudget, Controller, LocalOpenAIAdapter, Observation, Restart, ScriptedOracleController,
                         VLMValueBindingController, VLMPreparedController,
                         ScriptedPendingController, VLMChronotopeController, VLMDirectController,
                         VLMLateBindingController, VLMPendingController)
import json
import random
import time

from .hashing import canonical_json, hash_obj, sha256_hex
from .scenario import EVENT_EFFECT_VISIBLE, RESPONSE_TRIGGERING_EVENT_ID, Event, Manifest
from .spec import OBSERVATION_BOUNDARY_VERSION, SCHEMA_VERSION, SCORER_VERSION, TrialSpec
from .trace import Trace, TraceWriter
from .latest_frame import ActionAbstention


class PayloadStore:
    """Content-addressed store for observation payloads (the evidence itself)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def put(self, payload: bytes) -> str:
        h = sha256_hex(payload)
        p = self.root / f"{h}.bin"
        if not p.exists():
            p.write_bytes(payload)
        return h

    def get(self, payload_hash: str) -> bytes:
        return (self.root / f"{payload_hash}.bin").read_bytes()


@dataclass(frozen=True)
class Frame:
    due_harness_ns: int
    source_event_id: int
    source_time_ns: int
    payload: bytes


# ---- the two seams (cards/PHY2-instrument-split.md part 1) ------------------------------------
# A trial consumes frames through a FrameSource (where O comes from) and sends actions through a
# Route (where A comes from). The simulated implementations below and the physical ones in
# chronotope.phy (CaptureFrameSource, PhysicalRoute) are interchangeable; controller, binder,
# scorer and trace code never see which one is in use beyond the recorded name.
@runtime_checkable
class FrameSource(Protocol):
    name: str

    def peek(self) -> Frame | None:
        """The next frame that will be published, or None when the stream is exhausted."""
        ...

    def pop(self) -> Frame:
        """Consume the frame returned by peek()."""
        ...


@runtime_checkable
class Route(Protocol):
    route: str
    delay: int   # nominal D->A in ns (oracle true_at_A hook and the publish horizon after a lost delivery)

    def deliver(self, action_bytes: bytes, attempt: int, submit_ns: int) -> tuple[int, bytes] | None:
        """Return (receipt harness time A, bytes as received by the target), or None if the attempt was lost."""
        ...


class EnvFrameSource:
    """Frames rendered by the environment at frame cadence, published after a fixed delay."""

    name = "environment"

    def __init__(self, manifest: Manifest, task_clock, publication_delay_ns: int, h0: int, env_state=None) -> None:
        self.m, self.tc, self.delay, self.h0 = manifest, task_clock, publication_delay_ns, h0
        self.env_state = env_state   # multi-action tasks: () -> accepted_at_ns | None
        self.k = 0
        self._cached_frame: Frame | None = None
        self._cached_env_state: int | None = None

    def next_due_ns(self) -> int | None:
        """Read the next publication time without rendering a future frame."""
        t = self.k * self.m.frame_period_ns
        if t > self.m.episode_horizon_ns:
            return None
        pinned = getattr(self.tc, "frozen_at_ns", None)
        if pinned is not None and t > pinned:
            return None
        return self.h0 + t + self.delay

    def peek_due(self, upto_h: int) -> Frame | None:
        due = self.next_due_ns()
        return self.peek() if due is not None and due <= upto_h else None

    def peek(self) -> Frame | None:
        due = self.next_due_ns()
        if due is None:
            return None
        acc = self.env_state() if self.env_state is not None else None
        # A previously queued future frame may predate an accepted action.
        if self._cached_frame is not None and self._cached_env_state == acc:
            return self._cached_frame
        t = self.k * self.m.frame_period_ns
        if self.env_state is not None:
            st = self.m.state_env(t, acc)
            frame = Frame(due, st.latest_event_id, t, self.m.render_env(t, acc))
        else:
            st = self.m.state_at(t)
            frame = Frame(due, st.latest_event_id, t, self.m.render(t))
        self._cached_frame, self._cached_env_state = frame, acc
        return frame

    def pop(self) -> Frame:
        f = self.peek()
        assert f is not None
        self.k += 1
        self._cached_frame = None
        self._cached_env_state = None
        return f


class ReplayFrameSource:
    """The matched Live trace's observations, in order, at the same harness offsets."""

    name = "replay"

    def __init__(self, live: Trace, store: PayloadStore, h0: int) -> None:
        live_h0 = live.header()["harness_origin_ns"]
        self.frames = [Frame(h0 + (o["publication_time_ns"] - live_h0), o["source_event_id"],
                             o["source_time_ns"], store.get(o["payload_hash"]))
                       for o in live.all("observation")]
        self.i = 0

    def peek(self) -> Frame | None:
        return self.frames[self.i] if self.i < len(self.frames) else None

    def pop(self) -> Frame:
        f = self.frames[self.i]
        self.i += 1
        return f


class SimulatedFixedDelayRoute:
    """Delivery adapter (§10). deliver() returns the receipt harness time or None on failure."""

    route = "simulated_fixed_delay"

    def __init__(self, delivery_delay_ns: int) -> None:
        self.delay = delivery_delay_ns

    def deliver(self, action_bytes: bytes, attempt: int, submit_ns: int) -> tuple[int, bytes] | None:
        """Return (receipt harness time, bytes as received by the target) or None if the attempt failed."""
        return submit_ns + self.delay, action_bytes


class SimulatedDropFirstRoute(SimulatedFixedDelayRoute):
    """The very first delivery attempt of the episode is lost; later attempts deliver (WI-006 retry cost)."""

    route = "simulated_drop_first"

    def __init__(self, delivery_delay_ns: int) -> None:
        super().__init__(delivery_delay_ns)
        self.n = 0

    def deliver(self, action_bytes: bytes, attempt: int, submit_ns: int) -> tuple[int, bytes] | None:
        self.n += 1
        return None if self.n == 1 else super().deliver(action_bytes, attempt, submit_ns)


class SimulatedSeededJitterRoute:
    """Trial-level D->A variation (gym-outcome-contract card, H4.2 software half): each attempt's
    delay is drawn Uniform[min, max] from an explicitly seeded generator, so within ONE condition
    A is not a fixed offset from D. `delay` exposes the midpoint; it is read only by the oracle's
    true_at_A hook, whose "true box at receipt" is exact only under a fixed route (documented)."""

    route = "simulated_seeded_jitter"

    def __init__(self, delivery_delay_min_ns: int, delivery_delay_max_ns: int, delivery_seed: int) -> None:
        self.min_ns, self.max_ns = delivery_delay_min_ns, delivery_delay_max_ns
        self.rng = random.Random(delivery_seed)
        self.delay = (delivery_delay_min_ns + delivery_delay_max_ns) // 2

    def deliver(self, action_bytes: bytes, attempt: int, submit_ns: int) -> tuple[int, bytes] | None:
        return submit_ns + self.rng.randint(self.min_ns, self.max_ns), action_bytes


class SimulatedEmpiricalRoute:
    """Distribution replay (PHY2 part 3): each attempt's D->A is one sample, drawn with replacement by an
    explicitly seeded generator, from the D->A samples the physical instrument measured
    (results/PHY_route_<name>.json, field "da_ms"). `delay` is the sample median (the oracle hook's
    nominal delay). A negative or missing sample list is a spec error, never a silent zero."""

    route = "simulated_empirical"

    def __init__(self, samples_ms: list[float], delivery_seed: int, name: str = "") -> None:
        if not samples_ms:
            raise ValueError(f"simulated_empirical route {name!r}: no D->A samples")
        self.samples_ns = [int(round(x * 1_000_000)) for x in samples_ms]
        if any(x < 0 for x in self.samples_ns):
            raise ValueError(f"simulated_empirical route {name!r}: negative D->A sample")
        self.name = name
        self.rng = random.Random(delivery_seed)
        s = sorted(self.samples_ns)
        self.delay = s[len(s) // 2]

    @classmethod
    def from_json(cls, path: Path, delivery_seed: int) -> "SimulatedEmpiricalRoute":
        d = json.loads(Path(path).read_text())
        return cls(d["da_ms"], delivery_seed, name=Path(path).stem)

    def deliver(self, action_bytes: bytes, attempt: int, submit_ns: int) -> tuple[int, bytes] | None:
        return submit_ns + self.rng.choice(self.samples_ns), action_bytes


ROOT = Path(__file__).resolve().parents[2]


def route_samples_path(name: str) -> Path:
    """results/PHY_route_<name>.json (the instrument's output), resolved against the repository root."""
    p = Path(name)
    return p if p.is_absolute() or p.suffix == ".json" else ROOT / "results" / f"PHY_route_{name}.json"


def make_route(delivery: dict) -> Route:
    """Dispatch on delivery["route"]. The physical route is built by its study (it owns devices), not here."""
    kind = delivery["route"]
    if kind == "playwright_page":
        raise ValueError("the playwright_page route drives a live browser; pass live_page.PlaywrightRoute as run_trial(route=...)")
    if kind == "simulated_seeded_jitter":
        return SimulatedSeededJitterRoute(delivery["delivery_delay_min_ns"], delivery["delivery_delay_max_ns"],
                                          delivery["delivery_seed"])
    if kind == "simulated_empirical":
        return SimulatedEmpiricalRoute.from_json(route_samples_path(delivery["route_name"]), delivery["delivery_seed"])
    if kind == "physical":
        raise ValueError("delivery.route 'physical' must be passed to run_trial(route=PhysicalRoute(...)) by the study")
    cls = {"simulated_fixed_delay": SimulatedFixedDelayRoute, "simulated_drop_first": SimulatedDropFirstRoute}[kind]
    return cls(delivery["delivery_delay_ns"])


def make_source(spec: TrialSpec, manifest: Manifest, task_clock, h0: int, *, env_state=None,
                live_trace: Trace | None = None, store: PayloadStore | None = None) -> FrameSource:
    """Dispatch on spec["observation"].get("source", "environment"). Replay policy always reads the
    matched Live trace whatever the source name says (the source is what Live observed)."""
    if spec.policy == "replay":
        assert live_trace is not None and store is not None
        return ReplayFrameSource(live_trace, store, h0)
    src = spec.raw["observation"].get("source", "environment")
    if src == "environment":
        return EnvFrameSource(manifest, task_clock, spec.raw["observation"]["publication_delay_ns"], h0, env_state=env_state)
    if src == "capture":
        raise ValueError("observation.source 'capture' must be passed to run_trial(source=CaptureFrameSource(...)) by the study")
    raise ValueError(f"unknown observation.source {src!r}")


def prepared_xy(m: Manifest) -> tuple[int, int] | None:
    """A8-PREP: the skin's usual place for the target, the center of the fixed-layout box for this canvas, target size
    and skin (public geometry, the same for every seed; NOT the manifest's own seeded box). None when the action has
    no coordinate to prepare (rotating_code: type(value))."""
    import dataclasses
    if m.scenario_id == "rotating_code":
        return None
    return dataclasses.replace(m, fixed_layout=True).target_box().center()


def build_controller(spec: TrialSpec, *, vlm_adapter=None, budget: CallBudget | None = None,
                     true_box_at_receipt=None) -> Controller:
    c, m = spec.raw["controller"], spec.manifest()
    wall = spec.raw["timestamp_source"] == "monotonic_ns"
    if c["kind"] == "scripted_oracle":
        typed = m.scenario_id == "rotating_code"
        return ScriptedOracleController(c["compute_ns"], c["coordinate_offset_px"], m.canvas_w_px, m.canvas_h_px,
                                        wall_clock=wall, binding=c["binding"],
                                        expected_observation_delay_ns=c["expected_observation_delay_ns"],
                                        expected_delivery_delay_ns=c["expected_delivery_delay_ns"],
                                        true_box_at_receipt=true_box_at_receipt, target_color=m.target_color(),
                                        prepared_xy=prepared_xy(m), action_kind="type" if typed else "click",
                                        code_digits=getattr(m, "code_digits", 4))
    if c["kind"] == "scripted_pending":
        return ScriptedPendingController(compute_ns=c["compute_ns"], memory=c["memory"], pending_timeout_ns=c["pending_timeout_ns"],
                                         pending_radius_px=c["pending_radius_px"], canvas_w_px=m.canvas_w_px,
                                         canvas_h_px=m.canvas_h_px, target_color=m.target_color())
    if c["kind"] in ("vlm_direct", "vlm_late_binding", "vlm_pending", "vlm_chronotope", "vlm_value_binding", "vlm_prepared"):
        if budget is None:
            raise ValueError(f"{c['kind']} needs a call budget")
        if vlm_adapter is not None:
            adapter = vlm_adapter
        elif c.get("server") == "agents3_service":   # M4 phase 3: Agent S3 behind studies/agents3_service.py
            adapter = AgentS3Adapter(base_url=c["base_url"], model_id=c["model_id"])
        elif c.get("server") == "anthropic_api":     # 2026-08-28: Claude models through the Messages API
            from .anthropic_adapter import AnthropicAdapter
            adapter = AnthropicAdapter(model_id=c["model_id"], max_tokens=c["max_tokens"], temperature=c["temperature"],
                                       effort=(c.get("reasoning_effort") or "low"))
        else:
            adapter = LocalOpenAIAdapter(
                base_url=c["base_url"], model_id=c["model_id"], max_tokens=c["max_tokens"],
                temperature=c["temperature"], enable_thinking=c["enable_thinking"], reasoning_effort=c["reasoning_effort"])
        kw = dict(adapter=adapter, budget=budget, prompt_version=c["prompt_version"], image_scale=c["image_scale"],
                  invoke_on=c["invoke_on"], canvas_w_px=m.canvas_w_px, canvas_h_px=m.canvas_h_px, skin=m.skin,
                  instruction_color=getattr(m, "instruction_color", lambda: None)(),
                  target_color=m.target_color())   # the evaluator-only detector (V1 validator, A8 release)
        if c["kind"] == "vlm_direct":
            return VLMDirectController(**kw)
        if c["kind"] == "vlm_prepared":
            return VLMPreparedController(prepared_xy=prepared_xy(m), **kw)
        if c["kind"] == "vlm_value_binding":
            return VLMValueBindingController(at_commit_policy=c["at_commit_policy"], validate_kind=c.get("validate_kind", False),
                                             code_digits=getattr(m, "code_digits", 4), **kw)
        if c["kind"] == "vlm_pending":
            return VLMPendingController(memory=c["memory"], pending_timeout_ns=c["pending_timeout_ns"],
                                        pending_radius_px=c["pending_radius_px"], **kw)
        if c["kind"] == "vlm_chronotope":
            return VLMChronotopeController(memory=c["memory"], pending_timeout_ns=c["pending_timeout_ns"],
                                           pending_radius_px=c["pending_radius_px"],
                                           template_radius_px=c["template_radius_px"],
                                           min_match_score=c["min_match_score"], **kw)
        return VLMLateBindingController(at_commit_policy=c["at_commit_policy"], template_radius_px=c["template_radius_px"],
                                        min_match_score=c["min_match_score"], validate_kind=c.get("validate_kind", False),
                                        restart_threshold_px=c["restart_threshold_px"], **kw)
    raise ValueError(c["kind"])


def run_trial(spec: TrialSpec, trace_path: Path, store: PayloadStore, *,
              live_trace: Trace | None = None, controller: Controller | None = None,
              route: Route | None = None, source=None, manifest: Manifest | None = None,
              enforce_horizon: bool = False, timing_accounting: str = "auto") -> Path:
    """`source`: a FrameSource (PHY3 capture) or a factory (h0, task_clock) -> FrameSource (G2 live page) replacing the
    environment renderer; `manifest`: the manifest object to drive the trial with (one that reads the application's
    own log, G2)."""
    m = manifest if manifest is not None else spec.manifest()
    policy = spec.policy
    if policy == "replay":
        if live_trace is None:
            raise ValueError("replay requires the matched live trace")
        if live_trace.file_hash() != spec.raw["replay_source_live_trace_sha256"]:
            raise ValueError("live trace hash does not match spec.replay_source_live_trace_sha256")
    route = route if route is not None else make_route(spec.raw["delivery"])
    simulated = spec.raw["timestamp_source"] == "simulated"
    harness = SimulatedHarnessClock(0) if simulated else MonotonicHarnessClock()
    h0 = harness.now_ns()
    task_clock = LiveTaskClock(harness, h0) if policy == "live" else FrozenAfterTriggerTaskClock(harness, h0)
    # Oracle upper bound (S3.2 true_at_A): the harness reveals the true box at the receipt time.
    true_box_at_receipt = lambda commit_ns: m.target_box_at(task_clock.at_ns(commit_ns + route.delay))  # noqa: E731
    controller = controller if controller is not None else build_controller(spec, true_box_at_receipt=true_box_at_receipt)
    if hasattr(controller, "bind_clock"):
        controller.bind_clock(harness.now_ns)
    multi = bool(getattr(m, "multi_action", False))
    env = {"accepted_at_ns": None}     # env state owned by the runner (multi-action tasks)
    if multi and policy != "live":
        raise ValueError("multi-action tasks are defined under Live only (the evidence depends on the agent's actions)")
    if policy == "replay" and source is not None:
        raise ValueError("replay reads the matched Live trace; do not pass a source")
    if source is None:
        frames = make_source(spec, m, task_clock, h0, env_state=(lambda: env["accepted_at_ns"]) if multi else None,
                             live_trace=live_trace, store=store)
    elif callable(source) and not hasattr(source, "peek"):
        frames = source(h0, task_clock)   # G2: a factory bound to this trial's origin and clock
    else:
        frames = source
    if hasattr(frames, "bind_origin"):
        frames.bind_origin(h0)   # physical sources learn the harness origin at trial start

    # The old software track uses nominal frame-publication timestamps. The
    # wall-accounted option charges frames rendered after blocking inference to
    # the time they actually become available. External capture/receipt paths
    # use the wall-accounted mode by default. Historical traces are unchanged.
    if timing_accounting not in ("auto", "scheduled", "wall_accounted"):
        raise ValueError("timing_accounting must be auto, scheduled, or wall_accounted")
    if timing_accounting == "auto":
        timing_accounting = ("wall_accounted" if not simulated and
                             getattr(frames, "name", None) in ("capture", "live_page")
                             else "scheduled")
    if simulated and timing_accounting == "wall_accounted":
        raise ValueError("wall_accounted requires timestamp_source=monotonic_ns")

    tw = TraceWriter(trace_path)
    tw.append("trial_header", schema_version=SCHEMA_VERSION, spec=spec.raw, spec_hash=spec.spec_hash(),
              matching_key_hash=hash_obj(spec.matching_key()), manifest=spec.raw["scenario"],
              manifest_hash=m.manifest_hash(), condition=policy, task_clock_policy=policy,
              trial_class=spec.raw["trial_class"], timestamp_source=harness.source,
              controller_spec_hash=controller.spec_hash(), action_space_version=spec.raw["action_space_version"],
              scorer_version=SCORER_VERSION, observation_boundary_version=OBSERVATION_BOUNDARY_VERSION,
              replay_source_live_trace_sha256=spec.raw["replay_source_live_trace_sha256"], harness_origin_ns=h0,
              **({"timing_accounting": timing_accounting,
                  "receipt_source": "external" if route.route in ("physical", "playwright_page") else "simulated"}
                 if not simulated else {}))

    # ---- environment events -----------------------------------------------------------
    emitted: set[int] = set()
    if policy == "replay":
        for e in live_trace.all("event"):  # the replayed evidence depicts these events
            tw.append("event", event_id=e["event_id"], kind=e["kind"], task_time_ns=e["task_time_ns"],
                      harness_time_ns=h0 + (e["harness_time_ns"] - live_trace.header()["harness_origin_ns"]),
                      source="live_trace")
            emitted.add(e["event_id"])

    extra_events: list[Event] = []   # scheduled by the env during the episode (effect visible)

    def advance_to(h: int) -> None:
        harness.advance_to(h)
        if policy == "replay":
            return
        t = task_clock.now_ns()
        for ev in (*m.events(), *extra_events):
            if ev.event_id not in emitted and ev.task_time_ns <= t:
                emitted.add(ev.event_id)
                tw.append("event", event_id=ev.event_id, kind=ev.kind, task_time_ns=ev.task_time_ns,
                          harness_time_ns=h0 + ev.task_time_ns, source="environment")

    # ---- observation publication and admission (§7) -----------------------------------
    obs_version = 0
    next_obs_id = 1
    pending: list[Observation] = []
    admitted: list[Observation] = []
    control_state_version = 0

    # Sources with peek_due can check publication deadlines without rendering
    # a future frame. EnvFrameSource and physical capture sources use this path.
    peek_due = getattr(frames, "peek_due", None) or (lambda upto_h: frames.peek())

    def publish_due(upto_h: int) -> None:
        nonlocal obs_version, next_obs_id
        while (f := peek_due(upto_h)) is not None and f.due_harness_ns <= upto_h:
            frames.pop()
            advance_to(f.due_harness_ns)
            obs_version += 1
            ph = store.put(f.payload)
            # f.due_harness_ns is the original publication schedule. A synthetic
            # frame rendered on demand after a blocking model call cannot be
            # available before its generation has finished. External captures
            # already carry their independent availability timestamp.
            ready_ns = harness.now_ns() if not simulated else None
            actual_pub_ns = (max(f.due_harness_ns, ready_ns)
                             if timing_accounting == "wall_accounted" and frames.name == "environment"
                             else f.due_harness_ns)
            obs = Observation(next_obs_id, f.source_event_id, f.source_time_ns, actual_pub_ns, ph,
                              obs_version, f.payload)
            next_obs_id += 1
            tw.append("observation", observation_id=obs.observation_id, source_event_id=obs.source_event_id,
                      source_time_ns=obs.source_time_ns, publication_time_ns=obs.publication_time_ns,
                      payload_hash=ph, observation_version=obs_version,
                      **({"scheduled_publication_time_ns": f.due_harness_ns,
                          "runner_frame_ready_time_ns": ready_ns} if not simulated else {}))
            pending.append(obs)
            # Opt-in research observers retain inter-step frames before coalescing.
            # Existing controllers have no hook and retain their original behavior.
            buffer_observation = getattr(controller, "buffer_observation", None)
            if buffer_observation is not None and buffer_observation(obs):
                tw.append("observation_buffered", observation_id=obs.observation_id,
                          harness_time_ns=harness.now_ns(), observer=type(controller).__name__)
            if (policy != "live" and f.source_event_id == RESPONSE_TRIGGERING_EVENT_ID
                    and task_clock.frozen_at_ns is None):
                task_clock.freeze(f.source_time_ns)
        advance_to(upto_h)

    def admit_newest() -> None:
        """Coalesce: newest pending retained for state update; superseded ones recorded (§7.2)."""
        nonlocal control_state_version
        if not pending:
            return
        newest = pending[-1]
        for old in pending[:-1]:
            tw.append("observation_superseded", observation_id=old.observation_id, superseded_by=newest.observation_id)
        pending.clear()
        upd = controller.admit(newest)
        before = control_state_version
        if upd.relevance != "none":
            control_state_version += 1
        admitted.append(newest)
        tw.append("admission", observation_id=newest.observation_id, control_state_version_before=before,
                  control_state_version_after=control_state_version, relevance=upd.relevance,
                  controller_state_hash=upd.controller_state_hash, latest_source_time_ns=newest.source_time_ns,
                  harness_time_ns=harness.now_ns())

    def end(reason: str) -> Path:
        if hasattr(m, "trailer"):   # a scenario that learned its events from the application records them (G2)
            tw.append("scenario_trailer", **m.trailer())
        tw.append("trial_end", reason=reason, harness_time_ns=harness.now_ns(), task_time_ns=task_clock.now_ns())
        tw.close()
        return trace_path

    # ---- observe; decide (possibly several no-action decisions); commit; deliver; receive ------
    # Single-response tasks end at the first receipt. Multi-action tasks (pending_effect) loop:
    # every committed action gets its own action_id, delivery and receipt; the episode ends at the
    # horizon or POST_EFFECT_NS after the effect became visible.
    horizon_h = h0 + m.episode_horizon_ns
    decision_id = 0
    action_id = 0

    def drain_pending_events() -> None:
        dr = getattr(controller, "drain_events", None)
        if dr is not None:
            for ev in dr():
                tw.append("pending_event", **ev)

    while True:
        end_h = horizon_h
        if multi and env["accepted_at_ns"] is not None:
            end_h = min(horizon_h, h0 + m.effect_visible_ns(env["accepted_at_ns"]) + m.POST_EFFECT_NS)
        # ---- decision loop: until an action is proposed (and survives the at-commit step) ---------
        while True:
            admit_newest()
            drain_pending_events()
            if (multi or enforce_horizon) and harness.now_ns() >= end_h:
                # multi-action only: the episode's end fell inside the last decision (frames that keep changing after
                # the effect keep a controller deciding); end here rather than after the frame stream runs out.
                # Single-response trials keep their original semantics (a decision may still begin after the horizon).
                return end("horizon" if end_h == horizon_h else "effect_done")
            if not controller.ready_to_decide():
                f = frames.peek()
                if f is None or f.due_harness_ns > end_h:
                    publish_due(max(end_h, harness.now_ns()))
                    return end("horizon" if end_h == horizon_h else "effect_done")
                publish_due(f.due_harness_ns)
                continue
            decision_id += 1
            obs_b = admitted[-1]
            supporting_version = control_state_version
            b_ns = harness.now_ns()
            tw.append("decision_begin", decision_id=decision_id, begin_time_ns=b_ns, observation_id=obs_b.observation_id,
                      supporting_control_state_version=supporting_version, invocation_kind="fixed_input")
            try:
                decision = controller.decide()
            except Exception as e:  # infrastructure failure (e.g. API error): not an agent failure
                tw.append("decision_failed", decision_id=decision_id, error=f"{type(e).__name__}: {e}"[:500],
                          harness_time_ns=harness.now_ns())
                return end("decision_failed")
            if simulated:
                if decision.declared_compute_ns is None:
                    raise ValueError("a simulated clock needs the controller to declare its compute time")
                d_ns = b_ns + decision.declared_compute_ns
            else:
                d_ns = harness.now_ns()  # the action is complete when decide() returns
            tw.append("decision_invocation", decision_id=decision_id, invocation_id=decision.invocation_id,
                      invocation_kind=decision.invocation_kind, end_time_ns=d_ns, **decision.invocation_detail)
            drain_pending_events()
            # frames arriving during B->D are published at their due times and coalesced at D
            publish_due(d_ns)
            if decision.action is None:
                tw.append("decision_no_action", decision_id=decision_id, end_time_ns=d_ns)
                continue
            tw.append("action_proposed", decision_id=decision_id, action=decision.action,
                      coordinate_status="provisional", proposal_time_ns=d_ns)   # P: the model's output exists
            p_ns = d_ns
            admit_newest()
            # ---- at commit: resolve unresolved fields (§9.5 step 1) via the controller's separately
            # identified component, if any: a Correction (same decision), a Restart (new decision), or None
            correct = getattr(controller, "correct_at_commit", None)
            wall_synthetic = timing_accounting == "wall_accounted" and isinstance(frames, EnvFrameSource)
            if wall_synthetic:
                # Rendering frames deferred during inference consumes real wall time.
                # Catch up to the advancing clock, then rebind against the newest
                # admitted observation. If binding itself spans another frame due
                # time, repeat. Never commit with a known overdue frame.
                binder_start_ns = harness.now_ns()
                bind_cost_ns = 0
                for _ in range(4096):
                    # No peek/render is needed just to check a future deadline.
                    cutoff = harness.now_ns()
                    next_due = frames.next_due_ns()
                    if next_due is not None and next_due <= cutoff:
                        publish_due(cutoff)
                        continue
                    admit_newest()
                    t_bind0 = time.perf_counter_ns()
                    corr = correct(harness.now_ns()) if correct is not None else None
                    bind_cost_ns += time.perf_counter_ns() - t_bind0
                    if isinstance(corr, (Restart, ActionAbstention)):
                        d_ns = harness.now_ns()
                        break
                    cutoff = harness.now_ns()
                    next_due = frames.next_due_ns()
                    if next_due is None or next_due > cutoff:
                        # Logical commitment cutoff: every synthetic frame
                        # scheduled by this instant has been published/admitted.
                        d_ns = cutoff
                        break
                else:
                    # If rendering cannot catch up to the simulated producer,
                    # fail closed instead of scoring an action on stale evidence.
                    return end("commit_catch_up_exhausted")
            else:
                t_bind0 = time.perf_counter_ns() if not simulated else 0
                binder_start_ns = d_ns
                corr = correct(d_ns) if correct is not None else None
                bind_cost_ns = (time.perf_counter_ns() - t_bind0) if not simulated else 0
            if isinstance(corr, Restart):
                tw.append("decision_restarted", decision_id=decision_id, reason=corr.reason,
                          observation_id=corr.observation_id, harness_time_ns=harness.now_ns(), **corr.detail)
                controller.restart()
                continue
            if not wall_synthetic and not simulated and corr is not None:
                # Retain historical accounting in the scheduled mode.
                d_ns = d_ns + bind_cost_ns
                publish_due(d_ns)
            if isinstance(corr, ActionAbstention):
                tw.append("action_suppressed", decision_id=decision_id, reason=corr.reason,
                          observation_id=corr.observation_id, proposed_action=decision.action,
                          proposal_time_ns=p_ns, binder_start_ns=binder_start_ns,
                          binder_end_ns=d_ns, updater_wall_ns=bind_cost_ns, **corr.detail)
                return end("safe_no_action")
            break

        final_action = decision.action
        if corr is not None:
            if corr.action.get("kind") != decision.action.get("kind"):
                tw.append("commit_failed", decision_id=decision_id, reason="correction_changed_action_kind",
                          harness_time_ns=d_ns)
                return end("commit_failed")
            tw.append("coordinate_correction", decision_id=decision_id, binding=corr.binding,
                      observation_id=corr.observation_id, proposed_action=decision.action, corrected_action=corr.action,
                      harness_time_ns=d_ns, proposal_time_ns=p_ns, binder_start_ns=binder_start_ns, binder_end_ns=d_ns,
                      translation_x=corr.action.get("x", 0) - decision.action.get("x", 0),
                      translation_y=corr.action.get("y", 0) - decision.action.get("y", 0),
                      coordinate_status="resolved", **corr.detail)
            final_action = corr.action

        # ---- commitment (D), one atomic logical operation (§9.5) ---------------------------
        reason = m.validate_action(final_action)
        if timing_accounting == "wall_accounted" and not wall_synthetic:
            # The synthetic wall-accounted path already established its final
            # observation-consistent cutoff above. External sources use the
            # real clock here but require independent receipt instrumentation.
            d_ns = harness.now_ns()
        # This Python runner is single-threaded. There is no concurrent state
        # mutation between validation and commitment, so a read-then-compare
        # check cannot detect a race. Record the current version in the trace.
        # Concurrent runtimes must validate and commit under their own lock or
        # atomic version check.
        version_read = control_state_version
        if reason is not None:
            tw.append("commit_failed", decision_id=decision_id, reason=reason, harness_time_ns=harness.now_ns())
            return end("commit_failed")
        action_id += 1
        action_hash = hash_obj(final_action)
        committed = {"action_id": action_id, "decision_id": decision_id, "action": final_action,
                     "serialized_action_hash": action_hash, "supporting_control_state_version": supporting_version,
                     "control_state_version_at_commit": version_read, "commitment_time_ns": d_ns,
                     "task_time_at_commit_ns": task_clock.at_ns(d_ns)}
        if not simulated:
            committed["commit_observed_time_ns"] = harness.now_ns()
        tw.append("commit", **committed)
        action_bytes = canonical_json(final_action).encode()  # what leaves the controller; immutable from here

        # ---- delivery (after D) and application receipt (A) --------------------------------
        received_ok = False
        for attempt in range(1, spec.raw["delivery"]["max_attempts"] + 1):
            submit_ns = d_ns if attempt == 1 else harness.now_ns()
            tw.append("delivery", action_id=action_id, attempt=attempt, submit_time_ns=submit_ns,
                      submit_observed_time_ns=harness.now_ns(), route=route.route)
            outcome = route.deliver(action_bytes, attempt, submit_ns)
            if outcome is None:
                publish_due(submit_ns + route.delay)
                tw.append("delivery_failed", action_id=action_id, attempt=attempt, harness_time_ns=harness.now_ns())
                continue
            a_ns, received_bytes = outcome
            publish_due(a_ns)
            # The target application parses what it actually received (not the controller's object).
            received = json.loads(received_bytes)
            t_recv = task_clock.at_ns(a_ns)
            rec = {"action_id": action_id, "attempt": attempt, "receipt_time_ns": a_ns, "task_time_at_receipt_ns": t_recv,
                   "receipt_observed_time_ns": harness.now_ns(),
                   "validity_policy": "advancing" if policy == "live" else "frozen_at_trigger",
                   "received_action_hash": sha256_hex(received_bytes)}
            if multi:
                acc = env["accepted_at_ns"]
                st = m.state_env(t_recv, acc)
                out = m.receipt_outcome(t_recv, received, acc)
                if out == "accepted":
                    env["accepted_at_ns"] = t_recv
                    extra_events.append(Event(EVENT_EFFECT_VISIBLE, "effect_visible", m.effect_visible_ns(t_recv)))
                tw.append("receipt", **rec, target_visible=st.target_visible, window_open=st.window_open,
                          click_inside_target=m.target_box().contains(received["x"], received["y"]) if m.validate_action(received) is None else False,
                          valid_at_receipt=(out == "accepted"), outcome=out,
                          effect_visible_at_receipt=(st.latest_event_id == EVENT_EFFECT_VISIBLE))
            else:
                st = m.state_at(t_recv)
                inside = (m.validate_action(received) is None and received.get("kind") == "click"
                          and m.target_box_at(t_recv).contains(received["x"], received["y"]))
                tw.append("receipt", **rec, target_visible=st.target_visible, window_open=st.window_open,
                          click_inside_target=inside, valid_at_receipt=m.action_valid_at(t_recv, received))
            received_ok = True
            break
        if not multi:
            return end("receipt" if received_ok else "delivery_exhausted")
        # multi-action: the episode continues whether or not this action arrived
