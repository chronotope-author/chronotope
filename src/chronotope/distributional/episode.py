"""Run and score one episode of the distributional gym (cards/D1-distributional-gym.md, request §2, §5–§8, §16).

  run_episode(...)   drives run_trial with the row's own scenario object; returns the trace path
  score_trace(...)   THE scoring implementation (SCORER); every controller and condition is scored here, from the
                     trace, at application receipt A. Nothing is scored at commitment D.
  verify_trace_evidence(...)  the §16 checks that need the trace: event sequence and rendered evidence

The model's reasoning time is what it realized: T_reason = D - B. Under the simulated clock a scripted controller
declares it; under the wall clock it is measured. No deadline in this module is computed from any of it.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..controller import CallBudget, Controller, Restart, ScriptedOracleController
from ..evaluate import evaluate_pending_trace, evaluate_trace
from ..hashing import hash_obj
from ..runner import PayloadStore, build_controller, run_trial
from ..scenario import EVENT_TARGET_APPEARS, Manifest, payload_hash
from ..spec import TrialSpec
from ..trace import Trace
from .manifest import EpisodeRow, build_manifest, expected_event_hash, proxy_scenario, verify_row_against_manifest
from .scenarios import DistMovingTargetManifest, DistRotatingCodeManifest

SCORER = "distributional-1"
CONDITIONS = ("paused", "replay", "live")
EVIDENCE_SAMPLE = 24     # environment frames per trace whose payload hash is recomputed from the manifest
VLM_KINDS = ("vlm_direct", "vlm_late_binding", "vlm_pending", "vlm_chronotope", "vlm_value_binding", "vlm_prepared")
FAILURE_MODES = ("success", "duplicate", "decision_failed", "no_action", "commit_failed", "dropped", "late", "wrong_kind",
                 "stale_value", "wrong_value", "off_target", "outside_window_at_receipt")


@dataclass(frozen=True)
class ControllerSpec:
    """A controller as the strict spec loader sees it, plus a display name. The same dict runs on every episode."""
    name: str
    controller: dict[str, Any]

    @property
    def is_vlm(self) -> bool:
        return self.controller["kind"] in VLM_KINDS

    @property
    def is_resident(self) -> bool:
        return self.controller["kind"] == "chronotope"   # the CUDA-resident core (chronotope/python); consumes real time

    @property
    def timestamp_source(self) -> str:
        return "monotonic_ns" if (self.is_vlm or self.is_resident) else "simulated"

    def spec_hash(self) -> str:
        return hash_obj(self.controller)


class TypedRestartOracle(ScriptedOracleController):
    """restart_if_changed for a typed value: at D, if the newest decoded value differs from the one read at B, the
    proposal is discarded and a new decision starts (the scripted re-decision baseline of the fixed-gate study; the
    registered oracle implements restart_if_changed only for clicks)."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self._value_at_b: str | None = None

    def decide(self):
        d = super().decide()
        self._value_at_b = self._value
        return d

    def restart(self) -> None:
        super().restart()
        self._value_at_b = None

    def correct_at_commit(self, commit_time_ns: int):
        if self.action_kind == "type" and self.binding == "restart_if_changed":
            if self._newest_value is not None and self._value_at_b is not None and self._newest_value[1] != self._value_at_b:
                obs, v = self._newest_value
                return Restart("value_changed_since_B", obs.observation_id, {"value_at_B": self._value_at_b, "value_at_D": v})
            return None
        return super().correct_at_commit(commit_time_ns)


def trial_spec_for(row: EpisodeRow, ctrl: ControllerSpec, condition: str, replay_hash: str | None) -> TrialSpec:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if ctrl.is_resident:
        from chronotope_core.glue import register   # the "chronotope" kind's schema is registered at runtime (chronotope/python/chronotope_core/glue.py)
        register()
    return TrialSpec.from_dict({
        "schema_version": 1, "scenario": proxy_scenario(row), "task_clock_policy": condition,
        "replay_source_live_trace_sha256": replay_hash if condition == "replay" else None,
        "controller": json.loads(json.dumps(ctrl.controller)),
        "observation": {"boundary_version": 1, "publication_delay_ns": row.publication_delay_ns},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": ctrl.timestamp_source,
        "delivery": row.delivery, "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision",
    })


def make_controller(spec: TrialSpec, m: Manifest, ctrl: ControllerSpec, *, budget: CallBudget | None,
                    vlm_adapter: Any) -> Controller | None:
    """None lets run_trial build the registered controller (it binds the true_at_A hook to the row's manifest)."""
    c = ctrl.controller
    if c["kind"] == "scripted_oracle" and c["binding"] == "restart_if_changed" and isinstance(m, DistRotatingCodeManifest):
        return TypedRestartOracle(c["compute_ns"], c["coordinate_offset_px"], m.canvas_w_px, m.canvas_h_px,
                                  wall_clock=False, binding="restart_if_changed", target_color=m.target_color(),
                                  action_kind="type", code_digits=m.code_digits)
    if ctrl.is_vlm:
        return build_controller(spec, vlm_adapter=vlm_adapter, budget=budget)
    if ctrl.is_resident:
        from chronotope_core.controller import build as build_kgua   # chronotope/python on sys.path (studies add it); registered by trial_spec_for
        return build_kgua(spec, budget=budget)
    return None


def trace_path_for(out_dir: Path, ctrl: ControllerSpec, row: EpisodeRow, condition: str, repeat_index: int) -> Path:
    return Path(out_dir) / ctrl.name / f"{row.episode_id}_r{repeat_index}_{condition}.jsonl"


def run_episode(row: EpisodeRow, manifest_hash: str, ctrl: ControllerSpec, condition: str, out_dir: Path,
                store: PayloadStore, *, live_trace: Trace | None = None, repeat_index: int = 0,
                budget: CallBudget | None = None, vlm_adapter: Any = None) -> Path:
    """One (episode, controller, condition). Replay needs the controller's own Live trace of the same episode.
    The row is verified against the scenario object before the trial starts; the trace is verified after."""
    m = build_manifest(row)
    verify_row_against_manifest(row, m)
    p = trace_path_for(out_dir, ctrl, row, condition, repeat_index)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"{p} exists: an episode is never redrawn or rerun in place (use a new repeat_index)")
    spec = trial_spec_for(row, ctrl, condition, live_trace.file_hash() if live_trace is not None else None)
    controller = make_controller(spec, m, ctrl, budget=budget, vlm_adapter=vlm_adapter)
    run_trial(spec, p, store, live_trace=live_trace, controller=controller, manifest=m)
    t = Trace.load(p)
    verify_trace_evidence(row, m, t, manifest_hash)
    return p


# ---- §16 checks on the trace ------------------------------------------------------------------------------------
def verify_trace_evidence(row: EpisodeRow, m: Manifest, t: Trace, manifest_hash: str) -> None:
    h = t.header()
    if h["manifest_hash"] != m.manifest_hash():
        raise AssertionError(f"{row.episode_id}: trace manifest hash != the row's scenario hash")
    if h["spec"]["scenario"]["response_window_ns"] != row.window_ns:
        raise AssertionError(f"{row.episode_id}: the trial's deadline is not the frozen row's deadline")
    if h["spec"]["delivery"] != row.delivery:
        raise AssertionError(f"{row.episode_id}: delivery route differs from the frozen row")
    want = [[e.event_id, e.kind, e.task_time_ns] for e in m.events()]
    static_kinds = {e.kind for e in m.events()}
    # multi-action families (pending_effect): the effect's event depends on when the agent's click was accepted, so it
    # is not in the frozen row; the check covers the environment's own events
    got = sorted(([e["event_id"], e["kind"], e["task_time_ns"]] for e in t.all("event")
                  if not getattr(m, "multi_action", False) or e["kind"] in static_kinds), key=lambda e: e[2])
    if got != want[:len(got)] or hash_obj(want) != row.expected_event_hash:
        raise AssertionError(f"{row.episode_id}: event sequence in the trace differs from the frozen row")
    obs = t.all("observation")
    if h["task_clock_policy"] != "replay" and obs:
        step = max(1, len(obs) // EVIDENCE_SAMPLE)
        acc_at = None
        if getattr(m, "multi_action", False):   # the rendering after an accepted click depends on when it was accepted
            acc = next((r for r in sorted(t.all("receipt"), key=lambda r: r["receipt_time_ns"]) if r.get("outcome") == "accepted"), None)
            acc_at = acc["task_time_at_receipt_ns"] if acc else None
        for o in obs[::step] + [obs[-1]]:
            expected = m.render_env(o["source_time_ns"], acc_at) if getattr(m, "multi_action", False) else m.render(o["source_time_ns"])
            if o["payload_hash"] != payload_hash(expected):
                raise AssertionError(f"{row.episode_id}: observation {o['observation_id']} is not the manifest's rendering")


# ---- the scorer ----------------------------------------------------------------------------------------------------
@dataclass
class EpisodeOutcome:
    episode_id: str
    task_family: str
    controller: str
    condition: str
    repeat_index: int
    manifest_hash: str              # the frozen manifest file
    row_hash: str                   # this row
    scorer: str
    deadline_s: float
    # timestamps (harness ns) and intervals; None when the boundary does not exist in the trace
    E: int
    O: int | None
    B: int | None
    D: int | None
    A: int | None
    L_EO: int | None
    L_OB: int | None
    L_BD: int | None
    L_DA: int | None
    L_EA: int | None
    T_reason: int | None
    action: dict[str, Any] | None
    # outcome components (request §5, §6, §7, §8)
    C: int                          # semantic correctness of the produced action
    R: int                          # the application received the committed bytes
    V_A: int | None                 # A <= T_close (Live only; None under a frozen window)
    Y: int                          # the family's success: Y_live / Y_motion / Y_bind (see family)
    S_A_ns: int | None              # receipt slack T_close - A (Live only)
    C_kind: int | None = None       # moving target: action shape valid
    e_px: float | None = None       # receipt-time distance to the true centre
    x_norm: float | None = None     # e / r
    U: int | None = None            # harness time of the update
    G: int | None = None            # 1[B < U < D]
    z_latest: str | None = None     # newest value that reached O before the commit gate
    z_hat: str | None = None        # value committed
    valid_at_A_star: int | None = None
    n_restarts: int = 0
    n_decisions: int = 0
    speed_px_s: float | None = None     # the row's target speed (moving target)
    q_s: float | None = None            # the row's pre-commit update slack (fixed-gate study)
    failure_mode: str = "success"
    invalid_trial: bool = False     # infrastructure failure (counted as a failure, reported separately)
    contract_violations: list[str] = field(default_factory=list)
    G_return: float | None = None   # the task return of Section 3.2 (the old evaluator's return_G), for the tables
    duplicates: int | None = None      # pending_effect: clicks received after the accepted one and before the effect
    effect_delay_s: float | None = None
    reclick_gap_s: float | None = None # pending_effect: accepted receipt -> the next click's receipt (None: no re-click)
    n_suppressed: int | None = None    # pending_effect: proposals the memory suppressed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeOutcome":
        return cls(**d)


def _t_eval(task_time_ns: int, close_ns: int) -> int:
    """The last open state at or before task_time_ns: semantic correctness is judged against the state the action
    met (or the last state before closure), never against the closed window; timeliness is V_A, separately."""
    return min(task_time_ns, close_ns - 1)


def score_trace(row: EpisodeRow, t: Trace, manifest_hash: str, *, controller: str, repeat_index: int = 0) -> EpisodeOutcome:
    m = build_manifest(row)
    h = t.header()
    h0 = h["harness_origin_ns"]
    cond = h["task_clock_policy"]
    live = cond == "live"
    fam = row.task_family
    E = h0 + row.event_ns
    close_h = h0 + row.close_ns
    obs = t.all("observation")
    o1 = next((o for o in obs if o["source_event_id"] == EVENT_TARGET_APPEARS), None)
    O = o1["publication_time_ns"] if o1 else None
    decisions = t.all("decision_begin")
    commits = t.all("commit")
    receipts = t.all("receipt")
    restarts = t.all("decision_restarted")
    c = commits[0] if commits else None
    d0 = t.committing_decision()
    B = d0["begin_time_ns"] if d0 else (decisions[0]["begin_time_ns"] if decisions else None)
    D = c["commitment_time_ns"] if c else None
    r = receipts[0] if receipts else None
    A = r["receipt_time_ns"] if r else None
    action = c["action"] if c else None
    failed = t.first("decision_failed")

    if fam == "pending_effect":
        return _score_pending(row, t, manifest_hash, controller=controller, repeat_index=repeat_index, m=m, h=h, h0=h0, cond=cond)
    base = dict(episode_id=row.episode_id, task_family=fam, controller=controller, condition=cond, repeat_index=repeat_index,
                manifest_hash=manifest_hash, row_hash=hash_obj(row.to_dict()), scorer=SCORER, deadline_s=row.deadline_s,
                E=E, O=O, B=B, D=D, A=A,
                L_EO=(O - E) if O is not None else None, L_OB=(B - O) if (B is not None and O is not None) else None,
                L_BD=(D - B) if (D is not None and B is not None) else None, L_DA=(A - D) if (A is not None and D is not None) else None,
                L_EA=(A - E) if A is not None else None, T_reason=(D - B) if (D is not None and B is not None) else None,
                action=action, n_restarts=len(restarts), n_decisions=len(decisions),
                speed_px_s=(row.target_speed_px_s if fam in ("moving_target", "static_control") else None),
                q_s=row.task_payload.get("q_s"))
    U = None if row.update_ns is None else h0 + row.update_ns
    G = None
    if U is not None and B is not None and D is not None:
        G = int(B < U < D)

    # ---- nothing committed: C = R = 0 (timeouts, crashes, malformed and missing actions all score 0) ------------
    if c is None:
        if failed is not None:
            mode, invalid = "decision_failed", True
        elif t.first("commit_failed") is not None:
            mode, invalid = "commit_failed", False
        else:
            mode, invalid = "no_action", False
        return EpisodeOutcome(**base, C=0, R=0, V_A=(0 if live else None), Y=0, S_A_ns=None, U=U, G=G,
                              failure_mode=mode, invalid_trial=invalid)

    # ---- receipt R and the task time the application saw ------------------------------------------------------
    R = int(r is not None and r["received_action_hash"] == c["serialized_action_hash"])
    t_recv = r["task_time_at_receipt_ns"] if r is not None else c["task_time_at_commit_ns"]
    t_eval = _t_eval(t_recv, row.close_ns)
    C = int(m.action_valid_at(t_eval, action))
    V_A = (int(A is not None and A <= close_h) if live else None)
    S_A = (close_h - A) if (live and A is not None) else None
    valid_at_recv = bool(r is not None and m.action_valid_at(t_recv, action))
    kind_ok = m.validate_action(action) is None

    extra: dict[str, Any] = {"U": U, "G": G}
    if fam in ("moving_target", "static_control"):
        assert isinstance(m, DistMovingTargetManifest)
        cx, cy = m.center_f(t_recv)
        e = math.hypot(action["x"] - cx, action["y"] - cy) if kind_ok else None
        x = (e / m.half_width_px()) if e is not None else None
        Y = int(kind_ok and R == 1 and x is not None and x <= 1.0)
        extra.update(C_kind=int(kind_ok), e_px=e, x_norm=x)
    elif fam in ("late_binding_natural", "late_binding_gate"):
        assert isinstance(m, DistRotatingCodeManifest)
        gate_D = (h0 + row.task_payload["D_star_ns"]) if fam == "late_binding_gate" else D
        seen = [o for o in obs if o["publication_time_ns"] <= gate_D]
        z_latest = m.code_at(max(o["source_time_ns"] for o in seen)) if seen else None
        z_hat = action.get("value") if action.get("kind") == "type" else None
        if fam == "late_binding_gate":
            a_star = row.task_payload["A_star_ns"]
            valid_star = int(m.action_valid_at(a_star, action))
            bind_ok = (z_hat == z_latest) if z_latest is not None else True   # kind_change: nothing to bind; validity decides
            Y = int(bind_ok and valid_star == 1)
            extra.update(valid_at_A_star=valid_star)
        else:
            Y = int(C == 1 and R == 1 and V_A == 1) if live else int(C == 1 and R == 1)
        extra.update(z_latest=z_latest, z_hat=z_hat)
    else:   # closing_window
        Y = int(C == 1 and R == 1 and V_A == 1) if live else int(C == 1 and R == 1)

    # ---- one failure mode per episode -------------------------------------------------------------------------
    if Y == 1:
        mode = "success"
    elif R == 0:
        mode = "dropped"
    elif not kind_ok:
        mode = "wrong_kind"
    elif live and V_A == 0:
        mode = "late"
    elif fam in ("late_binding_natural", "late_binding_gate"):
        codes = m.codes()
        if action.get("kind") == "type" and action.get("value") in codes and not valid_at_recv:
            mode = "stale_value"
        elif action.get("kind") == "type" and not valid_at_recv:
            mode = "wrong_value"
        elif not valid_at_recv:
            mode = "off_target"
        else:
            mode = "stale_value"     # valid at receipt but not the newest eligible value at the gate
    elif fam in ("moving_target", "static_control"):
        mode = "off_target"
    else:
        mode = "off_target" if m.state_at(t_eval).target_visible else "outside_window_at_receipt"
    try:
        extra["G_return"] = evaluate_trace(t).return_G
    except Exception:   # noqa: BLE001 - the return is a convenience column; the gym score does not depend on it
        extra["G_return"] = None
    return EpisodeOutcome(**base, C=C, R=R, V_A=V_A, Y=Y, S_A_ns=S_A, failure_mode=mode, invalid_trial=False, **extra)


def _score_pending(row: EpisodeRow, t: Trace, manifest_hash: str, *, controller: str, repeat_index: int, m, h, h0, cond) -> EpisodeOutcome:
    """The memory axis (2026-08-27): the first accepted click and whether a duplicate followed before the effect. The
    stamps are the first decision's; A is the accepted receipt. Y = accepted and no duplicate (P1's criterion)."""
    pr = evaluate_pending_trace(t)
    E = h0 + row.event_ns
    obs = t.all("observation")
    o1 = next((o for o in obs if o["source_event_id"] == EVENT_TARGET_APPEARS), None)
    O = o1["publication_time_ns"] if o1 else None
    decisions = t.all("decision_begin"); commits = t.all("commit"); receipts = sorted(t.all("receipt"), key=lambda r: r["receipt_time_ns"])
    B = decisions[0]["begin_time_ns"] if decisions else None
    acc = next((r for r in receipts if r.get("outcome") == "accepted"), None)
    c_acc = next((c for c in commits if acc and c["action_id"] == acc["action_id"]), None)
    D = c_acc["commitment_time_ns"] if c_acc else (commits[0]["commitment_time_ns"] if commits else None)
    A = acc["receipt_time_ns"] if acc else None
    gap = None
    if acc is not None:
        later = [r for r in receipts if r["receipt_time_ns"] > acc["receipt_time_ns"] and r.get("outcome") in ("duplicate", "accepted")]
        if later:
            gap = (later[0]["receipt_time_ns"] - acc["receipt_time_ns"]) / 1e9
    mode = "success" if pr.exact_success else ("duplicate" if pr.duplicates > 0 else ("decision_failed" if not pr.valid_trial and pr.invalid_reason and pr.invalid_reason.startswith("decision_failed") else ("no_action" if not pr.accepted else "outside_window_at_receipt")))
    return EpisodeOutcome(episode_id=row.episode_id, task_family=row.task_family, controller=controller, condition=cond, repeat_index=repeat_index,
                          manifest_hash=manifest_hash, row_hash=hash_obj(row.to_dict()), scorer=SCORER, deadline_s=row.deadline_s,
                          E=E, O=O, B=B, D=D, A=A,
                          L_EO=(O - E) if O is not None else None, L_OB=(B - O) if (B is not None and O is not None) else None,
                          L_BD=(D - B) if (D is not None and B is not None) else None, L_DA=(A - D) if (A is not None and D is not None) else None,
                          L_EA=(A - E) if A is not None else None, T_reason=(D - B) if (D is not None and B is not None) else None,
                          action=(c_acc["action"] if c_acc else None), C=int(pr.accepted), R=int(pr.accepted), V_A=(1 if pr.accepted else 0),
                          Y=int(pr.exact_success), S_A_ns=None, n_restarts=0, n_decisions=pr.n_decisions, failure_mode=mode,
                          invalid_trial=(not pr.valid_trial), contract_violations=list(pr.contract_violations), G_return=pr.return_G,
                          duplicates=pr.duplicates, effect_delay_s=row.task_payload.get("effect_delay_s"), reclick_gap_s=gap, n_suppressed=pr.n_suppressed)


def load_outcomes(path: Path) -> list[EpisodeOutcome]:
    return [EpisodeOutcome.from_dict(json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l]


def check_fairness(outcomes: list[EpisodeOutcome]) -> None:
    """The cross-controller assertions of request §16 on a set of scored outcomes (all conditions, all controllers)."""
    if not outcomes:
        raise AssertionError("no outcomes")
    hashes = {o.manifest_hash for o in outcomes}
    if len(hashes) != 1:
        raise AssertionError(f"outcomes come from {len(hashes)} manifests; one frozen manifest is required")
    scorers = {o.scorer for o in outcomes}
    if scorers != {SCORER}:
        raise AssertionError(f"mixed scorers {sorted(scorers)}")
    by_ep: dict[str, list[EpisodeOutcome]] = {}
    for o in outcomes:
        by_ep.setdefault(o.episode_id, []).append(o)
    for eid, group in by_ep.items():
        if len({o.deadline_s for o in group}) != 1:
            raise AssertionError(f"{eid}: controllers saw different deadlines")
        if len({o.row_hash for o in group}) != 1:
            raise AssertionError(f"{eid}: controllers saw different rows (seed or payload differs)")
        us = {(o.U - o.E) for o in group if o.U is not None}   # harness origins differ per trial: compare U - E
        if len(us) > 1:
            raise AssertionError(f"{eid}: the update time U differs across controllers")
        seen: set[tuple[str, str, int]] = set()
        for o in group:
            key = (o.controller, o.condition, o.repeat_index)
            if key in seen:
                raise AssertionError(f"{eid}: {key} scored twice (a repeated run must carry its own repeat_index)")
            seen.add(key)
