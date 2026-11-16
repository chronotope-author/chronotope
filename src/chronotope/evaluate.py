"""Evaluator: from one immutable trace, derive validity, intervals, and overlapping failure labels.

Never estimates or interpolates a missing boundary: a required missing boundary makes the trial
invalid (§13). Also runs the trace-level contract checks that realize §16 items 7, 10, 11, 15, 16.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .scenario import EVENT_TARGET_APPEARS, Manifest, manifest_from_dict
from .trace import Trace


@dataclass
class TrialResult:
    policy: str
    manifest_hash: str
    spec_hash: str
    valid_trial: bool
    invalid_reason: str | None
    action_committed: bool
    semantic_correct: bool | None         # action right for the state the controller used at B
    valid_at_receipt: bool | None         # application-time task value (primary outcome)
    intervals_ns: dict[str, int | None]   # E_O, O_B, B_D, O_D, D_A, E_A, S_A, I_D, I_A
    labels: list[str] = field(default_factory=list)
    contract_violations: list[str] = field(default_factory=list)
    n_decisions: int = 0             # invocations, including those that produced no action
    b_to_d_ns: int | None = None     # B->D of the committing decision
    binding: str = "state_at_B"      # which coordinate binding produced the committed action
    spatial_error_px: float | None = None   # click distance from target center at receipt
    # WI-004 (S7.2): the frozen counterfactual scored at the SAME A from the same trace, when a
    # coordinate correction exists; restarts (new decisions begun after discarding a proposal).
    proposed_valid_at_receipt: bool | None = None
    proposed_spatial_error_px: float | None = None
    n_restarts: int = 0
    first_b_to_d_ns: int | None = None   # D minus the FIRST decision's B (includes restart cost)
    # Outcome contract (gym-outcome-contract card): the episodic return G_g, task value in the
    # state at application receipt. Defined for EVERY valid terminal trace (no-action, dropped,
    # late, invalid included); None only when the trial itself is invalid (§13).
    return_G: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _manifest(h: dict[str, Any], t: Trace | None = None) -> Manifest:
    scenario = h["manifest"]
    if "skin" not in scenario:
        # Early schema-v1 traces predate the skin field. The original renderer
        # used the undecorated target, now named "plain". Do not rewrite the
        # trace or its hashes, and keep current TrialSpec parsing strict.
        if h.get("schema_version") != 1:
            raise ValueError("manifest is missing skin outside the legacy schema")
        scenario = {**scenario, "skin": "plain"}
    m = manifest_from_dict(scenario)
    # a scenario whose events come from the application itself (live_page) appends what it learned at trial end
    tr = t.first("scenario_trailer") if t is not None else None
    return m.with_trailer(tr) if tr is not None and hasattr(m, "with_trailer") else m


def _model_mismatch(t: Trace, h: dict[str, Any]) -> bool:
    """True if any model invocation reported a model id other than the spec's controller.model_id."""
    want = h["spec"]["controller"].get("model_id")
    if want is None:
        return False
    return any(i.get("model_reported") not in (None, want) for i in t.all("decision_invocation"))


def evaluate_trace(t: Trace) -> TrialResult:
    h = t.header()
    m = _manifest(h, t)
    h0 = h["harness_origin_ns"]
    policy = h["task_clock_policy"]
    viol: list[str] = []
    base = dict(policy=policy, manifest_hash=h["manifest_hash"], spec_hash=h["spec_hash"])

    obs = t.all("observation")
    for i, o in enumerate(obs):
        if o["observation_id"] != i + 1 or o["observation_version"] != i + 1:
            viol.append("observation_ids_or_versions_not_sequential")
            break

    decisions = t.all("decision_begin")
    ids = [d["decision_id"] for d in decisions]
    if len(ids) != len(set(ids)):
        viol.append("decision_id_reused")  # §16.15: a restart is a new decision, not the same one
    if _model_mismatch(t, h):
        viol.append("model_reported_mismatch")   # WI-007: the served model must be the specified one
    commits = t.all("commit")
    receipts = t.all("receipt")
    deliveries = t.all("delivery")
    failed = t.all("delivery_failed")

    ev = {e["event_id"]: e for e in t.all("event")}
    window_frozen = policy != "live"
    e1_h = h0 + m.event_time_ns
    t_close = m.window_close_ns()
    close_h = h0 + t_close

    # Evidence of E1 reaching O. Under an advancing window, evidence that arrives after the
    # window closed is "missed"; under a frozen window nothing closes.
    evid_e1 = [o for o in obs if o["source_event_id"] == EVENT_TARGET_APPEARS]
    if window_frozen:
        o1 = evid_e1[0] if evid_e1 else None
    else:
        o1 = next((o for o in evid_e1 if o["publication_time_ns"] <= close_h), None)
    labels: list[str] = []
    if o1 is None:
        labels.append(f"missed_evidence:{m.events()[0].kind}")

    failed_dec = t.first("decision_failed")
    if failed_dec is not None and not commits:
        return TrialResult(**base, valid_trial=False, invalid_reason=f"decision_failed:{failed_dec['error']}",
                           action_committed=False, semantic_correct=None, valid_at_receipt=None, intervals_ns={},
                           labels=labels, contract_violations=viol, n_decisions=len(decisions))
    if not commits:
        if t.first("trial_end") is None:
            return TrialResult(**base, valid_trial=False, invalid_reason="no trial_end", action_committed=False,
                               semantic_correct=None, valid_at_receipt=None, intervals_ns={}, labels=labels,
                               contract_violations=viol)
        if not window_frozen and decisions:
            # A `wait` is a decision too: committed on a frame without the target while a newer admitted frame,
            # published before that decision's end, already showed it, the wait fit the state used at B and not
            # the newest pre-D state — outdated at commitment (paper §3.2), the same label a stale click gets.
            ends = {i["decision_id"]: i["end_time_ns"] for i in t.all("decision_invocation") if i.get("outcome") == "wait"}
            by_id = {o["observation_id"]: o for o in obs}
            for d in decisions:
                ob = by_id.get(d["observation_id"]); end = ends.get(d["decision_id"])
                if ob is None or end is None or m.state_at(ob["source_time_ns"]).target_visible:
                    continue
                if any(o["publication_time_ns"] < end and o["source_time_ns"] > ob["source_time_ns"]
                       and m.state_at(o["source_time_ns"]).target_visible for o in obs):
                    labels.append("outdated_at_commitment")
                    break
        return TrialResult(**base, valid_trial=True, invalid_reason=None, action_committed=False,
                           semantic_correct=None, valid_at_receipt=False,
                           intervals_ns={"E_O": (o1["publication_time_ns"] - e1_h) if o1 else None},
                           labels=labels, contract_violations=viol, n_decisions=len(decisions),
                           return_G=m.return_no_receipt())

    c = commits[0]
    d0 = t.committing_decision()
    if h["trial_class"] == "agent_decision" and d0 is None:
        return TrialResult(**base, valid_trial=False, invalid_reason="missing_B", action_committed=True,
                           semantic_correct=None, valid_at_receipt=None, intervals_ns={}, labels=labels,
                           contract_violations=viol, n_decisions=len(decisions))
    # §16.16: for a fixed-input invocation the supporting version must be the version at B.
    adm_at_b = [a for a in t.all("admission") if a["harness_time_ns"] <= d0["begin_time_ns"]]
    version_at_b = adm_at_b[-1]["control_state_version_after"] if adm_at_b else 0
    if d0["invocation_kind"] == "fixed_input" and c["supporting_control_state_version"] != version_at_b:
        viol.append("supporting_version_inconsistent_with_B")

    if not receipts:
        if failed and len(failed) >= h["spec"]["delivery"]["max_attempts"]:
            return TrialResult(**base, valid_trial=True, invalid_reason=None, action_committed=True,
                               semantic_correct=None, valid_at_receipt=False, intervals_ns={},
                               labels=labels + ["dropped"], contract_violations=viol, n_decisions=len(decisions),
                               return_G=m.return_no_receipt())
        return TrialResult(**base, valid_trial=False, invalid_reason="missing_A", action_committed=True,
                           semantic_correct=None, valid_at_receipt=None, intervals_ns={}, labels=labels,
                           contract_violations=viol, n_decisions=len(decisions))
    r = receipts[0]
    if not any(d["attempt"] == r["attempt"] for d in deliveries):
        viol.append("unrecorded_delivery_attempt")  # §16.11
    if r["received_action_hash"] != c["serialized_action_hash"]:
        labels.append("actuation_error")  # §16.10: what arrived is not what was committed

    action = c["action"]
    # the proposal of the COMMITTING decision (a restarted decision's proposal is a different decision)
    proposed = next((p for p in t.all("action_proposed") if p["decision_id"] == c["decision_id"]), None)
    corr = next((r for r in t.all("coordinate_correction") if r["decision_id"] == c["decision_id"]), None)
    binding = corr["binding"] if corr else "state_at_B"
    # A binder that found nothing to bind to (matched False) committed the proposal unchanged: that action was
    # formed from the state at B, so it is judged as a state-at-B commit (semantic at B; the outdated /
    # missed-evidence rules apply). 2026-08-26 (V1 card): needed so that type(value) into a field that has vanished
    # by D is labelled outdated_at_commitment, not semantic_error. Affects 6 earlier traces (LEDGER).
    fell_back = corr is not None and corr.get("matched") is False and corr["corrected_action"] == corr["proposed_action"]
    if proposed is not None and proposed["action"] != action and corr is None:
        viol.append("uncorrected_action_change")   # committed action differs from proposal with no record of why
    if corr is not None and corr["corrected_action"] != action:
        viol.append("correction_record_mismatch")
    if corr is not None and binding in ("tracked_at_D", "decoded_at_D") and not {"match_score", "matched", "moved_px", "track_ns"} <= set(corr):
        viol.append("correction_record_incomplete")   # the tracker must report what it saw
    restarts = t.all("decision_restarted")
    if any(rs["decision_id"] == c["decision_id"] for rs in restarts):
        viol.append("decision_id_reused")   # a restarted decision can never be the committing one
    ob = next((o for o in obs if o["observation_id"] == d0["observation_id"]), None)
    if ob is None:
        viol.append("decision_references_unknown_observation")
        tau = None
        semantic = None
    else:
        tau = ob["source_time_ns"]
        if binding == "state_at_B" or fell_back:
            semantic = m.action_valid_at(tau, action)
        elif binding in ("newest_at_D", "tracked_at_D", "decoded_at_D"):
            ob2 = next((o for o in obs if o["observation_id"] == corr["observation_id"]), None)
            semantic = m.action_valid_at(ob2["source_time_ns"], action) if ob2 else None
        else:
            semantic = None   # predicted/true-at-A actions are aimed at an unobserved future state
    if semantic is False:
        labels.append("semantic_error")

    D, A = c["commitment_time_ns"], r["receipt_time_ns"]
    t_d, t_a = c["task_time_at_commit_ns"], r["task_time_at_receipt_ns"]
    valid_at_d = m.action_valid_at(t_d, action)
    valid_at_a = r["valid_at_receipt"]
    # Harness self-check: the receipt record must agree with the pure state function.
    if valid_at_a != (m.action_valid_at(t_a, action) and "actuation_error" not in labels):
        viol.append("receipt_inconsistent_with_manifest")

    # Timing labels are defined against an ADVANCING response window. Under paused/replay the
    # triggering state is held valid until the action arrives, so no later state can "require a
    # different outcome" and none of these labels can apply (the test suite enforces this).
    if not window_frozen:
        if t_a >= t_close:
            labels.append("late")
        if valid_at_d and not valid_at_a and "actuation_error" not in labels:
            labels.append("invalidated_after_commitment")
        if semantic and tau is not None and (binding == "state_at_B" or fell_back):
            newer_invalidating = [o for o in obs if o["publication_time_ns"] < D and o["source_time_ns"] > tau
                                  and not m.action_valid_at(o["source_time_ns"], action)]
            if newer_invalidating:
                labels.append("outdated_at_commitment")
            elif not valid_at_d:
                # The state was already invalid at D, but no observation of the closing reached O before D.
                labels.append("missed_evidence:window_closes")

    B0 = d0["begin_time_ns"]   # B of the committing decision (restarted decisions are earlier, separate records)
    intervals = {
        "E_O": (o1["publication_time_ns"] - e1_h) if o1 else None,
        "O_B": (B0 - o1["publication_time_ns"]) if o1 else None,
        "B_D": D - B0,
        "O_D": (D - o1["publication_time_ns"]) if o1 else None,
        "D_A": A - D,
        "E_A": A - e1_h,
        "S_A": None if window_frozen else close_h - A,
        "I_D": (D - (h0 + tau)) if tau is not None else None,
        "I_A": (A - (h0 + tau)) if tau is not None else None,
    }
    visible_a = m.state_at(t_a).target_visible
    prop = corr["proposed_action"] if corr is not None else None
    return TrialResult(**base, valid_trial=True, invalid_reason=None, action_committed=True,
                       semantic_correct=semantic, valid_at_receipt=valid_at_a, intervals_ns=intervals,
                       labels=labels, contract_violations=viol, n_decisions=len(decisions),
                       b_to_d_ns=D - d0["begin_time_ns"], binding=binding,
                       spatial_error_px=m.spatial_error_px(t_a, action) if visible_a else None,
                       proposed_valid_at_receipt=(m.action_valid_at(t_a, prop) if prop is not None else None),
                       proposed_spatial_error_px=(m.spatial_error_px(t_a, prop) if (prop is not None and visible_a) else None),
                       n_restarts=len(restarts),
                       first_b_to_d_ns=(D - decisions[0]["begin_time_ns"]) if decisions else None,
                       return_G=m.return_on_receipt(t_a, action))


def summarize(results: list[TrialResult]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in results:
        s = out.setdefault(r.policy, {"n": 0, "invalid": 0, "success": 0, "sum_G": 0.0, "labels": {}, "violations": {}})
        s["n"] += 1
        if not r.valid_trial:
            s["invalid"] += 1
            continue
        if r.return_G is None:
            raise ValueError("outcome contract violated: a valid terminal trace has no return_G")
        s["success"] += int(bool(r.valid_at_receipt))
        s["sum_G"] += r.return_G
        for lab in r.labels:
            s["labels"][lab] = s["labels"].get(lab, 0) + 1
        for v in r.contract_violations:
            s["violations"][v] = s["violations"].get(v, 0) + 1
    for s in out.values():
        n_valid = s["n"] - s["invalid"]
        s["mean_G"] = s.pop("sum_G") / n_valid if n_valid else None
    return out


# ---- multi-action tasks (WI-006, pending_effect) -----------------------------------------------

@dataclass
class PendingResult:
    manifest_hash: str
    spec_hash: str
    valid_trial: bool
    invalid_reason: str | None
    n_decisions: int                    # model invocations
    n_clicks_committed: int
    n_suppressed: int                   # proposals the pending record suppressed (harness_pending only)
    n_waits: int
    accepted: bool                      # the first valid click was accepted by the application
    first_valid_E_A_ns: int | None      # E1 -> accepted receipt
    duplicates: int                     # clicks received inside the button after acceptance, before the effect
    invalid_clicks: int                 # clicks received after the effect was visible / outside / pre-appearance
    dropped: int                        # committed actions that never arrived
    exact_success: bool                 # accepted AND duplicates == 0
    pending_events: dict[str, int] = field(default_factory=dict)
    contract_violations: list[str] = field(default_factory=list)
    # Outcome contract on multi-action terminals: the manifest's return_pending(accepted_at) — time until the
    # accepted action's effect (G_A on sponsor_skip_pending; the same shape on pending_effect); the full window
    # when nothing was ever accepted. None only for an invalid trial.
    return_G: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_pending_trace(t: Trace) -> PendingResult:
    h = t.header()
    m = _manifest(h, t)
    h0 = h["harness_origin_ns"]
    base = dict(manifest_hash=h["manifest_hash"], spec_hash=h["spec_hash"])
    viol: list[str] = []
    decisions = t.all("decision_begin")
    inv = t.all("decision_invocation")
    ids = [d["decision_id"] for d in decisions]
    if len(ids) != len(set(ids)):
        viol.append("decision_id_reused")
    if _model_mismatch(t, h):
        viol.append("model_reported_mismatch")
    commits = t.all("commit")
    receipts = t.all("receipt")
    dfail = t.all("delivery_failed")
    suppressed = [i for i in inv if i.get("outcome") == "suppressed_pending"]
    waits = [i for i in inv if i.get("outcome") == "wait"]
    # a suppressed decision must never commit
    sup_ids = {i["decision_id"] for i in suppressed}
    if any(c["decision_id"] in sup_ids for c in commits):
        viol.append("suppressed_decision_committed")
    # every committed action has exactly one receipt or a recorded delivery failure
    rec_by_action = {r["action_id"]: r for r in receipts}
    fail_actions = {d["action_id"] for d in dfail}
    for c in commits:
        if c["action_id"] not in rec_by_action and c["action_id"] not in fail_actions:
            viol.append("action_without_receipt_or_failure")
            break
    if len(rec_by_action) != len(receipts):
        viol.append("duplicate_receipt_for_action")
    # replay the application's rule over the receipts in order: the receipt flags must agree
    acc_at = None
    dup = inv_clicks = 0
    first_A = None
    for r in sorted(receipts, key=lambda r: r["receipt_time_ns"]):
        c = next(cc for cc in commits if cc["action_id"] == r["action_id"])
        if r["received_action_hash"] != c["serialized_action_hash"]:
            viol.append("actuation_error")
        expected = m.receipt_outcome(r["task_time_at_receipt_ns"], c["action"], acc_at)
        if r["outcome"] != expected:
            viol.append("receipt_outcome_inconsistent_with_manifest")
        if expected == "accepted":
            acc_at = r["task_time_at_receipt_ns"]
            first_A = r["receipt_time_ns"]
        elif expected == "duplicate":
            dup += 1
        else:
            inv_clicks += 1
    failed = t.first("decision_failed")
    if failed is not None:
        return PendingResult(**base, valid_trial=False, invalid_reason=f"decision_failed:{failed['error']}",
                             n_decisions=len(decisions), n_clicks_committed=len(commits), n_suppressed=len(suppressed),
                             n_waits=len(waits), accepted=acc_at is not None, first_valid_E_A_ns=None, duplicates=dup,
                             invalid_clicks=inv_clicks, dropped=len(fail_actions - set(rec_by_action)), exact_success=False,
                             contract_violations=viol)
    pev: dict[str, int] = {}
    for e in t.all("pending_event"):
        pev[e["event"]] = pev.get(e["event"], 0) + 1
    return PendingResult(**base, valid_trial=t.first("trial_end") is not None, invalid_reason=None if t.first("trial_end") else "no trial_end",
                         n_decisions=len(decisions), n_clicks_committed=len(commits), n_suppressed=len(suppressed),
                         n_waits=len(waits), accepted=acc_at is not None,
                         first_valid_E_A_ns=(first_A - (h0 + m.event_time_ns)) if first_A is not None else None,
                         duplicates=dup, invalid_clicks=inv_clicks, dropped=len(fail_actions - set(rec_by_action)),
                         exact_success=(acc_at is not None and dup == 0), pending_events=pev, contract_violations=viol,
                         return_G=m.return_pending(acc_at))
