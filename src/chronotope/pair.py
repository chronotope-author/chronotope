"""Replay–Live pair validator (experiments.md §14, §15.6).

A pair is valid only when every required identity matches and the realized evidence is the
same. Any mismatch is `invalid_pair` with a specific reason — never an agent failure.
"""
from __future__ import annotations

from dataclasses import dataclass

from .trace import Trace


@dataclass(frozen=True)
class PairVerdict:
    valid: bool
    reason: str
    same_evidence_at_decision: bool | None = None  # did both decisions begin on the same observation?


def _evidence(t: Trace) -> list[tuple[int, int, str]]:
    return [(o["source_event_id"], o["source_time_ns"], o["payload_hash"]) for o in t.all("observation")]


def _events(t: Trace) -> list[tuple[int, str, int]]:
    return [(e["event_id"], e["kind"], e["task_time_ns"]) for e in t.all("event")]


def validate_pair(replay: Trace, live: Trace) -> PairVerdict:
    rh, lh = replay.header(), live.header()
    if {rh["task_clock_policy"], lh["task_clock_policy"]} != {"replay", "live"}:
        return PairVerdict(False, "policies must be exactly {replay, live}")
    if rh["task_clock_policy"] != "replay":
        return PairVerdict(False, "first argument must be the replay trace")
    if rh["replay_source_live_trace_sha256"] != live.file_hash():
        return PairVerdict(False, "replay_source_live_trace_sha256 does not match the live trace file")
    if rh["matching_key_hash"] != lh["matching_key_hash"]:
        return PairVerdict(False, "spec matching key differs (something other than task-clock policy changed)")
    for k in ("manifest_hash", "controller_spec_hash", "action_space_version", "scorer_version",
              "observation_boundary_version", "trial_class"):
        if rh[k] != lh[k]:
            return PairVerdict(False, f"{k} differs")
    if _events(replay) != _events(live):
        return PairVerdict(False, "event trace differs")
    ev_r, ev_l = _evidence(replay), _evidence(live)
    if len(ev_r) > len(ev_l):
        return PairVerdict(False, "replay published more observations than live recorded")
    for i, (a, b) in enumerate(zip(ev_r, ev_l)):
        if a != b:
            return PairVerdict(False, f"evidence mismatch at observation index {i}: {a} != {b}")
    db_r = replay.committing_decision() or replay.first("decision_begin")
    db_l = live.committing_decision() or live.first("decision_begin")
    same = None
    if db_r is not None and db_l is not None:
        same = db_r["observation_id"] == db_l["observation_id"]
        if db_r["observation_id"] > len(ev_r):
            return PairVerdict(False, "replay decision references an observation it did not publish")
    return PairVerdict(True, "ok", same_evidence_at_decision=same)
