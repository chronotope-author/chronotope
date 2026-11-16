"""U3: §16 negative contract tests. Each test introduces one defect and verifies rejection.
Items 8 and 9 (GPU residency) are not applicable before Gate 3. Items 15/16 are realized as
trace-level checks because no restart/buffer mechanism exists yet (a trace that *claims* one is
rejected)."""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest

from chronotope.__main__ import run_triplet
from chronotope.controller import Decision, Observation, ScriptedOracleController
from chronotope.evaluate import evaluate_trace
from chronotope.hashing import canonical_json
from chronotope.pair import validate_pair
from chronotope.runner import PayloadStore, SimulatedFixedDelayRoute, run_trial
from chronotope.scenario import Manifest
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict


def _spec(**over) -> TrialSpec:
    return TrialSpec.from_dict(base_spec_dict(**over))


def _tamper(src: Path, dst: Path, fn) -> Trace:
    """Write a corrupted COPY of a trace (traces themselves are append-only) and load it."""
    recs = [json.loads(l) for l in src.read_text().splitlines()]
    recs = fn(recs)
    for n, r in enumerate(recs, 1):   # a defective harness would still write a well-formed file
        r["seq"] = n
    dst.write_text("".join(canonical_json(r) + "\n" for r in recs))
    return Trace.load(dst)


# ---- §11 strict configuration: items 1, 2, 3, 12 and friends ------------------------------
def test_01_missing_seed():
    d = base_spec_dict(); del d["scenario"]["scenario_seed"]
    with pytest.raises(SpecError, match="missing"):
        TrialSpec.from_dict(d)


def test_02_unknown_field():
    d = base_spec_dict(); d["delivery"]["jitter_ns"] = 5
    with pytest.raises(SpecError, match="unknown"):
        TrialSpec.from_dict(d)


@pytest.mark.parametrize("name", ["delivery_delay", "delivery_delay_ms", "delivery_delay_s"])
def test_03_unitless_or_wrong_unit_delay(name):
    d = base_spec_dict(); del d["delivery"]["delivery_delay_ns"]; d["delivery"][name] = 20
    with pytest.raises(SpecError):
        TrialSpec.from_dict(d)


def test_12_generic_metadata_container_rejected():
    for k in ("metadata", "options", "extra"):
        d = base_spec_dict(); d[k] = {"anything": 1}
        with pytest.raises(SpecError, match="unknown"):
            TrialSpec.from_dict(d)


def test_strict_parsing_rejects_types_duplicates_nonfinite_unversioned():
    with pytest.raises(SpecError, match="expected int"):
        TrialSpec.from_dict(base_spec_dict(controller__compute_ns=True))       # bool is not int
    with pytest.raises(SpecError, match="expected int"):
        TrialSpec.from_dict(base_spec_dict(controller__compute_ns="5"))        # no str->int
    with pytest.raises(SpecError, match="expected int"):
        TrialSpec.from_dict(base_spec_dict(controller__compute_ns=5.0))        # no float->int
    with pytest.raises(SpecError, match="duplicate"):
        TrialSpec.from_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(SpecError, match="non-finite"):
        TrialSpec.from_text('{"schema_version": 1, "x": NaN}')
    d = base_spec_dict(); del d["schema_version"]
    with pytest.raises(SpecError, match="unversioned"):
        TrialSpec.from_dict(d)
    with pytest.raises(SpecError, match="non-negative"):
        TrialSpec.from_dict(base_spec_dict(controller__compute_ns=-1))
    with pytest.raises(SpecError, match="replay_source"):
        TrialSpec.from_dict(base_spec_dict(task_clock_policy="replay"))
    with pytest.raises(SpecError, match="replay_source"):
        TrialSpec.from_dict(base_spec_dict(replay_source_live_trace_sha256="abc"))


# ---- §6 environment independence: items 4, 5 ---------------------------------------------
def _forbid_wall_clock(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("wall clock read during a simulated trial")
    for name in ("time", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "time_ns"):
        monkeypatch.setattr(time, name, boom)


def test_04_no_wall_clock_in_task_logic(ws, monkeypatch):
    _forbid_wall_clock(monkeypatch)
    out = run_triplet(_spec(), ws)            # the whole harness runs without touching the wall clock
    assert out["pair"].valid

    class Defective(Manifest):                 # a scenario that peeks at the wall clock is caught
        def state_at(self, t):
            time.time()
            return super().state_at(t)
    with pytest.raises(AssertionError, match="wall clock"):
        Defective(**{k: v for k, v in base_spec_dict()["scenario"].items() if k != "scenario_id"}).state_at(0)


def test_05_no_condition_dependent_branch(ws):
    # Structural: the scenario's state and evidence functions take task time and nothing else.
    for fn in (Manifest.state_at, Manifest.render):
        assert list(inspect.signature(fn).parameters) == ["self", "task_time_ns"], fn
    assert list(inspect.signature(Manifest.action_valid_at).parameters) == ["self", "task_time_ns", "action"]
    out = run_triplet(_spec(controller__compute_ns=300_000_000), ws)
    evid = {}
    for p in ("paused", "replay", "live"):
        t = Trace.load(ws / f"{p}.jsonl")
        evid[p] = {o["source_time_ns"]: o["payload_hash"] for o in t.all("observation")}
    # At any depicted task time, all three conditions rendered identical evidence.
    for p in ("paused", "replay"):
        for st, h in evid[p].items():
            assert evid["live"][st] == h, (p, st)


# ---- §14 pair validation: item 6 -----------------------------------------------------------
def test_06_different_replay_live_evidence_is_invalid_pair(ws):
    run_triplet(_spec(), ws)
    live = Trace.load(ws / "live.jsonl")

    def corrupt(recs):
        obs = [r for r in recs if r["record"] == "observation"]
        obs[3]["payload_hash"] = "0" * 64
        return recs
    replay_bad = _tamper(ws / "replay.jsonl", ws / "replay_bad.jsonl", corrupt)
    v = validate_pair(replay_bad, live)
    assert not v.valid and "evidence mismatch at observation index 3" in v.reason

    # A replay run against a different Live trace (different seed) is refused by the runner...
    run_triplet(_spec(scenario__scenario_seed=8), ws / "other")
    other_live = Trace.load(ws / "other" / "live.jsonl")
    with pytest.raises(ValueError, match="does not match"):
        run_trial(_spec().derive("replay", live.file_hash()), ws / "x.jsonl", PayloadStore(ws / "payloads"),
                  live_trace=other_live)
    # ...and a pair whose specs differ in anything but the policy is invalid.
    replay_other = run_trial(_spec(scenario__scenario_seed=8).derive("replay", live.file_hash()),
                             ws / "y.jsonl", PayloadStore(ws / "payloads"), live_trace=live)
    v = validate_pair(Trace.load(replay_other), live)
    assert not v.valid and "matching key" in v.reason

    # Controls: the pair validator also rejects two lives, and a same-policy swap.
    assert not validate_pair(live, live).valid


# ---- §13 trace validity: item 7 --------------------------------------------------------------
def test_07_missing_A_makes_trial_invalid_not_interpolated(ws):
    run_triplet(_spec(), ws)
    t = _tamper(ws / "live.jsonl", ws / "noA.jsonl", lambda rs: [r for r in rs if r["record"] != "receipt"])
    r = evaluate_trace(t)
    assert not r.valid_trial and r.invalid_reason == "missing_A"
    assert r.valid_at_receipt is None and r.intervals_ns == {}


# ---- §9.6 / §10: items 10, 11 -----------------------------------------------------------------
def test_10_committed_action_is_immutable_and_mutation_in_transit_is_an_actuation_error(ws):
    with pytest.raises(Exception):
        Decision(action={"kind": "click", "x": 1, "y": 1}, declared_compute_ns=0).action = {}
    with pytest.raises(Exception):
        Observation(1, 1, 0, 0, "h", 1, b"").payload_hash = "x"

    class Tamper(SimulatedFixedDelayRoute):      # a route that "repairs" coordinates in flight (§10 forbids)
        def deliver(self, b, attempt, submit_ns):
            a = json.loads(b); a["x"] = 0; a["y"] = 0
            return submit_ns + self.delay, canonical_json(a).encode()
    p = run_trial(_spec(), ws / "t.jsonl", PayloadStore(ws / "p"), route=Tamper(20_000_000))
    r = evaluate_trace(Trace.load(p))
    assert "actuation_error" in r.labels and r.valid_at_receipt is False and r.semantic_correct is True


def test_11_retries_are_recorded_and_unrecorded_attempts_are_flagged(ws):
    class FailFirst(SimulatedFixedDelayRoute):
        def deliver(self, b, attempt, submit_ns):
            return None if attempt == 1 else (submit_ns + self.delay, b)
    spec = _spec(delivery__max_attempts=2)
    p = run_trial(spec, ws / "r.jsonl", PayloadStore(ws / "p"), route=FailFirst(20_000_000))
    t = Trace.load(p)
    assert [d["attempt"] for d in t.all("delivery")] == [1, 2]
    assert [f["attempt"] for f in t.all("delivery_failed")] == [1]
    assert t.first("receipt")["attempt"] == 2
    assert evaluate_trace(t).contract_violations == []
    hidden = _tamper(p, ws / "hidden.jsonl",
                     lambda rs: [r for r in rs if not (r["record"] == "delivery" and r["attempt"] == 2)])
    assert "unrecorded_delivery_attempt" in evaluate_trace(hidden).contract_violations

    class AlwaysFail(SimulatedFixedDelayRoute):
        def deliver(self, b, attempt, submit_ns):
            return None
    p2 = run_trial(spec, ws / "drop.jsonl", PayloadStore(ws / "p"), route=AlwaysFail(20_000_000))
    r = evaluate_trace(Trace.load(p2))
    assert r.valid_trial and r.valid_at_receipt is False and "dropped" in r.labels


# ---- §7.1, §7.2: items 13, 14 -------------------------------------------------------------
def test_13_irrelevant_motion_advances_observation_version_not_control_state(ws):
    spec = _spec(scenario__irrelevant_motion=True, controller__compute_ns=100_000_000)
    p = run_trial(spec, ws / "im.jsonl", PayloadStore(ws / "p"))
    t = Trace.load(p)
    c, d = t.first("commit"), t.first("decision_begin")
    obs_at_b = d["observation_id"]
    published_before_d = [o for o in t.all("observation") if o["publication_time_ns"] < c["commitment_time_ns"]]
    assert published_before_d[-1]["observation_version"] > obs_at_b         # raw version moved during B->D
    assert c["control_state_version_at_commit"] == c["supporting_control_state_version"]  # control state did not
    assert evaluate_trace(t).valid_at_receipt is True
    # A trace that uses the raw observation version as the commitment guard is rejected.
    bad = _tamper(p, ws / "badguard.jsonl", lambda rs: [
        ({**r, "supporting_control_state_version": published_before_d[-1]["observation_version"]}
         if r["record"] == "commit" else r) for r in rs])
    assert "supporting_version_inconsistent_with_B" in evaluate_trace(bad).contract_violations


def test_14_backlog_is_coalesced_to_newest(ws):
    spec = _spec(controller__compute_ns=100_000_000)  # ~6 frames arrive during B->D
    t = Trace.load(run_trial(spec, ws / "bl.jsonl", PayloadStore(ws / "p")))
    d, c = t.first("decision_begin"), t.first("commit")
    during = [o["observation_id"] for o in t.all("observation")
              if d["begin_time_ns"] < o["publication_time_ns"] <= c["commitment_time_ns"]]
    assert len(during) >= 5
    adm = [a for a in t.all("admission") if a["harness_time_ns"] == c["commitment_time_ns"]]
    assert adm and adm[-1]["observation_id"] == max(during)            # newest admitted at D
    superseded = {s["observation_id"] for s in t.all("observation_superseded")}
    assert superseded == set(during[:-1])                               # the rest recorded as superseded


# ---- §13 invocation identity: items 15, 16 -------------------------------------------------
def test_15_restart_recorded_as_same_decision_is_rejected(ws):
    run_triplet(_spec(), ws)
    p = ws / "live.jsonl"
    def dup(rs):
        db = next(r for r in rs if r["record"] == "decision_begin")
        i = rs.index(db)
        rs.insert(i + 1, {**db, "seq": 0})
        for n, r in enumerate(rs, 1):
            r["seq"] = n
        return rs
    assert "decision_id_reused" in evaluate_trace(_tamper(p, ws / "dup.jsonl", dup)).contract_violations


def test_16_buffered_observation_claimed_as_same_decision_update_is_rejected(ws):
    # compute spans the window close, so a relevant change (target gone) is admitted before D
    spec = _spec(controller__compute_ns=450_000_000)
    p = run_trial(spec, ws / "buf.jsonl", PayloadStore(ws / "p"))
    t = Trace.load(p)
    c = t.first("commit")
    assert c["control_state_version_at_commit"] > c["supporting_control_state_version"]
    assert "outdated_at_commitment" in evaluate_trace(t).labels
    bad = _tamper(p, ws / "claim.jsonl", lambda rs: [
        ({**r, "supporting_control_state_version": r["control_state_version_at_commit"]}
         if r["record"] == "commit" else r) for r in rs])
    assert "supporting_version_inconsistent_with_B" in evaluate_trace(bad).contract_violations


# ---- §9.2 concrete action contract: items 17, 18 -------------------------------------------
class _Emits(ScriptedOracleController):
    def __init__(self, action, spec):
        m = spec.manifest()
        super().__init__(spec.raw["controller"]["compute_ns"], 0, m.canvas_w_px, m.canvas_h_px)
        self._a = action

    def decide(self):
        self._decided = True
        return Decision(action=self._a, declared_compute_ns=self.compute_ns)


@pytest.mark.parametrize("action,reason", [
    ({"kind": "click_target", "label": "the square"}, "fields must be exactly"),   # 17: label, not click(x,y)
    ({"kind": "press_key", "x": 0, "y": 0}, "not in action space"),                # 18: invented mechanism
    ({"kind": "click", "x": 1.0, "y": 2}, "must be int"),
    ({"kind": "click", "x": 999, "y": 2}, "outside canvas"),
])
def test_17_18_non_contract_actions_fail_commitment(ws, action, reason):
    spec = _spec()
    p = run_trial(spec, ws / "a.jsonl", PayloadStore(ws / "p"), controller=_Emits(action, spec))
    t = Trace.load(p)
    assert t.first("commit") is None
    assert reason in t.first("commit_failed")["reason"]
    assert t.first("trial_end")["reason"] == "commit_failed"
    r = evaluate_trace(t)
    assert r.valid_trial and not r.action_committed and r.valid_at_receipt is False
