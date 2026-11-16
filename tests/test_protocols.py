"""PHY2 part 1: FrameSource / Route protocols and the no-behaviour-change guard (P3 hard stop).

The two golden shas were computed on 2026-08-26 BEFORE the protocol refactor (commit ef3d0cc, the
last state of runner.py without FrameSource/Route/make_source), from the same specs, by hashing the
complete trace file. A simulated-clock trial is deterministic, so any diff = the refactor changed
behaviour. Both goldens: `PYTHONPATH=src .venv/bin/python -c` of the two builders below at ef3d0cc.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronotope.runner import (EnvFrameSource, FrameSource, ReplayFrameSource, Route, SimulatedDropFirstRoute,
                               SimulatedEmpiricalRoute, SimulatedFixedDelayRoute, SimulatedSeededJitterRoute,
                               PayloadStore, make_route, make_source, run_trial)
from chronotope.spec import SpecError, TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict
from test_moving_target import mt_spec

GOLDEN_MOVING_TARGET = "30d606a3648998974569ec6f600003aaad3140adb1425c4c2cdb12b96a7e2826"
GOLDEN_B1_JITTER = "e82cfbcc7777d9aea2bcf0510a02d6c2cfbc28084ab64419926abb14cbac419c"


def b1_jitter_spec() -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "live_trading", "scenario_seed": 1001, "canvas_w_px": 128, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": 500_000_000, "response_window_ns": 2_000_000_000,
                     "frame_period_ns": 100_000_000, "episode_horizon_ns": 3_000_000_000,
                     "irrelevant_motion": False, "skin": "chart", "reversal_delay_ns": 250_000_000,
                     "u_valid_order": 1.0, "u_invalid_order": -1.0, "u_no_order": 0.0}
    d["controller"]["compute_ns"] = 200_000_000
    d["delivery"] = {"route": "simulated_seeded_jitter", "delivery_delay_min_ns": 0, "delivery_delay_max_ns": 60_000_000,
                     "delivery_seed": 1001, "max_attempts": 1}
    return TrialSpec.from_dict(d)


def _sha(spec: TrialSpec, ws: Path, tag: str) -> str:
    p = run_trial(spec, ws / f"{tag}.jsonl", PayloadStore(ws / "payloads"))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_golden_trace_sha_moving_target_oracle(ws):
    assert _sha(mt_spec(vx=120, delivery=60_000_000, seed=7), ws, "mt") == GOLDEN_MOVING_TARGET


def test_golden_trace_sha_b1_seeded_jitter(ws):
    assert _sha(b1_jitter_spec(), ws, "b1") == GOLDEN_B1_JITTER


def test_simulated_classes_satisfy_the_protocols(ws):
    spec = mt_spec()
    for r in (SimulatedFixedDelayRoute(1), SimulatedDropFirstRoute(1), SimulatedSeededJitterRoute(0, 5, 1),
              SimulatedEmpiricalRoute([1.0, 2.0], 1)):
        assert isinstance(r, Route) and isinstance(r.route, str) and isinstance(r.delay, int)
    m = spec.manifest()
    env = EnvFrameSource(m, None, 0, 0)
    assert isinstance(env, FrameSource) and env.name == "environment"
    live = Trace.load(run_trial(spec, ws / "live.jsonl", PayloadStore(ws / "p")))
    rep = ReplayFrameSource(live, PayloadStore(ws / "p"), 0)
    assert isinstance(rep, FrameSource) and rep.name == "replay"


def test_make_source_dispatches_on_observation_source(ws):
    spec = mt_spec()
    assert make_source(spec, spec.manifest(), None, 0).name == "environment"
    d = json.loads(json.dumps(spec.raw)); d["observation"]["source"] = "environment"
    s2 = TrialSpec.from_dict(d)
    assert make_source(s2, s2.manifest(), None, 0).name == "environment"
    d["observation"]["source"] = "capture"
    s3 = TrialSpec.from_dict(d)
    with pytest.raises(ValueError, match="capture"):
        make_source(s3, s3.manifest(), None, 0)
    d["observation"]["source"] = "telepathy"
    with pytest.raises(SpecError):
        TrialSpec.from_dict(d)


def test_optional_source_does_not_change_spec_hash_when_absent():
    """The only optional field: absent == default, so existing specs keep their spec_hash."""
    d = base_spec_dict()
    assert "source" not in d["observation"]
    assert TrialSpec.from_dict(d).spec_hash() == TrialSpec.from_dict(json.loads(json.dumps(d))).spec_hash()


def test_simulated_empirical_route_draws_seeded_samples(ws, tmp_path):
    (tmp_path / "PHY_route_t.json").write_text(json.dumps({"da_ms": [10.0, 20.0, 30.5]}))
    d = base_spec_dict()
    d["delivery"] = {"route": "simulated_empirical", "route_name": str(tmp_path / "PHY_route_t.json"),
                     "delivery_seed": 3, "max_attempts": 1}
    spec = TrialSpec.from_dict(d)
    r1, r2 = make_route(spec.raw["delivery"]), make_route(spec.raw["delivery"])
    assert r1.route == "simulated_empirical" and r1.delay == 20_000_000
    a = [r1.deliver(b"x", 1, 0)[0] for _ in range(20)]
    assert a == [r2.deliver(b"x", 1, 0)[0] for _ in range(20)]      # seeded: reproducible
    assert set(a) <= {10_000_000, 20_000_000, 30_500_000} and len(set(a)) > 1
    p = run_trial(spec, ws / "e.jsonl", PayloadStore(ws / "p"))
    t = Trace.load(p)
    assert t.first("delivery")["route"] == "simulated_empirical"
    rec = t.first("receipt")
    assert rec["receipt_time_ns"] - t.first("commit")["commitment_time_ns"] in {10_000_000, 20_000_000, 30_500_000}
    with pytest.raises(ValueError):
        SimulatedEmpiricalRoute([], 1)
    with pytest.raises(ValueError):
        SimulatedEmpiricalRoute([-1.0], 1)
