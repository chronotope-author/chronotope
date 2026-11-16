"""D1 distributional gym (cards/D1-distributional-gym.md): frozen manifests, fairness assertions, receipt-time scoring,
the three family scorers, retention, pairwise statistics and curves — all on scripted controllers under the
simulated clock, so every number here has a closed form."""
from __future__ import annotations

import json
import math

import pytest

from chronotope.distributional import (ROW_FIELDS, ControllerSpec, GenConfig, ManifestSet, build_manifest, build_report,
                                       check_fairness, deadline_curve, fit_logistic, generate, negative_control_check,
                                       pairwise, run_episode, score_trace, slack_curve, speed_curve, summarize_controller)
from chronotope.distributional.manifest import first_target_frame_ns, fixed_gates_ns
from chronotope.runner import PayloadStore
from chronotope.trace import Trace

S = 1_000_000_000
SMALL = dict(canvas_w_px=200, canvas_h_px=150, target_size_px=16)


def scripted(binding: str, compute_ns: int) -> dict:
    return {"kind": "scripted_oracle", "compute_ns": compute_ns, "coordinate_offset_px": 0, "binding": binding,
            "expected_observation_delay_ns": None, "expected_delivery_delay_ns": None}


def run_all(ms: ManifestSet, ctrl: ControllerSpec, out, conditions=("live", "replay")) -> dict[str, list]:
    store = PayloadStore(out / "payloads")
    by: dict[str, list] = {c: [] for c in conditions}
    for row in ms.rows:
        live = None
        for c in conditions:
            p = run_episode(row, ms.frozen_hash, ctrl, c, out, store, live_trace=live)
            t = Trace.load(p)
            if c == "live":
                live = t
            by[c].append(score_trace(row, t, ms.frozen_hash, controller=ctrl.name))
    return by


@pytest.fixture
def cw(tmp_path):
    ms = generate(GenConfig(task_family="closing_window", n=16, master_seed=11, **SMALL))
    ms.freeze(tmp_path / "cw.jsonl")
    return ms


# ---- manifest ---------------------------------------------------------------------------------------------------
def test_rows_have_exactly_the_required_fields_and_ranges(cw):
    for r in cw.rows:
        assert tuple(r.to_dict()) == ROW_FIELDS
        assert 1.0 <= r.deadline_s <= 16.0
        assert r.target_speed_px_s == 0.0 and r.update_type == "none" and r.update_time_s is None
    assert len({r.episode_id for r in cw.rows}) == len(cw.rows)
    assert len({r.deadline_s for r in cw.rows}) > 1          # a distribution, not a grid
    assert min(r.environment_seed for r in cw.rows) >= 5_000_000   # held out from every existing study


def test_generation_is_deterministic_and_freeze_hash_roundtrips(tmp_path):
    a = generate(GenConfig(task_family="closing_window", n=5, master_seed=3, **SMALL))
    b = generate(GenConfig(task_family="closing_window", n=5, master_seed=3, **SMALL))
    assert a.manifest_hash() == b.manifest_hash()
    h = a.freeze(tmp_path / "m.jsonl")
    loaded = ManifestSet.load(tmp_path / "m.jsonl")
    assert loaded.frozen_hash == h and [r.to_dict() for r in loaded.rows] == [r.to_dict() for r in a.rows]
    with pytest.raises(FileExistsError):
        a.freeze(tmp_path / "m.jsonl")            # never rewritten
    # editing a frozen row (a redrawn deadline) is detected
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    row = json.loads(lines[1]); row["deadline_s"] = 30.0
    lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    (tmp_path / "m.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        ManifestSet.load(tmp_path / "m.jsonl")


def test_moving_rows_are_log_uniform_in_speed_and_stay_on_canvas():
    ms = generate(GenConfig(task_family="moving_target", n=40, master_seed=5, generous_deadline_s=6.0, **SMALL))
    for r in ms.rows:
        assert 1.0 <= r.target_speed_px_s <= 16.0 and 0 <= r.target_direction < 2 * math.pi
        assert r.deadline_s == 6.0                            # generous for a scripted oracle: closure never confounds motion error
        m = build_manifest(r)
        for t in range(m.event_time_ns, m.window_close_ns(), 250_000_000):
            b = m.target_box_at(t)
            assert 0 <= b.x0 and b.x1 <= m.canvas_w_px and 0 <= b.y0 and b.y1 <= m.canvas_h_px
    st = generate(GenConfig(task_family="static_control", n=3, master_seed=5, **SMALL))
    assert all(r.target_speed_px_s == 0.0 for r in st.rows)


def test_gate_rows_order_and_slack_range():
    cfg = GenConfig(task_family="late_binding_gate", n=30, master_seed=8, **SMALL)
    ms = generate(cfg)
    b, d, a = fixed_gates_ns(cfg)
    for r in ms.rows:
        q = r.task_payload["q_s"]
        assert 1 / 60 <= q <= 1.0
        assert b < r.update_ns < d < a and r.update_ns == d - round(q * S)
    after = generate(GenConfig(task_family="late_binding_gate", n=5, master_seed=8, update_type="after_D", **SMALL))
    assert all(r.update_ns > a for r in after.rows)


def test_natural_update_is_inside_the_window_and_model_independent():
    ms = generate(GenConfig(task_family="late_binding_natural", n=20, master_seed=9, **SMALL))
    for r in ms.rows:
        assert r.event_ns < r.update_ns < r.close_ns
        assert math.isclose(r.update_ns, r.event_ns + r.task_payload["rho"] * r.window_ns, abs_tol=2)


# ---- fairness and the §16 assertions -----------------------------------------------------------------------------
def test_same_rows_same_deadline_same_evidence_across_controllers(cw, tmp_path):
    fast = run_all(cw, ControllerSpec("fast", scripted("state_at_B", 100_000_000)), tmp_path / "r")
    slow = run_all(cw, ControllerSpec("slow", scripted("state_at_B", 5_200_000_000)), tmp_path / "r")
    allo = fast["live"] + fast["replay"] + slow["live"] + slow["replay"]
    check_fairness(allo)
    for a, b in zip(fast["live"], slow["live"]):
        assert a.deadline_s == b.deadline_s and a.row_hash == b.row_hash and a.manifest_hash == cw.frozen_hash
        assert a.T_reason == 100_000_000 and b.T_reason == 5_200_000_000     # each at its own realized speed
    # the trace headers carry the row's deadline, not anything model-relative
    for row in cw.rows:
        h = Trace.load(tmp_path / "r" / "slow" / f"{row.episode_id}_r0_live.jsonl").header()
        assert h["spec"]["scenario"]["response_window_ns"] == row.window_ns


def test_fairness_rejects_mixed_manifests_and_duplicate_runs(cw, tmp_path):
    a = run_all(cw, ControllerSpec("a", scripted("state_at_B", 100_000_000)), tmp_path / "r", ("live",))["live"]
    bad = a[0].to_dict(); bad["manifest_hash"] = "0" * 64
    with pytest.raises(AssertionError, match="manifests"):
        check_fairness(a + [type(a[0]).from_dict(bad)])
    with pytest.raises(AssertionError, match="scored twice"):
        check_fairness(a + [a[0]])
    with pytest.raises(FileExistsError, match="never redrawn"):
        run_episode(cw.rows[0], cw.frozen_hash, ControllerSpec("a", scripted("state_at_B", 100_000_000)), "live",
                    tmp_path / "r", PayloadStore(tmp_path / "r" / "payloads"))


# ---- closing window: scored at A, never at D ----------------------------------------------------------------------
def test_closing_window_success_is_receipt_le_close(cw, tmp_path):
    slow = run_all(cw, ControllerSpec("slow", scripted("state_at_B", 5_200_000_000)), tmp_path / "r")
    n_late = 0
    for o, row in zip(slow["live"], cw.rows):
        close_h = o.E + row.window_ns
        assert o.C == 1 and o.R == 1                       # the click itself is right on every row
        assert o.V_A == int(o.A <= close_h) and o.Y == o.V_A
        assert o.S_A_ns == close_h - o.A
        n_late += o.failure_mode == "late"
        # the Replay twin of the same episode holds the window: semantic credit only
        rp = next(r for r in slow["replay"] if r.episode_id == o.episode_id)
        assert rp.C == 1 and rp.Y == 1 and rp.V_A is None
    assert 0 < n_late < len(cw.rows)                       # W log-uniform on [1, 16] straddles 5.2 s + delays
    s = summarize_controller(slow["live"], slow["replay"])
    assert s["retention"] == s["live_sr"] and s["semantic_sr"]["replay"] == 1.0 and s["temporal_loss"] == 1 - s["live_sr"]


def test_no_scoring_at_D_delivery_after_close_is_late(tmp_path):
    ms = generate(GenConfig(task_family="closing_window", n=20, master_seed=21, delivery_delay_ns=2 * S, **SMALL))
    ms.freeze(tmp_path / "m.jsonl")
    live = run_all(ms, ControllerSpec("c", scripted("state_at_B", 100_000_000)), tmp_path / "r", ("live",))["live"]
    saw_gap = False
    for o, row in zip(live, ms.rows):
        close_h = o.E + row.window_ns
        if o.D <= close_h < o.A:
            saw_gap = True
            assert o.Y == 0 and o.failure_mode == "late"    # committed in time, received late: no credit
    assert saw_gap


def test_retention_is_undefined_not_zero_when_nothing_is_replay_solvable(cw, tmp_path):
    off = dict(scripted("state_at_B", 100_000_000), coordinate_offset_px=190)   # clicks off the canvas edge
    r = run_all(cw, ControllerSpec("off", off), tmp_path / "r")
    s = summarize_controller(r["live"], r["replay"])
    assert s["semantic_sr"]["replay"] == 0.0 and s["retention"] is None and s["temporal_loss"] is None
    md = build_report("closing_window", {"off": r}, n_boot=50)
    assert "undefined" in md


def test_deadline_curve_recovers_the_oracle_latency(tmp_path):
    ms = generate(GenConfig(task_family="closing_window", n=60, master_seed=31, **SMALL))
    ms.freeze(tmp_path / "m.jsonl")
    live = run_all(ms, ControllerSpec("slow", scripted("state_at_B", 5_200_000_000)), tmp_path / "r", ("live",))["live"]
    cv = deadline_curve(live, n_boot=30)
    assert cv["fit"]["W50"] is not None and cv["beta"] > 0
    assert 4.0 < cv["fit"]["W50"] < 7.0                    # the threshold sits near 5.2 s + O and D->A delays
    assert cv["fit"]["W90"] > cv["fit"]["W50"]


# ---- moving target ---------------------------------------------------------------------------------------------------
def test_moving_target_error_law_and_normalized_displacement(tmp_path):
    ms = generate(GenConfig(task_family="moving_target", n=16, master_seed=41, generous_deadline_s=6.0, **SMALL))
    ms.freeze(tmp_path / "m.jsonl")
    fro = run_all(ms, ControllerSpec("frozen", scripted("state_at_B", 2 * S)), tmp_path / "r", ("live",))["live"]
    tru = run_all(ms, ControllerSpec("true_at_A", scripted("true_at_A", 2 * S)), tmp_path / "r", ("live",))["live"]
    for o, row in zip(fro, ms.rows):
        m = build_manifest(row)
        t1 = first_target_frame_ns(row.event_ns, row.task_payload["frame_period_ns"])
        travel = row.target_speed_px_s * (o.A - (o.E - row.event_ns) - t1) / 1e9
        assert o.e_px == pytest.approx(travel, abs=1.5)     # e = v (A - tau) up to pixel flooring
        assert o.x_norm == pytest.approx(o.e_px / m.half_width_px())
        assert o.Y == int(o.C_kind == 1 and o.R == 1 and o.x_norm <= 1.0)
        assert o.deadline_s == 6.0 and o.V_A == 1
    assert all(o.x_norm < 0.2 and o.Y == 1 for o in tru)
    pw = pairwise(tru, fro, n_boot=200)
    assert pw["n01"] == 0 and pw["n"] == 16 and pw["delta_sr"] >= 0
    cv = speed_curve(fro, n_boot=20)
    assert cv["n"] == 16


# ---- late binding: natural and fixed gate ----------------------------------------------------------------------------
def test_natural_late_binding_itt_and_mechanism_strata(tmp_path):
    ms = generate(GenConfig(task_family="late_binding_natural", n=16, master_seed=51, **SMALL))
    ms.freeze(tmp_path / "m.jsonl")
    fro = run_all(ms, ControllerSpec("frozen", scripted("state_at_B", 3 * S)), tmp_path / "r")
    lb = run_all(ms, ControllerSpec("late_bind", scripted("newest_at_D", 3 * S)), tmp_path / "r")
    check_fairness(fro["live"] + fro["replay"] + lb["live"] + lb["replay"])
    for a, b in zip(fro["live"], lb["live"]):
        assert a.U - a.E == b.U - b.E and a.G == b.G           # the same U for every controller
    for o in fro["live"]:
        if o.G == 1 and o.V_A == 1 and o.A - o.U > 60_000_000:
            assert o.Y == 0 and o.failure_mode == "stale_value"   # the code changed during B->D; frozen typed the old one
    for o in lb["live"]:
        if o.G == 1 and o.V_A == 1 and (o.D - o.U) > 60_000_000:
            assert o.Y == 1                                   # the update reached O before D: late binding used it
    s = summarize_controller(lb["live"], lb["replay"])
    assert "itt_sr" in s and s["mechanism_stratum_n"] + s["update_after_D"] <= 16
    md = build_report("late_binding_natural", {"frozen": fro, "late_bind": lb}, n_boot=50)
    assert "mechanism SR" in md


def test_fixed_gate_binding_success_and_slack_curve(tmp_path):
    cfg = GenConfig(task_family="late_binding_gate", n=24, master_seed=61, **SMALL)
    ms = generate(cfg)
    ms.freeze(tmp_path / "m.jsonl")
    comp = ms.rows[0].task_payload["gate_compute_ns"]
    fro = run_all(ms, ControllerSpec("frozen", scripted("state_at_B", comp)), tmp_path / "r", ("live",))["live"]
    lb = run_all(ms, ControllerSpec("late_bind", scripted("newest_at_D", comp)), tmp_path / "r", ("live",))["live"]
    rs = run_all(ms, ControllerSpec("restart", scripted("restart_if_changed", comp)), tmp_path / "r", ("live",))["live"]
    b, d, a = fixed_gates_ns(cfg)
    for o in fro + lb:
        assert o.B == b and o.D == d and o.A == a              # common gates
    for o in fro:
        assert o.Y == 0 and o.z_hat == build_manifest(ms.row(o.episode_id)).codes()[0]   # stale at A*
    for o, row in zip(lb, ms.rows):
        per = row.task_payload["frame_period_ns"]
        t_up = math.ceil(row.update_ns / per) * per                    # first frame that depicts the new code
        observable = t_up + row.publication_delay_ns <= d              # ... and it reached O before D*
        codes = row.task_payload["codes"]
        assert (o.z_latest == codes[1]) == observable
        if o.z_latest == codes[1]:
            assert o.z_hat == codes[1] and o.Y == 1
        else:
            assert o.Y == 0 and o.failure_mode == "stale_value"
    for o in rs:
        assert o.n_restarts in (0, 1)
        if o.n_restarts == 1:
            assert o.D > d and o.z_hat == ms.row(o.episode_id).task_payload["codes"][1]
    assert sum(o.Y for o in lb) > sum(o.Y for o in fro)
    cv = slack_curve(lb, n_boot=20)
    assert cv["n"] == 24 and (cv["fit"] is None or cv["fit"]["q50"] is None or cv["fit"]["q50"] > 0)
    md = build_report("late_binding_gate", {"frozen": {"live": fro}, "late_bind": {"live": lb}, "restart": {"live": rs}}, n_boot=50)
    assert "q50" in md


@pytest.mark.parametrize("stratum", ["irrelevant", "duplicate", "after_D", "kind_change"])
def test_negative_control_strata_give_late_binding_no_credit(tmp_path, stratum):
    cfg = GenConfig(task_family="late_binding_gate", n=8, master_seed=71, update_type=stratum, **SMALL)
    ms = generate(cfg)
    ms.freeze(tmp_path / "m.jsonl")
    comp = ms.rows[0].task_payload["gate_compute_ns"]
    fro = run_all(ms, ControllerSpec("frozen", scripted("state_at_B", comp)), tmp_path / "r", ("live",))["live"]
    lb = run_all(ms, ControllerSpec("late_bind", scripted("newest_at_D", comp)), tmp_path / "r", ("live",))["live"]
    r = negative_control_check(lb, fro)
    assert not r["late_binding_credited"] and r["delta_sr"] == 0.0
    if stratum == "kind_change":
        assert all(o.Y == 0 for o in fro + lb)             # a typed value into a vanished field is never valid
    else:
        assert all(o.Y == 1 for o in fro + lb)             # nothing to bind: both keep the (still valid) value


# ---- statistics --------------------------------------------------------------------------------------------------------
def test_pairwise_counts_and_mcnemar():
    from chronotope.distributional.episode import EpisodeOutcome

    def mk(eid, ctrl, y):
        return EpisodeOutcome(episode_id=eid, task_family="closing_window", controller=ctrl, condition="live", repeat_index=0,
                              manifest_hash="m", row_hash=eid, scorer="distributional-1", deadline_s=1.0, E=0, O=None, B=None, D=None,
                              A=None, L_EO=None, L_OB=None, L_BD=None, L_DA=None, L_EA=None, T_reason=None, action=None,
                              C=y, R=y, V_A=y, Y=y, S_A_ns=None)
    ya = [1, 1, 1, 0, 0, 1, 0, 1]
    yb = [1, 0, 0, 0, 1, 1, 0, 0]
    a = [mk(f"e{i}", "a", y) for i, y in enumerate(ya)]
    b = [mk(f"e{i}", "b", y) for i, y in enumerate(yb)]
    pw = pairwise(a, b, n_boot=500)
    assert (pw["n10"], pw["n01"], pw["n11"], pw["n00"]) == (3, 1, 2, 2)
    assert pw["delta_sr"] == pytest.approx(2 / 8) and pw["delta_ci95"][0] <= pw["delta_sr"] <= pw["delta_ci95"][1]
    assert 0 < pw["mcnemar_p"] <= 1
    # repeats of one episode are one cluster: the same episode twice does not double the count
    pw2 = pairwise(a + [mk("e0", "a", 1)], b + [mk("e0", "b", 1)], n_boot=100)
    assert pw2["n"] == 8


def test_penalized_logistic_is_finite_under_separation():
    xs = [math.log(w) for w in (1, 2, 3, 4, 8, 12, 16)]
    ys = [0, 0, 0, 1, 1, 1, 1]
    a, b = fit_logistic(xs, ys)
    assert math.isfinite(a) and math.isfinite(b) and b > 0
    assert fit_logistic(xs, [1] * 7) is None
