"""WI-004 study (S7.2, Gate 5): frozen vs restart-after-update vs late binding with a real agent on the
moving balloon. Predeclared parameters live HERE (see cards/C1-balloon-drift.md).

  PYTHONPATH=src .venv/bin/python studies/c1_balloon_drift.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/wi004_qwen38_nothink --max-calls 400
  ... --thinking --reasoning-effort low --max-tokens 4096 --subset-thinking --out runs/wi004_qwen38_think

Predeclared:
  calibration: Paused, v = 0, policy frozen, seeds 100..109 -> competence >= 0.80 else STOP
  evaluation:  Live, seeds 300..309 x velocities {0, 4, 8, 16} px/s x {frozen, restart_if_moved, late_bind}
  control:     Live, v = 0, irrelevant_motion = True, seeds 300..309 x {frozen, late_bind}
  --subset-thinking: velocities {4, 8} x {frozen, late_bind} only (evidence-age knob; ~24 s per call)
  balloon skin 320x240, target 24 px, prompt 5 (0-1000 grid), image scale 2; event 500 ms; window 16 s
  (lateness is not the failure under study); 60 Hz frames; 30 ms publication; 20 ms delivery;
  tracker: template radius 12 px, min NCC 0.8; restart threshold 6 px.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.controller import CallBudget  # noqa: E402
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

CALIBRATION_SEEDS = list(range(100, 110))
EVAL_SEEDS = list(range(300, 310))
VELOCITIES = [0, 4, 8, 16]
POLICIES = ["frozen", "restart_if_moved", "late_bind"]
COMPETENCE_THRESHOLD = 0.80
WINDOW_NS = 16_000_000_000
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def make_spec(a, seed: int, vx: int, policy: str, irrelevant: bool, clock: str) -> TrialSpec:
    return TrialSpec.from_dict({
        "schema_version": 1,
        "scenario": {"scenario_id": "moving_target", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 500_000_000, "response_window_ns": WINDOW_NS,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 500_000_000 + WINDOW_NS + 500_000_000,
                     "irrelevant_motion": irrelevant, "skin": "balloon",
                     "velocity_x_px_per_s": vx, "velocity_y_px_per_s": 0},
        "task_clock_policy": clock, "replay_source_live_trace_sha256": None,
        "controller": {"kind": "vlm_late_binding", "server": a.server, "base_url": a.base_url,
                       "model_id": a.model, "prompt_version": a.prompt_version, "image_scale": 2, "max_tokens": a.max_tokens,
                       "temperature": a.temperature, "enable_thinking": a.thinking, "reasoning_effort": a.reasoning_effort,
                       "invoke_on": "frame_change", "at_commit_policy": policy, "template_radius_px": 12,
                       "min_match_score": 0.8, "restart_threshold_px": a.restart_threshold if policy == "restart_if_moved" else None},
        "observation": {"boundary_version": 1, "publication_delay_ns": 30_000_000},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": 20_000_000, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision",
    })


def run_one(spec: TrialSpec, out: Path, name: str, budget: CallBudget):
    store = PayloadStore(out / "payloads")
    p = run_trial(spec, out / f"{name}.jsonl", store, controller=build_controller(spec, budget=budget))
    return evaluate_trace(Trace.load(p))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--thinking", action="store_true")
    g.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=None, help="first evaluation seed (default: the predeclared list)")
    ap.add_argument("--n-seeds", type=int, default=None, help="number of evaluation seeds (rule: >= 100 for a cited cell)")
    ap.add_argument("--calib-seed-start", type=int, default=None)
    ap.add_argument("--n-calib", type=int, default=None)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--server", default="openai_compatible_local", choices=["openai_compatible_local", "agents3_service"],
                    help="agents3_service: Agent S3 behind studies/agents3_service.py (pilot 2026-08-27)")
    ap.add_argument("--subset-thinking", action="store_true", help="velocities {4,8} x {frozen, late_bind} only; no calibration")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--velocities", default=None, help="comma list overriding the predeclared {0,4,8,16}")
    ap.add_argument("--policies", default=None, help="comma list overriding {frozen,restart_if_moved,late_bind}")
    ap.add_argument("--no-controls", action="store_true", help="skip the irrelevant-motion control cells")
    ap.add_argument("--window-s", type=int, default=16, help="response window (s); 32 for thinking-on (B->D ~24 s)")
    ap.add_argument("--restart-threshold", type=int, default=6, help="restart_if_moved threshold (px)")
    ap.add_argument("--prompt-version", type=int, default=5, help="5 = balloon, 0-1000 grid (Qwen-VL family); 3 = balloon, pixels; 7 = UI-TARS native grammar")
    a = ap.parse_args()
    global WINDOW_NS, EVAL_SEEDS, CALIBRATION_SEEDS
    WINDOW_NS = a.window_s * 1_000_000_000
    if a.seed_start is not None or a.n_seeds is not None:
        EVAL_SEEDS = list(range(a.seed_start or 300, (a.seed_start or 300) + (a.n_seeds or 10)))
    if a.calib_seed_start is not None or a.n_calib is not None:
        CALIBRATION_SEEDS = list(range(a.calib_seed_start or 100, (a.calib_seed_start or 100) + (a.n_calib or 10)))
    a.out.mkdir(parents=True, exist_ok=True)
    budget = CallBudget(a.max_calls)
    mode = f"{a.model} thinking={'on' if a.thinking else 'off'} effort={a.reasoning_effort} T={a.temperature} max_tokens={a.max_tokens}"
    say(f"# WI-004 — late binding with a real agent ({mode})")
    say()
    rows = []   # (velocity, policy, irrelevant, result)

    if not (a.subset_thinking or a.skip_calibration):
        say("## Competence (Paused, v = 0, frozen, seeds 100..109)")
        ok = 0
        for seed in CALIBRATION_SEEDS:
            r = run_one(make_spec(a, seed, 0, "frozen", False, "paused"), a.out, f"calib_s{seed}", budget)
            ok += int(bool(r.valid_trial and r.valid_at_receipt))
            say(f"- seed {seed}: {'valid' if r.valid_at_receipt else 'invalid'} (B→D {fmt((r.b_to_d_ns or 0)/1e9)} s, err {fmt(r.spatial_error_px)} px)")
        say(f"\nPaused competence: {ok}/{len(CALIBRATION_SEEDS)} = {ok/len(CALIBRATION_SEEDS):.2f} (threshold {COMPETENCE_THRESHOLD})")
        if ok / len(CALIBRATION_SEEDS) < COMPETENCE_THRESHOLD:
            say("\nSTOP: competence below threshold; timing conditions not run (card stop condition).")
            (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
            return 1
        say()

    vels = [4, 8] if a.subset_thinking else VELOCITIES
    pols = ["frozen", "late_bind"] if a.subset_thinking else POLICIES
    if a.velocities:
        vels = [int(x) for x in a.velocities.split(",")]
    if a.policies:
        pols = a.policies.split(",")
    cells = [(v, p, False) for v in vels for p in pols]
    if not (a.subset_thinking or a.no_controls):
        cells += [(0, p, True) for p in ("frozen", "late_bind")]
    say("## Live trials (one line per trial)")
    for v, pol, irr in cells:
        for seed in EVAL_SEEDS:
            name = f"live_v{v}_{pol}{'_irr' if irr else ''}_s{seed}"
            r = run_one(make_spec(a, seed, v, pol, irr, "live"), a.out, name, budget)
            rows.append((v, pol, irr, r))
            say(f"- {name}: valid={r.valid_at_receipt} err={fmt(r.spatial_error_px)} "
                f"counterfactual_frozen={r.proposed_valid_at_receipt} restarts={r.n_restarts} "
                f"B→D={fmt((r.b_to_d_ns or 0)/1e9)}s firstB→D={fmt((r.first_b_to_d_ns or 0)/1e9)}s "
                f"labels={','.join(r.labels) or '-'} viol={','.join(r.contract_violations) or '-'}"
                + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
            (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")

    say()
    say("## Summary (Live; n valid trials per cell)")
    say()
    say("| v (px/s) | irrelevant motion | policy | n | valid at A | mean err (px) | frozen counterfactual valid (same A) | restarts / trial | B→D (s) | first B→D (s) | no-match |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for v, pol, irr in cells:
        rs = [r for (vv, pp, ii, r) in rows if (vv, pp, ii) == (v, pol, irr) and r.valid_trial]
        n = len(rs)
        if n == 0:
            say(f"| {v} | {irr} | {pol} | 0 | - | - | - | - | - | - | - |")
            continue
        valid = sum(bool(r.valid_at_receipt) for r in rs) / n
        cf = [r.proposed_valid_at_receipt for r in rs if r.proposed_valid_at_receipt is not None]
        cf_s = f"{sum(cf)/len(cf):.2f}" if cf else "-"
        nomatch = "-"
        if pol == "late_bind":
            nomatch = "see below"
        say(f"| {v} | {irr} | {pol} | {n} | {valid:.2f} | {fmt(mean([r.spatial_error_px for r in rs]))} | {cf_s} | "
            f"{mean([r.n_restarts for r in rs]):.1f} | {fmt(mean([(r.b_to_d_ns or 0)/1e9 for r in rs]))} | "
            f"{fmt(mean([(r.first_b_to_d_ns or 0)/1e9 for r in rs]))} | {nomatch} |")
    # no-match and tracker cost from the traces
    say()
    nm_tot = nm_n = 0
    track_ms = []
    for p in sorted(a.out.glob("live_*late_bind*.jsonl")):
        t = Trace.load(p)
        for c in t.all("coordinate_correction"):
            nm_n += 1
            nm_tot += int(not c["matched"])
            track_ms.append(c["track_ns"] / 1e6)
    if nm_n:
        say(f"Tracker: {nm_n} bindings, {nm_tot} no-match ({nm_tot/nm_n:.2f}); median track time {statistics.median(track_ms):.1f} ms, max {max(track_ms):.1f} ms.")
    say(f"\nModel calls used: {budget.calls} of {a.max_calls}.")
    (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
