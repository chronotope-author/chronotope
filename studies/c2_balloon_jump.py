"""WI-009 study: pre-/post-commit change (jump_target).
  --oracle : simulated clock; bindings x jump time relative to B, D, A   (CPU only; seconds)
  --agent  : Live, real model (vlm_late_binding) frozen / late_bind / restart_if_moved on the balloon

  PYTHONPATH=src .venv/bin/python studies/c2_balloon_jump.py --oracle --out results/WI-009_jump_target_oracle.md
  PYTHONPATH=src .venv/bin/python studies/c2_balloon_jump.py --agent --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/wi009_qwen38_nothink --max-calls 300

Predeclared (cards/C2-balloon-jump.md):
  oracle: plain 96x64, target 6, event 500 ms, window 1.5 s, 60 Hz, 30 ms pub, compute 500 ms, delivery 60 ms,
          jump (40, 0) px, jump_delay ∈ {100, 300, 560, 900, 1400} ms (during B->D / in (D, A] / after A), seeds 0..9
  agent:  balloon 320x240, target 24, prompt 5, scale 2, window 16 s, jump (96, 0) at 2 s after appearance
          (during B->D ≈ 5.3 s); policies frozen / late_bind / restart_if_moved(threshold 6); seeds 500..509;
          control: jump_delay 14 s (after A) x {frozen, late_bind}.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.controller import CallBudget  # noqa: E402
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

PUB, PERIOD = 30_000_000, 16_666_667
ORACLE_SEEDS = list(range(0, 10))
ORACLE_JUMPS_MS = [100, 300, 560, 900, 1400]
BINDINGS = ["state_at_B", "newest_at_D", "predicted_at_A", "true_at_A"]
AGENT_SEEDS = list(range(500, 510))
POLICIES = ["frozen", "late_bind", "restart_if_moved"]
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def oracle_spec(seed: int, jump_ms: int, binding: str) -> TrialSpec:
    delivery = 60_000_000
    return TrialSpec.from_dict({
        "schema_version": 1,
        "scenario": {"scenario_id": "jump_target", "scenario_seed": seed, "canvas_w_px": 96, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": 500_000_000, "response_window_ns": 1_500_000_000,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": 2_500_000_000, "irrelevant_motion": False,
                     "skin": "plain", "jump_delay_ns": jump_ms * 1_000_000, "jump_dx_px": 40, "jump_dy_px": 0},
        "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": {"kind": "scripted_oracle", "compute_ns": 500_000_000, "coordinate_offset_px": 0, "binding": binding,
                       "expected_observation_delay_ns": PUB if binding == "predicted_at_A" else None,
                       "expected_delivery_delay_ns": delivery if binding == "predicted_at_A" else None},
        "observation": {"boundary_version": 1, "publication_delay_ns": PUB},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "simulated",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": delivery, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision",
    })


def agent_spec(a, seed: int, policy: str, jump_delay_ns: int) -> TrialSpec:
    window = 16_000_000_000
    return TrialSpec.from_dict({
        "schema_version": 1,
        "scenario": {"scenario_id": "jump_target", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 500_000_000, "response_window_ns": window,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": 500_000_000 + window + 500_000_000,
                     "irrelevant_motion": False, "skin": "balloon", "jump_delay_ns": jump_delay_ns,
                     "jump_dx_px": 96, "jump_dy_px": 0},
        "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": {"kind": "vlm_late_binding", "server": "openai_compatible_local", "base_url": a.base_url,
                       "model_id": a.model, "prompt_version": 5, "image_scale": 2, "max_tokens": a.max_tokens,
                       "temperature": a.temperature, "enable_thinking": a.thinking, "reasoning_effort": a.reasoning_effort,
                       "invoke_on": "frame_change", "at_commit_policy": policy, "template_radius_px": 12,
                       "min_match_score": 0.8, "restart_threshold_px": 6 if policy == "restart_if_moved" else None},
        "observation": {"boundary_version": 1, "publication_delay_ns": PUB},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": 20_000_000, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision",
    })


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def run_oracle(out: Path) -> None:
    import tempfile
    say("# WI-009 — jump_target, oracle bindings (simulated clock)")
    say()
    say("B ≈ 0.53 s (first frame depicting the target + 30 ms), D = B + 0.5 s ≈ 1.03 s, A = D + 60 ms ≈ 1.09 s; "
        f"jump at 0.5 s + jump_delay. {len(ORACLE_SEEDS)} seeds per cell; every cell's bindings share B, D, A exactly.")
    say()
    say("| jump delay (ms) | jump relative to | " + " | ".join(f"{b} valid" for b in BINDINGS) + " | labels (state_at_B) |")
    say("|---|---|" + "---|" * len(BINDINGS) + "---|")
    with tempfile.TemporaryDirectory() as td:
        store = PayloadStore(Path(td) / "p")
        for jm in ORACLE_JUMPS_MS:
            cells = {}
            labs = {}
            for b in BINDINGS:
                ok = 0
                for seed in ORACLE_SEEDS:
                    spec = oracle_spec(seed, jm, b)
                    p = run_trial(spec, Path(td) / f"{jm}_{b}_{seed}.jsonl", store)
                    r = evaluate_trace(Trace.load(p))
                    ok += int(bool(r.valid_at_receipt))
                    if b == "state_at_B":
                        for l in r.labels:
                            labs[l] = labs.get(l, 0) + 1
                cells[b] = ok / len(ORACLE_SEEDS)
            rel = ("before B (observed at B)" if jm + 500 < 530 else "during B→D (observed before D)" if jm + 500 < 1030
                   else "in (D, A]" if jm + 500 <= 1090 else "after A")
            say(f"| {jm} | {rel} | " + " | ".join(f"{cells[b]:.2f}" for b in BINDINGS) + " | " + (", ".join(f"{k}×{v}" for k, v in sorted(labs.items())) or "-") + " |")
    say()
    say("Reading: only the bindings that read the NEWER state at D (newest_at_D) or the true state at A succeed on a "
        "pre-commit jump; the velocity extrapolator (predicted_at_A) is fooled by a discrete change; nothing helps a jump "
        "inside (D, A] except the oracle; after A all bindings tie.")
    out.write_text("\n".join(lines) + "\n")


def run_agent(a) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    budget = CallBudget(a.max_calls)
    say(f"# WI-009 — jump_target with a real agent ({a.model}, thinking={'on' if a.thinking else 'off'})")
    say()
    cells = [(2_000_000_000, p) for p in POLICIES] + [(14_000_000_000, p) for p in ("frozen", "late_bind")]
    rows = []
    for jd, pol in cells:
        for seed in AGENT_SEEDS:
            name = f"live_jump{jd//1_000_000_000}s_{pol}_s{seed}"
            spec = agent_spec(a, seed, pol, jd)
            p = run_trial(spec, a.out / f"{name}.jsonl", PayloadStore(a.out / "payloads"),
                          controller=build_controller(spec, budget=budget))
            r = evaluate_trace(Trace.load(p))
            rows.append((jd, pol, r))
            say(f"- {name}: valid={r.valid_at_receipt} err={fmt(r.spatial_error_px)} counterfactual_frozen={r.proposed_valid_at_receipt} "
                f"restarts={r.n_restarts} B→D={fmt((r.b_to_d_ns or 0)/1e9)}s labels={','.join(r.labels) or '-'} "
                f"viol={','.join(r.contract_violations) or '-'}" + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
            (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    say()
    say("| jump delay (s) | policy | n | valid at A | mean err (px) | frozen counterfactual valid (same A) | restarts / trial | B→D (s) | outdated_at_commitment |")
    say("|---|---|---|---|---|---|---|---|---|")
    for jd, pol in cells:
        rs = [r for (j, p, r) in rows if (j, p) == (jd, pol) and r.valid_trial]
        n = len(rs)
        if not n:
            say(f"| {jd//1_000_000_000} | {pol} | 0 | - | - | - | - | - | - |"); continue
        cf = [r.proposed_valid_at_receipt for r in rs if r.proposed_valid_at_receipt is not None]
        say(f"| {jd//1_000_000_000} | {pol} | {n} | {sum(bool(r.valid_at_receipt) for r in rs)/n:.2f} | {fmt(mean([r.spatial_error_px for r in rs]))} | "
            f"{(f'{sum(cf)/len(cf):.2f}' if cf else '-')} | {mean([r.n_restarts for r in rs]):.1f} | {fmt(mean([(r.b_to_d_ns or 0)/1e9 for r in rs]))} | "
            f"{sum('outdated_at_commitment' in r.labels for r in rs)} |")
    say(f"\nModel calls used: {budget.calls} of {a.max_calls}.")
    (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--oracle", action="store_true")
    g.add_argument("--agent", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=None, help="first evaluation seed (default: the predeclared list)")
    ap.add_argument("--n-seeds", type=int, default=None, help="number of evaluation seeds (rule: >= 100 for a cited cell)")
    ap.add_argument("--calib-seed-start", type=int, default=None)
    ap.add_argument("--n-calib", type=int, default=None)
    ap.add_argument("--model"); ap.add_argument("--base-url")
    t = ap.add_mutually_exclusive_group()
    t.add_argument("--thinking", action="store_true"); t.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-calls", type=int, default=300)
    a = ap.parse_args()
    global AGENT_SEEDS, ORACLE_SEEDS
    if a.seed_start is not None or a.n_seeds is not None:
        AGENT_SEEDS = list(range(a.seed_start or 500, (a.seed_start or 500) + (a.n_seeds or 10)))
        ORACLE_SEEDS = list(range(a.seed_start or 0, (a.seed_start or 0) + (a.n_seeds or 10)))
    if a.oracle:
        run_oracle(a.out)
    else:
        if not (a.model and a.base_url):
            raise SystemExit("--agent needs --model and --base-url")
        run_agent(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
