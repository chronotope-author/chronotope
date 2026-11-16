"""WI-006 study: pending-action record vs memoryless vs model-informed on the pending_effect task with
a real agent (Live only). Predeclared parameters live HERE (see cards/P1-pending-button.md).

  PYTHONPATH=src .venv/bin/python studies/p1_pending_button.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/wi006_qwen38_nothink --max-calls 1500

Predeclared:
  cells: effect_delay {1, 2, 4, 8} s x memory {none, harness_pending, model_informed} x seeds 400..409
  dropped-click variant: route simulated_drop_first, effect_delay 2 s, memory {none, harness_pending}, seeds 400..409
  notification skin 320x240, target 24 px, prompt 4 (0-1000 grid), image scale 2; event 500 ms; window 16 s;
  60 Hz frames; 30 ms publication; 20 ms delivery; pending_timeout 10 s; pending_radius 10 px;
  episode ends 2 s after the effect is visible or at the 17-s horizon.
  Paused competence on this skin: WI-005 measured 1.00 (seeds 100..109) with the same prompt; reused.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.controller import CallBudget  # noqa: E402
from chronotope.evaluate import evaluate_pending_trace  # noqa: E402
from chronotope.runner import PayloadStore, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

EVAL_SEEDS = list(range(400, 410))
EFFECT_DELAYS_S = [1, 2, 4, 8]
MEMORIES = ["none", "harness_pending", "model_informed"]
WINDOW_NS = 16_000_000_000
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def make_spec(a, seed: int, effect_delay_s: int, memory: str, route: str) -> TrialSpec:
    return TrialSpec.from_dict({
        "schema_version": 1,
        "scenario": {"scenario_id": "pending_effect", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 500_000_000, "response_window_ns": WINDOW_NS,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 500_000_000 + WINDOW_NS + 500_000_000,
                     "irrelevant_motion": False, "skin": "notification", "effect_delay_ns": effect_delay_s * 1_000_000_000},
        "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": {"kind": "vlm_pending", "server": "openai_compatible_local", "base_url": a.base_url,
                       "model_id": a.model, "prompt_version": a.prompt_version, "image_scale": 2, "max_tokens": a.max_tokens,
                       "temperature": a.temperature, "enable_thinking": a.thinking, "reasoning_effort": a.reasoning_effort,
                       "invoke_on": "frame_change", "memory": memory, "pending_timeout_ns": 10_000_000_000,
                       "pending_radius_px": 10},
        "observation": {"boundary_version": 1, "publication_delay_ns": 30_000_000},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": route, "delivery_delay_ns": 20_000_000, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision",
    })


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, nd=2):
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
    ap.add_argument("--effect-delays", default=None, help="comma list (s) overriding {1,2,4,8}")
    ap.add_argument("--memories", default=None, help="comma list overriding {none,harness_pending,model_informed}")
    ap.add_argument("--no-drop", action="store_true", help="skip the dropped-click variant")
    ap.add_argument("--prompt-version", type=int, default=4, help="4 = notification, 0-1000 grid; 2 = notification, pixels; 7 = UI-TARS native grammar")
    a = ap.parse_args()
    global EVAL_SEEDS
    if a.seed_start is not None or a.n_seeds is not None:
        EVAL_SEEDS = list(range(a.seed_start or 400, (a.seed_start or 400) + (a.n_seeds or 10)))
    a.out.mkdir(parents=True, exist_ok=True)
    budget = CallBudget(a.max_calls)
    say(f"# WI-006 — pending-action record with a real agent ({a.model}, thinking={'on' if a.thinking else 'off'})")
    say()
    delays = [int(x) for x in a.effect_delays.split(",")] if a.effect_delays else EFFECT_DELAYS_S
    mems = a.memories.split(",") if a.memories else MEMORIES
    cells = [(d, mem, "simulated_fixed_delay") for d in delays for mem in mems]
    if not a.no_drop:
        cells += [(2, mem, "simulated_drop_first") for mem in ("none", "harness_pending")]
    rows = []
    say("## Live trials (one line per trial)")
    for d, mem, route in cells:
        for seed in EVAL_SEEDS:
            name = f"live_d{d}_{mem}{'_drop' if route != 'simulated_fixed_delay' else ''}_s{seed}"
            spec = make_spec(a, seed, d, mem, route)
            p = run_trial(spec, a.out / f"{name}.jsonl", PayloadStore(a.out / "payloads"),
                          controller=build_controller(spec, budget=budget))
            r = evaluate_pending_trace(Trace.load(p))
            rows.append((d, mem, route, r))
            say(f"- {name}: accepted={r.accepted} dup={r.duplicates} invalid_clicks={r.invalid_clicks} "
                f"suppressed={r.n_suppressed} waits={r.n_waits} decisions={r.n_decisions} "
                f"firstE→A={fmt((r.first_valid_E_A_ns or 0)/1e9)}s exact={r.exact_success} "
                f"dropped={r.dropped} viol={','.join(r.contract_violations) or '-'}"
                + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
            (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    say()
    say("## Summary (Live; n valid trials per cell)")
    say()
    say("| effect delay (s) | route | memory | n | accepted | mean duplicates | trials with ≥1 duplicate | exact success | mean first E→A (s) | mean invocations | mean suppressed | mean invalid clicks |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d, mem, route in cells:
        rs = [r for (dd, mm, rr, r) in rows if (dd, mm, rr) == (d, mem, route) and r.valid_trial]
        n = len(rs)
        if n == 0:
            say(f"| {d} | {route} | {mem} | 0 | - | - | - | - | - | - | - | - |")
            continue
        say(f"| {d} | {route} | {mem} | {n} | {sum(r.accepted for r in rs)/n:.2f} | {mean([r.duplicates for r in rs]):.2f} | "
            f"{sum(r.duplicates > 0 for r in rs)/n:.2f} | {sum(r.exact_success for r in rs)/n:.2f} | "
            f"{fmt(mean([(r.first_valid_E_A_ns or 0)/1e9 for r in rs if r.accepted]))} | {mean([r.n_decisions for r in rs]):.1f} | "
            f"{mean([r.n_suppressed for r in rs]):.1f} | {mean([r.invalid_clicks for r in rs]):.2f} |")
    say(f"\nModel calls used: {budget.calls} of {a.max_calls}.")
    (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
