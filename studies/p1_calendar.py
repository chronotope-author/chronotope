"""WI-010 study: does the memoryless duplicate replicate on the realistic 'calendar' skin?
Predeclared parameters live HERE (see cards/P1-reschedule.md).

  PYTHONPATH=src .venv/bin/python studies/p1_calendar.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/wi010_qwen38_nothink

Predeclared:
  gate:  revealed_target, calendar skin 640x480, target 48 px, Paused, seeds 100..109; pass >= 0.80.
  live:  pending_effect, calendar skin, effect_delay 8 s, memory none, Live, seeds 400..409 (pilot);
         full run 2026-08-24: --memories none,harness_pending --gate-seeds 100 --seed-start 1000 --n-seeds 100.
  prompt 4, image_scale 1 (model input 640x480 = WI-006's input size), temperature 0, thinking off;
  event 500 ms; window 16 s; 60 Hz; 30 ms publication; 20 ms delivery; horizon 17 s.
  The Live cell runs regardless of the gate; a gate < 0.80 voids the timing claim (card).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.controller import CallBudget  # noqa: E402
from chronotope.evaluate import evaluate_pending_trace, evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

GATE_SEEDS = list(range(100, 110))   # overridden by --gate-seeds
LIVE_SEEDS = list(range(400, 410))   # overridden by --seed-start/--n-seeds
W = 16_000_000_000
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def scenario(seed: int, pending: bool) -> dict:
    d = {"scenario_id": "pending_effect" if pending else "revealed_target", "scenario_seed": seed,
         "canvas_w_px": 640, "canvas_h_px": 480, "target_size_px": 48, "event_time_ns": 500_000_000,
         "response_window_ns": W, "frame_period_ns": 16_666_667, "episode_horizon_ns": 17_000_000_000,
         "irrelevant_motion": False, "skin": "calendar"}
    if pending:
        d["effect_delay_ns"] = 8_000_000_000
    return d


def make_spec(a, seed: int, pending: bool, memory: str = "none") -> TrialSpec:
    ctrl = {"kind": "vlm_pending" if pending else "vlm_direct", "server": "openai_compatible_local",
            "base_url": a.base_url, "model_id": a.model, "prompt_version": 4, "image_scale": 1,
            "max_tokens": a.max_tokens, "temperature": 0.0, "enable_thinking": a.thinking,
            "reasoning_effort": None, "invoke_on": "frame_change"}
    if pending:
        ctrl.update({"memory": memory, "pending_timeout_ns": 10_000_000_000, "pending_radius_px": 10})
    return TrialSpec.from_dict({
        "schema_version": 1, "scenario": scenario(seed, pending),
        "task_clock_policy": "live" if pending else "paused", "replay_source_live_trace_sha256": None,
        "controller": ctrl,
        "observation": {"boundary_version": 1, "publication_delay_ns": 30_000_000},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": 20_000_000, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--thinking", action="store_true")
    g.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-calls", type=int, default=400)
    ap.add_argument("--memories", default="none", help="comma list of pending memories: none,harness_pending,model_informed")
    ap.add_argument("--gate-seeds", type=int, default=10, help="number of Paused gate seeds starting at 100 (0 = skip)")
    ap.add_argument("--seed-start", type=int, default=400)
    ap.add_argument("--n-seeds", type=int, default=10)
    a = ap.parse_args()
    a.thinking = bool(a.thinking)
    global GATE_SEEDS, LIVE_SEEDS
    GATE_SEEDS = list(range(100, 100 + a.gate_seeds))
    LIVE_SEEDS = list(range(a.seed_start, a.seed_start + a.n_seeds))
    memories = [m for m in a.memories.split(",") if m]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = PayloadStore(out / "payloads")
    budget = CallBudget(a.max_calls)

    ok = 0
    if GATE_SEEDS:
        say(f"# WI-010 gate: revealed_target paused, calendar skin, seeds {GATE_SEEDS[0]}..{GATE_SEEDS[-1]}")
    for seed in GATE_SEEDS:
        spec = make_spec(a, seed, pending=False)
        p = run_trial(spec, out / f"gate_paused_s{seed}.jsonl", store,
                      controller=build_controller(spec, budget=budget))
        r = evaluate_trace(Trace.load(p))
        ok += int(bool(r.valid_at_receipt))
        say(f"gate seed {seed}: valid_at_receipt={r.valid_at_receipt}")
    if GATE_SEEDS:
        n = len(GATE_SEEDS)
        say(f"gate: {ok}/{n}  ({'PASS' if ok >= 0.8 * n else 'BELOW GATE — timing claim void (card)'})")

    for memory in memories:
        say(f"# WI-010 live: pending_effect d8 memory={memory}, seeds {LIVE_SEEDS[0]}..{LIVE_SEEDS[-1]}")
        dups = accepted = 0
        ea = []
        for seed in LIVE_SEEDS:
            spec = make_spec(a, seed, pending=True, memory=memory)
            p = run_trial(spec, out / f"live_d8_{memory}_s{seed}.jsonl", store,
                          controller=build_controller(spec, budget=budget))
            r = evaluate_pending_trace(Trace.load(p))
            accepted += int(bool(r.accepted))
            dups += r.duplicates
            if r.first_valid_E_A_ns is not None:
                ea.append(r.first_valid_E_A_ns / 1e9)
            say(f"live seed {seed}: accepted={r.accepted} duplicates={r.duplicates} "
                f"first_E→A={'%.2f' % (r.first_valid_E_A_ns / 1e9) if r.first_valid_E_A_ns else '-'} "
                f"invalid={r.invalid_clicks} decisions={r.n_decisions} violations={r.contract_violations}")
        n = len(LIVE_SEEDS)
        say(f"live {memory}: accepted {accepted}/{n}, duplicates/trial {dups / n:.2f}, "
            f"first E→A mean {sum(ea) / len(ea):.2f} s (WI-006: 5.30 s)" if ea else f"live {memory}: no accepted clicks")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
