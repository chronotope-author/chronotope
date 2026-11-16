"""WI-002 study: competence qualification under Paused, then Paused–Replay–Live across response
windows, for one VLM direct controller on revealed_target. Predeclared parameters live HERE.

Usage (a LOCAL OpenAI-compatible server, e.g. vLLM serving Qwen/Qwen3.8-27B on 127.0.0.1:8000; no keys):
  PYTHONPATH=src .venv/bin/python studies/r1_revealed_target.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/wi002_qwen38_nothink --max-calls 200
  PYTHONPATH=src .venv/bin/python studies/r1_revealed_target.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --thinking --reasoning-effort low --out runs/wi002_qwen38_think --max-calls 200

Predeclared (cards/R1-revealed-target.md):
  calibration seeds 100..109 (Paused only)  -> competence threshold >= 0.80, else STOP
  evaluation  seeds 200..209 x windows {1, 2, 4, 8, 16} s x {paused, replay, live}
  event_time 500 ms, 60 Hz frames, 30 ms publication delay, 20 ms delivery delay, image scale 10
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.__main__ import run_triplet  # noqa: E402
from chronotope.controller import CallBudget  # noqa: E402
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

CALIBRATION_SEEDS = list(range(100, 110))
EVAL_SEEDS = list(range(200, 210))
WINDOWS_NS = [1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000, 16_000_000_000]  # amended pre-data, see card
COMPETENCE_THRESHOLD = 0.80
SERVER = "openai_compatible_local"


SKIN_GEOMETRY = {  # skin -> (canvas_w, canvas_h, target_size, image_scale, prompt_version)
    "plain": (64, 48, 6, 10, 1), "notification": (320, 240, 24, 2, 2), "balloon": (320, 240, 24, 2, 3)}


def make_spec(model: str, base_url: str, thinking: bool, effort: str | None, temperature: float,
              max_tokens: int, seed: int, window_ns: int, skin: str = "plain") -> TrialSpec:
    cw, ch, ts, scale, pv = SKIN_GEOMETRY[skin]
    return TrialSpec.from_dict({
        "schema_version": 1,
        "scenario": {"scenario_id": "revealed_target", "scenario_seed": seed, "canvas_w_px": cw, "canvas_h_px": ch,
                     "target_size_px": ts, "event_time_ns": 500_000_000, "response_window_ns": window_ns,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 500_000_000 + window_ns + 500_000_000,
                     "irrelevant_motion": False, "skin": skin},
        "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": {"kind": "vlm_direct", "server": SERVER, "base_url": base_url,
                       "model_id": model, "prompt_version": pv, "image_scale": scale, "max_tokens": max_tokens,
                       "temperature": temperature, "enable_thinking": thinking, "reasoning_effort": effort,
                       "invoke_on": "frame_change"},
        "observation": {"boundary_version": 1, "publication_delay_ns": 30_000_000},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": 20_000_000, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision",
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--thinking", action="store_true")
    g.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64, help="64 without thinking; thinking needs more (e.g. 4096)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--skin", default="plain", choices=list(SKIN_GEOMETRY))
    ap.add_argument("--prompt-version", type=int, default=None, help="override the skin's default prompt")
    ap.add_argument("--seed-start", type=int, default=None); ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--calib-seed-start", type=int, default=None); ap.add_argument("--n-calib", type=int, default=None)
    ap.add_argument("--windows-s", default=None, help="comma-separated windows in seconds (R2 fine sweep); default 1,2,4,8,16")
    ap.add_argument("--policies", default="paused,replay,live", help="R2: 'live' only (Paused/Replay are window-independent)")
    ap.add_argument("--server", default="openai_compatible_local", choices=["openai_compatible_local", "agents3_service"],
                    help="M4 phase 3: agents3_service posts to studies/agents3_service.py (--model = 'planner+grounder')")
    ap.add_argument("--windows-ratio", default=None,
                    help="M4: comma-separated multiples of the agent's own median calibration B->D, e.g. 0.5,0.75,1,1.25,1.5; "
                         "the windows are fixed from the Paused calibration before any evaluation trial runs")
    a = ap.parse_args()
    global EVAL_SEEDS, CALIBRATION_SEEDS
    if a.seed_start is not None or a.n_seeds is not None:
        EVAL_SEEDS = list(range(a.seed_start or 200, (a.seed_start or 200) + (a.n_seeds or 10)))
    global WINDOWS_NS, SERVER
    SERVER = a.server
    if a.windows_s:
        WINDOWS_NS = [int(round(float(x) * 1e9)) for x in a.windows_s.split(",")]
    if a.calib_seed_start is not None or a.n_calib is not None:
        CALIBRATION_SEEDS = list(range(a.calib_seed_start or 100, (a.calib_seed_start or 100) + (a.n_calib or 10)))
    if a.prompt_version is not None:
        cw, ch, ts, sc, _ = SKIN_GEOMETRY[a.skin]
        SKIN_GEOMETRY[a.skin] = (cw, ch, ts, sc, a.prompt_version)
    thinking = bool(a.thinking)
    if thinking and a.max_tokens < 1024:
        print("warning: thinking enabled with max_tokens < 1024; answers may be cut off", file=sys.stderr)
    budget = CallBudget(a.max_calls)
    mk = lambda spec: build_controller(spec, budget=budget)  # noqa: E731  (adapter built from the spec)
    a.out.mkdir(parents=True, exist_ok=True)
    store = PayloadStore(a.out / "payloads")
    report: dict = {"model": a.model, "skin": a.skin, "prompt_version": SKIN_GEOMETRY[a.skin][4], "effort": a.reasoning_effort, "temperature": a.temperature,
                    "base_url": a.base_url, "thinking": thinking, "calibration": [], "evaluation": []}

    # ---- Phase 1: competence under Paused (calibration seeds) --------------------------------
    for seed in CALIBRATION_SEEDS:
        spec = make_spec(a.model, a.base_url, thinking, a.reasoning_effort, a.temperature, a.max_tokens, seed,
                         WINDOWS_NS[-1], a.skin).derive("paused", None)
        p = run_trial(spec, a.out / f"cal_s{seed}_paused.jsonl", store, controller=mk(spec))
        r = evaluate_trace(Trace.load(p))
        report["calibration"].append({"seed": seed, **r.to_dict()})
        print(f"[cal] seed={seed} valid={r.valid_trial} success={r.valid_at_receipt} B->D={r.b_to_d_ns} "
              f"labels={r.labels} calls={budget.calls}")
    valid = [c for c in report["calibration"] if c["valid_trial"]]
    comp = sum(bool(c["valid_at_receipt"]) for c in valid) / max(1, len(valid))
    report["competence"] = {"rate": comp, "n_valid": len(valid), "threshold": COMPETENCE_THRESHOLD,
                            "passed": comp >= COMPETENCE_THRESHOLD and len(valid) >= 8}
    print(f"competence {comp:.2f} on {len(valid)} valid trials -> {'PASS' if report['competence']['passed'] else 'STOP'}")
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    if not report["competence"]["passed"]:
        return 1
    if a.windows_ratio:  # M4 (cards/M4-cross-agent-prl.md U61): windows scaled to this agent's deliberation
        bd = sorted(c["b_to_d_ns"] for c in valid if c.get("b_to_d_ns"))
        med = int(statistics.median(bd))
        WINDOWS_NS = [int(round(float(x) * med)) for x in a.windows_ratio.split(",")]
        report["windows_ratio"] = {"ratios": [float(x) for x in a.windows_ratio.split(",")], "median_calibration_b_to_d_ns": med,
                                   "windows_ns": WINDOWS_NS}
        print(f"windows from ratio x median calibration B->D {med/1e9:.2f}s: {[w/1e9 for w in WINDOWS_NS]}")
        (a.out / "report.json").write_text(json.dumps(report, indent=2))

    # ---- R2 (cards/R2-window-sweep.md): Live-only sweep; files eval_w<ns>_s<seed>_live.jsonl for analyze_r2.py ----
    if a.policies == "live":
        for w in WINDOWS_NS:
            for seed in EVAL_SEEDS:
                spec = make_spec(a.model, a.base_url, thinking, a.reasoning_effort, a.temperature, a.max_tokens, seed, w, a.skin)
                try:
                    p = run_trial(spec, a.out / f"eval_w{w}_s{seed}_live.jsonl", PayloadStore(a.out / "payloads"), controller=mk(spec))
                except RuntimeError as e:
                    print("stopping:", e); return 3
                r = evaluate_trace(Trace.load(p))
                report["evaluation"].append({"window_ns": w, "seed": seed, "live": r.to_dict()})
                print(f"[eval-live] w={w/1e9:.2f}s seed={seed} L={r.valid_at_receipt} E→A={(r.intervals_ns.get('E_A') or 0)/1e9:.2f}s "
                      f"B→D={(r.b_to_d_ns or 0)/1e9:.2f}s labels={','.join(r.labels) or '-'}", flush=True)
                (a.out / "report.json").write_text(json.dumps(report, indent=2))
        return 0

    # ---- Phase 2: Paused–Replay–Live across windows (evaluation seeds) -----------------------
    for w in WINDOWS_NS:
        for seed in EVAL_SEEDS:
            spec = make_spec(a.model, a.base_url, thinking, a.reasoning_effort, a.temperature, a.max_tokens, seed, w, a.skin)
            tag = f"eval_w{w}_s{seed}_"
            try:
                out = run_triplet(spec, a.out, tag, controller_factory=mk)
            except RuntimeError as e:  # budget exhausted
                print("stopping:", e)
                (a.out / "report.json").write_text(json.dumps(report, indent=2))
                return 3
            row = {"window_ns": w, "seed": seed, "pair": out["pair"].__dict__,
                   **{p: out["results"][p].to_dict() for p in ("paused", "replay", "live")}}
            report["evaluation"].append(row)
            live = out["results"]["live"]
            print(f"[eval] w={w/1e9:.1f}s seed={seed} P={out['results']['paused'].valid_at_receipt} "
                  f"R={out['results']['replay'].valid_at_receipt} L={live.valid_at_receipt} "
                  f"B->D={live.b_to_d_ns} labels={live.labels} pair={out['pair'].valid} calls={budget.calls}")
            (a.out / "report.json").write_text(json.dumps(report, indent=2))

    # ---- summary table -------------------------------------------------------------------------
    lines = [f"# WI-002 {a.model} skin={a.skin} prompt_v{SKIN_GEOMETRY[a.skin][4]} (thinking={thinking}, effort={a.reasoning_effort}, temperature={a.temperature})",
             f"competence (Paused, calibration): {comp:.2f} n={len(valid)}", "",
             "| window | Paused | Replay | Live | median live B->D | live labels |", "|---|---|---|---|---|---|"]
    for w in WINDOWS_NS:
        rows = [r for r in report["evaluation"] if r["window_ns"] == w]
        def rate(p):
            v = [r[p] for r in rows if r[p]["valid_trial"]]
            return f"{sum(bool(x['valid_at_receipt']) for x in v)}/{len(v)}"
        bd = [r["live"]["b_to_d_ns"] for r in rows if r["live"]["b_to_d_ns"] is not None]
        labs: dict[str, int] = {}
        for r in rows:
            for l in r["live"]["labels"]:
                labs[l] = labs.get(l, 0) + 1
        lines.append(f"| {w/1e9:.1f} s | {rate('paused')} | {rate('replay')} | {rate('live')} | "
                     f"{statistics.median(bd)/1e6:.0f} ms | {labs or '-'} |" if bd else
                     f"| {w/1e9:.1f} s | {rate('paused')} | {rate('replay')} | {rate('live')} | - | {labs or '-'} |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
