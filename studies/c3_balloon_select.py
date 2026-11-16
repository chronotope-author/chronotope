"""C3 balloon select: three balloons drift, the instruction names one by colour. The model picks the balloon
(semantic choice); the binder may only track the picked patch (coordinate currency). Predeclared here.

  PYTHONPATH=src .venv/bin/python studies/c3_balloon_select.py --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 \
      --no-thinking --seed-start 1000 --n-seeds 100 --calib-seed-start 2000 --n-calib 100 --out runs/c3_qwen38_nothink_100 --max-calls 2500

Predeclared (cards/C3-balloon-select.md):
  calibration: Paused, v = 0, frozen, named colour cycles red/blue/yellow by seed -> competence >= 0.80 else STOP
  evaluation:  Live, seeds x {frozen, late_bind, restart_if_moved}, target v = 8 px/s, distractors at seeded
               velocities of similar magnitude (one opposite), named colour cycles by seed; balloon skin 320x240,
               target 24 px, prompt 8 (0-1000 grid) or 7 (UI-TARS native), image scale 2, window 16 s, 60 Hz,
               30 ms publication, 20 ms delivery; binder: RGB template radius 12, min NCC 0.8; restart threshold 6 px.
  outcome:     valid at A (click inside the NAMED balloon); which balloon was clicked (named / distractor / none);
               frozen counterfactual at the same A; a wrong pick must stay wrong after binding.
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
from chronotope.scenario import BALLOON_ORDER, manifest_from_dict  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

POLICIES = ["frozen", "late_bind", "restart_if_moved"]
COMPETENCE_THRESHOLD = 0.80
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); lines.append(s)


def make_spec(a, seed: int, vx: int, policy: str, clock: str) -> TrialSpec:
    color = BALLOON_ORDER[seed % 3]
    return TrialSpec.from_dict({
        "schema_version": 1,
        "scenario": {"scenario_id": "balloon_select", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 500_000_000, "response_window_ns": 16_000_000_000,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 17_000_000_000, "irrelevant_motion": False,
                     "skin": "balloon", "named_color": color, "velocity_x_px_per_s": vx},
        "task_clock_policy": clock, "replay_source_live_trace_sha256": None,
        "controller": {"kind": "vlm_late_binding", "server": "openai_compatible_local", "base_url": a.base_url,
                       "model_id": a.model, "prompt_version": a.prompt_version, "image_scale": 2, "max_tokens": a.max_tokens,
                       "temperature": a.temperature, "enable_thinking": a.thinking, "reasoning_effort": a.reasoning_effort,
                       "invoke_on": "frame_change", "at_commit_policy": policy, "template_radius_px": 12,
                       "min_match_score": 0.8, "restart_threshold_px": 6 if policy == "restart_if_moved" else None},
        "observation": {"boundary_version": 1, "publication_delay_ns": 30_000_000},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": 20_000_000, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision",
    })


def which_balloon(t: Trace, m, action_key: str = "action"):
    c = t.first("commit"); r = t.first("receipt")
    if not (c and r):
        return "none"
    return m.clicked_balloon(r["task_time_at_receipt_ns"], c["action"]) or "none"


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--base-url", required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--thinking", action="store_true"); g.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-version", type=int, default=8, help="8 = 0-1000 grid (Qwen); 7 = UI-TARS native")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--seed-start", type=int, default=1000); ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--calib-seed-start", type=int, default=2000); ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--velocity", type=int, default=8); ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--skip-calibration", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); budget = CallBudget(a.max_calls)
    say(f"# C3 balloon select ({a.model}, thinking={'on' if a.thinking else 'off'}, prompt {a.prompt_version})"); say()
    if not a.skip_calibration:
        ok = 0; seeds = list(range(a.calib_seed_start, a.calib_seed_start + a.n_calib)); picks = {"named": 0, "distractor": 0, "none": 0}
        for seed in seeds:
            spec = make_spec(a, seed, 0, "frozen", "paused"); m = spec.manifest()
            p = run_trial(spec, a.out / f"calib_s{seed}.jsonl", PayloadStore(a.out / "payloads"), controller=build_controller(spec, budget=budget))
            t = Trace.load(p); r = evaluate_trace(t); wb = which_balloon(t, m)
            ok += int(bool(r.valid_at_receipt)); picks["named" if wb == m.named_color else ("distractor" if wb != "none" else "none")] += 1
            say(f"- calib seed {seed} ({m.named_color}): {'valid' if r.valid_at_receipt else 'invalid'}, clicked {wb}, err {fmt(r.spatial_error_px)} px")
        say(f"\nPaused competence: {ok}/{len(seeds)} = {ok/len(seeds):.2f} (threshold {COMPETENCE_THRESHOLD}); picks {picks}")
        (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
        if ok / len(seeds) < COMPETENCE_THRESHOLD:
            say("\nSTOP: competence below threshold; timing conditions not run."); (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n"); return 1
    rows = []
    pols = a.policies.split(",")
    for pol in pols:
        for seed in range(a.seed_start, a.seed_start + a.n_seeds):
            spec = make_spec(a, seed, a.velocity, pol, "live"); m = spec.manifest()
            name = f"live_v{a.velocity}_{pol}_s{seed}"
            p = run_trial(spec, a.out / f"{name}.jsonl", PayloadStore(a.out / "payloads"), controller=build_controller(spec, budget=budget))
            t = Trace.load(p); r = evaluate_trace(t); wb = which_balloon(t, m)
            corr = t.first("coordinate_correction")
            prop_wb = m.clicked_balloon(t.first("receipt")["task_time_at_receipt_ns"], corr["proposed_action"]) if (corr and t.first("receipt")) else None
            # which balloon did the PROPOSAL point at on the frame at B (the model's semantic pick)?
            d0 = t.committing_decision(); ob = next((o for o in t.all("observation") if d0 and o["observation_id"] == d0["observation_id"]), None)
            pick_at_B = m.clicked_balloon(ob["source_time_ns"], t.first("action_proposed")["action"]) if (ob and t.first("action_proposed")) else None
            rows.append((pol, m.named_color, r, wb, prop_wb, pick_at_B))
            say(f"- {name} ({m.named_color}): valid={r.valid_at_receipt} picked_at_B={pick_at_B} clicked_at_A={wb} "
                f"counterfactual_frozen={r.proposed_valid_at_receipt} restarts={r.n_restarts} B→D={fmt((r.b_to_d_ns or 0)/1e9)}s "
                f"labels={','.join(r.labels) or '-'} viol={','.join(r.contract_violations) or '-'}")
            (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    say(); say("## Summary"); say()
    say("| policy | n | valid at A | picked the named balloon at B | clicked named at A | clicked a distractor at A | no balloon | frozen counterfactual valid | restarts/trial | B→D (s) |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for pol in pols:
        rs = [x for x in rows if x[0] == pol and x[2].valid_trial]; n = len(rs)
        if not n:
            continue
        cf = [x[2].proposed_valid_at_receipt for x in rs if x[2].proposed_valid_at_receipt is not None]
        say(f"| {pol} | {n} | {sum(bool(x[2].valid_at_receipt) for x in rs)/n:.2f} | {sum(x[5] == x[1] for x in rs)/n:.2f} | "
            f"{sum(x[3] == x[1] for x in rs)/n:.2f} | {sum(x[3] not in (x[1], 'none') for x in rs)/n:.2f} | {sum(x[3] == 'none' for x in rs)/n:.2f} | "
            f"{(f'{sum(cf)/len(cf):.2f}' if cf else '-')} | {statistics.mean([x[2].n_restarts for x in rs]):.1f} | {fmt(statistics.mean([(x[2].b_to_d_ns or 0)/1e9 for x in rs]), 2)} |")
    # the key separation: among late_bind trials, a wrong pick at B stays wrong; a right pick is recovered
    lb = [x for x in rows if x[0] == "late_bind" and x[2].valid_trial]
    if lb:
        right = [x for x in lb if x[5] == x[1]]; wrong = [x for x in lb if x[5] not in (x[1], None)]
        say(f"\nlate_bind: picks right at B = {len(right)} → valid at A {sum(bool(x[2].valid_at_receipt) for x in right)}; "
            f"picks wrong at B = {len(wrong)} → valid at A {sum(bool(x[2].valid_at_receipt) for x in wrong)} (must be 0); "
            f"frozen counterfactual among right picks: {sum(bool(x[2].proposed_valid_at_receipt) for x in right)}")
    say(f"\nModel calls used: {budget.calls} of {a.max_calls}.")
    (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
