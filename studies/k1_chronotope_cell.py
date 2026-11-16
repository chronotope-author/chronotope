"""K1c (U-K3): what residency buys — the binder's latency as a term of the error law.

Moving target, balloon skin, as C1 (320x240, target 24 px, event 500 ms, 60 Hz, 30 ms publication, 20 ms delivery)
but with a 1.3-s window and a 1.0-s emulated deliberation: at 100-200 px/s the 27B model's 5.2 s would carry the
target off the 320-px canvas (the manifest refuses that geometry); 1.0 s still makes the frozen click miss by
100-200 px, which is all the mechanism needs. Planner: the resident detector at B. Cells: velocity x binder {tracker_py, cuda} x policy late_bind, plus frozen; n seeds per cell.
Wall clock. Writes runs/<out>/live_v{v}_{binder}_{policy}_s{seed}.jsonl and RESULTS.md.

  PYTHONPATH=src:chronotope/python .venv/bin/python studies/k1_kgua_cell.py --out runs/k1_kgua_cell --n-seeds 100
  ... --pilot     (2 seeds, velocities 8 and 200, for a smoke run)

Never run concurrently with another GPU job (the bind latency and the trial timing are wall-clock).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "chronotope" / "python"))

from chronotope_core import glue  # noqa: E402
from chronotope_core.controller import build  # noqa: E402
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.stats import wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

WINDOW_NS = 1_300_000_000   # set from --window-s
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def make_spec(seed: int, vx: int, binder: str, policy: str, deliberation_ns: int) -> TrialSpec:
    glue.register()
    return TrialSpec.from_dict({
        "schema_version": 1,
        "scenario": {"scenario_id": "moving_target", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": 500_000_000, "response_window_ns": WINDOW_NS,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 500_000_000 + WINDOW_NS + 500_000_000,
                     "irrelevant_motion": False, "skin": "balloon", "velocity_x_px_per_s": vx, "velocity_y_px_per_s": 0},
        "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": glue.controller_dict(planner="detector", binder=binder, at_commit_policy=policy, compute_ns=deliberation_ns),
        "observation": {"boundary_version": 1, "publication_delay_ns": 30_000_000},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": 20_000_000, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision",
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--velocities", default="8,100,200")
    ap.add_argument("--deliberation-s", type=float, default=1.0, help="emulated deliberation; 1.0 s keeps a 200 px/s target on the 320-px canvas (5.2 s cannot)")
    ap.add_argument("--window-s", type=float, default=1.3)
    ap.add_argument("--pilot", action="store_true", help="2 seeds, velocities 8 and 200")
    ap.add_argument("--results", type=Path, default=None, help="also write results/<name>.md")
    a = ap.parse_args()
    vels = [int(v) for v in a.velocities.split(",")]
    n = a.n_seeds
    if a.pilot:
        vels, n = [8, 200], 2
    seeds = list(range(a.seed_start, a.seed_start + n))
    delib = int(a.deliberation_s * 1e9)
    global WINDOW_NS
    WINDOW_NS = int(a.window_s * 1e9)
    a.out.mkdir(parents=True, exist_ok=True)
    store = PayloadStore(a.out / "payloads")
    cells = [(v, b, "late_bind") for v in vels for b in ("tracker_py", "cuda")] + [(v, "cuda", "frozen") for v in vels]
    say(f"# K1c — the binder's latency in the error law (detector planner, deliberation {a.deliberation_s} s, {n} seeds per cell)")
    say()
    rows = defaultdict(list)   # (v, binder, policy) -> (result, correction record, observation source times)
    say("## Trials")
    for v, b, pol in cells:
        for seed in seeds:
            name = f"live_v{v}_{b}_{pol}_s{seed}"
            spec = make_spec(seed, v, b, pol, delib)
            ctrl = build(spec)
            try:
                p = run_trial(spec, a.out / f"{name}.jsonl", store, controller=ctrl)
            finally:
                ctrl.close()
            t = Trace.load(p)
            r = evaluate_trace(t)
            corr = next((c for c in t.all("coordinate_correction") if c["decision_id"] == (t.first("commit") or {}).get("decision_id")), None)
            obs = {o["observation_id"]: o["source_time_ns"] for o in t.all("observation")}
            rows[(v, b, pol)].append((r, corr, obs, t))
            say(f"- {name}: valid={r.valid_at_receipt} err={r.spatial_error_px if r.spatial_error_px is None else round(r.spatial_error_px, 1)} "
                f"B→D={(r.b_to_d_ns or 0)/1e9:.2f}s track={(corr or {}).get('track_ns', 0)/1e6:.1f}ms "
                f"device={(corr or {}).get('track_device_ns', 0)/1e6:.2f}ms residency_ok={(corr or {}).get('residency_ok')} "
                f"labels={','.join(r.labels) or '-'} viol={','.join(r.contract_violations) or '-'}")
            (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    say()
    say("## Per cell (Live; Wilson 95%)")
    say()
    say("| v (px/s) | binder | policy | n | SR | Wilson | mean G (px) | mean err (px) | median track ms (charged to D) | median device bind ms | "
        "median t_A − τ_eff (ms) | mean error-law residual (px) | max residual (px) | residency ok |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for key in cells:
        rs = [x for x in rows[key] if x[0].valid_trial]
        if not rs:
            say(f"| {key[0]} | {key[1]} | {key[2]} | 0 | - | - | - | - | - | - | - | - | - | - |"); continue
        v, b, pol = key
        k = sum(bool(r.valid_at_receipt) for r, *_ in rs)
        p_, lo, hi = wilson(k, len(rs))
        errs = [r.spatial_error_px for r, *_ in rs if r.spatial_error_px is not None]
        g = statistics.mean(r.return_G for r, *_ in rs)
        tracks = [c["track_ns"] / 1e6 for _, c, *_ in rs if c]
        devs = [c["track_device_ns"] / 1e6 for _, c, *_ in rs if c and c.get("track_device_ns")]
        resid, slack = [], []
        for r, c, obs, t in rs:
            if r.spatial_error_px is None:
                continue
            commit, rec = t.first("commit"), t.first("receipt")
            h0 = t.header()["harness_origin_ns"]
            t_a = rec["receipt_time_ns"] - h0
            if c and c["binding"] == "tracked_at_D" and c.get("matched"):
                tau = obs[c["observation_id"]]
            else:
                tau = obs[t.committing_decision()["observation_id"]]
            predicted = v * (t_a - tau) / 1e9
            resid.append(abs(r.spatial_error_px - predicted)); slack.append((t_a - tau) / 1e6)
        ok = all(bool(c.get("residency_ok")) for _, c, *_ in rs if c)
        summary[key] = {"sr": k, "n": len(rs), "err": statistics.mean(errs) if errs else None, "track_ms": statistics.median(tracks) if tracks else None,
                        "device_ms": statistics.median(devs) if devs else None, "resid": statistics.mean(resid) if resid else None}
        say(f"| {v} | {b} | {pol} | {len(rs)} | {k}/{len(rs)} | [{lo:.2f}, {hi:.2f}] | {g:.1f} | {statistics.mean(errs) if errs else float('nan'):.1f} | "
            f"{statistics.median(tracks) if tracks else float('nan'):.1f} | {statistics.median(devs) if devs else float('nan'):.2f} | "
            f"{statistics.median(slack) if slack else float('nan'):.0f} | {statistics.mean(resid) if resid else float('nan'):.1f} | "
            f"{max(resid) if resid else float('nan'):.1f} | {ok} |")
    say()
    say("τ_eff = the source time of the frame bound at D (late binding, matched) or of the frame at B (frozen / no match); "
        "the residual is |error − v (t_A − τ_eff)| per trial. The host binder's cost enters t_A − τ_eff because D moves by the "
        "binder's measured cost (LEDGER WI-004 part 1).")
    (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (a.out / "summary.json").write_text(json.dumps({f"v{k[0]}_{k[1]}_{k[2]}": s for k, s in summary.items()}, indent=1))
    if a.results:
        a.results.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
