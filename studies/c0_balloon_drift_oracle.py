"""WI-003 study (oracle controllers, simulated clock): S3.1 post-commit delay sweep on a moving
target; S3.2 four bindings with matched B, D, A; predictor delivery-estimate perturbation.
Writes results/WI-003_moving_target.md. Deterministic; reproducible with one command:
  PYTHONPATH=src .venv/bin/python studies/c0_balloon_drift_oracle.py
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from chronotope.__main__ import run_triplet  # noqa: E402
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, run_trial  # noqa: E402
from chronotope.trace import Trace  # noqa: E402
from test_moving_target import PUB, mt_spec  # noqa: E402

OUT = ROOT / "runs" / "wi003"
SEEDS = [11, 12, 13, 14, 15]
lines: list[str] = ["# WI-003 — moving target (oracle controllers, simulated clock)", "",
                    "canvas 96x64, target 6 px, event 500 ms, window 400 ms, 60 Hz frames, 30 ms publication delay, "
                    "oracle compute 100 ms, seeds 11–15 (mean over seeds).", ""]


def mean(xs):
    return statistics.mean(xs)


def main() -> int:
    global lines
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    # ---- S3.1: post-commit delay sweep --------------------------------------------------------
    lines += ["## S3.1 — post-commit delay sweep, binding = state_at_B", "",
              "| velocity (px/s) | D→A delay (ms) | Paused | Replay | Live valid | mean spatial error at A (px) |",
              "|---|---|---|---|---|---|"]
    for vx in (0, 60, 120):
        for delivery in (0, 20_000_000, 60_000_000, 100_000_000, 150_000_000, 200_000_000):
            res = [run_triplet(mt_spec(vx=vx, delivery=delivery, seed=s), OUT / f"s31_v{vx}_d{delivery}_s{s}")["results"]
                   for s in SEEDS]
            P = sum(r["paused"].valid_at_receipt for r in res)
            R = sum(r["replay"].valid_at_receipt for r in res)
            L = sum(r["live"].valid_at_receipt for r in res)
            err = mean([r["live"].spatial_error_px for r in res])
            lines.append(f"| {vx} | {delivery/1e6:.0f} | {P}/{len(SEEDS)} | {R}/{len(SEEDS)} | {L}/{len(SEEDS)} | {err:.1f} |")
    # ---- S3.2: four bindings, matched B, D, A -------------------------------------------------
    lines += ["", "## S3.2 — four bindings, matched B, D, A (delivery 60 ms; predictor told the true delays)", "",
              "| velocity (px/s) | binding | Live valid | mean spatial error at A (px) | trace record |", "|---|---|---|---|---|"]
    for vx in (0, 60, 120):
        for b in ("state_at_B", "newest_at_D", "predicted_at_A", "true_at_A"):
            rs = []
            for s in SEEDS:
                spec = mt_spec(vx=vx, binding=b, seed=s, exp_obs=PUB if b == "predicted_at_A" else None,
                               exp_del=60_000_000 if b == "predicted_at_A" else None)
                t = Trace.load(run_trial(spec, OUT / f"s32_v{vx}_{b}_s{s}.jsonl", PayloadStore(OUT / "payloads")))
                rs.append((evaluate_trace(t), t))
            rec = "coordinate_correction" if rs[0][1].first("coordinate_correction") else "—"
            lines.append(f"| {vx} | {b} | {sum(r.valid_at_receipt for r, _ in rs)}/{len(SEEDS)} | "
                         f"{mean([r.spatial_error_px for r, _ in rs]):.1f} | {rec} |")
    # ---- predictor perturbation ----------------------------------------------------------------
    lines += ["", "## S3.2 — predictor with a wrong delivery-delay estimate (true 60 ms, velocity 120 px/s)", "",
              "| estimate error (ms) | Live valid | mean spatial error at A (px) | v·|error| (px) |", "|---|---|---|---|"]
    for delta in (-60_000_000, -40_000_000, -20_000_000, 0, 20_000_000, 40_000_000, 60_000_000):
        rs = [evaluate_trace(Trace.load(run_trial(
            mt_spec(vx=120, binding="predicted_at_A", seed=s, exp_obs=PUB, exp_del=60_000_000 + delta),
            OUT / f"pert_{delta}_s{s}.jsonl", PayloadStore(OUT / "payloads")))) for s in SEEDS]
        lines.append(f"| {delta/1e6:+.0f} | {sum(r.valid_at_receipt for r in rs)}/{len(SEEDS)} | "
                     f"{mean([r.spatial_error_px for r in rs]):.1f} | {abs(120*delta/1e9):.1f} |")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "WI-003_moving_target.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
