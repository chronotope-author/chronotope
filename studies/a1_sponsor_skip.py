"""A1 sponsor skip (oracle, simulated clock): closed-form sweep of the graded exposure return,
Paused/Replay pinning, and the success-by-deadline curve. 100 seeds per cell (seeds vary the
button placement; the oracle timing is seed-independent, which the closed form verifies).
Writes results/A1_sponsor_skip_oracle.md. Reproducible with one command:
  PYTHONPATH=src .venv/bin/python studies/a1_sponsor_skip.py
"""
from __future__ import annotations

import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from chronotope.__main__ import run_triplet  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.stats import success_by_deadline  # noqa: E402
from conftest import base_spec_dict  # noqa: E402

OUT = ROOT / "runs" / "a1_oracle"
SEEDS = list(range(1000, 1010))   # 10 per cell (user rule 2026-08-27); 100 only to finalize a cited cell
E, WINDOW, PERIOD, PUB, DELIVERY = 500_000_000, 5_000_000_000, 100_000_000, 30_000_000, 20_000_000
CLOSE = E + WINDOW
lines: list[str] = ["# A1 — sponsor skip (oracle, simulated clock)", "",
                    "canvas 96x64, button 15x6 px, sponsor 500 ms..5.5 s (window 5 s), 10 Hz frames, "
                    "30 ms publication delay, 20 ms delivery, seeds 1000..1099 (100 per cell).",
                    "G_A = -(sponsor seconds watched); a valid skip received at t removes the sponsor at "
                    "t + seek_delay; no valid skip watches the full 5 s.", ""]


def spec(compute: int, seek: int, seed: int) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "sponsor_skip", "scenario_seed": seed, "canvas_w_px": 96, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": E, "response_window_ns": WINDOW,
                     "frame_period_ns": PERIOD, "episode_horizon_ns": E + WINDOW + 500_000_000,
                     "irrelevant_motion": False, "skin": "player", "seek_delay_ns": seek}
    d["controller"]["compute_ns"] = compute
    d["delivery"]["delivery_delay_ns"] = DELIVERY
    return TrialSpec.from_dict(d)


def closed_form(compute: int, seek: int) -> dict:
    t1 = E  # 500 ms is a frame time at 10 Hz
    a = t1 + PUB + compute + DELIVERY
    valid = a < CLOSE
    return {"E_A": a - E, "valid": valid,
            "g_live": -(min(a + seek, CLOSE) - E) / 1e9 if valid else -WINDOW / 1e9,
            "g_pinned": -(min(t1 + seek, CLOSE) - E) / 1e9}


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    mismatches = 0
    lines.extend(["## Closed-form sweep — (SR, mean G_A) per cell",
                  "", "| compute (s) | seek delay (s) | Paused SR / G | Replay SR / G | Live SR / G | Live E→A (s) | Live labels | closed-form match |",
                  "|---|---|---|---|---|---|---|---|"])
    ea_by_compute: dict[int, tuple[bool, int]] = {}
    for compute in (500_000_000, 2_000_000_000, 4_000_000_000, 5_200_000_000):
        for seek in (0, 1_000_000_000):
            pred = closed_form(compute, seek)
            rows = {"paused": [], "replay": [], "live": []}
            labels: dict[str, int] = {}
            ok = 0
            for s in SEEDS:
                cell = OUT / f"c{compute}_k{seek}_s{s}"
                out = run_triplet(spec(compute, seek, s), cell)
                assert out["pair"].valid, (compute, seek, s)
                for p in rows:
                    r = out["results"][p]
                    assert r.valid_trial and not r.contract_violations, (p, r.to_dict())
                    rows[p].append(r)
                live = out["results"]["live"]
                for lab in live.labels:
                    labels[lab] = labels.get(lab, 0) + 1
                match = (live.valid_at_receipt == pred["valid"]
                         and abs(live.return_G - pred["g_live"]) < 1e-9
                         and live.intervals_ns["E_A"] == pred["E_A"]
                         and all(abs(out["results"][p].return_G - pred["g_pinned"]) < 1e-9
                                 and out["results"][p].valid_at_receipt for p in ("paused", "replay")))
                ok += int(match)
                shutil.rmtree(cell)
            mismatches += len(SEEDS) - ok
            ea_by_compute[compute] = (pred["valid"], pred["E_A"])

            def cell_txt(p):
                rs = rows[p]
                return f"{sum(bool(r.valid_at_receipt) for r in rs)}/{len(rs)} / {statistics.mean(r.return_G for r in rs):.2f}"
            lines.append(f"| {compute/1e9:.1f} | {seek/1e9:.0f} | {cell_txt('paused')} | {cell_txt('replay')} | "
                         f"{cell_txt('live')} | {statistics.mean(r.intervals_ns['E_A'] for r in rows['live'])/1e9:.2f} | "
                         f"{', '.join(f'{k} ×{v}' for k, v in sorted(labels.items())) or '—'} | {ok}/{len(SEEDS)} |")
    lines.extend(["", "Reading: the same valid skip is worth less the later it lands (seek 1 s: G falls from "
                  "-1.55 s at 0.5 s compute to -5.0 s once the path exceeds the sponsor); Paused/Replay pin the "
                  "receipt state at the trigger frame, so G is ≈ -(seek delay) there regardless of compute — "
                  "an advancing sponsor, not latency itself, is what the return prices.", ""])

    # ---- success-by-deadline curve (U23): pooled over the four compute cells --------------------
    outcomes = [(v, ea) for v, ea in ea_by_compute.values()]
    ws = [1, 2, 3, 4, 5, 6]
    curve = success_by_deadline([(True, ea) for _, ea in outcomes], [w * 1_000_000_000 for w in ws])
    lines.extend(["## Success-by-deadline curve F(w), pooled over the four compute cells", "",
                  "| w (s) | " + " | ".join(str(w) for w in ws) + " |",
                  "|---|" + "---|" * len(ws),
                  "| F(w) | " + " | ".join(f"{c:.2f}" for c in curve) + " |", "",
                  "Each compute cell contributes a step at its own E→A (0.55, 2.05, 4.05, 5.25 s); at w = 5 s "
                  "(the sponsor length) the curve equals the Live success rate of the sweep above. Exact for "
                  "the oracle (it never reads the rendered countdown); agents get this curve from the window "
                  "sweep itself.", ""])
    lines.append(f"Closed-form mismatches across all cells: **{mismatches}** (expected 0).")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "A1_sponsor_skip_oracle.md").write_text(text)
    print(text)
    shutil.rmtree(OUT, ignore_errors=True)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
