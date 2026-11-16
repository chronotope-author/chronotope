"""Paired comparison of two P/R/L run directories that share cells and seeds (e.g. A1 on the animated player vs
the player_static control): per cell and mode, SR and mean G in each run, discordant counts and exact p on the
paired outcomes, and the paired mean difference in G.

  PYTHONPATH=src .venv/bin/python studies/compare_prl_runs.py RUN_A RUN_B OUT.md --labels animated,static [--pilot]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_prl import load  # noqa: E402
from chronotope.stats import paired_exact  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path); ap.add_argument("run_b", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--labels", default="A,B"); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--cell-order", default=None)
    a = ap.parse_args()
    la, lb = a.labels.split(",")
    ma, ca, _, _ = load(a.run_a); mb, cb, _, _ = load(a.run_b)
    cells = a.cell_order.split(",") if a.cell_order else sorted({k[0] for k in ca} | {k[0] for k in cb})
    L = [f"# Paired comparison: {la} ({a.run_a.name}) vs {lb} ({a.run_b.name}) — {ma}" + (" — PILOT" if a.pilot else ""), "",
         "Same cells and seeds; the only difference between the runs is the one named in the labels. Outcome pair per run, then the "
         "paired contrast (exact McNemar on the discordant seeds; paired mean ΔG).", "",
         f"| cell | mode | n pairs | {la} SR | {lb} SR | {la}-only | {lb}-only | exact p | {la} Ḡ | {lb} Ḡ | mean ΔG ({lb}−{la}) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        for pol in ("paused", "replay", "live"):
            da, db = ca.get((cell, pol), {}), cb.get((cell, pol), {})
            common = [s for s in sorted(set(da) & set(db)) if da[s][0].valid_trial and db[s][0].valid_trial]
            if not common:
                continue
            xa = [bool(da[s][0].valid_at_receipt) for s in common]; xb = [bool(db[s][0].valid_at_receipt) for s in common]
            t = paired_exact(xa, xb)
            ga = [da[s][0].return_G for s in common]; gb = [db[s][0].return_G for s in common]
            L.append(f"| {cell} | {pol} | {len(common)} | {sum(xa)}/{len(common)} | {sum(xb)}/{len(common)} | {t['a_only']} | {t['b_only']} | "
                     f"{t['p_two_sided']:.3g} | {statistics.mean(ga):+.2f} | {statistics.mean(gb):+.2f} | {statistics.mean(y - x for x, y in zip(ga, gb)):+.2f} |")
    # decision timing: when did the first decision start, and how many decisions per Live trial
    L += ["", "## Live decision cadence (per run): median first-decision start (task s), decisions per trial", ""]
    for lab, cc in ((la, ca), (lb, cb)):
        for cell in cells:
            rs = [(r, t) for r, t in cc.get((cell, "live"), {}).values() if r.valid_trial]
            if not rs:
                continue
            starts = []
            for r, t in rs:
                d = t.first("decision_begin"); h0 = t.header()["harness_origin_ns"]
                if d:
                    starts.append((d["begin_time_ns"] - h0) / 1e9)
            L.append(f"- {lab} {cell}: first decision at {statistics.median(starts):.2f} s (median), {statistics.mean(r.n_decisions for r, _ in rs):.1f} decisions/trial")
    a.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
