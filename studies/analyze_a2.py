"""A2 analysis: per (seek delay, arm) the outcome pair FIRST — SR_exact (accepted and no duplicate) and mean G_A —
then duplicates, first E→A, invocations, suppressed proposals, binder diagnostics (chronotope arm); paired exact tests
none vs record and record vs chronotope on exact success; G_A and first-A equality seed by seed.

  PYTHONPATH=src .venv/bin/python studies/analyze_a2.py runs/a2_qwen38_nothink_pilot10 results/A2_qwen38_nothink_pilot10.md [--pilot]
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_pending_trace  # noqa: E402
from chronotope.stats import bootstrap_ci, paired_exact, wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"live_k(\d+)_(none|record|chronotope)_s(\d+)\.jsonl$")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("run", type=Path); ap.add_argument("out", type=Path); ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cells, binder, model = defaultdict(dict), defaultdict(list), None
    for p in sorted(a.run.glob("live_*.jsonl")):
        m = NAME.search(p.name)
        if not m:
            continue
        t = Trace.load(p)
        model = model or t.header()["spec"]["controller"]["model_id"]
        key = (int(m.group(1)), m.group(2))
        cells[key][int(m.group(3))] = evaluate_pending_trace(t)
        for c in t.all("coordinate_correction"):
            binder[key].append((c["translation_x"], c["translation_y"], c.get("ambiguous", False), c["matched"]))
    if not cells:
        print("no traces"); return 1
    L = [f"# A2 sponsor skip (pending) — {model} ({a.run.name})" + (" — PILOT, 10 seeds: not a citable cell" if a.pilot else ""), "",
         "Outcome pair first: SR_exact = the first skip was accepted AND no duplicate skip was received; mean G_A = −(seconds of "
         "sponsor watched), full window when no skip was accepted (cards/A2-sponsor-skip-pending.md). Duplicates are not charged in G_A.", "",
         "| seek delay (s) | arm | n | SR_exact (Wilson 95%) | mean G_A (s) | accepted | duplicates/trial (bootstrap 95%) | first E→A (s) | invocations | suppressed | invalid clicks | binder: zero-translation / ambiguous / bindings; max |translation| |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells, key=lambda k: (k[0], ["none", "record", "chronotope"].index(k[1]))):
        rs = [r for r in cells[key].values() if r.valid_trial]
        n = len(rs)
        if not n:
            L.append(f"| {key[0]} | {key[1]} | 0 | - | - | - | - | - | - | - | - | - |"); continue
        k = sum(r.exact_success for r in rs); _, lo, hi = wilson(k, n)
        dups = [r.duplicates for r in rs]; dm, dlo, dhi = bootstrap_ci(dups, lambda s: sum(s) / len(s), n_boot=5000, seed=0)
        fa = [r.first_valid_E_A_ns / 1e9 for r in rs if r.first_valid_E_A_ns is not None]
        b = binder.get(key, [])
        bs = (f"{sum(1 for x in b if x[0] == 0 and x[1] == 0)} / {sum(1 for x in b if x[2])} / {len(b)}; max |t| "
              f"{max(max(abs(x[0]), abs(x[1])) for x in b)} px") if b else "-"
        L.append(f"| {key[0]} | {key[1]} | {n} | {k}/{n} [{lo:.2f}, {hi:.2f}] | {statistics.mean(r.return_G for r in rs):+.2f} | "
                 f"{sum(r.accepted for r in rs)}/{n} | {dm:.2f} [{dlo:.2f}, {dhi:.2f}] | {(statistics.mean(fa) if fa else float('nan')):.2f} | "
                 f"{statistics.mean([r.n_decisions for r in rs]):.1f} | {statistics.mean([r.n_suppressed for r in rs]):.1f} | "
                 f"{statistics.mean([r.invalid_clicks for r in rs]):.2f} | {bs} |")
    L += ["", "## Paired comparisons (same seeds)", "",
          "| seek delay (s) | comparison | n pairs | A exact | B exact | B wins | A wins | exact McNemar p | max |ΔG_A| (s) | max |Δ first A| (s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted({k[0] for k in cells}):
        for x, y in (("none", "record"), ("record", "chronotope")):
            cx, cy = cells.get((d, x), {}), cells.get((d, y), {})
            common = [s for s in sorted(set(cx) & set(cy)) if cx[s].valid_trial and cy[s].valid_trial]
            if not common:
                continue
            t = paired_exact([cx[s].exact_success for s in common], [cy[s].exact_success for s in common])
            eq = max(abs(cx[s].return_G - cy[s].return_G) for s in common)
            fa = [abs((cx[s].first_valid_E_A_ns or 0) - (cy[s].first_valid_E_A_ns or 0)) / 1e9 for s in common
                  if cx[s].first_valid_E_A_ns is not None and cy[s].first_valid_E_A_ns is not None]
            L.append(f"| {d} | {x} vs {y} | {t['n_pairs']} | {t['rate_a']:.2f} | {t['rate_b']:.2f} | {t['b_only']} | {t['a_only']} | "
                     f"{t['p_two_sided']:.3g} | {eq:.2f} | {(max(fa) if fa else float('nan')):.2f} |")
    a.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
