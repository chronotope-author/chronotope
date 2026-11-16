"""Analysis of WI-006 traces: per-cell duplicates / exact success / first E→A with CIs; paired exact tests
none vs harness_pending and none vs model_informed on exact success (same seeds); dropped-click cost.

  PYTHONPATH=src .venv/bin/python studies/analyze_p1.py runs/wi006_qwen38_nothink results/WI-006_qwen38_nothink.md
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_pending_trace  # noqa: E402
from chronotope.stats import bootstrap_ci, paired_exact, wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"live_d(\d+)_(none|harness_pending|model_informed)(_drop)?_s(\d+)\.jsonl$")


def main(run_dir: Path, out: Path) -> int:
    cells = defaultdict(dict)
    model = None
    for p in sorted(run_dir.glob("live_*.jsonl")):
        m = NAME.search(p.name)
        if not m:
            continue
        t = Trace.load(p)
        model = model or t.header()["spec"]["controller"].get("model_id", t.header()["spec"]["controller"]["kind"])
        cells[(int(m.group(1)), m.group(2), bool(m.group(3)))][int(m.group(4))] = evaluate_pending_trace(t)
    if not cells:
        print("no traces"); return 1
    L = [f"# WI-006 — pending-action record: analysis ({model}; {run_dir.name})", "",
         "## Per cell (Live; valid trials)", "",
         "Outcome pair first: SR_exact = accepted and no duplicate; mean G = −(seconds until the accepted click's effect was visible), "
         "the full window when no click was accepted (time-based, the shape of Task A's G_A; duplicates are not charged in G).", "",
         "| effect delay (s) | dropped first click | memory | n | SR_exact (Wilson 95%) | mean G (s) | accepted | duplicates/trial (mean, 95% bootstrap CI) | trials with ≥1 duplicate | first E→A (s) | invocations/trial | suppressed/trial | invalid clicks/trial |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        d, mem, drop = key
        rs = [r for r in cells[key].values() if r.valid_trial]
        n = len(rs)
        if not n:
            L.append(f"| {d} | {drop} | {mem} | 0 | - | - | - | - | - | - | - | - | - |"); continue
        dups = [r.duplicates for r in rs]
        dm, dlo, dhi = bootstrap_ci(dups, lambda s: sum(s) / len(s), n_boot=5000, seed=0)
        k = sum(r.exact_success for r in rs)
        p, lo, hi = wilson(k, n)
        fa = [r.first_valid_E_A_ns / 1e9 for r in rs if r.first_valid_E_A_ns is not None]
        L.append(f"| {d} | {drop} | {mem} | {n} | {p:.2f} [{lo:.2f}, {hi:.2f}] | {statistics.mean(r.return_G for r in rs):+.2f} | "
                 f"{sum(r.accepted for r in rs)/n:.2f} | {dm:.2f} [{dlo:.2f}, {dhi:.2f}] | "
                 f"{sum(x > 0 for x in dups)/n:.2f} | {(statistics.mean(fa) if fa else float('nan')):.2f} | "
                 f"{statistics.mean([r.n_decisions for r in rs]):.1f} | {statistics.mean([r.n_suppressed for r in rs]):.1f} | "
                 f"{statistics.mean([r.invalid_clicks for r in rs]):.2f} |")
    L += ["", "## Paired exact tests on exact success (same seeds, same frames until the first divergence)", "",
          "| effect delay (s) | comparison | n pairs | A | B | B wins | A wins | exact McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for d in sorted({k[0] for k in cells}):
        base = cells.get((d, "none", False), {})
        for other in ("harness_pending", "model_informed"):
            oc = cells.get((d, other, False), {})
            common = [s for s in sorted(set(base) & set(oc)) if base[s].valid_trial and oc[s].valid_trial]
            if not common:
                continue
            a = [base[s].exact_success for s in common]; b = [oc[s].exact_success for s in common]
            t = paired_exact(a, b)
            L.append(f"| {d} | none vs {other} | {t['n_pairs']} | {t['rate_a']:.2f} | {t['rate_b']:.2f} | {t['b_only']} | {t['a_only']} | {t['p_two_sided']:.3g} |")
    L += ["", "## First valid A: does the record delay the first action? (same seeds; seconds from E1)", "",
          "| effect delay (s) | none | harness_pending | model_informed | max |Δ| none−harness (s) |", "|---|---|---|---|---|"]
    for d in sorted({k[0] for k in cells}):
        vals = {}
        for mem in ("none", "harness_pending", "model_informed"):
            c = cells.get((d, mem, False), {})
            vals[mem] = {s: r.first_valid_E_A_ns / 1e9 for s, r in c.items() if r.valid_trial and r.first_valid_E_A_ns is not None}
        if not vals["none"]:
            continue
        common = sorted(set(vals["none"]) & set(vals["harness_pending"]))
        mx = max((abs(vals["none"][s] - vals["harness_pending"][s]) for s in common), default=float("nan"))
        L.append(f"| {d} | " + " | ".join((f"{statistics.mean(v.values()):.2f}" if v else "-") for v in (vals["none"], vals["harness_pending"], vals["model_informed"])) + f" | {mx:.2f} |")
    dk = [k for k in cells if k[2]]
    if dk:
        L += ["", "## Dropped first click (route simulated_drop_first, effect delay 2 s)", "",
              "| memory | n | accepted | first E→A (s) | duplicates/trial | pending expired/trial |", "|---|---|---|---|---|---|"]
        for key in sorted(dk):
            rs = [r for r in cells[key].values() if r.valid_trial]
            if not rs:
                continue
            fa = [r.first_valid_E_A_ns / 1e9 for r in rs if r.first_valid_E_A_ns is not None]
            L.append(f"| {key[1]} | {len(rs)} | {sum(r.accepted for r in rs)/len(rs):.2f} | {(statistics.mean(fa) if fa else float('nan')):.2f} | "
                     f"{statistics.mean([r.duplicates for r in rs]):.2f} | {statistics.mean([r.pending_events.get('pending_expired', 0) for r in rs]):.1f} |")
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
