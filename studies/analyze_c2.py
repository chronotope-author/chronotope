"""Analysis of WI-009 agent traces (jump_target): per-cell success, frozen counterfactual at the same A,
paired exact tests, restart cost, labels. Writes results/WI-009_<tag>.md.

  PYTHONPATH=src .venv/bin/python studies/analyze_c2.py runs/wi009_qwen38_nothink results/WI-009_qwen38_nothink.md
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.stats import paired_exact, wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"live_jump(\d+)s_(frozen|restart_if_moved|late_bind)_s(\d+)\.jsonl$")


def main(run: Path, out: Path) -> int:
    cells = defaultdict(dict); model = None
    for p in sorted(run.glob("live_*.jsonl")):
        m = NAME.search(p.name)
        if not m:
            continue
        t = Trace.load(p); model = model or t.header()["spec"]["controller"]["model_id"]
        cells[(int(m.group(1)), m.group(2))][int(m.group(3))] = evaluate_trace(t)
    L = [f"# WI-009 — jump target with a real agent: analysis ({model}; {run.name})", "",
         "Balloon skin, 320×240, 24-px target, 16-s window; the balloon jumps 96 px to the right at 2 s after appearance (during the ~5.3-s deliberation) or at 14 s (after A: control).", "",
         "| jump at (s) | policy | n | SR (valid at A) | Wilson 95% | mean G_C (px) | mean err (px) | restarts/trial | B→D of committing decision (s) | first B→D (s) | frozen counterfactual SR (same A) | labels | contract violations |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        rs = [r for r in cells[key].values() if r.valid_trial]; n = len(rs)
        k = sum(bool(r.valid_at_receipt) for r in rs); p, lo, hi = wilson(k, n)
        err = [r.spatial_error_px for r in rs if r.spatial_error_px is not None]
        cf = [r.proposed_valid_at_receipt for r in rs if r.proposed_valid_at_receipt is not None]
        labs = defaultdict(int); viol = defaultdict(int)
        for r in rs:
            for x in r.labels: labs[x] += 1
            for x in r.contract_violations: viol[x] += 1
        L.append(f"| {key[0]} | {key[1]} | {n} | {p:.2f} | [{lo:.2f}, {hi:.2f}] | {statistics.mean(r.return_G for r in rs):.1f} | {(statistics.mean(err) if err else float('nan')):.1f} | "
                 f"{statistics.mean([r.n_restarts for r in rs]):.1f} | {statistics.mean([(r.b_to_d_ns or 0)/1e9 for r in rs]):.2f} | "
                 f"{statistics.mean([(r.first_b_to_d_ns or 0)/1e9 for r in rs]):.2f} | {(f'{sum(cf)/len(cf):.2f}' if cf else '-')} | "
                 f"{', '.join(f'{a}×{b}' for a, b in sorted(labs.items())) or '-'} | {', '.join(f'{a}×{b}' for a, b in sorted(viol.items())) or '-'} |")
    L += ["", "## Same-deliberation pairs: frozen counterfactual vs late_bind (B, D, A shared)", "",
          "| jump at (s) | n pairs | frozen SR | late_bind SR | late_bind wins | frozen wins | exact p | late_bind Ḡ_C (px) |", "|---|---|---|---|---|---|---|---|"]
    for (j, pol), d in sorted(cells.items()):
        if pol != "late_bind":
            continue
        rs = [r for r in d.values() if r.valid_trial and r.proposed_valid_at_receipt is not None]
        t = paired_exact([bool(r.proposed_valid_at_receipt) for r in rs], [bool(r.valid_at_receipt) for r in rs])
        L.append(f"| {j} | {t['n_pairs']} | {t['rate_a']:.2f} | {t['rate_b']:.2f} | {t['b_only']} | {t['a_only']} | {t['p_two_sided']:.3g} | {statistics.mean(r.return_G for r in rs):.1f} |")
    L += ["", "## Counterfactual vs independent frozen run (same seeds)", "", "| jump at (s) | frozen run | counterfactual | disagreeing |", "|---|---|---|---|"]
    for j in sorted({k[0] for k in cells}):
        fr, lb = cells.get((j, "frozen"), {}), cells.get((j, "late_bind"), {})
        common = [s for s in sorted(set(fr) & set(lb)) if fr[s].valid_trial and lb[s].valid_trial and lb[s].proposed_valid_at_receipt is not None]
        if common:
            a = [bool(fr[s].valid_at_receipt) for s in common]; b = [bool(lb[s].proposed_valid_at_receipt) for s in common]
            L.append(f"| {j} | {sum(a)/len(a):.2f} | {sum(b)/len(b):.2f} | {sum(x != y for x, y in zip(a, b))}/{len(common)} |")
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
