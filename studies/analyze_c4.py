"""C4 analysis: per-cell SR with Wilson CIs and mean G; frozen counterfactual vs late_bind paired exact test (same
trace: B, D, A shared); the independent frozen run vs the counterfactual on the same seeds; restart cost.

  PYTHONPATH=src .venv/bin/python studies/analyze_c4.py runs/c4_qwen38_nothink_100 results/C4_qwen38_nothink_100.md
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.stats import paired_difference_ci, paired_exact, wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"live_(ch2|ch14|static)_(frozen|late_bind|restart_if_changed)_s(\d+)\.jsonl$")


def main(run_dir: Path, out: Path) -> int:
    cells = defaultdict(dict)
    model = None
    for p in sorted(run_dir.glob("live_*.jsonl")):
        m = NAME.search(p.name)
        if not m:
            continue
        t = Trace.load(p)
        model = model or t.header()["spec"]["controller"]["model_id"]
        cells[(m.group(1), m.group(2))][int(m.group(3))] = evaluate_trace(t)
    L = [f"# C4 rotating code — {model}", "",
         "Scenario `rotating_code`, notification rendering, 320x240, 4-digit code from E (0.5 s), replaced at E + change; "
         "window 16 s; prompt 11 (type the code currently shown); publication 30 ms; delivery 20 ms. Every cell reports the "
         "outcome pair first: SR = valid at receipt (the typed value equals the code shown at A), mean G = minus the seconds "
         "from E to the receipt of a correct value (window charged when none).", "",
         "| cell | policy | n | SR | Wilson 95% | mean G (s) | restarts/trial | B→D (s) | first B→D (s) | outdated_at_commitment | semantic_error | invalid trials |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (cell, pol), rs in sorted(cells.items()):
        v = [r for r in rs.values() if r.valid_trial]; n = len(v)
        k = sum(bool(r.valid_at_receipt) for r in v)
        lo, _, hi = wilson(k, n) if n else (0, 0, 0)
        L.append(f"| {cell} | {pol} | {n} | {k/n:.2f} | [{lo:.2f}, {hi:.2f}] | {statistics.mean(r.return_G for r in v):.2f} | "
                 f"{statistics.mean(r.n_restarts for r in v):.1f} | {statistics.median((r.b_to_d_ns or 0)/1e9 for r in v):.2f} | "
                 f"{statistics.median((r.first_b_to_d_ns or 0)/1e9 for r in v):.2f} | {sum('outdated_at_commitment' in r.labels for r in v)} | "
                 f"{sum('semantic_error' in r.labels for r in v)} | {len(rs) - n} |")
    L += ["", "## Same-decision comparison: frozen counterfactual vs late_bind (B, D, A shared exactly)", "",
          "| cell | n pairs | frozen SR | late_bind SR | late_bind wins | frozen wins | exact McNemar p | Δ SR (bootstrap 95% CI) |",
          "|---|---|---|---|---|---|---|---|"]
    for cell in ("ch2", "ch14", "static"):
        rs = {s: r for s, r in cells.get((cell, "late_bind"), {}).items() if r.valid_trial and r.proposed_valid_at_receipt is not None}
        if not rs:
            continue
        seeds = sorted(rs)
        a = [bool(rs[s].proposed_valid_at_receipt) for s in seeds]; b = [bool(rs[s].valid_at_receipt) for s in seeds]
        ex = paired_exact(a, b); lo, mid, hi = paired_difference_ci(a, b)
        L.append(f"| {cell} | {len(seeds)} | {sum(a)/len(a):.2f} | {sum(b)/len(b):.2f} | {ex['b_only']} | {ex['a_only']} | {ex['p_two_sided']:.3g} | "
                 f"{mid:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    L += ["", "## Independent frozen run vs the counterfactual (same seeds)", ""]
    for cell in ("ch2", "ch14", "static"):
        fr = cells.get((cell, "frozen"), {}); lb = cells.get((cell, "late_bind"), {})
        common = [s for s in fr if s in lb and fr[s].valid_trial and lb[s].valid_trial and lb[s].proposed_valid_at_receipt is not None]
        agree = sum(bool(fr[s].valid_at_receipt) == bool(lb[s].proposed_valid_at_receipt) for s in common)
        L.append(f"- {cell}: counterfactual agrees with the independent frozen run on {agree}/{len(common)} seeds")
    out.write_text("\n".join(L) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
