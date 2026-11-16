"""O3 part 2 analysis: the repair matrix (rows: inferred label; columns: intervention) from the rerun traces,
selected vs wrong intervention per label, and per-source-cell detail. Every cell is repaired/n (SR) with mean G.

  PYTHONPATH=src .venv/bin/python studies/analyze_o3.py runs/o3_qwen38_nothink_100 results/O3_auto_diagnosis_100.md
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

NAME = re.compile(r"o3_(\w+?)_(.+?)_s(\d+)_(outdated_field|outdated_kind|invalidated_after_commitment|late|missed_evidence)_(selected|wrong)_(late_bind|re_decision|earlier_delivery|earlier_observation)\.jsonl$")
LABELS = ["outdated_field", "outdated_kind", "invalidated_after_commitment", "late", "missed_evidence"]
INTERVENTIONS = ["late_bind", "re_decision", "earlier_delivery", "earlier_observation"]


def cell(rs) -> str:
    v = [r for r in rs if r.valid_trial]
    if not v:
        return "-"
    k = sum(bool(r.valid_at_receipt) for r in v); _, lo, hi = wilson(k, len(v))
    return f"{k}/{len(v)} ({k/len(v):.2f} [{lo:.2f}, {hi:.2f}]), G {statistics.mean(r.return_G for r in v):.2f}"


def main(run: Path, out: Path) -> int:
    rows = []
    model = None
    for p in sorted(run.glob("o3_*.jsonl")):
        m = NAME.search(p.name)
        if not m:
            continue
        t = Trace.load(p); model = model or t.header()["spec"]["controller"]["model_id"]
        rows.append({"study": m.group(1), "cell": m.group(2), "seed": int(m.group(3)), "label": m.group(4),
                     "which": m.group(5), "intervention": m.group(6), "r": evaluate_trace(t)})
    if not rows:
        print("no traces"); return 1
    mat = defaultdict(list)
    for x in rows:
        mat[(x["label"], x["intervention"])].append(x["r"])
    L = [f"# O3 auto-diagnosis — repair matrix on held-out Live failures ({model})", "",
         "Each failing seed of a baseline arm was rerun under the intervention selected by its events-only label and under one "
         "wrong intervention (rotated). A cell is the fraction of reruns that succeeded (valid at receipt) with the mean return; "
         "the diagonal of the label→intervention map is marked with *.", "",
         "## Repair matrix (rows: inferred label of the original failure; columns: intervention on the rerun)", "",
         "| label \\ intervention | " + " | ".join(INTERVENTIONS) + " |", "|---|" + "---|" * len(INTERVENTIONS)]
    diag = {"outdated_field": "late_bind", "outdated_kind": "re_decision", "invalidated_after_commitment": "earlier_delivery",
            "late": "earlier_delivery", "missed_evidence": "earlier_observation"}
    for lab in LABELS:
        if not any((lab, i) in mat for i in INTERVENTIONS):
            continue
        L.append(f"| {lab} | " + " | ".join(("*" if diag[lab] == i else "") + cell(mat.get((lab, i), [])) for i in INTERVENTIONS) + " |")
    L += ["", "## Selected vs wrong intervention, per label (paired on the seed)", "",
          "| label | n seeds | selected repaired | wrong repaired | selected only | wrong only | exact McNemar p |", "|---|---|---|---|---|---|---|"]
    for lab in LABELS:
        sel = {(x["study"], x["cell"], x["seed"]): x["r"] for x in rows if x["label"] == lab and x["which"] == "selected"}
        wr = {(x["study"], x["cell"], x["seed"]): x["r"] for x in rows if x["label"] == lab and x["which"] == "wrong"}
        keys = sorted(set(sel) & set(wr))
        if not keys:
            continue
        a_ = [bool(sel[k].valid_at_receipt) for k in keys]; b_ = [bool(wr[k].valid_at_receipt) for k in keys]
        ex = paired_exact(b_, a_)
        L.append(f"| {lab} | {len(keys)} | {sum(a_)}/{len(keys)} | {sum(b_)}/{len(keys)} | {ex['b_only']} | {ex['a_only']} | {ex['p_two_sided']:.3g} |")
    L += ["", "## Per source cell", "", "| study | cell | label | intervention | which | repaired |", "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for x in rows:
        by[(x["study"], x["cell"], x["label"], x["intervention"], x["which"])].append(x["r"])
    for k, rs in sorted(by.items()):
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {cell(rs)} |")
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
