"""O2 remedy matrix (agent) from traces: (SR, mean G_C) matrices, the card's pattern check, closing-column margins.
Lets the matrix be regenerated when the return rule changes without re-running the model.

  PYTHONPATH=src .venv/bin/python studies/analyze_o2.py runs/o2_qwen38_nothink_pilot10 results/O2_qwen38_nothink_pilot10.md [--pilot]
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.stats import wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"o2_(pre_d|closing|post_d)_(none|newest|earlier|both)_s(\d+)\.jsonl$")
COLS = ["pre_d", "closing", "post_d"]
ROWS = ["none", "newest", "earlier", "both"]
PREDICTED = {"pre_d": [0, 1, 0, 1], "closing": [0, 0, 1, 1], "post_d": [0, 0, 0, 1]}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("run", type=Path); ap.add_argument("out", type=Path); ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cells = defaultdict(list); model = None; close = None
    for p in sorted(a.run.glob("o2_*.jsonl")):
        m = NAME.search(p.name)
        if not m:
            continue
        t = Trace.load(p); h = t.header(); model = model or h["spec"]["controller"]["model_id"]
        r = evaluate_trace(t); c = t.first("commit"); rc = t.first("receipt")
        if m.group(1) == "closing":
            close = (h["spec"]["scenario"]["event_time_ns"] + h["spec"]["scenario"]["response_window_ns"]) / 1e9
        cells[(m.group(1), m.group(2))].append((r, c["task_time_at_commit_ns"] / 1e9 if c else None, rc["task_time_at_receipt_ns"] / 1e9 if rc else None))
    if not cells:
        print("no traces"); return 1
    L = [f"# O2 remedy matrix (agent) — {model} ({a.run.name})" + (" — PILOT, 10 seeds: not a citable cell" if a.pilot else ""), "",
         "Live, balloon skin 320x240, 24-px balloon, prompt 5; slow route 1.5 s (rows none, newest), fast route 20 ms (rows earlier, both); "
         "frozen commit (rows none, earlier) vs late binding (rows newest, both). SR = valid at receipt; G_C = −min(click error at receipt, 400 px), "
         "−400 when the balloon is absent at receipt or no click arrived. Predicted (none/newest/earlier/both): "
         + "; ".join(f"{c} {'/'.join(map(str, v))}" for c, v in PREDICTED.items()) + ".", "",
         "## Success rate (Wilson 95% in brackets)", "", "| remedy | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for row in ROWS:
        cells_row = []
        for c in COLS:
            rs = [r for r, _, _ in cells.get((c, row), []) if r.valid_trial]
            if not rs:
                cells_row.append("-"); continue
            k = sum(bool(r.valid_at_receipt) for r in rs); _, lo, hi = wilson(k, len(rs))
            cells_row.append(f"{k}/{len(rs)} [{lo:.2f}, {hi:.2f}]")
        L.append(f"| {row} | " + " | ".join(cells_row) + " |")
    L += ["", "## Mean return G_C (px)", "", "| remedy | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for row in ROWS:
        L.append(f"| {row} | " + " | ".join((f"{statistics.mean(r.return_G for r, _, _ in cells[(c, row)] if r.valid_trial):.1f}" if cells.get((c, row)) else "-") for c in COLS) + " |")
    L += ["", "## Pattern check vs the card's prediction", ""]
    for c, want in PREDICTED.items():
        got = [round(sum(bool(r.valid_at_receipt) for r, _, _ in cells.get((c, row), [])) / max(1, len(cells.get((c, row), [])))) for row in ROWS]
        L.append(f"- {c}: predicted {want}, observed (rounded) {got} {'OK' if got == want else 'DEVIATES'}")
    L += ["", "## Labels per cell", ""]
    for c in COLS:
        for row in ROWS:
            labs = defaultdict(int)
            for r, _, _ in cells.get((c, row), []):
                for lab in r.labels:
                    labs[lab] += 1
            L.append(f"- {c} × {row}: " + (", ".join(f"{k}×{v}" for k, v in sorted(labs.items())) or "-"))
    A = [x for (c, row), rs in cells.items() if c == "closing" for _, _, x in rs if x is not None]
    D = [d for (c, row), rs in cells.items() if c == "closing" for _, d, _ in rs if d is not None]
    if A and close:
        L += ["", f"Closing-column margins: window closes at {close:.2f} s; D ∈ [{min(D):.2f}, {max(D):.2f}] s; A ∈ [{min(A):.2f}, {max(A):.2f}] s "
              "(slow rows must have A > close, fast rows A < close)."]
    a.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
