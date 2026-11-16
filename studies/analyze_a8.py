"""A8-PREP analysis: the 4-regime table (which controller succeeds when the content is known before E / revealed before
D / changes after D / the kind changes) from the scripted results (results/A8_preparation_scripted_100.md), the model
run directory (runs/a8_qwen38_nothink_100, cells written by studies/a8_preparation.py --model) and, for the kind row,
the V1 run directory (runs/v1_qwen38_nothink_100) when present. Every cell is (SR, mean G).

  PYTHONPATH=src .venv/bin/python studies/analyze_a8.py results/A8_preparation_100.md \\
      [--model-run runs/a8_qwen38_nothink_100] [--v1-run runs/v1_qwen38_nothink_100] [--scripted results/A8_preparation_scripted_100.md]
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

MODEL_NAME = re.compile(r"a8_(fixed_w\d+|seeded_w8)_(prepared|direct|late_bind)_s(\d+)\.jsonl$")
V1_NAME = re.compile(r"live_code_kc2_(late_bind|validate|restart_if_changed)_s(\d+)\.jsonl$")


def load_cells(run: Path, rx: re.Pattern) -> dict:
    cells = defaultdict(list)
    if not run or not run.exists():
        return cells
    for p in sorted(run.glob("*.jsonl")):
        m = rx.search(p.name)
        if m:
            t = Trace.load(p)
            cells[(m.group(1), m.group(2))].append((evaluate_trace(t), t))
    return cells


def pair(rs: list) -> str:
    v = [r for r, _ in rs if r.valid_trial]
    if not v:
        return "-"
    k = sum(bool(r.valid_at_receipt) for r in v)
    _, lo, hi = wilson(k, len(v))
    return f"{k}/{len(v)} ({k/len(v):.2f} [{lo:.2f}, {hi:.2f}]), G {statistics.mean(r.return_G for r in v):.2f}"


def not_available(rs: list) -> int:
    return sum(1 for _, t in rs if any(i.get("outcome") == "prepare_unavailable" for i in t.all("decision_invocation")))


def scripted_cell(md: str, section_row: str, col: int) -> str:
    """Pull one cell out of the scripted results table by its row label (first column) and column index."""
    for line in md.splitlines():
        if line.startswith(f"| {section_row} |"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            return cells[col] if col < len(cells) else "-"
    return "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--scripted", type=Path, default=Path("results/A8_preparation_scripted_100.md"))
    ap.add_argument("--model-run", type=Path, default=Path("runs/a8_qwen38_nothink_100"))
    ap.add_argument("--v1-run", type=Path, default=Path("runs/v1_qwen38_nothink_100"))
    a = ap.parse_args()
    md = a.scripted.read_text() if a.scripted.exists() else ""
    mc = load_cells(a.model_run, MODEL_NAME)
    v1 = load_cells(a.v1_run, V1_NAME)
    model = next((t.header()["spec"]["controller"]["model_id"] for rs in mc.values() for _, t in rs), None)
    L = [f"# A8-PREP — which remedy, in which information regime" + (f" ({model})" if model else ""), "",
         "Scripted cells: results/A8_preparation_scripted_100.md (simulated clock, compute 5.0 s for the deliberating "
         "bindings). Model cells: one pre-event call with prompt 14 (vlm_prepared) vs the deliberating rows at the same "
         "E = 8 s. Kind row: the V1 rotating-code kind_change cells (prompt 13). Every cell is (SR, mean G).", "",
         "## Regime table", "",
         "| regime | task (cell) | prepared | late_bind | predicted | re-decision (restart / validate) |",
         "|---|---|---|---|---|---|"]
    # scripted rows (columns of table (a): window | prepared | frozen | late_bind | restart | predicted)
    L.append("| content known before E (scripted) | fixed-layout notification, window 1 s | "
             f"{scripted_cell(md, '1', 1)} | {scripted_cell(md, '1', 3)} | {scripted_cell(md, '1', 5)} | {scripted_cell(md, '1', 4)} |")
    L.append("| content revealed before D (scripted) | seeded notification, window 8 s | "
             f"{scripted_cell(md, 'notification, seeded position, window 8 s', 1)} | {scripted_cell(md, 'notification, seeded position, window 8 s', 2)} | - | - |")
    L.append("| content revealed before D (scripted) | rotating code, change 2 s | "
             f"{scripted_cell(md, 'rotating code, change at 2 s, window 16 s', 1)} | {scripted_cell(md, 'rotating code, change at 2 s, window 16 s', 2)} | - | - |")
    # table (b') is a single-row table without a label column: its cells are prepared | late_bind | predicted | restart
    bb = "-"
    for i, line in enumerate(md.splitlines()):
        if line.startswith("| prepared | late_bind | predicted | restart |"):
            bb = md.splitlines()[i + 2]
    if bb != "-":
        c = [x.strip() for x in bb.strip("|").split("|")]
        L.append(f"| change continues after D (scripted) | balloon 8 px/s, delivery 1.5 s | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
    # model rows
    if mc:
        for w in (1, 2):
            L.append(f"| content known before E (model) | fixed-layout notification, window {w} s, E 8 s | "
                     f"{pair(mc[(f'fixed_w{w}', 'prepared')])} (not available: {not_available(mc[(f'fixed_w{w}', 'prepared')])}) | "
                     f"deliberating vlm_direct: {pair(mc[(f'fixed_w{w}', 'direct')])} | - | - |")
        L.append("| content revealed before D (model) | seeded notification, window 8 s, E 8 s | "
                 f"{pair(mc[('seeded_w8', 'prepared')])} | {pair(mc[('seeded_w8', 'late_bind')])} | - | - |")
    else:
        L.append("| content known before E (model) | fixed-layout notification, windows 1, 2 s | queued (studies/queue_ctrl_20260826.sh) | | | |")
    if v1:
        by_pol = defaultdict(list)
        for (pol, _seed), rs in v1.items():
            by_pol[pol] += rs
        L.append("| kind changes (model, V1) | rotating code kind_change, change 2 s | - | "
                 f"{pair(by_pol['late_bind'])} | - | restart_if_changed {pair(by_pol['restart_if_changed'])}; "
                 f"validate {pair(by_pol['validate'])} |")
    else:
        L.append("| kind changes (model, V1) | rotating code kind_change, change 2 s | - | queued | - | queued |")
    L += ["", "Reading: preparation wins only where the action content exists before the event (fixed layout); it has "
          "wrong or no content once the event reveals the content; late binding needs the content visible before D; "
          "prediction needs the change to continue after D with a known delivery delay; re-decision is needed when "
          "the newer frame changes the KIND of the action, which no binder can absorb."]
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
