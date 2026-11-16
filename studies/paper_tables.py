"""Emit LaTeX rows for paper.tex result tables, derived from run directories at generation time
(same policy as fig_c1_story.py: numbers are computed from traces, never transcribed).

  PYTHONPATH=src .venv/bin/python studies/paper_tables.py balloon   # tab:results-balloon rows
  PYTHONPATH=src .venv/bin/python studies/paper_tables.py paired    # tab:results-paired rows (R1 PRL)

Balloon G_C convention: G_C = -e_click at A; when the trial's committed click arrives after the
window closed (target no longer exists at receipt) the evaluator has no distance, and the paper's
no-click rule is applied: e_click = the display diagonal. Stated in the table caption.
"""
from __future__ import annotations

import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"

C1_NAME = re.compile(r"live_v(\d+)_(frozen|next_decision_buffer|restart_if_moved|late_bind)(_irr)?_s(\d+)\.jsonl$")
C2_NAME = re.compile(r"live_(jump\w+?)_(frozen|next_decision_buffer|restart_if_moved|late_bind)_s(\d+)\.jsonl$")
R1_NAME = re.compile(r"eval_w(\d+)_s(\d+)_(paused|replay|live)\.jsonl$")

LABEL = {
    "frozen": "Initial coordinate",
    "next_decision_buffer": "Next-decision buffer",
    "restart_if_moved": "Complete restart",
    "late_bind": "Same-decision coordinate update",
}
ORDER = ["frozen", "next_decision_buffer", "restart_if_moved", "late_bind"]


def load_cell(path: Path):
    t = Trace.load(path)
    r = evaluate_trace(t)
    calls = sum('"decision_invocation"' in line for line in open(path))
    h = t.header()["spec"]["scenario"]
    diag = math.hypot(h["canvas_w_px"], h["canvas_h_px"])
    return r, calls, diag


def balloon_rows(run_dir: Path, name_re, jump_main: str | None = None):
    """-> {policy: (n, n_valid, mean_G, mean_calls)} pooled over v>0 (C1) / the during-deliberation jump (C2)."""
    cells = defaultdict(list)
    for p in sorted(run_dir.glob("live_*.jsonl")):
        m = name_re.search(p.name)
        if not m:
            continue
        if name_re is C1_NAME:
            v, pol, irr = int(m.group(1)), m.group(2), bool(m.group(3))
            if irr or v == 0:  # principal cells: moving target only
                continue
        else:
            jtag, pol = m.group(1), m.group(2)
            if jump_main is not None and jtag != jump_main:
                continue
        r, calls, diag = load_cell(p)
        if not r.valid_trial:
            continue
        e = r.spatial_error_px if r.spatial_error_px is not None else diag
        cells[pol].append((bool(r.valid_at_receipt), -e, calls))
    out = {}
    for pol, rows in cells.items():
        out[pol] = (len(rows), sum(v for v, _, _ in rows),
                    statistics.mean(g for _, g, _ in rows),
                    statistics.mean(c for _, _, c in rows))
    return out


def emit_balloon():
    c1 = balloon_rows(RUNS / "c1_qwen38_nothink_100", C1_NAME)
    # C2 file naming: discover the during-deliberation jump tag (the non-control cell)
    tags = defaultdict(int)
    for p in (RUNS / "c2_qwen38_nothink_100").glob("live_*.jsonl"):
        m = C2_NAME.search(p.name)
        if m:
            tags[m.group(1)] += 1
    print(f"% C2 jump tags found: {dict(tags)} — principal cell: jump2s (during deliberation)")
    c2 = balloon_rows(RUNS / "c2_qwen38_nothink_100", C2_NAME, jump_main="jump2s")
    for sub, cells in (("C1: continuous drift", c1), ("C2: single position change", c2)):
        first = True
        for pol in ORDER:
            head = sub if first else ""
            if pol not in cells:
                print(f"{head} & {LABEL[pol]} & \\textsc{{TBD}} & \\textsc{{TBD}} & \\textsc{{TBD}} \\\\")
            else:
                n, k, g, calls = cells[pol]
                print(f"{head} & {LABEL[pol]} & {100.0 * k / n:.0f} ({k}/{n}) & {g:.1f} & {calls:.1f} \\\\")
            first = False
        if sub.startswith("C1"):
            print("\\addlinespace")


def emit_paired():
    run = RUNS / "r1_qwen38_notification_100"
    if not run.exists() or not list(run.glob("eval_*.jsonl")):
        run = RUNS / "wi005_notification_p4"
    print(f"% source: {run.name}")
    cells = defaultdict(dict)  # (window_s, cond) -> seed -> valid
    for p in sorted(run.glob("eval_*.jsonl")):
        m = R1_NAME.search(p.name)
        if not m:
            continue
        r = evaluate_trace(Trace.load(p))
        if r.valid_trial:
            cells[(int(m.group(1)) / 1e9, m.group(3))][int(m.group(2))] = bool(r.valid_at_receipt)
    for w in sorted({k[0] for k in cells}):
        row = []
        for cond in ("paused", "replay", "live"):
            d = cells.get((w, cond), {})
            row.append(f"{sum(d.values())}/{len(d)}" if d else "--")
        rep = cells.get((w, "replay"), {})
        liv = cells.get((w, "live"), {})
        common = set(rep) & set(liv)
        lr = (f"{sum(liv[s] for s in common) / max(1, sum(rep[s] for s in common)):.2f}"
              if common and sum(rep[s] for s in common) else "--")
        wtxt = f"{w:.0f}" if w == int(w) else f"{w}"
        print(f"{wtxt} s & {row[0]} & {row[1]} & {row[2]} & {lr} \\\\")


if __name__ == "__main__":
    {"balloon": emit_balloon, "paired": emit_paired}[sys.argv[1]]()
