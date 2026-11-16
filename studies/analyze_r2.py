"""R2: the fine window sweep against the prediction from the recorded E→A of the reference run (cards/R2-window-sweep.md).
Prediction: a Live trial at window w succeeds iff its E→A at w = 16 s would be < w (deliberation does not depend on w).

  PYTHONPATH=src .venv/bin/python studies/analyze_r2.py runs/r1_qwen38_notification_100 runs/r2_window_sweep_100 results/R2_window_sweep_100.md
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.stats import wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"eval_w(\d+)_s(\d+)_live\.jsonl$")


def load(run: Path) -> dict:
    cells = defaultdict(dict)
    for p in sorted(run.glob("eval_w*_s*_live.jsonl")):
        m = NAME.search(p.name)
        if m:
            cells[int(m.group(1))][int(m.group(2))] = evaluate_trace(Trace.load(p))
    return cells


def main(ref: Path, run: Path, out: Path) -> int:
    ref_cells = load(ref); w_ref = max(ref_cells)
    ea = {s: r.intervals_ns.get("E_A") for s, r in ref_cells[w_ref].items() if r.valid_trial and r.intervals_ns.get("E_A") is not None}
    cells = load(run)
    L = [f"# R2 fine window sweep — reference E→A from {ref.name} at w = {w_ref/1e9:.0f} s (n = {len(ea)}, median {statistics.median(ea.values())/1e9:.2f} s, "
         f"range {min(ea.values())/1e9:.2f}–{max(ea.values())/1e9:.2f} s)", "",
         "| window (s) | n | Live SR | Wilson 95% | predicted SR (E→A < w) | seed-level agreement | median Live E→A (s) | labels |",
         "|---|---|---|---|---|---|---|---|"]
    for w in sorted(cells):
        rs = {s: r for s, r in cells[w].items() if r.valid_trial}; n = len(rs)
        k = sum(bool(r.valid_at_receipt) for r in rs.values()); lo, _, hi = wilson(k, n) if n else (0, 0, 0)
        pred = {s: ea[s] < w for s in rs if s in ea}
        agree = sum(pred[s] == bool(rs[s].valid_at_receipt) for s in pred)
        labs = defaultdict(int)
        for r in rs.values():
            for lab in r.labels:
                labs[lab] += 1
        med = statistics.median((r.intervals_ns.get("E_A") or 0) / 1e9 for r in rs.values()) if n else 0
        L.append(f"| {w/1e9:.1f} | {n} | {k}/{n} | [{lo:.2f}, {hi:.2f}] | {sum(pred.values())}/{len(pred)} | {agree}/{len(pred)} | {med:.2f} | "
                 f"{', '.join(f'{a} ×{c}' for a, c in sorted(labs.items())) or '-'} |")
    out.write_text("\n".join(L) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
