"""K2 interval tables (2026-08-27): E -> O -> B -> D -> A per cell, from the traces already on disk (no rerun).

    PYTHONPATH=src .venv/bin/python studies/k2_intervals.py runs/k2_c1 runs/k2_c2 ... -o results/K2_intervals.md

E is the environment's event (harness clock), O the publication of the first frame that shows it, B the begin of the
committing decision, D its commitment, A the receipt of the action; all on the harness clock (monotonic_ns), from the
evaluator (src/chronotope/evaluate.py intervals_ns). The resident core's own stamps are NOT intervals on that clock:
after decide() returns at D the runner publishes the frames that fell due during the deliberation before it calls the
binder (runner.py: lazy catch-up, not charged to the controller), so the device's %globaltimer at commit is later than
the harness's D by that catch-up (~1 s for a 5.2 s deliberation at 60 Hz). Reported as its own column so the gap is
visible; the device quantity that IS an interval is the binder's kernel time (track_device_ns).
"""
from __future__ import annotations
import argparse, re, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "src")
from chronotope.evaluate import evaluate_trace
from chronotope.trace import Trace

KEYS = ("E_O", "O_B", "B_D", "D_A", "E_A")


def cell_of(path: Path) -> str:
    return re.sub(r"_s\d+", "", path.stem)   # v0_frozen_s1000 -> v0_frozen; w16_s1000_live -> w16_live


def fmt(xs: list[int]) -> str:
    if not xs:
        return "-"
    return f"{statistics.mean(xs) / 1e6:.1f} / {statistics.median(xs) / 1e6:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    lines = ["# K2 intervals: E → O → B → D → A on the software track (resident agent)", "",
             "Per cell, over valid trials with a committed action: mean / median in ms unless marked. E→O = publication of the "
             "first frame showing the event; O→B = that frame's wait for the committing decision to begin (restart cells: after "
             "the restarts); B→D = the committing decision; D→A = delivery; E→A = the whole chain; bind (device µs) = the "
             "binder kernel's own time; catch-up = the runner's frame catch-up between the harness's D and the core's commit, "
             "wall time not on the harness clock (see the note above). SR is the receipt-valid rate over all trials of the cell.", ""]
    for run in a.runs:
        run = Path(run)
        per: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        n_all: dict[str, int] = defaultdict(int)
        sr: dict[str, int] = defaultdict(int)
        for p in sorted(run.glob("*.jsonl")):
            if p.name == "trials.jsonl":
                continue
            cell = cell_of(p)
            n_all[cell] += 1
            t = Trace.load(p)
            r = evaluate_trace(t)
            if not r.valid_trial or not r.action_committed:
                continue
            sr[cell] += bool(r.valid_at_receipt)
            iv = r.intervals_ns or {}
            for k in KEYS:
                if iv.get(k) is not None:
                    per[cell][k].append(int(iv[k]))
            corr = t.first("coordinate_correction") or {}
            com = t.first("commit") or {}
            if corr.get("track_device_ns") is not None:
                per[cell]["bind_dev"].append(int(corr["track_device_ns"]))
            if corr.get("device_commit_host_ns") is not None and com.get("commitment_time_ns") is not None:
                per[cell]["catchup"].append(int(corr["device_commit_host_ns"]) - int(com["commitment_time_ns"]))
        lines += [f"## {run.name}", "",
                  "| cell | n | SR | E→O | O→B | B→D | D→A | E→A | bind (device µs) | catch-up (wall ms) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for cell in sorted(n_all):
            v = per[cell]
            us = v["bind_dev"]
            bind = f"{statistics.mean(us) / 1e3:.0f} / {statistics.median(us) / 1e3:.0f}" if us else "-"
            lines.append(f"| {cell} | {n_all[cell]} | {sr[cell]}/{n_all[cell]} | {fmt(v['E_O'])} | {fmt(v['O_B'])} | "
                         f"{fmt(v['B_D'])} | {fmt(v['D_A'])} | {fmt(v['E_A'])} | {bind} | {fmt(v['catchup'])} |")
        lines.append("")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
