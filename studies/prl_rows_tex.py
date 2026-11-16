"""Emit Table 1 (tab:prl-main) rows from an analyze_prl.py results file. Numbers are read from the results file,
which analyze_prl.py derives from the traces; nothing is transcribed by hand.

  .venv/bin/python studies/prl_rows_tex.py results/C_prl_qwen38_nothink_100.md C
  .venv/bin/python studies/prl_rows_tex.py results/B1_qwen38_nothink_100.md B

Prints a group header line and one row per cell in the order given by the results table, using the paper's
cell names. Cells not in CELLS are skipped with a note on stderr (diagnostic cells stay out of Table 1).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CELLS = {
    "C": ("Task C, balloon ($\\bar G$ in px)", {
        "c1v4": "C1 drift 4 px/s", "c1v8": "C1 drift 8 px/s", "c1v16": "C1 drift 16 px/s",
        "c2j2": "C2 jump at 2 s", "c3v8": "C3 select, 8 px/s"}),
    "B": ("Task B, trading ($\\bar G$ in utility)", {
        "rnone": "B no reversal", "r2000": "B reversal at 2 s", "r5200": "B reversal at 5.2 s",
        "r6680": "B reversal at 6.7 s (near $D$)", "r6930s": "B reversal at 6.9 s, 500-ms delivery",
        "r10000": "B reversal at 10 s"}),
}
LABELS = {"late": "late", "outdated_at_commitment": "outdated", "invalidated_after_commitment": "invalidated",
          "missed_evidence": "missed"}


def g(x: str) -> str:
    x = x.strip()
    if x in ("-", "—", ""):
        return "---"
    v = float(x)
    s = f"{v:+.1f}" if abs(v) < 10 else f"{v:+.0f}"
    return "$" + s.replace("+", "+").replace("-", "-") + "$"


def labels(x: str) -> str:
    x = x.strip()
    if x in ("-", ""):
        return "---"
    out = []
    for part in x.split(","):
        name = part.strip().split("×")[0]
        out.append(LABELS.get(name, name.replace("_", " ")))
    return ", ".join(out)


def main(path: str, task: str) -> None:
    header, names = CELLS[task]
    print(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{header}}}}} \\\\")
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\|\s*([\w]+)( \(diagnostic\))?\s*\|" + r"\s*([^|]*)\|" * 11, line)
        if not m:
            continue
        cell, diag = m.group(1), m.group(2)
        f = [m.group(i) for i in range(3, 14)]
        if diag or cell not in names:
            print(f"skip {cell}{diag or ''}", file=sys.stderr)
            continue
        psr, pg, rsr, rg, lsr, lg, _ci, _p, _ret, bd, lab = f
        bd = bd.strip()
        print(f"{names[cell]} & {psr.strip()} & {g(pg)} & {rsr.strip()} & {g(rg)} & {lsr.strip()} & {g(lg)} & "
              f"{bd if bd not in ('-', '') else '---'} & {labels(lab)} \\\\")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
