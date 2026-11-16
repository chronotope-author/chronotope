"""Paused / Replay / Live table with the outcome pair FIRST — (SR, mean G) per cell and mode — from a run directory
written by agentlib.Study.triplet (files prl_<cell>_s<seed>_<policy>.jsonl; gate files cal_s<seed>_paused.jsonl).
Paired Replay–Live exact test per cell, Live-to-Replay retention, median Live B→D, labels; then one line per Live
trial (task times of D and A, G, labels) so straddle cells can be read trial by trial.

  PYTHONPATH=src .venv/bin/python studies/analyze_prl.py RUN_DIR OUT.md --title "A1 sponsor skip (agent)" \
      [--cell-order w1,w2,...] [--diagnostic c1v8lb]
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
from chronotope.stats import paired_exact, wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"prl_([a-z0-9]+)_s(\d+)_(paused|replay|live)\.jsonl$")
CAL = re.compile(r"cal_s(\d+)_paused\.jsonl$")


def load(run: Path):
    cells, calib, model, hdr = defaultdict(dict), {}, None, None
    for p in sorted(run.glob("*.jsonl")):
        m = NAME.search(p.name); c = CAL.search(p.name)
        if not (m or c):
            continue
        t = Trace.load(p)
        h = t.header()
        model = model or h["spec"]["controller"].get("model_id")
        hdr = hdr or h
        r = evaluate_trace(t)
        if m:
            cells[(m.group(1), m.group(3))][int(m.group(2))] = (r, t)
        else:
            calib[int(c.group(1))] = r
    return model, cells, calib, hdr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--title", default="Paused / Replay / Live")
    ap.add_argument("--cell-order", default=None, help="comma list; default = sorted")
    ap.add_argument("--diagnostic", default="", help="comma list of cells reported as diagnostics, not results")
    ap.add_argument("--pilot", action="store_true", help="label the file as a 10-seed pilot")
    a = ap.parse_args()
    model, cells, calib, hdr = load(a.run)
    if not cells:
        print("no traces"); return 1
    all_cells = a.cell_order.split(",") if a.cell_order else sorted({k[0] for k in cells})
    diag = set(x for x in a.diagnostic.split(",") if x)
    sc = hdr["spec"]["scenario"]
    L = [f"# {a.title} — {model} ({a.run.name})" + (" — PILOT, 10 seeds: not a citable cell" if a.pilot else ""), "",
         f"Scenario `{sc['scenario_id']}`, skin `{sc['skin']}`, canvas {sc['canvas_w_px']}x{sc['canvas_h_px']}, target {sc['target_size_px']} px, "
         f"event {sc['event_time_ns']/1e9:g} s, {1e9/sc['frame_period_ns']:.0f} Hz frames; publication {hdr['spec']['observation']['publication_delay_ns']/1e6:.0f} ms; "
         f"delivery {hdr['spec']['delivery'].get('delivery_delay_ns', 0)/1e6:.0f} ms; controller `{hdr['spec']['controller']['kind']}`, "
         f"prompt {hdr['spec']['controller']['prompt_version']}. Every cell reports the outcome pair first: SR = valid at receipt, "
         f"mean G = the task return at receipt (contract: logic.md).", ""]
    if calib:
        k = sum(bool(r.valid_at_receipt) for r in calib.values())
        L += [f"Paused competence gate: {k}/{len(calib)} (seeds {min(calib)}..{max(calib)})", ""]
    L += ["## Outcome pair per cell and mode", "",
          "| cell | Paused SR | Paused Ḡ | Replay SR | Replay Ḡ | Live SR | Live Ḡ | Live Wilson 95% | Replay vs Live (discordant, exact p) | Live/Replay retention | median Live B→D (s) | Live labels |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in all_cells:
        row = [cell + (" (diagnostic)" if cell in diag else "")]
        rates = {}
        for pol in ("paused", "replay", "live"):
            rs = [r for r, _ in cells.get((cell, pol), {}).values() if r.valid_trial]
            if rs:
                k = sum(bool(r.valid_at_receipt) for r in rs)
                rates[pol] = k / len(rs)
                row += [f"{k}/{len(rs)}", f"{statistics.mean(r.return_G for r in rs):+.2f}"]
            else:
                row += ["-", "-"]
        lv = [r for r, _ in cells.get((cell, "live"), {}).values() if r.valid_trial]
        if lv:
            k = sum(bool(r.valid_at_receipt) for r in lv); _, lo, hi = wilson(k, len(lv))
            row.append(f"[{lo:.2f}, {hi:.2f}]")
        else:
            row.append("-")
        rp, lvd = cells.get((cell, "replay"), {}), cells.get((cell, "live"), {})
        common = [s for s in sorted(set(rp) & set(lvd)) if rp[s][0].valid_trial and lvd[s][0].valid_trial]
        if common:
            t = paired_exact([bool(rp[s][0].valid_at_receipt) for s in common], [bool(lvd[s][0].valid_at_receipt) for s in common])
            row.append(f"{t['a_only']}/{t['b_only']}, p = {t['p_two_sided']:.3g}")
        else:
            row.append("-")
        row.append(f"{rates['live']/rates['replay']:.2f}" if rates.get("replay") else "-")
        bd = [r.b_to_d_ns / 1e9 for r in lv if r.b_to_d_ns]
        row.append(f"{statistics.median(bd):.2f}" if bd else "-")
        labs = defaultdict(int)
        for r in lv:
            for lab in r.labels:
                labs[lab] += 1
        row.append(", ".join(f"{k}×{v}" for k, v in sorted(labs.items())) or "-")
        L.append("| " + " | ".join(row) + " |")
    inv = sum(1 for d in cells.values() for r, _ in d.values() if not r.valid_trial)
    bad_pairs = 0
    L += ["", f"Invalid trials: {inv}. Pairs: every Replay–Live pair was validated at run time (see RESULTS.md 'pair=' fields).", ""]
    L += ["## Live trials, one line each (task-time D and A in s; G; labels)", ""]
    for cell in all_cells:
        for seed, (r, t) in sorted(cells.get((cell, "live"), {}).items()):
            c = t.first("commit"); rc = t.first("receipt")
            td = f"{c['task_time_at_commit_ns']/1e9:.3f}" if c else "-"
            ta = f"{rc['task_time_at_receipt_ns']/1e9:.3f}" if rc else "-"
            L.append(f"- {cell} s{seed}: valid={r.valid_at_receipt} G={r.return_G:+.2f} D={td} A={ta} "
                     f"B→D={(r.b_to_d_ns or 0)/1e9:.2f} labels={','.join(r.labels) or '-'}"
                     + (f" INVALID:{r.invalid_reason}" if not r.valid_trial else ""))
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
