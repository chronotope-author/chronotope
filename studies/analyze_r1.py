"""Cross-agent Paused/Replay/Live table from one or more WI-002-protocol run directories
(files eval_w<window_ns>_s<seed>_<policy>.jsonl, calib_s<seed>.jsonl). Paired exact tests Replay vs
Live per window; median B->D; agent ordering under Paused vs Live.

  PYTHONPATH=src .venv/bin/python studies/analyze_r1.py results/WI-007_prl_cross_agent.md \
      runs/wi005_notification_p4 runs/wi007_ui_tars_1_5_7b_prl runs/wi007_qwen3_vl_8b_instruct_prl
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

NAME = re.compile(r"eval_w(\d+)_s(\d+)_(paused|replay|live)\.jsonl$")
CAL = re.compile(r"cal_s(\d+)_paused\.jsonl$")


def load(run: Path):
    cells = defaultdict(dict)   # (window_ns, policy) -> seed -> result
    calib = {}
    model = None
    for p in sorted(run.glob("*.jsonl")):
        m = NAME.search(p.name); c = CAL.search(p.name)
        if not (m or c):
            continue
        t = Trace.load(p)
        model = model or t.header()["spec"]["controller"].get("model_id")
        r = evaluate_trace(t)
        if m:
            cells[(int(m.group(1)), m.group(3))][int(m.group(2))] = r
        else:
            calib[int(c.group(1))] = r
    return model, cells, calib


def main(out: Path, runs: list[Path]) -> int:
    L = ["# Paused / Replay / Live across agents (WI-002 protocol, notification skin)", ""]
    agents = []
    for run in runs:
        if not run.exists():
            L.append(f"_{run}: missing_"); continue
        model, cells, calib = load(run)
        agents.append((model, run.name, cells, calib))
    for model, name, cells, calib in agents:
        L += [f"## {model} ({name})", ""]
        if calib:
            k = sum(bool(r.valid_at_receipt) for r in calib.values())
            L.append(f"Paused competence (calibration seeds): {k}/{len(calib)}")
            L.append("")
        L += ["Outcome pair first per mode: SR = valid at receipt; Ḡ = mean task return at receipt (this task: G_C in px).", "",
              "| window (s) | Paused SR | Paused Ḡ | Replay SR | Replay Ḡ | Live SR | Live Ḡ | Replay vs Live (discordant, exact p) | median Live B→D (s) | Live labels |", "|---|---|---|---|---|---|---|---|---|---|"]
        for w in sorted({k[0] for k in cells}):
            row = []
            for pol in ("paused", "replay", "live"):
                rs = [r for r in cells.get((w, pol), {}).values() if r.valid_trial]
                row.append(f"{sum(bool(r.valid_at_receipt) for r in rs)}/{len(rs)}" if rs else "-")
                row.append(f"{statistics.mean(r.return_G for r in rs):+.1f}" if rs else "-")
            rp, lv = cells.get((w, "replay"), {}), cells.get((w, "live"), {})
            common = [s for s in sorted(set(rp) & set(lv)) if rp[s].valid_trial and lv[s].valid_trial]
            if common:
                t = paired_exact([bool(rp[s].valid_at_receipt) for s in common], [bool(lv[s].valid_at_receipt) for s in common])
                pe = f"{t['a_only']}/{t['b_only']}, p = {t['p_two_sided']:.3g}"
            else:
                pe = "-"
            lrs = [r for r in lv.values() if r.valid_trial]
            bd = statistics.median([r.b_to_d_ns / 1e9 for r in lrs if r.b_to_d_ns]) if lrs and any(r.b_to_d_ns for r in lrs) else None
            labs = defaultdict(int)
            for r in lrs:
                for lab in r.labels:
                    labs[lab] += 1
            L.append(f"| {w/1e9:g} | " + " | ".join(row) + f" | {pe} | {('%.2f' % bd) if bd else '-'} | {', '.join(f'{k}×{v}' for k, v in sorted(labs.items())) or '-'} |")
        L.append("")
    if len(agents) > 1:
        L += ["## Ordering of agents: Paused vs Live (success summed over windows)", "", "| agent | Paused total | Live total | median Live B→D (s) |", "|---|---|---|---|"]
        for model, name, cells, calib in agents:
            tot = {pol: sum(bool(r.valid_at_receipt) for (w, p), d in cells.items() if p == pol for r in d.values() if r.valid_trial) for pol in ("paused", "live")}
            bds = [r.b_to_d_ns / 1e9 for (w, p), d in cells.items() if p == "live" for r in d.values() if r.valid_trial and r.b_to_d_ns]
            L.append(f"| {model} | {tot['paused']} | {tot['live']} | {statistics.median(bds):.2f} |" if bds else f"| {model} | {tot['paused']} | {tot['live']} | - |")
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), [Path(x) for x in sys.argv[2:]]))
