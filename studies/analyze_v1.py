"""V1 analysis (cards/V1-action-kind-validation.md): the player windows with the recorded arm (A1's run) beside the
validate arm (P/R/L, (SR, mean G_A) first), validator precision on the controls (discards must be 0), the
discard/outdated-wait identity on the player (a discard iff the evaluator labels the wait outdated), and the
rotating-code kind_change cells (late_bind / validate / restart_if_changed).

  PYTHONPATH=src .venv/bin/python studies/analyze_v1.py runs/v1_qwen38_nothink_100 results/V1_qwen38_nothink_100.md \\
      [--recorded runs/a1_qwen38_nothink_100]
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

PRL = re.compile(r"prl_(w\d+)_s(\d+)_(paused|replay|live)\.jsonl$")
CODE = re.compile(r"live_code_kc2_(late_bind|validate|restart_if_changed)_s(\d+)\.jsonl$")
CTRL = re.compile(r"live_ctrl_(player_static|notification)_s(\d+)\.jsonl$")


def discards(t: Trace) -> int:
    return sum(1 for e in t.all("pending_event") if e.get("event") == "proposal_discarded")


def load_prl(run: Path) -> dict:
    cells = defaultdict(dict)
    for p in sorted(run.glob("prl_*.jsonl")):
        m = PRL.search(p.name)
        if m:
            t = Trace.load(p)
            cells[(m.group(1), m.group(3))][int(m.group(2))] = (evaluate_trace(t), t)
    return cells


def pair(rs) -> str:
    v = [r for r, _ in rs if r.valid_trial]
    if not v:
        return "-"
    k = sum(bool(r.valid_at_receipt) for r in v)
    return f"{k}/{len(v)}, G {statistics.mean(r.return_G for r in v):.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--recorded", type=Path, default=Path("runs/a1_qwen38_nothink_100"))
    a = ap.parse_args()
    val = load_prl(a.run)
    rec = load_prl(a.recorded) if a.recorded.exists() else {}
    model = next((t.header()["spec"]["controller"]["model_id"] for c in val.values() for _, t in c.values()), None)
    L = [f"# V1 action-kind validation — {model}", "",
         "Player = sponsor_skip on the animated player (320x240, seek 1 s, prompt 9); recorded = A1's vlm_direct run on the same "
         "seeds; validate = the same call with validate_kind (a wait is discarded when a frame published before its decision "
         "ended already showed the sponsor; a click whose target is gone at D is discarded; a new decision starts from that "
         "frame). Every cell is (SR, mean G_A) with G_A = minus the seconds of sponsor watched.", "",
         "## Player windows: recorded vs validate, Paused / Replay / Live", "",
         "| window | arm | Paused | Replay | Live | discards/trial (Live) | outdated waits (Live) | median Live E→A (s) |",
         "|---|---|---|---|---|---|---|---|"]
    windows = sorted({w for w, _ in val} | {w for w, _ in rec}, key=lambda s: int(s[1:]))
    for w in windows:
        for arm, cells in (("recorded", rec), ("validate", val)):
            if (w, "live") not in cells:
                continue
            live = list(cells[(w, "live")].values())
            ea = [r.intervals_ns["E_A"] / 1e9 for r, _ in live if r.valid_trial and r.intervals_ns.get("E_A") is not None]
            L.append(f"| {w} | {arm} | {pair(cells.get((w, 'paused'), {}).values())} | {pair(cells.get((w, 'replay'), {}).values())} | "
                     f"{pair(live)} | {statistics.mean(discards(t) for _, t in live):.2f} | "
                     f"{sum('outdated_at_commitment' in r.labels and not r.action_committed for r, _ in live)} | "
                     f"{(f'{statistics.median(ea):.2f}' if ea else '-')} |")
    L += ["", "### Paired Live comparison, recorded vs validate (same seeds)", "", "| window | n | recorded SR | validate SR | validate wins | recorded wins | exact McNemar p |", "|---|---|---|---|---|---|---|"]
    for w in windows:
        if (w, "live") in rec and (w, "live") in val:
            seeds = sorted(set(rec[(w, "live")]) & set(val[(w, "live")]))
            x = [bool(rec[(w, "live")][s][0].valid_at_receipt) for s in seeds]; y = [bool(val[(w, "live")][s][0].valid_at_receipt) for s in seeds]
            if seeds:
                ex = paired_exact(x, y)
                L.append(f"| {w} | {len(seeds)} | {sum(x)/len(x):.2f} | {sum(y)/len(y):.2f} | {ex['b_only']} | {ex['a_only']} | {ex['p_two_sided']:.3g} |")
    # discard / outdated-wait identity on the validate arm
    L += ["", "### Validator recall and precision on the player (validate arm, Live)", ""]
    for w in windows:
        if (w, "live") not in val:
            continue
        n_d = n_out = n_both = 0
        for r, t in val[(w, "live")].values():
            d = discards(t) > 0
            waits = [i for i in t.all("decision_invocation") if i.get("outcome") == "wait"]
            obs = t.all("observation")
            straddled = any(any(o["source_event_id"] == 1 and o["publication_time_ns"] <= i["end_time_ns"] for o in obs) for i in waits)
            n_d += d; n_out += straddled; n_both += d and straddled
        L.append(f"- {w}: trials with a discard {n_d}, trials with a wait that straddled the appearance {n_out}, both {n_both}")
    # controls
    ctrl = defaultdict(list)
    for p in sorted(a.run.glob("live_ctrl_*.jsonl")):
        m = CTRL.search(p.name)
        if m:
            t = Trace.load(p); ctrl[m.group(1)].append((evaluate_trace(t), t))
    L += ["", "## Controls (validate arm, window 8 s): the validator must never discard", "", "| control | n | SR, mean G | discards (total) |", "|---|---|---|---|"]
    for k, rs in ctrl.items():
        L.append(f"| {k} | {len(rs)} | {pair(rs)} | {sum(discards(t) for _, t in rs)} |")
    # rotating code kind_change
    code = defaultdict(list)
    for p in sorted(a.run.glob("live_code_kc2_*.jsonl")):
        m = CODE.search(p.name)
        if m:
            t = Trace.load(p); code[m.group(1)].append((evaluate_trace(t), t))
    L += ["", "## Rotating code, kind change at 2 s (window 16 s, prompt 13)", "",
          "| policy | n | SR | Wilson 95% | mean G (s) | committed click | committed type | discards/trial | restarts/trial | first B→D (s) | outdated_at_commitment | semantic_error | invalid |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pol in ("late_bind", "validate", "restart_if_changed"):
        rs = code.get(pol, [])
        v = [x for x in rs if x[0].valid_trial]
        if not v:
            L.append(f"| {pol} | 0 | - | - | - | - | - | - | - | - | - | - | {len(rs)} |"); continue
        k = sum(bool(r.valid_at_receipt) for r, _ in v); _, lo, hi = wilson(k, len(v))
        kinds = [(t.first("commit") or {}).get("action", {}).get("kind") for _, t in v]
        fbd = [r.first_b_to_d_ns / 1e9 for r, _ in v if r.first_b_to_d_ns is not None]
        L.append(f"| {pol} | {len(v)} | {k/len(v):.2f} | [{lo:.2f}, {hi:.2f}] | {statistics.mean(r.return_G for r, _ in v):.2f} | "
                 f"{kinds.count('click')} | {kinds.count('type')} | {statistics.mean(discards(t) for _, t in v):.2f} | "
                 f"{statistics.mean(r.n_restarts for r, _ in v):.2f} | {(f'{statistics.median(fbd):.2f}' if fbd else '-')} | "
                 f"{sum('outdated_at_commitment' in r.labels for r, _ in v)} | {sum('semantic_error' in r.labels for r, _ in v)} | {len(rs) - len(v)} |")
    a.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
