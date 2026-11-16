"""C3 balloon select analysis with the outcome pair FIRST: per policy SR (valid at A), Wilson 95%, mean G_C; then the
semantic pick at B vs the object under the click at A, the frozen counterfactual, restarts, B→D; paired exact test
frozen-counterfactual vs late_bind; the wrong-pick guard.

  PYTHONPATH=src .venv/bin/python studies/analyze_c3.py runs/c3_qwen38_nothink_100 results/C3_qwen38_nothink_100.md
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.scenario import manifest_from_dict  # noqa: E402
from chronotope.stats import paired_exact, wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"live_v(\d+)_(frozen|restart_if_moved|late_bind)_s(\d+)\.jsonl$")
CAL = re.compile(r"calib_s(\d+)\.jsonl$")


def main(run: Path, out: Path) -> int:
    rows = defaultdict(dict); calib = {}; model = None; v = None
    for p in sorted(run.glob("*.jsonl")):
        m = NAME.search(p.name); c = CAL.search(p.name)
        if not (m or c):
            continue
        t = Trace.load(p); h = t.header(); mf = manifest_from_dict(h["manifest"])
        model = model or h["spec"]["controller"]["model_id"]
        r = evaluate_trace(t)
        if c:
            calib[int(c.group(1))] = r; continue
        v = int(m.group(1))
        cm, rc = t.first("commit"), t.first("receipt")
        clicked = (mf.clicked_balloon(rc["task_time_at_receipt_ns"], cm["action"]) if (cm and rc) else None) or "none"
        d0 = t.committing_decision()
        ob = next((o for o in t.all("observation") if d0 and o["observation_id"] == d0["observation_id"]), None)
        prop = t.first("action_proposed")
        pick = (mf.clicked_balloon(ob["source_time_ns"], prop["action"]) if (ob and prop) else None)
        rows[m.group(2)][int(m.group(3))] = (r, mf.named_color, clicked, pick)
    if not rows:
        print("no traces"); return 1
    L = [f"# C3 balloon select — analysis ({model}; {run.name})", "",
         f"Three balloons, the instruction names one; the named balloon drifts at {v} px/s with crossing distractors; 320×240, 24-px balloons, 16-s window. "
         "Outcome pair first: SR = valid at receipt (the click is inside the NAMED balloon at A); G_C = −min(click error to the named balloon at receipt, 400 px).", ""]
    if calib:
        L += [f"Paused competence gate: {sum(bool(r.valid_at_receipt) for r in calib.values())}/{len(calib)}", ""]
    L += ["| policy | n | SR (valid at A) | Wilson 95% | mean G_C (px) | picked the named balloon at B | clicked named at A | clicked a distractor at A | no balloon | frozen counterfactual SR (same A) | restarts/trial | B→D (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pol in ("frozen", "late_bind", "restart_if_moved"):
        rs = [x for x in rows.get(pol, {}).values() if x[0].valid_trial]; n = len(rs)
        if not n:
            continue
        k = sum(bool(x[0].valid_at_receipt) for x in rs); _, lo, hi = wilson(k, n)
        cf = [x[0].proposed_valid_at_receipt for x in rs if x[0].proposed_valid_at_receipt is not None]
        L.append(f"| {pol} | {n} | {k/n:.2f} | [{lo:.2f}, {hi:.2f}] | {statistics.mean(x[0].return_G for x in rs):.1f} | "
                 f"{sum(x[3] == x[1] for x in rs)/n:.2f} | {sum(x[2] == x[1] for x in rs)/n:.2f} | {sum(x[2] not in (x[1], 'none') for x in rs)/n:.2f} | "
                 f"{sum(x[2] == 'none' for x in rs)/n:.2f} | {(f'{sum(cf)/len(cf):.2f}' if cf else '-')} | "
                 f"{statistics.mean([x[0].n_restarts for x in rs]):.1f} | {statistics.mean([(x[0].b_to_d_ns or 0)/1e9 for x in rs]):.2f} |")
    lb = [x for x in rows.get("late_bind", {}).values() if x[0].valid_trial and x[0].proposed_valid_at_receipt is not None]
    if lb:
        t = paired_exact([bool(x[0].proposed_valid_at_receipt) for x in lb], [bool(x[0].valid_at_receipt) for x in lb])
        L += ["", f"Same-deliberation pairs (late_bind trace, frozen counterfactual at the same A): frozen {t['rate_a']:.2f}, late_bind {t['rate_b']:.2f}; "
              f"{t['b_only']} discordant pairs favour late binding, {t['a_only']} favour frozen (exact McNemar p = {t['p_two_sided']:.3g}); late_bind Ḡ_C = {statistics.mean(x[0].return_G for x in lb):.1f} px."]
        right = [x for x in lb if x[3] == x[1]]; wrong = [x for x in lb if x[3] not in (x[1], None)]
        L.append(f"late_bind: right pick at B = {len(right)} → valid at A {sum(bool(x[0].valid_at_receipt) for x in right)}; wrong pick at B = {len(wrong)} → valid at A "
                 f"{sum(bool(x[0].valid_at_receipt) for x in wrong)} (must be 0).")
    fr = [x for x in rows.get("frozen", {}).values() if x[0].valid_trial and x[2] not in (x[1], "none")]
    if fr:
        L.append("Distractor hits under the initial coordinate: " + ", ".join(f"seed {s} (named {x[1]}, landed on {x[2]})" for s, x in sorted(rows["frozen"].items()) if x[2] not in (x[1], "none")) + ".")
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
