"""Analysis of WI-004 traces: per-cell success with Wilson CIs; frozen-vs-late_bind paired exact test
(the frozen counterfactual comes from the SAME trace, so pairs share B, D, A exactly); bootstrap CI of
the paired difference over seeds; restart cost. Writes results/WI-004_<tag>.md.

  PYTHONPATH=src .venv/bin/python studies/analyze_c1.py runs/wi004_qwen38_nothink results/WI-004_qwen38_nothink.md
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.stats import paired_difference_ci, paired_exact, wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"live_v(\d+)_(frozen|restart_if_moved|late_bind)(_irr)?_s(\d+)\.jsonl$")


def main(run_dir: Path, out: Path) -> int:
    cells = defaultdict(dict)   # (v, policy, irr) -> seed -> result
    hdr_model = None
    for p in sorted(run_dir.glob("live_*.jsonl")):
        m = NAME.search(p.name)
        if not m:
            continue
        t = Trace.load(p)
        hdr_model = hdr_model or t.header()["spec"]["controller"]["model_id"]
        r = evaluate_trace(t)
        cells[(int(m.group(1)), m.group(2), bool(m.group(3)))][int(m.group(4))] = r
    if not cells:
        print("no traces"); return 1
    L = [f"# WI-004 — late binding with a real agent: analysis ({hdr_model}; {run_dir.name})", ""]
    L += ["## Per cell (Live; valid trials; Wilson 95% CI)", "",
          "| v (px/s) | irrelevant motion | policy | n | SR (valid at A) | Wilson 95% | mean G_C (px) | mean err (px) | restarts/trial | B→D (s) | first B→D (s) | frozen counterfactual SR (same A) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        v, pol, irr = key
        rs = [r for r in cells[key].values() if r.valid_trial]
        n = len(rs)
        if not n:
            L.append(f"| {v} | {irr} | {pol} | 0 | - | - | - | - | - | - | - | - |"); continue
        k = sum(bool(r.valid_at_receipt) for r in rs)
        p, lo, hi = wilson(k, n)
        err = [r.spatial_error_px for r in rs if r.spatial_error_px is not None]
        cf = [r.proposed_valid_at_receipt for r in rs if r.proposed_valid_at_receipt is not None]
        g = statistics.mean(r.return_G for r in rs)
        L.append(f"| {v} | {irr} | {pol} | {n} | {p:.2f} | [{lo:.2f}, {hi:.2f}] | {g:.1f} | {(f'{statistics.mean(err):.1f}' if err else '-')} | "
                 f"{statistics.mean([r.n_restarts for r in rs]):.1f} | {statistics.mean([(r.b_to_d_ns or 0)/1e9 for r in rs]):.2f} | "
                 f"{statistics.mean([(r.first_b_to_d_ns or 0)/1e9 for r in rs]):.2f} | {(f'{sum(cf)/len(cf):.2f}' if cf else '-')} |")
    L += ["", "## Same-deliberation comparison: frozen counterfactual vs late_bind (B, D, A shared exactly)", "",
          "Pairs are seeds of the late_bind cell; the frozen outcome is the proposed action scored at the same A.", "",
          "| v (px/s) | irrelevant motion | n pairs | frozen SR | late_bind SR | late_bind wins | frozen wins | exact McNemar p | Δ SR (bootstrap 95% CI over seeds) | late_bind Ḡ_C (px) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        v, pol, irr = key
        if pol != "late_bind":
            continue
        rs = [r for r in cells[key].values() if r.valid_trial and r.proposed_valid_at_receipt is not None]
        if not rs:
            continue
        a = [bool(r.proposed_valid_at_receipt) for r in rs]
        b = [bool(r.valid_at_receipt) for r in rs]
        t = paired_exact(a, b)
        d, lo, hi = paired_difference_ci(a, b, n_boot=5000, seed=0)
        L.append(f"| {v} | {irr} | {t['n_pairs']} | {t['rate_a']:.2f} | {t['rate_b']:.2f} | {t['b_only']} | {t['a_only']} | {t['p_two_sided']:.3g} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | {statistics.mean(r.return_G for r in rs):.1f} |")
    L += ["", "## Independent frozen run vs the counterfactual (same seeds): does the counterfactual agree with a real frozen run?", "",
          "| v (px/s) | frozen run valid | counterfactual valid (from late_bind traces) | seeds disagreeing |", "|---|---|---|---|"]
    for v in sorted({k[0] for k in cells}):
        fr = cells.get((v, "frozen", False), {})
        lb = cells.get((v, "late_bind", False), {})
        common = sorted(set(fr) & set(lb))
        common = [s for s in common if fr[s].valid_trial and lb[s].valid_trial and lb[s].proposed_valid_at_receipt is not None]
        if not common:
            continue
        a = [bool(fr[s].valid_at_receipt) for s in common]
        b = [bool(lb[s].proposed_valid_at_receipt) for s in common]
        L.append(f"| {v} | {sum(a)/len(a):.2f} | {sum(b)/len(b):.2f} | {sum(x != y for x, y in zip(a, b))}/{len(common)} |")
    L += ["", "## Restart cost (restart_if_moved vs late_bind, same seeds)", "",
          "| v (px/s) | restart valid | late_bind valid | mean restarts | restart first B→D (s) | late_bind B→D (s) |", "|---|---|---|---|---|---|"]
    for v in sorted({k[0] for k in cells}):
        rr = [r for r in cells.get((v, "restart_if_moved", False), {}).values() if r.valid_trial]
        lb = [r for r in cells.get((v, "late_bind", False), {}).values() if r.valid_trial]
        if not rr or not lb:
            continue
        L.append(f"| {v} | {sum(bool(r.valid_at_receipt) for r in rr)/len(rr):.2f} | {sum(bool(r.valid_at_receipt) for r in lb)/len(lb):.2f} | "
                 f"{statistics.mean([r.n_restarts for r in rr]):.1f} | {statistics.mean([(r.first_b_to_d_ns or 0)/1e9 for r in rr]):.2f} | "
                 f"{statistics.mean([(r.b_to_d_ns or 0)/1e9 for r in lb]):.2f} |")
    # ---- reviewer-requested analyses -------------------------------------------------------------
    from chronotope.scenario import manifest_from_dict
    # (a) control cells: coordinate identity, not just success
    L += ["", "## Static / irrelevant-motion controls: did the binder change the coordinate?", "",
          "| v (px/s) | irrelevant motion | n | trials with translation (0,0) | max |translation| (px) | min NCC |", "|---|---|---|---|---|---|"]
    for key in sorted(cells):
        v, pol, irr = key
        if pol != "late_bind" or v != 0:
            continue
        tr = []
        for seed in cells[key]:
            t = Trace.load(run_dir / f"live_v{v}_late_bind{'_irr' if irr else ''}_s{seed}.jsonl")
            c = t.first("coordinate_correction")
            if c:
                dx = c.get("translation_x", c["corrected_action"]["x"] - c["proposed_action"]["x"]); dy = c.get("translation_y", c["corrected_action"]["y"] - c["proposed_action"]["y"])
                tr.append((dx, dy, c["match_score"]))
        if tr:
            L.append(f"| {v} | {irr} | {len(tr)} | {sum(1 for d in tr if d[0] == 0 and d[1] == 0)}/{len(tr)} | {max(max(abs(d[0]), abs(d[1])) for d in tr)} | {min(d[2] for d in tr):.3f} |")
    # (b) signed error along the motion axis vs the measurement model  e ≈ v·(t_A − t_O,used)
    L += ["", "## Measurement model: signed error along the motion axis vs v·(t_A − t_O,used)", "",
          "Frozen binds to the frame at B; late_bind to the newest frame at D. Residual = measured − predicted (px); the model's own aiming error is included in the residual.", "",
          "| v (px/s) | policy | n | mean age of the frame used at A (s) | predicted (px) | measured mean (px) | mean residual (px) | max |residual| (px) |", "|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        v, pol, irr = key
        if irr or pol == "restart_if_moved" or v == 0:
            continue
        ages, pred, meas = [], [], []
        for seed, r in cells[key].items():
            if not r.valid_trial:
                continue
            t = Trace.load(run_dir / f"live_v{v}_{pol}_s{seed}.jsonl")
            c = t.first("commit"); rc = t.first("receipt"); d0 = t.committing_decision()
            if not (c and rc and d0):
                continue
            obs = {o["observation_id"]: o for o in t.all("observation")}
            corr = t.first("coordinate_correction")
            src = obs[corr["observation_id"]]["source_time_ns"] if (corr and pol != "frozen") else obs[d0["observation_id"]]["source_time_ns"]
            m = manifest_from_dict(t.header()["manifest"]); tA = rc["task_time_at_receipt_ns"]
            if not m.state_at(tA).target_visible:
                continue
            tx, _ = m.target_box_at(tA).center()
            age = (tA - src) / 1e9; ages.append(age); pred.append(-v * age); meas.append(c["action"]["x"] - tx)
        if ages:
            res = [mv - pv for mv, pv in zip(meas, pred)]
            L.append(f"| {v} | {pol} | {len(ages)} | {statistics.mean(ages):.2f} | {statistics.mean(pred):.1f} | {statistics.mean(meas):.1f} | {statistics.mean(res):+.1f} | {max(abs(x) for x in res):.1f} |")
    # (c) error distributions per cell (not only means)
    L += ["", "## Click-error distribution at A (px; valid trials with a visible target)", "", "| v | policy | n | median | 90th pct | max | trials > 12 px |", "|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        v, pol, irr = key
        errs = sorted(r.spatial_error_px for r in cells[key].values() if r.valid_trial and r.spatial_error_px is not None)
        if errs:
            L.append(f"| {v}{' +irr' if irr else ''} | {pol} | {len(errs)} | {errs[len(errs)//2]:.1f} | {errs[min(len(errs)-1, int(0.9*len(errs)))]:.1f} | {errs[-1]:.1f} | {sum(e > 12 for e in errs)} |")
    # (d) end-to-end vs conditional (thinking-on style runs): late/outdated breakdown per cell
    L += ["", "## Outcome breakdown per cell (labels may overlap)", "", "| v | policy | n | valid | late | outdated_at_commitment | semantic_error | other invalid |", "|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        v, pol, irr = key
        rs = [r for r in cells[key].values() if r.valid_trial]
        if not rs:
            continue
        late = sum("late" in r.labels for r in rs); outd = sum("outdated_at_commitment" in r.labels for r in rs); sem = sum("semantic_error" in r.labels for r in rs)
        ok = sum(bool(r.valid_at_receipt) for r in rs); other = sum((not r.valid_at_receipt) and not (set(r.labels) & {"late", "outdated_at_commitment", "semantic_error"}) for r in rs)
        L.append(f"| {v}{' +irr' if irr else ''} | {pol} | {len(rs)} | {ok} | {late} | {outd} | {sem} | {other} |")
    L += ["", "Paired tests: exact two-sided paired (sign) test on the discordant seeds — under the null that either policy is equally likely to win a discordant pair. Wilson 95% intervals for single proportions.", ""]
    # tracker facts
    nm = n = 0; tms = []
    for p in run_dir.glob("live_*late_bind*.jsonl"):
        for c in Trace.load(p).all("coordinate_correction"):
            n += 1; nm += int(not c["matched"]); tms.append(c["track_ns"] / 1e6)
    if n:
        L += ["", f"Tracker: {n} bindings, {nm} no-match; median track time {statistics.median(tms):.0f} ms (max {max(tms):.0f} ms)."]
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
