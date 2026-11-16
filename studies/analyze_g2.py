"""G2 analysis (cards/G2-live-page.md): step 1 per policy under Live and Replay with Wilson CIs and mean G; the frozen
counterfactual vs late_bind paired exact test (same trace: B, D, A shared); the independent frozen run vs the
counterfactual on the same seeds; the two-step cells (SR = completed, G = minus seconds to completion); the O boundary
(achieved screenshot rate) and the application-side receipt latency A − submit (H-G2-3); page hit-test vs DOM geometry.

  PYTHONPATH=src .venv/bin/python studies/analyze_g2.py runs/g2_live_qwen38_nothink_100 results/G2_live_page_100.md
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.live_page import evaluate_live_two_step  # noqa: E402
from chronotope.scenario import manifest_from_dict  # noqa: E402
from chronotope.stats import paired_difference_ci, paired_exact, wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NAME = re.compile(r"(live|replay)_(step1|two_step)_(frozen|late_bind|restart_if_moved|chronotope)_s(\d+)\.jsonl$")


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


def main(run_dir: Path, out: Path) -> int:
    step1 = defaultdict(dict)     # (policy, cond) -> seed -> TrialResult
    two = defaultdict(dict)       # policy -> seed -> LiveTwoStepResult
    hz, lat, lost, agree, n_clicks_checked, model, sched = [], [], 0, 0, 0, None, None
    for p in sorted(run_dir.glob("*.jsonl")):
        mm = NAME.search(p.name)
        if not mm:
            continue
        cond, cell, pol, seed = mm.group(1), mm.group(2), mm.group(3), int(mm.group(4))
        t = Trace.load(p)
        h = t.header()
        model = model or h["spec"]["controller"]["model_id"]
        sched = sched or h["manifest"]["frame_period_ns"]
        tr = t.first("scenario_trailer")
        if cond == "live" and tr is not None:
            if tr.get("capture", {}).get("achieved_hz"):
                hz.append(tr["capture"]["achieved_hz"])
            subs = {d["action_id"]: d["submit_time_ns"] for d in t.all("delivery") if d["attempt"] == 1}
            for r in t.all("receipt"):
                if r["attempt"] == 1 and r["action_id"] in subs:
                    lat.append(r["receipt_time_ns"] - subs[r["action_id"]])
            lost += len(t.all("delivery_failed"))
            m = manifest_from_dict(h["manifest"]).with_trailer(tr)
            for c in t.all("commit"):
                r = next((x for x in t.all("receipt") if x["action_id"] == c["action_id"]), None)
                if r is not None:
                    n_clicks_checked += 1
                    agree += int(bool(r["valid_at_receipt"]) == m.geometry_valid_at(r["task_time_at_receipt_ns"], c["action"]))
        if cell == "step1":
            step1[(pol, cond)][seed] = evaluate_trace(t)
        else:
            two[pol][seed] = evaluate_live_two_step(t)
    L = [f"# G2 live Chromium page — {model}", "",
         f"Scenario `live_page` (site/live_page.html in headless Chromium, 640x480), screenshots scheduled every {sched/1e6:.1f} ms through the O "
         "boundary (PNG), clicks through the browser's input path (Playwright), events and geometry from the page's own log (E = the banner "
         "insert the page reports; boxes = DOM rects); A = the page's pointerdown timestamp mapped to the harness clock. Paused is not "
         "available on a live page (the page cannot hold its state without instrumentation); Replay replays the recorded screenshot stream "
         "and holds the response-triggering state valid. Every table leads with (SR, mean G): step 1 G = minus the click error in px at "
         "receipt (minus the display diagonal when no valid click); two-step G = minus the seconds from the reveal to the page's completion "
         "(minus the window when never completed).", ""]
    if hz:
        L.append(f"O boundary: achieved screenshot rate median {statistics.median(hz):.1f} Hz (min {min(hz):.1f}, max {max(hz):.1f}) over {len(hz)} live trials "
                 f"(scheduled {1e9/sched:.1f} Hz).")
    if lat:
        L.append(f"Application-side receipt latency A − submit (page pointerdown minus harness commit), {len(lat)} deliveries: median {statistics.median(lat)/1e6:.2f} ms, "
                 f"p10 {q(lat, 0.1)/1e6:.2f} ms, p90 {q(lat, 0.9)/1e6:.2f} ms, max {max(lat)/1e6:.2f} ms; lost deliveries {lost}.")
    if n_clicks_checked:
        L.append(f"Page hit-test agrees with the DOM geometry at receipt on {agree}/{n_clicks_checked} received clicks.")
    L += ["", "## Step 1: the banner reflows the page during deliberation", "",
          "| cond | policy | n | SR | Wilson 95% | mean G (px) | mean err (px) | restarts/trial | B→D (s) | first B→D (s) | D→A (ms) | outdated_at_commitment | semantic_error | invalid trials |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond in ("live", "replay"):
        for pol in ("frozen", "late_bind", "restart_if_moved"):
            rs = step1.get((pol, cond), {})
            if not rs:
                continue
            v = [r for r in rs.values() if r.valid_trial]; n = len(v)
            k = sum(bool(r.valid_at_receipt) for r in v)
            lo, _, hi = wilson(k, n) if n else (0, 0, 0)
            errs = [r.spatial_error_px for r in v if r.spatial_error_px is not None]
            das = [r.intervals_ns.get("D_A") for r in v if r.intervals_ns.get("D_A") is not None]
            L.append(f"| {cond} | {pol} | {n} | {k/n:.2f} | [{lo:.2f}, {hi:.2f}] | {statistics.mean(r.return_G for r in v):.1f} | "
                     f"{(statistics.mean(errs) if errs else float('nan')):.1f} | {statistics.mean(r.n_restarts for r in v):.1f} | "
                     f"{statistics.median((r.b_to_d_ns or 0)/1e9 for r in v):.2f} | {statistics.median((r.first_b_to_d_ns or 0)/1e9 for r in v):.2f} | "
                     f"{(statistics.median(das)/1e6 if das else float('nan')):.1f} | {sum('outdated_at_commitment' in r.labels for r in v)} | "
                     f"{sum('semantic_error' in r.labels for r in v)} | {len(rs) - n} |")
    L += ["", "Replay holds the pre-banner state valid until the action arrives, so under Replay the frozen click is right and a click "
          "re-bound to the newest frame (late_bind, restart) is wrong by construction; the Replay–Live reversal is the diagnosis, not a "
          "controller failure.", "",
          "## Same-decision comparison (Live): frozen counterfactual vs late_bind (B, D, A shared exactly)", "",
          "| n pairs | frozen SR | late_bind SR | late_bind wins | frozen wins | exact McNemar p | Δ SR (bootstrap 95% CI) |", "|---|---|---|---|---|---|---|"]
    rs = {s: r for s, r in step1.get(("late_bind", "live"), {}).items() if r.valid_trial and r.proposed_valid_at_receipt is not None}
    if rs:
        seeds = sorted(rs)
        a = [bool(rs[s].proposed_valid_at_receipt) for s in seeds]; b = [bool(rs[s].valid_at_receipt) for s in seeds]
        ex = paired_exact(a, b); lo, mid, hi = paired_difference_ci(a, b)
        L.append(f"| {len(seeds)} | {sum(a)/len(a):.2f} | {sum(b)/len(b):.2f} | {ex['b_only']} | {ex['a_only']} | {ex['p_two_sided']:.3g} | {mid:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    fr, lb = step1.get(("frozen", "live"), {}), step1.get(("late_bind", "live"), {})
    common = [s for s in fr if s in lb and fr[s].valid_trial and lb[s].valid_trial and lb[s].proposed_valid_at_receipt is not None]
    if common:
        ag = sum(bool(fr[s].valid_at_receipt) == bool(lb[s].proposed_valid_at_receipt) for s in common)
        L += ["", f"Independent frozen run vs the counterfactual (same seeds): agree on {ag}/{len(common)} seeds."]
    fv = [r for r in fr.values() if r.valid_trial and r.spatial_error_px is not None]
    if fv:
        # the error law: the frozen click misses by the reflow displacement (needs the per-seed dy from the traces)
        devs = []
        for s, r in fr.items():
            p = run_dir / f"live_step1_frozen_s{s}.jsonl"
            t = Trace.load(p); tr = t.first("scenario_trailer")
            if r.valid_trial and r.spatial_error_px is not None and tr and tr["measured"]["button_after"]:
                dy = tr["measured"]["button_after"][1] - tr["measured"]["button_before"][1]
                devs.append(abs(r.spatial_error_px - dy))
        if devs:
            L.append(f"Frozen error vs reflow displacement: |err − dy| median {statistics.median(devs):.1f} px, max {max(devs):.1f} px over {len(devs)} seeds.")
    if two:
        L += ["", "## Two-step: the dialog is placed from the first click and slides while the agent deliberates", "",
              "| policy | n | SR (completed) | Wilson 95% | mean G (−s to completion) | median s | clicks/trial | hits/trial | decisions/trial | step-2 B→D (s) | dialog moved at step 2 (px) | step-2 outdated_at_commitment | step-1 outdated | invalid |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for pol in ("frozen", "late_bind", "chronotope"):
            rs = two.get(pol, {})
            if not rs:
                continue
            v = [r for r in rs.values() if r.valid_trial]; n = len(v)
            k = sum(r.completed for r in v)
            lo, _, hi = wilson(k, n) if n else (0, 0, 0)
            secs = [r.seconds_to_completion for r in v if r.seconds_to_completion is not None]
            mv = [r.dialog_moved_px for r in v if r.dialog_moved_px is not None]
            b2 = [r.step2_b_to_d_ns for r in v if r.step2_b_to_d_ns is not None]
            L.append(f"| {pol} | {n} | {k/n:.2f} | [{lo:.2f}, {hi:.2f}] | {statistics.mean(r.return_G for r in v):.2f} | "
                     f"{(statistics.median(secs) if secs else float('nan')):.2f} | {statistics.mean(r.n_clicks for r in v):.1f} | {statistics.mean(r.n_hits for r in v):.1f} | "
                     f"{statistics.mean(r.n_decisions for r in v):.1f} | {(statistics.median(b2)/1e9 if b2 else float('nan')):.2f} | "
                     f"{(statistics.mean(mv) if mv else float('nan')):.1f} | {sum('outdated_at_commitment' in r.step2_labels for r in v)} | "
                     f"{sum('outdated_at_commitment' in r.step1_labels for r in v)} | {len(rs) - n} |")
        L += ["", "Step-2 labels are those of the first click decided on a frame that showed the dialog. A duplicate Continue click decided "
              "on the reflowed frame the first click preceded counts as a step-1 click (memory none re-clicks; harness_pending suppresses it)."]
    out.write_text("\n".join(L) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
