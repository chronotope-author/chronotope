"""Rebuild Section 5 evidence from complete, immutable historical cohorts.

No model calls, no invented latency shifts, and no changes to source traces.
The deadline curve re-scores fixed recorded executions, not new window trials.
Run: .venv/bin/python studies/section5_results.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronotope.evaluate import evaluate_trace
from chronotope.stats import paired_exact, wilson
from chronotope.trace import Trace

OUT = ROOT / "results/section5"
PAPER = ROOT / "paper_chronotope_iclr2027"
POLICIES = {"frozen": "Keep", "restart_if_moved": "Restart", "late_bind": "Update"}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def deadline_count(rows, deadline_ns):
    """Fixed-execution deadline budget; failures/missing receipts stay in n."""
    return sum(r["success"] and r["E_A_ns"] is not None and r["E_A_ns"] < deadline_ns for r in rows)


def boundary(rows, fraction):
    """Smallest integer-ns budget reaching a fraction of ALL trials, or None."""
    receipts = sorted(r["E_A_ns"] for r in rows if r["success"] and r["E_A_ns"] is not None)
    rank = math.ceil(fraction * len(rows))
    return receipts[rank - 1] + 1 if len(receipts) >= rank else None


def load_cell(run, pattern, *, seeds=range(1000, 1100)):
    paths = sorted((ROOT / "runs" / run).glob(pattern))
    rows = []
    for path in paths:
        t = Trace.load(path)
        r = evaluate_trace(t)
        if not r.valid_trial or r.contract_violations or t.all("decision_failed"):
            raise ValueError(f"Invalid source trace: {path}: {r.to_dict()}")
        spec = t.header()["spec"]
        invocations = t.all("decision_invocation")
        have_tokens = all(type(v.get(k)) is int for v in invocations for k in ("input_tokens", "output_tokens"))
        corrections = t.all("coordinate_correction")
        row = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": spec["scenario"]["scenario_seed"],
            "success": bool(r.valid_at_receipt), "return": r.return_G,
            "proposed_success": r.proposed_valid_at_receipt,
            "calls": len(invocations),
            "input_tokens": sum(v["input_tokens"] for v in invocations) if have_tokens else None,
            "output_tokens": sum(v["output_tokens"] for v in invocations) if have_tokens else None,
            "first_B_D_ns": r.first_b_to_d_ns,
            "update_wall_ns": sum(c["binder_end_ns"] - c["binder_start_ns"] for c in corrections)
                if corrections and all("binder_end_ns" in c for c in corrections) else None,
            "update_device_ns": sum(c["track_device_ns"] for c in corrections)
                if corrections and all("track_device_ns" in c for c in corrections) else None,
            **{k + "_ns": r.intervals_ns.get(k) for k in ("E_O", "O_D", "D_A", "E_A")},
        }
        rows.append(row)
    expected = set(seeds)
    if len(rows) != len(expected) or {r["seed"] for r in rows} != expected:
        raise ValueError(f"Incomplete/duplicate cohort: {run}/{pattern}, n={len(rows)}")
    return rows


def summary(rows):
    k, n = sum(r["success"] for r in rows), len(rows)
    _, lo, hi = wilson(k, n)
    out = dict(n=n, successes=k, wilson_low=lo, wilson_high=hi)
    for key in ("calls", "input_tokens", "output_tokens", "first_B_D_ns", "E_O_ns", "O_D_ns", "D_A_ns", "E_A_ns", "update_wall_ns", "update_device_ns"):
        vals = [r[key] for r in rows if r[key] is not None]
        out[key + "_mean"] = mean(vals) if vals else None
        out[key + "_median"] = median(vals) if vals else None
        out[key + "_n"] = len(vals)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generated = PAPER / "generated"
    generated.mkdir(exist_ok=True)
    cells = {}
    for task, run, tag in [
        ("Jump", "c2_qwen38_nothink_100", "jump2s"),
        ("Drift8", "c1_qwen38_nothink_100", "v8"),
        ("Select", "c3_qwen38_nothink_100", "v8"),
        ("Button", "g1_reflow_qwen38_nothink_100", "ch2"),
        ("Drift1", "c1b_qwen38_tau6_100", "v1"),
        ("Drift2", "c1b_qwen38_tau6_100", "v2"),
        ("Drift4", "c1b_qwen38_tau6_100", "v4"),
    ]:
        for policy in POLICIES:
            # The slow-speed cohort contains keep/restart, while updates were run separately.
            if task in ("Drift1", "Drift2", "Drift4") and policy == "late_bind":
                continue
            cells[f"{task}/{POLICIES[policy]}"] = load_cell(run, f"live_{tag}_{policy}_s*.jsonl")
    for policy in ("frozen", "restart_if_changed", "late_bind"):
        name = {"frozen": "Keep", "restart_if_changed": "Restart", "late_bind": "Update"}[policy]
        cells[f"Code/{name}"] = load_cell("c4_qwen38_nothink_100", f"live_ch2_{policy}_s*.jsonl")
    for speed in (4, 8, 16):
        cells[f"SameTime{speed}/Update"] = load_cell("c1_qwen38_nothink_100", f"live_v{speed}_late_bind_s*.jsonl")
    for policy in ("frozen", "late_bind"):
        cells[f"UITARS/{POLICIES[policy]}"] = load_cell("c1_uitars_native_100", f"live_v8_{policy}_s*.jsonl")

    prl = [
        ("Notification4", "r1_qwen38_notification_100", "eval_w4000000000", range(1000, 1100)),
        ("Notification8", "r1_qwen38_notification_100", "eval_w8000000000", range(1000, 1100)),
        ("Balloon8", "c_prl_qwen38_nothink_100", "prl_c1v8", range(1000, 1100)),
        ("Trading2", "b1_qwen38_nothink_100", "prl_r2000", range(200, 300)),
        ("S3Notification16", "m4_agents3_evocua_8b_notification_100", "eval_w16000000000", range(200, 300)),
    ]
    for name, run, tag, seeds in prl:
        for mode in ("paused", "replay", "live"):
            cells[f"{name}/{mode}"] = load_cell(run, f"{tag}_s*_{mode}.jsonl", seeds=seeds)
    for v in (8, 100, 200):
        for policy in ("late_bind_cuda", "late_bind_tracker_py", "frozen_cuda"):
            cells[f"Resident{v}/{policy}"] = load_cell("k2_k1c", f"v{v}_{policy}_s*.jsonl")

    summaries = {name: summary(rows) for name, rows in cells.items()}
    write_csv(OUT / "cells.csv", [{"cell": name, **s} for name, s in summaries.items()])
    write_csv(OUT / "trials.csv", [{"cell": name, **row} for name, rows in cells.items() for row in rows])
    boundaries = {p: {f"W{int(q*100)}_ns": boundary(cells[f"Jump/{p}"], q) for q in (.5, .9, .95, 1)} for p in POLICIES.values()}
    (OUT / "boundaries.json").write_text(json.dumps(boundaries, indent=2) + "\n")
    comparisons = {}
    for v in (4, 8, 16):
        rows = cells[f"SameTime{v}/Update"]
        if any(r["proposed_success"] is None for r in rows):
            raise ValueError("Missing same-decision counterfactual")
        comparisons[f"same_time_v{v}"] = paired_exact([r["proposed_success"] for r in rows], [r["success"] for r in rows])
    for v in (8, 100, 200):
        a = sorted(cells[f"Resident{v}/late_bind_tracker_py"], key=lambda r: r["seed"])
        b = sorted(cells[f"Resident{v}/late_bind_cuda"], key=lambda r: r["seed"])
        comparisons[f"resident_v{v}"] = paired_exact([r["success"] for r in a], [r["success"] for r in b])
    (OUT / "paired.json").write_text(json.dumps(comparisons, indent=2) + "\n")

    budgets = sorted(set(range(0, 16_000_000_001, 10_000_000)) | {
        r["E_A_ns"] + d for p in POLICIES.values() for r in cells[f"Jump/{p}"]
        if r["E_A_ns"] is not None for d in (0, 1)})
    curve = []
    for p in POLICIES.values():
        for w in budgets:
            rows = cells[f"Jump/{p}"]
            k = deadline_count(rows, w)
            rate, lo, hi = wilson(k, len(rows))
            curve.append(dict(policy=p, budget_ns=w, n=len(rows), successes=k, rate=rate, lo=lo, hi=hi))
    write_csv(OUT / "deadline_curve.csv", curve)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), layout="constrained")
    colors = {"Keep": "#707070", "Restart": "#d56e23", "Update": "#23799d"}
    for p in ("Update", "Restart", "Keep"):
        rows = [r for r in curve if r["policy"] == p]
        x = [r["budget_ns"] / 1e9 for r in rows]
        axes[0].step(x, [100*r["rate"] for r in rows], where="post", label=p, color=colors[p], lw=1.8)
        axes[0].fill_between(x, [100*r["lo"] for r in rows], [100*r["hi"] for r in rows], step="post", color=colors[p], alpha=.10)
    axes[0].set(title="(a) Recorded actions by time budget", xlabel="Budget from event to receipt (s)", ylabel="Successful trials (%)", xlim=(0,16), ylim=(-3,104))
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    for policy, label, color in [("late_bind_cuda", "GPU update", colors["Update"]), ("late_bind_tracker_py", "CPU update", colors["Restart"]), ("frozen_cuda", "Keep", colors["Keep"])]:
        x = [8, 100, 200]
        ss = [summaries[f"Resident{v}/{policy}"] for v in x]
        y = [s["successes"] for s in ss]
        axes[1].errorbar(x, y, yerr=[[max(0, a-100*s["wilson_low"]) for a,s in zip(y,ss)], [max(0, 100*s["wilson_high"]-a) for a,s in zip(y,ss)]], fmt="o-", color=color, label=label, capsize=2, markersize=4)
    axes[1].set(title="(b) Measured control-path boundary", xlabel="Balloon speed (pixels/s)", xlim=(0,210), ylim=(-3,104))
    axes[1].legend(frameon=False, fontsize=8, loc="center left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    fig.savefig(PAPER / "figures/section5_deadline_and_control.pdf")
    fig.savefig(OUT / "deadline_and_control.png", dpi=180)
    plt.close(fig)

    efficiency = []
    for task, label in [("Jump", "Balloon Jump"), ("Drift8", "Balloon Drift, 8 pixels/s"), ("Code", "Authenticator Code")]:
        for p in POLICIES.values():
            s = summaries[f"{task}/{p}"]
            efficiency.append(f"{label} & {p} & {s['successes']}/100 & {s['calls_mean']:.0f} & {s['input_tokens_mean']:.0f}+{s['output_tokens_mean']:.0f} & {s['first_B_D_ns_mean']/1e9:.2f} & {s['E_A_ns_mean']/1e9:.2f} " + r"\\")
    (generated / "section5_efficiency_rows.tex").write_text("% Generated by studies/section5_results.py; input + output tokens.\n" + "\n".join(efficiency) + "\n")
    prl_rows = []
    for name, label in [("Notification4", "Notification, 4 s"), ("Notification8", "Notification, 8 s"), ("Balloon8", "Balloon, 8 pixels/s"), ("Trading2", "Trading, reversal at 2 s"), ("S3Notification16", "Agent S3: Notification, 16 s")]:
        counts = " & ".join(f"{summaries[f'{name}/{mode}']['successes']}/100" for mode in ("paused", "replay", "live"))
        prl_rows.append(f"{label} & {counts} " + r"\\")
    (generated / "section5_prl_rows.tex").write_text("% Generated by studies/section5_results.py\n" + "\n".join(prl_rows) + "\n")
    resident_rows = []
    for v in (8, 100, 200):
        for policy, label in [("frozen_cuda", "Keep"), ("late_bind_tracker_py", "Update / host"), ("late_bind_cuda", "Update / CUDA")]:
            s = summaries[f"Resident{v}/{policy}"]
            resident_rows.append(f"{v} & {label} & {s['successes']}/100 & {s['E_A_ns_mean']/1e6:.2f} & {s['update_wall_ns_median']/1e6:.2f} " + r"\\")
    (generated / "section5_resident_rows.tex").write_text("% Generated by studies/section5_results.py\n" + "\n".join(resident_rows) + "\n")
    provenance = {
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256((ROOT / "src/chronotope/evaluate.py").read_bytes()).hexdigest(),
        "unique_trace_count": len({r["path"] for rows in cells.values() for r in rows}),
        "source_trace_hashes": "trials.csv", "cells": list(cells),
        "curve_semantics": "Post-hoc deadline scoring of fixed 16-second Jump executions. Not a rerun at changed windows; no inference about an adaptive controller under a different observation stream.",
        "resident_semantics": "Historical detector planner plus 1-second emulated deliberation, fixed 30-ms publication and 20-ms delivery; shared GPU. Not native Agent S3 or real-model latency.",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"traces": provenance["unique_trace_count"], "boundaries": boundaries,
                      "jump_at_6s": {p: deadline_count(cells[f'Jump/{p}'], 6_000_000_000) for p in POLICIES.values()},
                      "outputs": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
