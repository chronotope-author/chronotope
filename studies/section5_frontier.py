"""Measured multi-model success/latency frontier on one frozen task cohort.

Uses the distributional scorer and its floating-point trajectories. The fixed-
grid evaluator uses a proxy manifest and must not be used for these episodes.
Preserves all configurations, including those with poor Replay competence.
No native Agent S3 or CPU/GPU speedup is inferred from these ablations.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median

from section5_results import ROOT, OUT, PAPER, write_csv
from chronotope.distributional.episode import score_trace, verify_trace_evidence
from chronotope.distributional.manifest import ManifestSet, build_manifest
from chronotope.stats import paired_exact, wilson
from chronotope.trace import Trace

MODELS = [
    ("cpp_qwen35_0p8b", "Qwen3.5-0.8B", .8),
    ("edgellm_fara4b", "Fara1.5-4B", 4),
    ("edgellm_uitars7b", "UI-TARS-1.5-7B", 7),
    ("edgellm_fara9b", "Fara1.5-9B", 9),
    ("edgellm_qwen38_27b", "Qwen3.8-27B", 27),
]


def frontier(points):
    """Nondominated observed (latency, success) points, minimizing latency."""
    best = -1
    result = []
    for p in sorted(points, key=lambda p: (p["latency_p50_s"], -p["successes"])):
        if p["successes"] > best:
            result.append(p)
            best = p["successes"]
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "runs/d1/manifests/moving_target_100.jsonl"
    ms = ManifestSet.load(manifest_path)
    if len(ms.rows) != 100:
        raise ValueError("Expected the full 100-episode manifest")
    trials, points, paired = [], [], {}
    for prefix, label, size in MODELS:
        by_policy = {}
        configs = []
        for suffix, policy in [("frozen", "Keep"), ("late_bind", "Update")]:
            run = ROOT / "runs/d1_100/moving_target" / f"{prefix}_{suffix}"
            config = json.loads((run / "controller.json").read_text())
            configs.append({k: v for k, v in config.items() if k != "at_commit_policy"})
            stored = [json.loads(l) for l in (run / "outcomes.jsonl").read_text().splitlines()]
            stored_map = {(r["episode_id"], r["condition"]): r for r in stored}
            if len(stored_map) != 200 or len(stored) != 200:
                raise ValueError(f"Incomplete or duplicate outcomes: {run}")
            rows_by_mode = {}
            for mode in ("live", "replay"):
                rs = []
                for index, row in enumerate(ms.rows):
                    path = run / f"{row.episode_id}_r0_{mode}.jsonl"
                    t = Trace.load(path)
                    m = build_manifest(row)
                    if t.header()["manifest_hash"] != m.manifest_hash():
                        raise ValueError(f"Scenario mismatch: {path}")
                    if t.header()["spec"]["delivery"] != row.delivery:
                        raise ValueError(f"Delivery mismatch: {path}")
                    # Full event/evidence check on the first episode of every arm;
                    # every episode is re-scored from its own frozen geometry below.
                    if index == 0:
                        verify_trace_evidence(row, m, t, ms.frozen_hash)
                    o = score_trace(row, t, ms.frozen_hash, controller=run.name)
                    saved = stored_map[(row.episode_id, mode)]
                    if o.invalid_trial or o.contract_violations or t.all("decision_failed"):
                        raise ValueError(f"Invalid trial: {path}")
                    for key in ("Y", "C", "R", "L_EA", "row_hash", "manifest_hash"):
                        if getattr(o, key) != saved[key]:
                            raise ValueError(f"Historical scoring drift ({key}): {path}")
                    if mode == "live" and o.Y and (o.A is None or o.A >= o.E + row.window_ns):
                        raise ValueError(f"Success outside response window: {path}")
                    receipt, commit = t.first("receipt"), t.first("commit")
                    if receipt:
                        valid = m.action_valid_at(receipt["task_time_at_receipt_ns"], commit["action"])
                        if valid != receipt["valid_at_receipt"]:
                            raise ValueError(f"Receipt inconsistent with actual distributional manifest: {path}")
                    r = {"model": label, "parameters_B": size, "policy": policy,
                         "condition": mode, "episode_id": row.episode_id, "success": o.Y,
                         "latency_s": o.L_EA / 1e9 if o.L_EA is not None else None,
                         "source": str(path.relative_to(ROOT)),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    trials.append(r)
                    rs.append(r)
                rows_by_mode[mode] = rs
            by_policy[policy] = rows_by_mode
            live = rows_by_mode["live"]
            latencies = sorted(r["latency_s"] for r in live if r["latency_s"] is not None)
            # No latency-conditioned omission: all 100 trials must have receipts.
            if len(latencies) != 100:
                raise ValueError(f"Missing receipt; define censored latency before plotting {run}")
            k = sum(r["success"] for r in live)
            _, lo, hi = wilson(k, 100)
            points.append(dict(model=label, parameters_B=size, policy=policy, n=100, successes=k,
                wilson_low=lo, wilson_high=hi, latency_p50_s=median(latencies),
                latency_p95_s=latencies[94], runtime=config["server"],
                config=str((run / "controller.json").relative_to(ROOT))))
        if configs[0] != configs[1]:
            raise ValueError(f"Keep/update differs beyond policy: {label}")
        competence = sum(r["success"] for r in by_policy["Keep"]["replay"])
        for p in points[-2:]:
            p["keep_replay_successes"] = competence
            p["replay_competent"] = competence >= 80
        paired[label] = paired_exact([r["success"] for r in by_policy["Keep"]["live"]],
                                    [r["success"] for r in by_policy["Update"]["live"]])
    write_csv(OUT / "frontier_points.csv", points)
    write_csv(OUT / "frontier_trials.csv", trials)
    (OUT / "frontier_paired.json").write_text(json.dumps(paired, indent=2) + "\n")
    (OUT / "frontier_provenance.json").write_text(json.dumps({
        "manifest": str(manifest_path.relative_to(ROOT)), "manifest_hash": ms.frozen_hash,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256((ROOT / "src/chronotope/distributional/episode.py").read_bytes()).hexdigest(),
        "source_hashes": "frontier_trials.csv", "traces": len(trials),
        "scoring": "distributional-1: Euclidean distance to continuous target center <= 12 pixels; all successes also verified before the 16-second deadline",
        "competence": "Keep-policy Replay accuracy on the same cohort; descriptive control, not independent Paused qualification",
        "scope": "Within-model keep/update ablation on the same resident core. Across-model points differ in architecture/prompt/runtime. No native Agent S3 comparison; no causal latency acceleration claim.",
    }, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {"Keep": "#777777", "Update": "#167fa0"}
    markers = {"Keep": "s", "Update": "o"}
    for policy in ("Keep", "Update"):
        ps = [p for p in points if p["policy"] == policy]
        for p in ps:
            for ax, x in [(axes[0], p["latency_p50_s"]), (axes[1], p["parameters_B"])]:
                k = p["successes"]
                ax.errorbar(x, k, yerr=[[max(0, k-100*p["wilson_low"])], [max(0, 100*p["wilson_high"]-k)]],
                    marker=markers[policy], color=colors[policy], mfc=colors[policy] if p["replay_competent"] else "white",
                    ms=5, capsize=2, linestyle="none", alpha=1 if p["replay_competent"] else .55)
        fs = frontier([p for p in ps if p["replay_competent"]])
        # Horizontal segments are the best observed point allowed by a median-
        # latency budget, not interpolated models or new experimental outcomes.
        if fs:
            axes[0].step([p["latency_p50_s"] for p in fs] + [10], [p["successes"] for p in fs] + [fs[-1]["successes"]],
                         where="post", color=colors[policy], linestyle="--", alpha=.65, label=policy)
    offsets = {"Fara1.5-4B": (-28,-23), "Fara1.5-9B": (-27,-24), "Qwen3.8-27B": (5,-42),
               "Qwen3.5-0.8B": (6,12), "UI-TARS-1.5-7B": (-22,13)}
    for p in points:
        if p["policy"] == "Update":
            label = p["model"].replace("Fara1.5", "Fara").replace("Qwen3.8", "Qwen").replace("Qwen3.5", "Qwen").replace("UI-TARS-1.5", "UI-TARS")
            axes[0].annotate(label, (p["latency_p50_s"], p["successes"]), xytext=offsets[p["model"]], textcoords="offset points", fontsize=7.3)
    axes[0].set(xlabel="Median event-to-receipt latency (s)", ylabel="Live success (%)", title="(a) Success versus measured latency", xscale="log", xlim=(.4,10.5))
    axes[0].set_xticks([.5,1,2,4,6,10], labels=["0.5","1","2","4","6","10"])
    axes[0].legend(frameon=False, loc="center left", fontsize=8)
    axes[1].set(xlabel="Model parameters (billions)", title="(b) Parameter count", xscale="log", xlim=(.6,35))
    axes[1].set_xticks([.8,4,7,9,27], labels=["0.8","4","7","9","27"])
    for ax in axes:
        ax.set_ylim(-4,106)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    fig.savefig(PAPER / "figures/section5_performance_envelope.pdf")
    fig.savefig(OUT / "performance_envelope.png", dpi=180)
    plt.close(fig)

    lines = []
    for _, label, _ in MODELS:
        keep, update = [p for p in points if p["model"] == label]
        lines.append(f"{label} & {keep['keep_replay_successes']}/100 & {keep['successes']}/100 & {update['successes']}/100 & {keep['latency_p50_s']:.3f} & {update['latency_p50_s']:.3f} " + r"\\")
    (PAPER / "generated/section5_frontier_rows.tex").write_text("% Generated by studies/section5_frontier.py\n" + "\n".join(lines) + "\n")
    print(json.dumps(points, indent=2))


if __name__ == "__main__":
    main()
