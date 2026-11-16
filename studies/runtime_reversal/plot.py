#!/usr/bin/env python3
"""Export CPU-only figures from the fixed existing-evidence analysis snapshot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"legacy_cpp": "#1866a3", "agents3_stripped": "#ca6a18"}
LABELS = {"legacy_cpp": "Legacy C++ / TensorRT", "agents3_stripped": "Stripped Agent S3 / vLLM"}
MODEL_LABELS = {"qwen08": "Qwen3.5-0.8B · FP16", "radixark27b": "RadixArk Qwen3.8-27B · mixed precision"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/runtime_reversal_20261029")
    args = parser.parse_args()
    source = args.out / "analysis.json"
    analysis = json.loads(source.read_text())
    n = len(analysis["cohort_indices"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for ax, model in zip(axes, MODEL_LABELS):
        for arm in LABELS:
            rows = [r for r in analysis["rows"] if (r["model"], r["arm"], r["mode"]) == (model, arm, "live")]
            fit = analysis["models"][model]["live"][arm]["fit"]
            x = np.array([r["output_tokens"] for r in rows])
            y = np.array([r["call_ms"] for r in rows]) / 1000
            ax.scatter(x, y, s=23, alpha=.65, color=COLORS[arm], label=LABELS[arm], edgecolors="none")
            line = np.geomspace(x.min(), x.max(), 100)
            ax.plot(line, (fit["intercept_ms"] + fit["slope_ms_per_additional_output_token"] * line) / 1000,
                    color=COLORS[arm], lw=1.4)
        ax.set(xscale="log", yscale="log", xlabel="Generated output tokens (log scale)",
               ylabel="First request duration, seconds (log scale)", title=MODEL_LABELS[model])
        ax.grid(alpha=.18, which="both")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Matched first requests · Live W16 · {n} cases per model and runtime\nAll completed failures retained; lines are observational Theil–Sen fits", fontsize=11)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.out / f"first_request_duration.{extension}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    table = []
    ys, labels = [], []
    for model_index, model in enumerate(MODEL_LABELS):
        for mode_index, mode in enumerate(("live", "replay", "paused")):
            group = analysis["models"][model][mode]
            native = group["legacy_cpp"]["fit"]["slope_ms_per_additional_output_token"]
            host = group["agents3_stripped"]["fit"]["slope_ms_per_additional_output_token"]
            ratio = native / host
            low, high = group["paired"]["slope_ratio_native_over_host_bootstrap95"]
            y = 6 - model_index * 3.6 - mode_index
            ax.errorbar(ratio, y, xerr=[[ratio - low], [high - ratio]], fmt="o", capsize=4,
                        color="#1866a3" if model_index == 0 else "#713c8b", markersize=6)
            ys.append(y)
            labels.append(("0.8B FP16" if model_index == 0 else "27B mixed") + " / " + mode.title())
            ax.annotate(f"{native:.2f} / {host:.2f} ms per token", (ratio, y),
                        xytext=(9, 5), textcoords="offset points", fontsize=9)
            table.append({"model": analysis["identities"][model]["model"], "condition": mode,
                "cases_per_runtime": n, "native_slope_ms_per_output_token": native,
                "host_slope_ms_per_output_token": host, "slope_ratio_native_over_host": ratio,
                "slope_ratio_bootstrap95_low": low, "slope_ratio_bootstrap95_high": high,
                "paired_median_native_minus_host_request_ms": group["paired"]["median_native_minus_host_call_ms"],
                "all_successes_and_failures_retained": True})
    ax.axvline(1, color="#777777", ls="--", lw=1)
    ax.set(yticks=ys, yticklabels=labels, xlim=(.78, 1.25), ylim=(-.5, 7),
           xlabel="Native / host request-duration slope (lower favors native)",
           title="The reversal remains in Live, Replay and Paused conditions")
    ax.grid(axis="x", alpha=.2)
    fig.supxlabel(f"{n} paired cases per cell · 998 input tokens each · 95% paired bootstrap intervals, 500 resamples\nRequest-duration slopes include server/CPU/GPU work; they do not identify a specific kernel or overhead.", fontsize=8)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.out / f"request_slope_reversal.{extension}", dpi=180)
    plt.close(fig)
    with (args.out / "request_slope_reversal.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (args.out / "plot_provenance.json").write_text(json.dumps({
        "analysis": str(source), "analysis_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases_per_model_arm_condition": n, "first_requests": len(analysis["rows"]),
        "inference_calls": 0, "experimental_trials": 0,
    }, indent=2) + "\n")
    print(f"Exported two figures in PNG/SVG/PDF and six-row CSV to {args.out}")


if __name__ == "__main__":
    main()
