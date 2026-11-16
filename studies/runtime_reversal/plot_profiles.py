#!/usr/bin/env python3
"""CPU-only comparison of four already captured Nsight decode diagnostics.

Matrix intervals use a union across all matrix kernel families. The two other
segments are GPU busy minus matrix union and activity span minus GPU busy, so
concurrent streams are never double-counted. No profiling/inference is run.
"""
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
MODELS = ("qwen08", "radixark27b")
BACKENDS = ("native", "vllm")
TITLES = {"qwen08": "Qwen3.5-0.8B · FP16", "radixark27b": "RadixArk Qwen3.8-27B · mixed precision"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile(folder, backend, model):
    paths = {name: folder / name for name in ("launch_plan.json", "capture_result.json", "analysis/summary.json")}
    plan, capture, summary = [read(p) for p in paths.values()]
    assert plan["task"] == "Disappearing Notification" and plan["task_family"] == "notification"
    assert plan["source_job"]["index"] == 7 and plan["source_job"]["phase"] == "evaluation"
    assert plan["application_episodes"] == plan["qualification_trials"] == 0
    assert plan["expected_input_tokens"] == 998
    assert capture["application_episodes"] == capture["qualification_trials"] == 0
    assert capture["model_requests_executed"] == 1
    steps = summary["expected_decode_steps"]
    assert steps == summary["cuda_graph_launch_calls"] == 32
    if backend == "native":
        if capture["state"] != "captured_request":
            paths["recovery_audit.json"] = folder / "recovery_audit.json"
            evidence = read(paths["recovery_audit.json"])
            assert evidence["state"] == "valid_capture_with_shutdown_error"
        else:
            evidence = capture
        scope = evidence["capture_scope"]
        assert scope["started"] and scope["stopped"] and scope["first_request"]
        assert scope["delay_tokens"] == 5 and scope["requested_decode_steps"] == steps
        assert scope["cuda_start_status"] == scope["cuda_stop_status"] == 0
    else:
        paths["provenance_audit.json"] = folder / "provenance_audit.json"
        evidence = read(paths["provenance_audit.json"])
        assert evidence["state"] == "passed"
        assert evidence["observed_generation_nvtx_count"] == evidence["observed_graph_launches"] == steps
        assert sum(r["count"] for r in summary["generation_nvtx_ranges"]) == steps
        assert evidence["observed_prompt_tokens"] == 998
    matrix = summary["kernel_category_partition"]["matrix_kernel_union_ms"]
    busy, span = summary["gpu_busy_union_ms"], summary["gpu_activity_span_ms"]
    assert 0 <= matrix <= busy <= span
    other, uncovered = busy - matrix, span - busy
    assert np.isclose(uncovered, summary["gpu_uncovered_ms"])
    assert np.isclose(matrix + other + uncovered, span)
    row = {"model_key": model, "model": plan["model"], "revision": plan["revision"],
           "precision": plan["precision"], "backend": backend, "case_index": 7,
           "decode_steps": steps, "first_generated_token": 6, "last_generated_token": 37,
           "matrix_union_ms_per_step": matrix / steps,
           "other_gpu_busy_ms_per_step": other / steps,
           "gpu_uncovered_ms_per_step": uncovered / steps,
           "gpu_busy_union_ms_per_step": busy / steps,
           "gpu_activity_span_ms_per_step": span / steps,
           "evaluation_trials": 0, "diagnostic_model_requests": 1,
           "capture_state": capture["state"], "evidence_state": evidence.get("state"),
           "initial_input_token_ids_sha256": plan["initial_input_token_ids_sha256"],
           "summary": str(paths["analysis/summary.json"])}
    hashes = {str(path): digest(path) for path in paths.values()}
    return row, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=ROOT / "runs/runtime_reversal_20261029")
    parser.add_argument("--out", type=Path, default=ROOT / "results/runtime_reversal_20261029")
    args = parser.parse_args()
    rows, inputs = [], {}
    for model in MODELS:
        for backend in BACKENDS:
            row, hashes = load_profile(args.profiles / f"{backend}_{model}_case7", backend, model)
            rows.append(row)
            inputs.update(hashes)
        native, host = rows[-2:]
        assert native["revision"] == host["revision"]
        assert native["precision"] == host["precision"]
        assert native["initial_input_token_ids_sha256"] == host["initial_input_token_ids_sha256"]
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.5), layout="constrained")
    categories = [
        ("matrix_union_ms_per_step", "Matrix kernels (interval union)", "#276fa3"),
        ("other_gpu_busy_ms_per_step", "Other GPU activity outside matrix intervals", "#74b7a2"),
        ("gpu_uncovered_ms_per_step", "Intervals without captured GPU activity", "#d7dce2"),
    ]
    for ax, model in zip(axes, MODELS):
        group = [row for row in rows if row["model_key"] == model]
        x, bottom = np.arange(2), np.zeros(2)
        for key, label, color in categories:
            height = np.array([row[key] for row in group])
            ax.bar(x, height, bottom=bottom, width=.55, label=label, color=color,
                   edgecolor="white", linewidth=.5)
            bottom += height
        for index, row in enumerate(group):
            ax.text(index, row["matrix_union_ms_per_step"] / 2,
                    f"Matrix\n{row['matrix_union_ms_per_step']:.2f} ms", color="white",
                    ha="center", va="center", fontsize=11)
            ax.annotate(f"{row['gpu_activity_span_ms_per_step']:.2f} ms", (index, bottom[index]),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11)
        ax.set(xticks=x, xticklabels=["Legacy C++ / TensorRT", "Stripped Agent S3 / vLLM"],
               ylabel="Captured activity span per decode step (ms)", title=TITLES[model],
               ylim=(0, max(bottom) * 1.16), xlim=(-.6, 1.6))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=1, frameon=False, fontsize=9)
    fig.supxlabel("One recorded Disappearing Notification request per model/runtime · 32 decode steps (tokens 6–37)\nNsight instrumentation; excludes leading/trailing CPU-only time. Diagnostic timing, not evaluation latency.", fontsize=9)
    outputs = []
    for extension in ("png", "svg", "pdf"):
        path = args.out / f"gpu_decode_partition.{extension}"
        fig.savefig(path, dpi=180)
        outputs.append(str(path))
    plt.close(fig)
    csv_path = args.out / "gpu_decode_partition.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = {}
    for model in MODELS:
        native, host = [row for row in rows if row["model_key"] == model]
        comparisons[model] = {key + "_native_minus_host": native[key] - host[key]
            for key in ("matrix_union_ms_per_step", "other_gpu_busy_ms_per_step",
                        "gpu_uncovered_ms_per_step", "gpu_activity_span_ms_per_step")}
    provenance = {"script": str(Path(__file__).resolve()), "script_sha256": digest(__file__),
        "inputs": inputs, "outputs": outputs + [str(csv_path)], "rows": rows,
        "comparisons": comparisons, "plot_inference_calls": 0, "plot_experimental_trials": 0,
        "scope": "One diagnostic request per model/runtime; 32 captured decode steps, tokens6–37.",
        "partition": "Matrix union; GPU busy union minus matrix union; activity span minus GPU busy union.",
        "limitations": ["Diagnostic instrumented intervals, not evaluation request or E-to-A latency.",
            "Captured GPU activity span excludes leading/trailing CPU-only work.",
            "Kernel names define matrix categories; overlapping matrix families counted once.",
            "Matrix intervals can overlap other kernels, which are not added again.",
            "Uncovered intervals do not by themselves identify a CPU, synchronization, or scheduler cause.",
            "One request per configuration is not a repeated-run population estimate."]}
    (args.out / "gpu_decode_partition.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"state": "complete", "figure": outputs[0], "comparisons": comparisons}))


if __name__ == "__main__":
    main()
