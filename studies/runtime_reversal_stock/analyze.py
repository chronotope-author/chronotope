#!/usr/bin/env python3
"""CPU-only analysis of completed pristine-upstream diagnostic captures.

No inference, process control, profiling, or network calls. Refuses a complete
report until all four captures have succeeded and their Nsight exports exist.
Reuses the previous interval-union analysis; never sums overlapping GPU work.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen08", "radixark27b")
BACKENDS = ("native", "vllm")
COMMIT = "bb291453bc10d9bd1142070c605e090d4e483ee2"
EXPERIMENT = ROOT / "runs/agents3_policy_radixark27b_20261028/experiment"
PAUSE = EXPERIMENT / "pauses/20261029T180600_876683Z/journal.json"
METRICS = ("gpu_activity_span_ms_per_step", "gpu_busy_union_ms_per_step",
           "matrix_union_ms_per_step", "other_gpu_busy_ms_per_step", "gpu_uncovered_ms_per_step")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def remember(inputs, *paths):
    for path in paths:
        inputs[str(Path(path).resolve())] = digest(path)


def readiness(profiles):
    rows = []
    for model in MODELS:
        for backend in BACKENDS:
            folder = profiles / f"{backend}_{model}_case7"
            result = profiles / f"native_capture_{model}_result.json" if backend == "native" else folder / "capture_result.json"
            prefix = "native_timeline" if backend == "native" else "vllm_timeline"
            required = [result, folder / "response.json", folder / f"{prefix}.sqlite", folder / f"{prefix}.nsys-rep"]
            missing = [str(p) for p in required if not p.is_file() or not p.stat().st_size]
            state = read(result).get("state") if result.is_file() else "not_started"
            rows.append({"model": model, "backend": backend, "state": state, "missing": missing,
                         "ready": not missing and state == ("complete" if backend == "native" else "captured_request")})
    return {"state": "ready" if all(r["ready"] for r in rows) else "incomplete",
            "captures": rows, "analysis_model_requests": 0, "analysis_application_episodes": 0}


def checked_summary(folder, backend, summarize):
    database = folder / ("native_timeline.sqlite" if backend == "native" else "vllm_timeline.sqlite")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        assert connection.execute("pragma quick_check").fetchone()[0] == "ok"
    summary = summarize.summarize(database, folder / "analysis", 32)
    assert summary["expected_decode_steps"] == summary["cuda_graph_launch_calls"] == 32
    if backend == "vllm":
        assert sum(r["count"] for r in summary["generation_nvtx_ranges"]) == 32
        with (folder / "analysis/nvtx_events.csv").open() as stream:
            # vLLM labels decode-only iterations execute_context_0(0)_generation_1(1).
            # The word context itself does not establish a prefill operation.
            context = [r for r in csv.DictReader(stream) if "prefill" in r["name"].lower()
                       or any(int(n) > 0 for n in re.findall(r"context_(\d+)", r["name"].lower()))]
        assert not context, context
    return summary, database


def new_profile(profiles, model, backend, native_plan, summarize, inputs):
    folder = profiles / f"{backend}_{model}_case7"
    host_plan = read(profiles / f"vllm_{model}_case7/launch_plan.json")
    assert host_plan["task_family"] == "notification" and host_plan["source_job"]["index"] == 7
    assert host_plan["source_job"]["phase"] == "evaluation"
    assert host_plan["application_episodes"] == host_plan["qualification_trials"] == 0
    assert host_plan["expected_input_tokens"] == 998
    precision = read(profiles / f"export_{model}_precision_audit.json")
    assert precision["state"] == "passed" and precision["upstream_commit"] == COMMIT
    response = read(folder / "response.json")
    if backend == "native":
        result_path = profiles / f"native_capture_{model}_result.json"
        capture = read(result_path)
        assert capture["state"] == "complete"
        assert response["ok"] and response["model_requests_executed"] == 1
        scope = response["capture"]
        assert scope["started"] and scope["stopped"] and scope["full_generation_preserved"]
        assert scope["cuda_start_status"] == scope["cuda_stop_status"] == 0
        assert scope["after_output_callback"] == 5 and scope["through_output_callback"] == 37
        assert scope["requested_decode_steps"] == 32 and scope["observed_output_callbacks"] >= 37
        input_audit = read(folder / "input_audit.json")
        cpu_audit = read(profiles / f"raw_prompt_{model}_cpu_audit.json")
        assert cpu_audit["checkpoint_rendered_input_ids_match_recorded"]
        assert cpu_audit["stock_raw_api_formatted_text_matches_checkpoint_rendered_text"]
        assert cpu_audit["predicted_expanded_input_tokens"] == 998
        assert response["input_token_counts"] == [998], response["input_token_counts"]
        assert input_audit["observed_input_tokens"] == input_audit["expected_input_tokens"] == 998
        assert input_audit["runtime_formatted_text_matches_checkpoint_render"]
        assert input_audit["cpu_audit_input_ids_match_recorded"]
        # The upstream input-ID getter is empty without thinker embeddings. Do
        # not turn those on merely to audit: they change inference work.
        original = read(folder / "expected_first_request.json")
        runtime_ids = response.get("input_token_ids", [])
        if runtime_ids:
            assert runtime_ids == [original["input_token_ids"]]
        formatted = response["formatted_requests"]
        assert len(formatted) == 1
        formatted_sha = hashlib.sha256(formatted[0]["complete"].encode()).hexdigest()
        assert formatted_sha == cpu_audit["formatted_text_sha256"], (formatted_sha, cpu_audit)
        runtime_ids_available = bool(runtime_ids)
        output_tokens = len(response["output_token_ids"][0])
        effective_precision = precision["runtime_precision"]
        request = native_plan["models"][model]["request"]
        assert request["apply_chat_template"] is False
        assert request["max_generate_length"] == 4096 and request["temperature"] == 0
        remember(inputs, result_path, folder / "input_audit.json", folder / "expected_first_request.json",
                 profiles / f"raw_prompt_{model}_cpu_audit.json")
    else:
        result_path = folder / "capture_result.json"
        capture = read(result_path)
        assert capture["state"] == "captured_request" and capture["model_requests_executed"] == 1
        assert capture["response_sha256"] == digest(folder / "response.json")
        assert response["usage"]["prompt_tokens"] == 998
        assert host_plan["request_file_sha256"] == digest(folder / "request.json")
        request = read(folder / "request.json")
        assert request["repetition_penalty"] == 1.0 and request["max_tokens"] == 4096
        assert request["temperature"] == 0 and request["top_p"] == 0.8
        assert host_plan["request_changes"] == [{"path": "repetition_penalty", "old": 1.05, "new": 1.0,
            "reason": "User requested removal of local backend code; pristine Edge-LLM has no repetition-penalty request field."}]
        runtime_ids_available = False
        output_tokens = response["usage"]["completion_tokens"]
        effective_precision = host_plan["precision"]
        remember(inputs, folder / "launch_plan.json", result_path)
    assert capture["application_episodes"] == capture["qualification_trials"] == 0
    summary, database = checked_summary(folder, backend, summarize)
    matrix = summary["kernel_category_partition"]["matrix_kernel_union_ms"]
    busy, span = summary["gpu_busy_union_ms"], summary["gpu_activity_span_ms"]
    assert 0 <= matrix <= busy <= span
    remember(inputs, database, folder / "analysis/summary.json", folder / "request.json", folder / "response.json",
             profiles / f"export_{model}_precision_audit.json")
    row = {"series": "stock", "model_key": model, "model": host_plan["model"], "revision": host_plan["revision"],
        "backend": backend, "checkpoint_precision": precision["source_checkpoint_precision"],
        "effective_runtime_precision": effective_precision,
        "kv_cache": host_plan["kv_cache"], "repetition_penalty": 1.0, "case_index": 7,
        "decode_steps": 32, "first_generated_token": 6, "last_generated_token": 37,
        "input_tokens": 998, "fresh_runtime_input_ids_available": runtime_ids_available,
        "input_id_proof": "CPU checkpoint-template audit plus runtime formatted-text/count" if backend == "native" else "Recorded request provenance plus response prompt-token count; fresh IDs not exposed",
        "output_tokens": output_tokens, "gpu_activity_span_ms_per_step": span / 32,
        "gpu_busy_union_ms_per_step": busy / 32, "matrix_union_ms_per_step": matrix / 32,
        "other_gpu_busy_ms_per_step": (busy - matrix) / 32, "gpu_uncovered_ms_per_step": (span - busy) / 32,
        "kernel_count_per_step": summary["kernel_count"] / 32,
        "initial_input_token_ids_sha256": host_plan["initial_input_token_ids_sha256"],
        "evaluation_trials": 0, "diagnostic_model_requests": 1, "summary": str(folder / "analysis/summary.json")}
    families = [{"series": "stock", "model_key": model, "backend": backend, "family": f["family"],
                 "kernel_count_per_step": f["count"] / 32, "family_union_ms_per_step": f["union_ms"] / 32,
                 "kernel_sum_ms_per_step": f["summed_ms"] / 32} for f in summary["kernel_family_summary"]]
    audit = {"state": "passed", "model": model, "backend": backend, "capture_steps": 32,
             "input_tokens": 998, "fresh_runtime_input_ids_available": runtime_ids_available,
             "application_episodes": 0, "qualification_trials": 0, "diagnostic_model_requests": 1,
             "limits": ["Single instrumented request; not evaluation latency or E-to-A.",
                        "Matching input token provenance does not establish equal visual embeddings or generated outputs."]}
    write(folder / "analysis/provenance_audit.json", audit)
    remember(inputs, folder / "analysis/provenance_audit.json")
    return row, families


def batch_evidence(profiles, inputs):
    journal, status = read(PAUSE), read(EXPERIMENT / "status.json")
    assert journal["completed_evidence"]["records"] == 598
    preserved = journal["completed_evidence"]["hashes"]
    assert all(digest(EXPERIMENT / p) == expected for p, expected in preserved.items())
    session = read(profiles / "session_status.json")
    remember(inputs, PAUSE, EXPERIMENT / "status.json", profiles / "session_status.json")
    result = {"pause_journal": str(PAUSE), "preserved_completed_trials": 598,
              "preserved_evidence_files_verified": len(preserved),
              "archived_unscored_files": len(journal["archived_unscored_files"]),
              "journal_state": journal["state"], "session_state": session["state"],
              "evaluation_state": status["state"], "evaluation_completed_trials": status["completed_trials"],
              "evaluation_planned_trials": status["planned_trials"],
              "resume_confirmed": journal["state"] == "resumed" and session["state"] == "profiles_complete_evaluation_resumed"}
    if result["resume_confirmed"]:
        result["resume_evidence"] = {k: v for k, v in journal.items() if k.startswith("resume")}
    return result


def figures(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    categories = [("matrix_union_ms_per_step", "Matrix kernels (interval union)", "#276fa3"),
                  ("other_gpu_busy_ms_per_step", "Other GPU activity outside matrix intervals", "#74b7a2"),
                  ("gpu_uncovered_ms_per_step", "Intervals without captured GPU activity", "#d7dce2")]
    for ax, model in zip(axes, MODELS):
        selected = [next(r for r in rows if r["series"] == series and r["backend"] == backend and r["model_key"] == model)
                    for series, backend in (("prior", "native"), ("prior", "vllm"), ("stock", "native"), ("stock", "vllm"))]
        bottoms = [0.] * 4
        for key, label, color in categories:
            values = [r[key] for r in selected]
            ax.bar(range(4), values, bottom=bottoms, width=.64, label=label, color=color, edgecolor="white", linewidth=.5)
            bottoms = [a + b for a, b in zip(bottoms, values)]
        for i, value in enumerate(bottoms):
            ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 5), textcoords="offset points", ha="center")
        ax.set(xticks=range(4), xticklabels=["Prior native\npenalty 1.05", "Prior vLLM\npenalty 1.05", "Stock native\npenalty 1.0", "Fresh vLLM\npenalty 1.0"],
               ylabel="Captured activity span / decode step (ms)",
               title="Qwen3.5-0.8B · FP16" if model == "qwen08" else "RadixArk Qwen3.8-27B · mixed precision",
               ylim=(0, max(bottoms) * 1.16))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", frameon=False, fontsize=9)
    fig.supxlabel("One recorded Notification request per configuration · tokens 6–37 · Nsight instrumentation\nStock native uses rebuilt pristine upstream; 27B vision is FP16 (prior native and vLLM: BF16). Not a patch-only causal experiment.", fontsize=9)
    outputs = []
    for ext in ("png", "svg", "pdf"):
        path = out / f"stock_vs_prior_decode.{ext}"
        fig.savefig(path, dpi=180)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def report(out, rows, comparisons, batch, commit, profiles):
    new = [r for r in rows if r["series"] == "stock"]
    lines = ["# Pristine TensorRT-Edge-LLM diagnostic", "",
        "Four completed diagnostic requests replay Disappearing Notification case 7. Each capture contains 32 decode steps (generated tokens 6–37), after five outputs. These are instrumented backend measurements, with zero application episodes or qualification trials; they do not enter evaluation denominators.", "",
        f"Native source is unmodified NVIDIA TensorRT-Edge-LLM commit `{commit}`. The external harness invokes public request/runtime APIs and starts/stops Nsight using the output callback. It is a backend diagnostic, not the full GPU-history Chronotope system.", "",
        "| Model | Runtime | GPU span, ms/step | GPU busy union | Matrix union | GPU-uncovered intervals | Full output tokens |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in new:
        lines.append(f"| {r['model']} | {'Stock C++/TensorRT-Edge-LLM' if r['backend']=='native' else 'vLLM'} | {r[METRICS[0]]:.3f} | {r[METRICS[1]]:.3f} | {r[METRICS[2]]:.3f} | {r[METRICS[4]]:.3f} | {r['output_tokens']} |")
    lines += ["", "Matrix union is contained within GPU busy union; do not add these columns. The stacked figure instead partitions time into matrix union, remaining GPU activity, and intervals without captured GPU activity.", "",
              "![Stock versus prior decode captures](stock_vs_prior_decode.png)", "",
              "| Model | Runtime | Prior span, ms/step | Fresh span, ms/step | Difference | Change |",
              "|---|---|---:|---:|---:|---:|"]
    for c in comparisons:
        if c["comparison"] == "stock_minus_prior" and c["metric"] == METRICS[0]:
            lines.append(f"| {c['model_key']} | {c['backend']} | {c['reference']:.3f} | {c['current']:.3f} | {c['difference']:+.3f} | {c['percent_change']:+.1f}% |")
    lines += ["", "The fresh native/host differences partition as follows:", ""]
    for model in MODELS:
        pair = {r["backend"]: r for r in new if r["model_key"] == model}
        d = {key: pair["native"][key] - pair["vllm"][key] for key in METRICS}
        lines.append(f"- {model}: native minus vLLM span **{d[METRICS[0]]:+.3f} ms/step** = matrix union {d[METRICS[2]]:+.3f}, other GPU activity {d[METRICS[3]]:+.3f}, uncovered intervals {d[METRICS[4]]:+.3f}.")
    lines += ["", "## Inputs and precision", "",
        "Qwen3.5-0.8B uses the local FP16 source checkpoint with weight SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. RadixArk/Qwen3.8-27B-NVFP4 uses revision `319f741cce68d7914884900c138a1fbb70a42f30`: NVFP4 MLP/lm_head, FP8 attention, BF16 source vision. Source checkpoint and effective runtime arithmetic are recorded separately.", "",
        "The stock Radix exporter retains FP4/FP8 quantized initializer counts but converts vision and unquantized LLM tensors to FP16. The prior modified native export preserved BF16 vision; vLLM still uses BF16 vision. Qwen0.8B remains FP16. KV cache is FP16 for 0.8B and FP8 E4M3/static scale 1 for Radix. This is not exact component-arithmetic parity for the 27B diagnostic.", "",
        f"[Export comparison]({profiles / 'export_comparison.json'}) finds both LLM computation graphs identical to their prior exports after removing only non-computational metadata and external-data path names; external weight bytes, embeddings and configs match exactly. The 0.8B vision graph/weights also match. Radix vision changes: 223 initializers (460,603,376 elements) change BF16→FP16, 27 generic attention nodes become 27 stock `ViTAttentionPlugin` nodes, and graph size falls from 1,701 to 1,469 nodes. Thus the re-export does not change either LLM computation graph, while the 27B vision computation changes.", "",
        "Both fresh arms use repetition penalty 1.0, greedy generation, top_p 0.8, the original 4096-token budget and stopping behavior, with thinking disabled. Prior captures and the original evaluation use repetition penalty 1.05. Full generation is allowed to finish; outputs and their lengths may differ.", "",
        "Prompts use the recorded Agent S3 Worker first request and screenshot. Native uses supported `apply_chat_template=false` with the original checkpoint Jinja-rendered text supplied through text/image/text inputs. This avoids lossy upstream extracted-template formatting without changing source, templates, image loading or preprocessing. CPU audits reproduce the original 998 input IDs; runtime formatted text and input counts are verified. The stock runtime input-ID getter is unavailable unless thinker embeddings are enabled, which would change work; those embeddings are not enabled. Fresh vLLM IDs are also not exposed, so HTTP provenance and its 998-token response count are used. Equal token IDs do not establish equal image embeddings or backend numerics.", "",
        "## Interpretation limits", "",
        "Removing local patches, rebuilding engines/tactics, changing repetition penalty, changing native 27B vision precision, and run order occurred together. Differences from prior profiles cannot be attributed solely to patch removal. The figures characterize these configurations; identifying a particular kernel or patch as causal needs a separate controlled replacement.", "",
        f"The [runtime patch audit]({profiles / 'removed_runtime_patches_audit.json'}) identifies the removed local repetition-penalty code, strict token-budget guard, input-ID getter relocation, and image copy/27B normalization paths. The stock decoder still copies sampled tokens from GPU to CPU and calls `cudaStreamSynchronize` at [vanillaDecoder.cpp:144]({profiles / 'source/TensorRT-Edge-LLM/cpp/runtime/decoding/vanillaDecoder.cpp'}:144), then performs CPU token bookkeeping. These CPU operations are upstream code and remain in this diagnostic; removing local patches does not remove CPU participation.", "",
        "Only one recorded request per model/backend is captured. Nsight node tracing perturbs execution, and this is not E→A, request latency, live success or a repeated-run population estimate. The span excludes leading/trailing CPU-only work. Matrix families use kernel symbols and their interval unions can overlap; family sums must not be used as elapsed time. CUDA API synchronization waits overlap GPU work and are not an independent overhead term. Uncovered GPU intervals alone do not identify CPU, scheduler or synchronization causes.", "",
        "## Evaluation preservation", "",
        f"The original 4,800-trial evaluation was paused at **598 completed trials**. All {batch['preserved_evidence_files_verified']} recorded evidence hashes still match; {batch['archived_unscored_files']} partial unscored files were archived. At report generation, the evaluation status is `{batch['evaluation_state']}` at **{batch['evaluation_completed_trials']}/{batch['evaluation_planned_trials']}**, and the diagnostic coordinator reports `{batch['session_state']}`.", ""]
    lines.append("The pause journal and coordinator confirm restoration of the original service and resumption of the original frozen evaluation engines/settings." if batch["resume_confirmed"] else "Restoration/resumption is not yet confirmed by both the pause journal and coordinator. Rerun this CPU-only analysis after the coordinator completes to update the evidence.")
    lines += ["", f"Pause/resume journal: [{PAUSE.name}]({PAUSE}). Machine-readable evidence and file hashes: [analysis.json](analysis.json). Per-step and kernel-family data: [decode_comparison.csv](decode_comparison.csv), [metric_changes.csv](metric_changes.csv), [kernel_families.csv](kernel_families.csv).", "",
              "```bash", ".venv/bin/python -B studies/runtime_reversal_stock/analyze.py", "```", "",
              "This command only reads existing captures and writes derived analysis; it runs no models or experiments.", ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=ROOT / "runs/runtime_reversal_stock_20261029")
    parser.add_argument("--prior", type=Path, default=ROOT / "runs/runtime_reversal_20261029")
    parser.add_argument("--out", type=Path, default=ROOT / "results/runtime_reversal_stock_20261029")
    parser.add_argument("--check-ready", action="store_true", help="Only inspect capture availability; write no report")
    args = parser.parse_args()
    state = readiness(args.profiles)
    if args.check_ready or state["state"] != "ready":
        print(json.dumps(state, indent=2))
        return 0 if state["state"] == "ready" else 2
    summarize = module("_stock_nsys_summary", ROOT / "studies/runtime_reversal/summarize_nsys.py")
    prior_loader = module("_prior_profile_plot", ROOT / "studies/runtime_reversal/plot_profiles.py")
    source = read(args.profiles / "upstream_source_manifest.json")
    native_plan = read(args.profiles / "native_plan.json")
    assert source["commit"] == native_plan["upstream_commit"] == COMMIT
    assert not source["local_source_patches"] and not native_plan["source_patches"]
    assert all(digest(Path(source["source"]) / p) == meta["sha256"] for p, meta in source["files"].items())
    exports = read(args.profiles / "export_comparison.json")
    for model in MODELS:
        llm = exports["models"][model]["llm"]
        assert llm["normalized_graph_equal"]
        assert all(llm["file_hashes"][name]["equal"] for name in ("model.onnx.data", "embedding.safetensors", "config.json"))
    assert exports["models"]["qwen08"]["visual"]["normalized_graph_equal"]
    visual = exports["models"]["radixark27b"]["visual"]
    assert len(visual["initializer_dtype_changes"]) == 223
    assert sum(r["elements"] for r in visual["initializer_dtype_changes"]) == 460603376
    assert (visual["old"]["node_count"], visual["stock"]["node_count"]) == (1701, 1469)
    kernel_audit = read(args.profiles / "official_cutedsl_artifact_audit.json")
    assert kernel_audit["state"] == "verified" and not kernel_audit["existing_local_artifacts_reused"]
    inputs, rows, families = {}, [], []
    remember(inputs, Path(__file__), ROOT / "studies/runtime_reversal/summarize_nsys.py",
             ROOT / "studies/runtime_reversal/plot_profiles.py", args.profiles / "upstream_source_manifest.json",
             args.profiles / "native_plan.json", args.profiles / "reviewed_session_plan.json",
             args.profiles / "reviewed_runtime_continuation.json", args.profiles / "export_comparison.json",
             args.profiles / "removed_runtime_patches_audit.json", args.profiles / "official_cutedsl_artifact_audit.json",
             args.profiles / "post_runtime_build_source_audit.json")
    for model in MODELS:
        for backend in BACKENDS:
            prior, hashes = prior_loader.load_profile(args.prior / f"{backend}_{model}_case7", backend, model)
            inputs.update(hashes)
            fresh, new_families = new_profile(args.profiles, model, backend, native_plan, summarize, inputs)
            assert prior["revision"] == fresh["revision"]
            assert prior["initial_input_token_ids_sha256"] == fresh["initial_input_token_ids_sha256"]
            old = {key: prior.get(key, "") for key in fresh}
            old.update(series="prior", checkpoint_precision=prior["precision"],
                       effective_runtime_precision=("NVFP4 MLP/head; FP8 attention; FP16 unquantized LLM; BF16 vision" if model == "radixark27b" and backend == "native" else prior["precision"]),
                       kv_cache=fresh["kv_cache"], repetition_penalty=1.05, input_tokens=998,
                       fresh_runtime_input_ids_available=backend == "native", input_id_proof="See archived provenance audit")
            old_summary = read(prior["summary"])
            old["kernel_count_per_step"] = old_summary["kernel_count"] / 32
            rows += [old, fresh]
            families += [{"series": "prior", "model_key": model, "backend": backend, "family": f["family"],
                          "kernel_count_per_step": f["count"] / 32, "family_union_ms_per_step": f["union_ms"] / 32,
                          "kernel_sum_ms_per_step": f["summed_ms"] / 32} for f in old_summary["kernel_family_summary"]] + new_families
    comparisons = []
    for model in MODELS:
        for comparison, pairs in (("stock_minus_prior", [(backend, ("stock", backend), ("prior", backend)) for backend in BACKENDS]),
                                  ("native_minus_vllm", [(series, (series, "native"), (series, "vllm")) for series in ("prior", "stock")])):
            for label, current_key, reference_key in pairs:
                current, reference = [next(r for r in rows if r["model_key"] == model and (r["series"], r["backend"]) == key) for key in (current_key, reference_key)]
                for metric in METRICS:
                    diff = current[metric] - reference[metric]
                    comparisons.append({"model_key": model, "comparison": comparison, "backend": label,
                        "metric": metric, "reference": reference[metric], "current": current[metric],
                        "difference": diff, "percent_change": 100 * diff / reference[metric]})
    batch = batch_evidence(args.profiles, inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_write(args.out / "decode_comparison.csv", rows)
    csv_write(args.out / "metric_changes.csv", comparisons)
    csv_write(args.out / "kernel_families.csv", families)
    outputs = figures(rows, args.out)
    report(args.out, rows, comparisons, batch, COMMIT, args.profiles)
    write(args.out / "analysis.json", {"state": "complete", "generated_at": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": COMMIT, "source_files_verified": len(source["files"]), "inputs": inputs,
        "rows": rows, "comparisons": comparisons, "batch": batch, "figures": outputs,
        "analysis_inference_calls": 0, "analysis_application_episodes": 0,
        "diagnostic_requests_represented": {"stock": 4, "prior": 4},
        "scope": "32 captured decode steps per configuration; single recorded Notification case7 request",
        "causal_limit": "Not patch-removal-only: settings, precision, rebuilt engines and run order differ."})
    print(json.dumps({"state": "complete", "report": str(args.out / "report.md"), "batch": batch}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
