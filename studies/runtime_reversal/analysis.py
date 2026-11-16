#!/usr/bin/env python3
"""CPU-only analysis of existing, completed Notification evaluation records.

Creates an immutable input snapshot on first use. Later invocations verify and
reuse it, so an advancing experiment cannot silently change the comparison.
No model calls, GPU operations, qualification episodes, or task filtering.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("legacy_cpp", "agents3_stripped")
MODES = ("live", "replay", "paused")
RUNS = {
    "qwen08": ROOT / "runs/agents3_policy_qwen08_20261028/experiment",
    "radixark27b": ROOT / "runs/agents3_policy_radixark27b_20261028/experiment",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def snapshot(path):
    if path.exists():
        return read(path)
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "runs": {}}
    for key, root in RUNS.items():
        # List paths once; atomic record publication makes these complete files.
        files = sorted((root / "records").glob("*.json"))
        entries = []
        for file in files:
            record = read(file)
            job = record["job"]
            assert job["phase"] == "evaluation" and job["task"] == "notification"
            if job["window_s"] != 16.0:
                continue
            ident = file.stem
            calls = root / "calls" / (ident + ".jsonl")
            trace = root / record["trace"]
            assert digest(trace) == record["trace_sha256"]
            entries.append({
                "record": str(file.relative_to(root)), "record_sha256": digest(file),
                "calls": str(calls.relative_to(root)), "calls_sha256": digest(calls),
                "trace": record["trace"], "trace_sha256": record["trace_sha256"],
            })
        execution = read(root / "execution_manifest.json")
        result["runs"][key] = {
            "root": str(root), "status": read(root / "status.json"),
            "plan_sha256": digest(root / "plan.json"),
            "execution_manifest_sha256": digest(root / "execution_manifest.json"),
            "identity": {k: execution.get(k) for k in ("model", "revision", "precision", "kv_cache")},
            "all_completed_record_count_when_listed": len(files), "entries": entries,
        }
    write(path, result)
    return result


def load_rows(snap):
    result = {}
    for model, info in snap["runs"].items():
        root = Path(info["root"])
        assert digest(root / "plan.json") == info["plan_sha256"]
        assert digest(root / "execution_manifest.json") == info["execution_manifest_sha256"]
        rows = []
        for entry in info["entries"]:
            for key in ("record", "calls", "trace"):
                assert digest(root / entry[key]) == entry[key + "_sha256"]
            record = read(root / entry["record"])
            job, outcome = record["job"], record["outcome"]
            assert not outcome["invalid_trial"] and not outcome["contract_violations"]
            calls = [json.loads(line) for line in (root / entry["calls"]).read_text().splitlines()]
            events = [x for line in (root / entry["trace"]).read_text().splitlines()
                      if (x := json.loads(line)).get("record") == "decision_invocation"]
            assert calls and events and record["request_audit"]
            call, event, audit = calls[0], events[0], record["request_audit"][0]
            native = job["arm"] == "legacy_cpp"
            if native:
                call_ms = (call["t_end_ns"] - call["t_start_ns"]) / 1e6
                input_tokens, output_tokens = call["input_tokens"], call["output_tokens"]
                first_decision_calls = sum(c["turn"] == calls[0]["turn"] for c in calls)
            else:
                call_ms = (call["end_ns"] - call["start_ns"]) / 1e6
                input_tokens = call["response"]["usage"]["prompt_tokens"]
                output_tokens = call["response"]["usage"]["completion_tokens"]
                first_decision_calls = event["planner_calls"] + event["reflection_calls"]
            assert input_tokens == audit["input_tokens"] and output_tokens == audit["output_tokens"]
            rows.append({
                "model": model, "arm": job["arm"], "mode": job["mode"], "index": job["index"],
                "episode_id": job["episode_id"], "source_record": str(root / entry["record"]),
                "input_tokens": input_tokens, "output_tokens": output_tokens, "call_ms": call_ms,
                "input_ids_sha256": audit["input_token_ids_sha256"],
                "messages_sha256": audit["messages_sha256"], "rgb_sha256": audit["source_rgb_sha256"],
                "calls_in_trial": len(calls), "decisions_in_trial": len(events),
                "calls_in_first_decision": first_decision_calls,
                "native_prefill_ms": event.get("prefill_ms") if native and first_decision_calls == 1 else None,
                "native_decode_ms": event.get("decode_ms") if native and first_decision_calls == 1 else None,
                "EA_ms": outcome["L_EA"] / 1e6 if outcome["L_EA"] is not None else None,
                "BD_ms": outcome["L_BD"] / 1e6 if outcome["L_BD"] is not None else None,
                "received": bool(outcome["R"]), "success": bool(outcome["Y"]),
                "failure_mode": outcome["failure_mode"],
            })
        result[model] = rows
    return result


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def nearest_rank(values, percentile):
    values = sorted(v for v in values if v is not None)
    return values[max(0, int(np.ceil(len(values) * percentile)) - 1)] if values else None


def robust_fit(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    if not np.any(dx):
        return float("nan"), float("nan")
    slope = float(np.median(dy[dx != 0] / dx[dx != 0]))
    return slope, float(np.median(y - slope * x))


def fit(rows):
    x = np.array([r["output_tokens"] for r in rows], dtype=float)
    y = np.array([r["call_ms"] for r in rows], dtype=float)
    slope, intercept = robust_fit(x, y)
    residual = y - (intercept + slope * x)
    return {
        "method": "Theil-Sen median pairwise slope; median residual intercept",
        "n": len(rows), "slope_ms_per_additional_output_token": slope,
        "intercept_ms": intercept, "median_absolute_residual_ms": float(np.median(np.abs(residual))),
        "maximum_absolute_residual_ms": float(np.max(np.abs(residual))),
        "output_token_range": [int(x.min()), int(x.max())],
        "input_token_values": sorted({r["input_tokens"] for r in rows}),
        "warning": "Observational request-time fit, not pure GPU decode or measured fixed CPU overhead. No requests excluded as outliers.",
    }


def summarize(rows):
    return {
        "n_all_first_requests": len(rows), "receipts": sum(r["received"] for r in rows),
        "successes": sum(r["success"] for r in rows),
        "outcomes": dict(Counter(r["failure_mode"] for r in rows)),
        "first_call_p50_ms": nearest_rank([r["call_ms"] for r in rows], .5),
        "first_call_median_ms": median(r["call_ms"] for r in rows),
        "first_call_p95_ms": nearest_rank([r["call_ms"] for r in rows], .95),
        "output_tokens_median": median(r["output_tokens"] for r in rows),
        "call_ms_per_output_token_median": median(r["call_ms"] / r["output_tokens"] for r in rows),
        "native_single_call_prefill_n": sum(r["native_prefill_ms"] is not None for r in rows),
        "native_single_call_prefill_median_ms": median(r["native_prefill_ms"] for r in rows),
        "native_single_call_decode_median_ms": median(r["native_decode_ms"] for r in rows),
        "native_decode_ms_per_output_token_median": median(
            r["native_decode_ms"] / r["output_tokens"] if r["native_decode_ms"] is not None else None for r in rows),
        "receipt_EA_p50_ms": nearest_rank([r["EA_ms"] for r in rows], .5),
        "receipt_EA_p95_ms": nearest_rank([r["EA_ms"] for r in rows], .95),
        "multiple_call_trials": sum(r["calls_in_trial"] > 1 for r in rows),
        "multiple_decision_trials": sum(r["decisions_in_trial"] > 1 for r in rows),
        "fit": fit(rows),
    }


def paired(left, right, repetitions, seed):
    assert [r["index"] for r in left] == [r["index"] for r in right]
    for a, b in zip(left, right):
        for key in ("input_ids_sha256", "messages_sha256", "rgb_sha256"):
            assert a[key] == b[key], (key, a["index"])
    x1 = np.array([r["output_tokens"] for r in left], float)
    x2 = np.array([r["output_tokens"] for r in right], float)
    y1 = np.array([r["call_ms"] for r in left], float)
    y2 = np.array([r["call_ms"] for r in right], float)
    rng = np.random.default_rng(seed)
    differences, ratios, normalized = [], [], []
    for _ in range(repetitions):
        i = rng.integers(0, len(left), len(left))
        a, _ = robust_fit(x1[i], y1[i])
        b, _ = robust_fit(x2[i], y2[i])
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            ratios.append(a / b)
        differences.append(float(np.median(y1[i] - y2[i])))
        normalized.append(float(np.median((y1[i] / x1[i]) / (y2[i] / x2[i]))))
    interval = lambda values: np.quantile(values, [.025, .975]).tolist()
    examples = [{"index": a["index"], "output_tokens": a["output_tokens"],
                 "cpp_call_ms": a["call_ms"], "vllm_call_ms": b["call_ms"],
                 "cpp_record": a["source_record"], "vllm_record": b["source_record"]}
                for a, b in zip(left, right) if a["output_tokens"] == b["output_tokens"]]
    return {
        "n": len(left), "initial_request_ids_messages_rgb_match": True,
        "native_faster_first_calls": int(np.sum(y1 < y2)),
        "native_shorter_outputs": int(np.sum(x1 < x2)),
        "median_native_minus_host_call_ms": float(np.median(y1 - y2)),
        "median_native_minus_host_call_ms_bootstrap95": interval(differences),
        "median_normalized_call_ratio_native_over_host": float(np.median((y1 / x1) / (y2 / x2))),
        "normalized_call_ratio_bootstrap95": interval(normalized),
        "slope_ratio_native_over_host_bootstrap95": interval(ratios),
        "bootstrap": {"resamples": repetitions, "seed": seed, "unit": "paired episode index", "interval": "percentile95"},
        "equal_output_count_examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/runtime_reversal_20261029")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snap_path = args.out / "analysis_snapshot.json"
    snap = snapshot(snap_path)
    rows = load_rows(snap)
    available = {(model, arm, mode): {r["index"] for r in data if r["arm"] == arm and r["mode"] == mode}
                 for model, data in rows.items() for arm in ARMS for mode in MODES}
    # Same index cohort in BOTH models/arms and ALL conditions. Includes every
    # completed success/failure; only unavailable/incomplete cases are excluded.
    shared = sorted(set.intersection(*available.values()))
    assert shared and shared == list(range(len(shared)))
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(snap_path), "snapshot_sha256": digest(snap_path),
        "script_sha256": digest(__file__), "window_s": 16, "cohort_indices": shared,
        "inference_calls": 0, "experimental_trials": 0,
        "selection": "All available paired prefix cases shared across both models, arms and conditions; no competence or receipt selection for first-call analyses.",
        "identities": {key: info["identity"] for key, info in snap["runs"].items()},
        "models": {}, "rows": [],
        "limitations": [
            "Models, precisions, KV precision, architectures and inference kernels differ between model sizes.",
            "Within-model prompts/settings/initial input IDs and RGB match; generated tokens need not match.",
            "Native audit interval contains request construction, GPU work, CPU decoding loop/sync and extraction; host interval contains HTTP, server processing and response handling.",
            "Request-time fits and their intercepts do not isolate GPU decode, CPU scheduling, HTTP, or synchronization.",
            "Output length varies naturally with request content and runtime; observational slopes are not randomized interventions on generation length.",
            "Bootstrap describes the sampled cases, not independent hardware repetitions; time/order/systematic effects can remain.",
            "E-to-A summaries condition on receipt; first-call comparisons retain no-action/competence failures.",
            "No per-trial thermal/clock traces or GPU kernel profiles were found in these paired-run directories.",
        ],
    }
    for model, data in rows.items():
        output["models"][model] = {}
        for mode in MODES:
            groups = {arm: sorted([r for r in data if r["arm"] == arm and r["mode"] == mode
                                   and r["index"] in shared], key=lambda r: r["index"]) for arm in ARMS}
            output["models"][model][mode] = {arm: summarize(group) for arm, group in groups.items()}
            output["models"][model][mode]["paired"] = paired(groups[ARMS[0]], groups[ARMS[1]], args.bootstrap, 20261029)
            output["rows"].extend(groups[ARMS[0]] + groups[ARMS[1]])
    write(args.out / "analysis.json", output)
    print(json.dumps({"state": "complete", "cohort_n_per_model_arm_condition": len(shared),
                      "first_requests_analyzed": len(output["rows"]), "out": str(args.out / "analysis.json")}))


if __name__ == "__main__":
    main()
