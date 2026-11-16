"""Audit the evidence shipped with the original-submission supplement.

Usage: python studies/audit_supplement.py [--root PATH] [--output PATH]
The audit reads immutable inputs. It never reconstructs missing event logs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chronotope.evaluate import evaluate_pending_trace, evaluate_trace
from chronotope.trace import Trace


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit(root: Path) -> dict:
    evidence = root / "evidence"
    inventory = json.loads((root / "FILE_SHA256.json").read_text(encoding="utf-8"))["files"]
    inventory_bad = []
    for name, expected in inventory.items():
        path = root.parent / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            inventory_bad.append(name)

    selected = sorted((evidence / "selected_raw_events").rglob("*.jsonl"))
    counts = Counter()
    exceptions = []
    violations = []
    receipt_mismatches = []
    legacy_no_skin = 0
    deferred_gap_ms = []
    for path in selected:
        try:
            trace = Trace.load(path)
            header = trace.header()
            if "skin" not in header["manifest"]:
                legacy_no_skin += 1
            is_pending = header["manifest"]["scenario_id"] == "pending_effect"
            result = evaluate_pending_trace(trace) if is_pending else evaluate_trace(trace)
            counts["pending_action" if is_pending else "single_action"] += 1
            if not result.valid_trial or result.contract_violations:
                violations.append({"path": str(path.relative_to(root)),
                                   "reason": result.invalid_reason,
                                   "violations": result.contract_violations})
            if not is_pending:
                receipt = trace.first("receipt")
                if receipt and result.valid_at_receipt != receipt["valid_at_receipt"]:
                    receipt_mismatches.append(str(path.relative_to(root)))
            group = path.relative_to(evidence / "selected_raw_events").parts[0]
            if group in {"c1_qwen38_nothink_100", "c2_qwen38_nothink_100", "c3_qwen38_nothink_100"}:
                receipt = trace.first("receipt")
                end = trace.first("trial_end")
                if receipt and end:
                    deferred_gap_ms.append((end["harness_time_ns"] - receipt["receipt_time_ns"]) / 1e6)
        except (ValueError, TypeError, KeyError) as exc:
            exceptions.append({"path": str(path.relative_to(root)), "error": str(exc)})

    primary = list(read_jsonl(evidence / "table_2_8" / "per_trial_clock_scores.jsonl"))
    supplemental = list(read_jsonl(evidence / "table_2_8" / "notification_extra_per_trial_scores.jsonl"))
    table9 = json.loads((evidence / "table_9" / "archived_aggregates.json").read_text())
    table10 = list(read_jsonl(evidence / "table_10" / "archived_pilot_cells.jsonl"))
    trial_keys = [(r["values"].get("run"), r["values"].get("cell"),
                   r["values"].get("policy"), r["values"].get("seed")) for r in primary]
    return {
        "source": "original-submission supplemental archive",
        "integrity": {"sha256_inventory_files": len(inventory), "sha256_mismatches": inventory_bad},
        "original_traces": {
            "count": len(selected), "scorer_group_counts": dict(counts),
            "schema_v1_traces_missing_skin": legacy_no_skin,
            "load_or_rescore_exceptions": exceptions,
            "contract_warnings": violations, "receipt_outcome_mismatches": receipt_mismatches,
            "byte_rewrite_performed": False,
        },
        "tables": {
            "2_and_8": {"derived_per_trial_rows": len(primary) + len(supplemental),
                        "primary_rows": len(primary), "extra_rows": len(supplemental),
                        "primary_duplicate_keys": len(primary) - len(set(trial_keys)),
                        "original_full_event_logs_present": False},
            "9": {"archived_aggregate_reports": len(table9["reports"]),
                  "original_per_trial_logs_present": False},
            "10": {"archived_pilot_cells": len(table10),
                   "original_per_trial_logs_present": False, "trials_per_cell": 10},
        },
        "timing_diagnostic": {
            "selected_C1_C2_C3_receipt_to_trial_end_count": len(deferred_gap_ms),
            "receipt_to_trial_end_gap_ms_median": statistics.median(deferred_gap_ms) if deferred_gap_ms else None,
            "receipt_to_trial_end_gap_ms_p95": sorted(deferred_gap_ms)[int(0.95 * (len(deferred_gap_ms) - 1))] if deferred_gap_ms else None,
            "gap_over_500_ms": sum(x > 500 for x in deferred_gap_ms),
            "meaning": "Trial-end minus nominal receipt is a post-receipt wall-time gap, not a measured delivery delay.",
        },
        "revision_scope": "No raw EODA/EDM, DynaCU-Bench, or War Thunder experiment evidence in this archive.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.root.resolve())
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)
    problems = (result["integrity"]["sha256_mismatches"] or
                result["original_traces"]["load_or_rescore_exceptions"] or
                result["original_traces"]["contract_warnings"] or
                result["original_traces"]["receipt_outcome_mismatches"])
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
