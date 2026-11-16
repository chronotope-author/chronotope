"""Continue a frozen matched run using only the project's five permitted tasks.

Retain completed allowed jobs with their original traces. The source remains
archived, including any completed jump/select jobs; none of those jobs enter
the continuation. Content-addressed payloads are immutable and may be hardlinked.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil


CORE_TASKS = frozenset(("notification", "drift", "code", "button", "trading"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restrict_manifest(source):
    result = copy.deepcopy(source)
    result["jobs"] = [job for job in result["jobs"] if job["task"] in CORE_TASKS]
    if {job["task"] for job in result["jobs"]} != CORE_TASKS:
        raise ValueError("Source must contain all five core tasks")
    result["templates"] = {key: value for key, value in result["templates"].items()
                           if key in CORE_TASKS}
    result["planned_trials"] = sum(len(job["modes"]) for job in result["jobs"])
    result["allowed_tasks"] = sorted(CORE_TASKS)
    if "motion_override" in result:
        result["motion_override"]["variants"] = ["drift"]
    return result


def prepare(source_run, out):
    source_run, out = Path(source_run).resolve(), Path(out).resolve()
    source = json.loads((source_run / "manifest.json").read_text())
    result = restrict_manifest(source)
    for rel, expected in result["code_sha256"].items():
        if digest(source_run / "code" / rel) != expected:
            raise ValueError(f"Frozen code changed: {rel}")
    allowed = {job["id"]: job for job in result["jobs"]}
    completed = []
    for path in (source_run / "trials").glob("*/result.json"):
        record = json.loads(path.read_text())
        if record["job"]["id"] not in allowed:
            continue
        if record["job"] != allowed[record["job"]["id"]]:
            raise ValueError(f"Completed job differs from its manifest: {path}")
        for value in record["results"].values():
            evaluation = value["evaluation"]
            if not evaluation["valid_trial"] or evaluation["contract_violations"]:
                raise ValueError(f"Invalid completed trial: {path}")
            if digest(source_run / value["trace"]) != value["sha256"]:
                raise ValueError(f"Completed trace changed: {value['trace']}")
        if "replay" in record["results"] and record["pair_valid"] is not True:
            raise ValueError(f"Invalid replay pair: {path}")
        completed.append((path, record))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_run / "code", out / "code")
    shutil.copytree(source_run / "payloads", out / "payloads", copy_function=os.link)
    for path, record in completed:
        shutil.copytree(path.parent, out / "trials" / record["job"]["id"])
    shutil.copy2(source_run / "environment.json", out / "environment.json")
    shutil.copytree(source_run / "services", out / "services")
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("run_matched_agents3_serial.py"),
                 out / "run_matched_agents3_serial.py")
    result.update(
        created_utc=datetime.now(timezone.utc).isoformat(), parent_run=str(source_run),
        parent_manifest_sha256=digest(source_run / "manifest.json"),
        preparation_script_sha256=digest(__file__),
        carried_completed_jobs=len(completed),
        carried_completed_trials=sum(len(record["results"]) for _, record in completed),
        scope_instruction="AGENTS.md: core five tasks only; Moving Balloon uses drift only.",
        scope_instruction_sha256=digest(Path(__file__).resolve().parents[1] / "AGENTS.md"),
        selected_configuration="64-second windows on notification, drift, code, button and trading; drift retains 8 px/s on a 640-pixel canvas.",
    )
    result["limitations"] = [value.replace("Drift/select", "Drift")
                             for value in result["limitations"]]
    (out / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_run, args.out)
    print(json.dumps({key: result[key] for key in
                      ("allowed_tasks", "planned_trials", "carried_completed_jobs", "carried_completed_trials")}))
