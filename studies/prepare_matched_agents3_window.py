"""Prepare a uniform-deadline batch from an existing frozen matched S3 run.

The source batch is read-only. Keep its code, models, seeds, policies and task
events; change response windows and horizons, and omit the redundant Jump
deadline sweep. Run the copied matched_agents3.py only when its shared model
services are free.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uniform_manifest(source: dict, window_s: float, *, motion_speed_px_s: int | None = None,
                     motion_canvas_w_px: int | None = None) -> dict:
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("Response window must be finite and positive")
    window_ns = round(window_s * 1e9)
    if window_ns < 1:
        raise ValueError("Response window must be at least one nanosecond")
    if motion_speed_px_s is not None and motion_canvas_w_px is not None:
        raise ValueError("Choose either a motion-speed override or a wider motion canvas")
    for value in (motion_speed_px_s, motion_canvas_w_px):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("Motion overrides must be positive integers")
    manifest = copy.deepcopy(source)

    def change_scenario(scenario):
        scenario["response_window_ns"] = window_ns
        scenario["episode_horizon_ns"] = scenario["event_time_ns"] + window_ns + 500_000_000
        if scenario.get("scenario_id") in ("moving_target", "balloon_select"):
            if motion_speed_px_s is not None:
                scenario["velocity_x_px_per_s"] = motion_speed_px_s
            if motion_canvas_w_px is not None:
                scenario["canvas_w_px"] = motion_canvas_w_px

    for template in manifest["templates"].values():
        change_scenario(template["spec"]["scenario"])
    manifest["jobs"] = [job for job in manifest["jobs"] if job["phase"] != "window"]
    for job in manifest["jobs"]:
        change_scenario(job["spec"]["scenario"])
        job["id"] = (f'{job["phase"]}_{job["task"]}_{job["implementation"]}'
                     f'_{job["policy"]}_w{window_ns}_s{job["seed"]}')
    if not manifest["jobs"]:
        raise ValueError("Source has no qualification, PRL or policy jobs")
    if len({job["id"] for job in manifest["jobs"]}) != len(manifest["jobs"]):
        raise ValueError("Uniform windows would create duplicate jobs")
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["response_window_override_s"] = window_s
    manifest["motion_override"] = {
        "variants": ["drift", "select"], "velocity_x_px_per_s": motion_speed_px_s,
        "canvas_w_px": motion_canvas_w_px,
    }
    manifest["planned_trials"] = sum(len(job["modes"]) for job in manifest["jobs"])
    manifest["window_protocol"] = (
        "Uniform response windows in qualification, PRL and policy phases. "
        "Episode horizon = event time + response window + 0.5 seconds. "
        "Task change/reversal times, seeds, policies and model settings are preserved. "
        "The separate Jump window sweep is omitted to avoid duplicate cells."
    )
    return manifest


def prepare(source_run: Path, out: Path, window_s: float, *,
            motion_speed_px_s: int | None = None, motion_canvas_w_px: int | None = None) -> dict:
    source_run, out = source_run.resolve(), out.resolve()
    source_manifest = source_run / "manifest.json"
    source = json.loads(source_manifest.read_text())
    for rel, expected in source["code_sha256"].items():
        if digest(source_run / "code" / rel) != expected:
            raise ValueError(f"Source snapshot changed: {rel}")
    manifest = uniform_manifest(source, window_s, motion_speed_px_s=motion_speed_px_s,
                                motion_canvas_w_px=motion_canvas_w_px)
    manifest["parent_run"] = str(source_run)
    manifest["parent_manifest_sha256"] = digest(source_manifest)
    manifest["preparation_script_sha256"] = digest(Path(__file__))
    environment = json.loads((source_run / "environment.json").read_text())
    services = json.loads((source_run / "services/processes.json").read_text())
    manifest["model_serving"] = {
        role: {"model": services[role]["model"], "runtime": "vllm",
               "runtime_version": environment["vllm"]["packages"]["vllm"],
               "endpoint": f'http://127.0.0.1:{services[role]["port"]}/v1',
               "snapshot": services[role]["snapshot"]}
        for role in ("planner", "grounder")
    }
    out.mkdir(parents=True, exist_ok=False)
    # Copy even bytecode when it is part of the source manifest's frozen hashes.
    shutil.copytree(source_run / "code", out / "code")
    shutil.copy2(Path(__file__), out / "prepare_matched_agents3_window.py")
    shutil.copy2(source_run / "environment.json", out / "environment.json")
    # These are provenance records for the shared services, not new launches.
    (out / "services").mkdir()
    shutil.copy2(source_run / "services/processes.json", out / "services/shared_processes.json")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-s", type=float, required=True)
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-speed-px-s", type=int)
    motion.add_argument("--motion-canvas-w-px", type=int)
    args = parser.parse_args()
    manifest = prepare(args.source_run, args.out, args.window_s,
                       motion_speed_px_s=args.motion_speed_px_s,
                       motion_canvas_w_px=args.motion_canvas_w_px)
    print(json.dumps({"out": str(args.out.resolve()), "jobs": len(manifest["jobs"]),
                      "trials": manifest["planned_trials"],
                      "window_s": manifest["response_window_override_s"]}))
