"""Launch a prepared batch after exclusively reserving its parent run's services.

Hold the parent runner lock for the new batch's lifetime, so the old batch cannot
resume concurrently. Drain any HTTP request left by an interrupted runner before
starting measured trials. This launcher does not stop the parent itself.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def active_service_connections(port):
    """Include CLOSE_WAIT: a disconnected client's model request may still run."""
    result = []
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(table).read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].rsplit(":", 1)[1], 16) == port and fields[3] in ("01", "08"):
                result.append({"local": fields[1], "remote": fields[2], "state": fields[3]})
    return result


def ancestor_runs(out, manifest):
    ancestors, seen = [], {out.resolve()}
    while manifest.get("parent_run"):
        parent = Path(manifest["parent_run"]).resolve()
        if parent in seen:
            raise ValueError("Cycle in experiment parent manifests")
        seen.add(parent)
        ancestors.append(parent)
        manifest = json.loads((parent / "manifest.json").read_text())
    return list(reversed(ancestors))


def main(out):
    out = out.resolve()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads((out / "validation.json").read_text())
    if validation["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("The manifest must be validated again before launch")
    allowed = {"notification", "drift", "code", "button", "trading"}
    excluded = {job["task"] for job in manifest["jobs"]} - allowed
    if excluded:
        raise ValueError(f"Tasks excluded by AGENTS.md: {sorted(excluded)}")
    parents = ancestor_runs(out, manifest)
    launcher_lock = (out / "launcher.lock").open("a")
    fcntl.flock(launcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent_locks = [(parent / "runner.lock").open("a") for parent in parents]

    def status(state):
        current = json.loads((out / "status.json").read_text())
        current.update(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                       pid=os.getpid(), waiting_for=[str(parent) for parent in parents])
        write_json(out / "status.json", current)

    status("waiting_for_parent")
    for lock in parent_locks:
        fcntl.flock(lock, fcntl.LOCK_EX)
    status("draining_shared_service")
    while active_service_connections(8110):
        time.sleep(1)
    with urllib.request.urlopen("http://127.0.0.1:8110/models", timeout=10) as response:
        service = json.load(response)
    if service["planner"] + "+" + service["grounder"] != manifest["model_pair"]:
        raise ValueError("Shared Agent S3 service has the wrong models")
    command = [sys.executable, "-B", str(out / "code/matched_agents3.py"),
               "--out", str(out), "--run"]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    with (out / "runner.log").open("a") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment, start_new_session=True)
    launch = {"pid": child.pid, "supervisor_pid": os.getpid(), "command": command,
              "launched_utc": datetime.now(timezone.utc).isoformat(),
              "parallel_measured_trials": 1, "exclusive_parent_runs": [str(parent) for parent in parents],
              "service_health": service, "manifest_sha256": validation["manifest_sha256"]}
    write_json(out / "launch.json", launch)
    print(json.dumps(launch), flush=True)
    code = child.wait()
    launch.update(exit_code=code, exited_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "launch.json", launch)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    sys.exit(main(parser.parse_args().out))
