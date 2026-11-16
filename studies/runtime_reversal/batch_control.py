#!/usr/bin/env python3
"""Explicit, journaled pause/resume for the current paired Notification batch.

No automatic resume, service replacement, SIGSTOP, or edits to frozen code.
--pause sends SIGTERM only to the verified runner, lets its handler close owned
controllers, waits for outstanding model requests, then archives unscored partial
files. --resume requires the original service command/environment and rechecks
all completed evidence before starting the frozen runner with --resume.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "runs/agents3_policy_radixark27b_20261028/experiment"
MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
REVISION = "319f741cce68d7914884900c138a1fbb70a42f30"
PERFORMANCE_ENV = (
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_LAUNCH_BLOCKING",
    "VLLM_USE_V1", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_TORCH_PROFILER_DIR",
    "VLLM_PROFILER_CONFIG", "VLLM_ATTENTION_BACKEND", "VLLM_NSYS_PROFILING",
    "VLLM_ENABLE_FLASHINFER_AUTOTUNE", "PYTORCH_CUDA_ALLOC_CONF",
    "TORCHINDUCTOR_COMPILE_THREADS", "OMP_NUM_THREADS", "LD_PRELOAD",
)
LAUNCH_ENV = PERFORMANCE_ENV + (
    "PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "EDGELLM_PLUGIN_PATH",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process(pid):
    try:
        root = Path("/proc") / str(int(pid))
        stat = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        command = [x.decode() for x in root.joinpath("cmdline").read_bytes().split(b"\0") if x]
        return {"pid": int(pid), "starttime_ticks": int(stat[19]), "command": command}
    except FileNotFoundError:
        return None


def matches(actual, expected):
    return bool(actual) and len(actual["command"]) >= len(expected) and actual["command"][-len(expected):] == expected


def checked_identity(launch):
    actual = process(launch["pid"])
    assert matches(actual, launch["command"]), "Recorded PID/command does not identify the expected process"
    if launch.get("process_starttime_ticks") is not None:
        assert actual["starttime_ticks"] == launch["process_starttime_ticks"], "PID was reused"
    return actual


def still_same(identity):
    return process(identity["pid"]) == identity


def environment_subset(pid, keys=PERFORMANCE_ENV):
    pairs = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    values = {}
    for pair in pairs:
        if b"=" not in pair:
            continue
        key, value = pair.split(b"=", 1)
        key = key.decode()
        if key in keys:
            values[key] = value.decode()
    return values


def processes_matching(command):
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            actual = process(path.name)
            if matches(actual, command):
                result.append(actual)
    return result


def native_processes(out):
    binary = str(out / "code/chronotope/build/chronotope")
    return [p for path in Path("/proc").iterdir() if path.name.isdigit()
            and (p := process(path.name)) and binary in p["command"] and "--serve" in p["command"]]


def service_idle(base_url):
    url = base_url.removesuffix("/v1") + "/metrics"
    with urllib.request.urlopen(url, timeout=10) as response:
        text = response.read().decode()
    groups = {"num_requests_running": [], "num_requests_waiting": []}
    for line in text.splitlines():
        for name in groups:
            if line.startswith("vllm:" + name + "{") or line.startswith("vllm:" + name + " "):
                groups[name].append(float(line.rsplit(None, 1)[1]))
    assert all(groups.values()), "Cannot establish model-service idle state"
    return all(value == 0 for values in groups.values() for value in values)


def job_id(job):
    return f"{job['arm']}_w{round(job['window_s'] * 1e9)}_{job['index']:03d}_{job['mode']}"


def validate_plan(out):
    plan = read(out / "plan.json")
    execution = read(out / "execution_manifest.json")
    assert (execution["model"], execution["revision"]) == (MODEL, REVISION)
    assert execution["planned_trials"] == 4800 and execution["qualification_trials"] == 0
    assert len(plan["jobs"]) == 4800
    seen = set()
    for job in plan["jobs"]:
        assert job["phase"] == "evaluation" and job["task"] == "notification" and job["family"] == "closing_window"
        assert job["model"] == "radixark_27b" and job["arm"] in ("legacy_cpp", "agents3_stripped")
        assert job["mode"] in ("live", "replay", "paused") and type(job["index"]) is int and 0 <= job["index"] < 100
        assert job["window_s"] in (.125, .25, .5, 1, 2, 4, 8, 16)
        key = job_id(job)
        assert key not in seen
        if job["mode"] == "replay":
            assert key.removesuffix("replay") + "live" in seen
        seen.add(key)
    return plan, execution


def verify_artifacts(out, execution):
    for relative, sha in execution["hashes"].items():
        assert digest(out / relative) == sha, relative
    for path, sha in {**execution["checkpoint_hashes"], **execution["engine_hashes"]}.items():
        assert digest(path) == sha, path
    assert digest(execution["plugin"]["path"]) == execution["plugin"]["sha256"]


AUDIT = r'''
import json,sys,hashlib
from pathlib import Path
out=Path(sys.argv[1]);code=out/'code'
sys.path[:0]=[str(code),str(code/'src'),str(code/'chronotope/python')]
from legacy_d1_w8 import setup
setup(code)
from chronotope.distributional.manifest import ManifestSet,build_manifest
from chronotope.trace import Trace
from audit_agents3_policy import audit_trial_requests
from watch_same_model_deadline import audit_record
execution=json.loads((out/'execution_manifest.json').read_text())
plan=json.loads((out/'plan.json').read_text())
ms={name:ManifestSet.load(out/'manifests'/name) for name in plan['manifest_hashes']}
for name,m in ms.items():assert m.frozen_hash==plan['manifest_hashes'][name]
count=0
for path in sorted((out/'records').glob('*.json')):
 r=json.loads(path.read_text());j=r['job'];audit_record(out,r)
 scenario=build_manifest(ms[j['manifest']].rows[j['index']])
 actual=audit_trial_requests(out,path.stem,j['arm'],Trace.load(out/r['trace']),scenario,execution)
 assert actual==r['request_audit'],path
 count+=1
print(json.dumps({'state':'passed','records':count,'request_audits_recomputed':True,'inference_calls':0}))
'''


def verify_records(out, plan, log):
    planned = {job_id(job): job for job in plan["jobs"]}
    records = sorted((out / "records").glob("*.json"))
    proof = {}
    for path in records:
        record = read(path)
        assert record["job"] == planned[path.stem]
        assert not record["outcome"]["invalid_trial"] and not record["outcome"]["contract_violations"]
        assert isinstance(record["request_audit"], list)
        if record["job"]["mode"] == "replay":
            assert record["pair_valid"] is True
        trace = out / record["trace"]
        assert digest(trace) == record["trace_sha256"]
        for p in (path, trace, out / "calls" / (path.stem + ".jsonl")):
            proof[str(p.relative_to(out))] = digest(p)
    assert {p.stem for p in records} == {job_id(j) for j in plan["jobs"][:len(records)]}, "Completed records are not an intact prefix"
    result = subprocess.run([(_CHRONOTOPE_HOME + '/agents3-venv/bin/python'), "-B", "-c", AUDIT, str(out)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    Path(log).write_text(result.stdout)
    assert result.returncode == 0, f"Completed request audit failed: {log}"
    return {"records": len(records), "hashes": proof, "request_audits_recomputed": True}


def exclusive_gpu_lock():
    lock = (ROOT / "runs/legacy_d1_gpu.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lock


def pause(out, timeout):
    plan, execution = validate_plan(out)
    initial = read(out / "status.json")
    assert initial["state"] == "running", "Only a running batch can be paused"
    launch = read(out / "launch.json")
    runner = checked_identity(launch)
    assert initial["pid"] == runner["pid"]
    service_launch = read(out / "service_launch.json")
    services = processes_matching(service_launch["command"])
    assert len(services) == 1, "Exact original model service must be identifiable"
    folder = out / "pauses" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True)
    identities = {}
    for name, path in (("auditor", out / "audit_launch.json"), ("queue", out.parent / "queue_launch.json")):
        if path.exists():
            entry = read(path)
            if process(entry["pid"]):
                identities[name] = checked_identity(entry)
    journal = {"state": "pause_requested", "requested_at": now(), "reason": "User-requested runtime-reversal investigation",
        "out": str(out), "runner": runner, "observers": identities, "initial_status": initial,
        "service": services[0], "service_command": service_launch["command"],
        "service_performance_environment": environment_subset(services[0]["pid"]),
        "runner_launch_environment": environment_subset(runner["pid"], LAUNCH_ENV),
        "completed_records_before_signal": {p.name: digest(p) for p in (out / "records").glob("*.json")},
        "plan_sha256": digest(out / "plan.json"), "execution_manifest_sha256": digest(out / "execution_manifest.json"),
        "automatic_resume": False, "signal_policy": "Only verified runner receives SIGTERM; its frozen close handler owns controller cleanup"}
    for name in ("launch.json", "audit_launch.json", "status.json", "audit_status.json"):
        if (out / name).exists():
            shutil.copy2(out / name, folder / ("before_" + name))
    write(folder / "journal.json", journal)
    assert still_same(runner), "Runner changed before signal"
    os.kill(runner["pid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while still_same(runner) or native_processes(out) or not service_idle(service_launch["base_url"]):
        if time.monotonic() > deadline:
            raise TimeoutError("Runner/owned native processes/outstanding HTTP request did not drain; no additional signals sent")
        time.sleep(1)
    terminal = read(out / "status.json")
    assert terminal["state"] in ("stopped", "complete"), terminal
    for identity in identities.values():
        while still_same(identity):
            if time.monotonic() > deadline:
                raise TimeoutError("Observer has not exited; no additional signals sent")
            time.sleep(1)
    with exclusive_gpu_lock():
        verify_artifacts(out, execution)
        proof = verify_records(out, plan, folder / "completed_request_audit.log")
        for name, sha in journal["completed_records_before_signal"].items():
            assert digest(out / "records" / name) == sha, "Completed record changed while pausing"
        recorded = {p.stem for p in (out / "records").glob("*.json")}
        recorded_traces = {read(p)["trace"] for p in (out / "records").glob("*.json")}
        current = terminal.get("current_job")
        allowed_call = job_id(current) if current else None
        allowed_trace = (f"trials/w{round(current['window_s'] * 1e9)}/{current['arm']}/{current['episode_id']}_{current['mode']}.jsonl" if current else None)
        partial = [p for p in (out / "trials").rglob("*.jsonl") if str(p.relative_to(out)) not in recorded_traces]
        partial += [p for p in (out / "calls").glob("*.jsonl") if p.stem not in recorded]
        archived = []
        for path in partial:
            relative = str(path.relative_to(out))
            assert relative == allowed_trace or relative == f"calls/{allowed_call}.jsonl", f"Unexpected unrecorded file requires review: {relative}"
            item = {"source": relative, "sha256": digest(path), "reason": "Interrupted, unrecorded trial; excluded from evaluation denominators and retried on explicit resume"}
            dest = folder / "incomplete" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            item["archive"] = str(dest.relative_to(out))
            archived.append(item)
            journal["archived_unscored_files"] = archived
            write(folder / "journal.json", journal)
        journal.update(state="complete" if terminal["state"] == "complete" else "paused", paused_at=now(), terminal_status=terminal,
            completed_evidence=proof, archived_unscored_files=archived, native_processes_remaining=[], model_requests_idle=True)
        write(folder / "journal.json", journal)
        if terminal["state"] != "complete":
            write(out / "status.json", {**terminal, "state": "paused", "updated_at": now(), "pid": None,
                "completed_trials": proof["records"], "error": None, "pause_reason": journal["reason"],
                "pause_journal": str(folder / "journal.json"), "automatic_resume": False})
    print(json.dumps({"state": journal["state"], "completed_trials": proof["records"], "journal": str(folder / "journal.json"), "archived_unscored_files": len(archived)}))


def resume(out, journal_path):
    journal = read(journal_path)
    assert journal["state"] == "paused" and journal["out"] == str(out)
    assert read(out / "status.json")["state"] == "paused"
    assert digest(out / "plan.json") == journal["plan_sha256"]
    assert digest(out / "execution_manifest.json") == journal["execution_manifest_sha256"]
    plan, execution = validate_plan(out)
    services = processes_matching(journal["service_command"])
    assert len(services) == 1, "Restore the exact original model-service command first"
    assert environment_subset(services[0]["pid"]) == journal["service_performance_environment"], "Model-service performance/profiling environment differs"
    assert service_idle(execution["model_service"]), "Model service is busy"
    with urllib.request.urlopen(execution["model_service"] + "/models", timeout=10) as response:
        assert [m["id"] for m in json.load(response)["data"]] == [MODEL]
    assert not native_processes(out)
    runner_script = str(out / "code/run_radix_agents3_policy.py")
    assert not any(p and runner_script in p["command"] for path in Path("/proc").iterdir()
                   if path.name.isdigit() for p in [process(path.name)]), "Runner already active"
    with exclusive_gpu_lock():
        verify_artifacts(out, execution)
        for relative, sha in journal["completed_evidence"]["hashes"].items():
            assert digest(out / relative) == sha, relative
        proof = verify_records(out, plan, journal_path.parent / "resume_request_audit.log")
        assert proof == journal["completed_evidence"], "Completed evidence changed while paused"
        base = [(_CHRONOTOPE_HOME + '/agents3-venv/bin/python'), "-B", str(out / "code/run_radix_agents3_policy.py"), "--out", str(out), "--run", "--resume"]
        audit = [str(ROOT / ".venv/bin/python"), "-B", str(out / "code/watch_same_model_deadline.py"), "--out", str(out)]
        env = dict(os.environ)
        for key in LAUNCH_ENV:
            if key in journal["runner_launch_environment"]:
                env[key] = journal["runner_launch_environment"][key]
            else:
                env.pop(key, None)
        journal.update(state="resume_requested", resume_requested_at=now(), restored_service=services[0])
        write(journal_path, journal)
    # Release the GPU lock immediately before the runner acquires it itself.
    with (out / "runner.log").open("ab") as log:
        runner = subprocess.Popen(base, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    write(out / "launch.json", {"pid": runner.pid, "command": base, "started_at": now(), "pause_journal": str(journal_path), "process_starttime_ticks": process(runner.pid)["starttime_ticks"]})
    deadline = time.monotonic() + 900
    while read(out / "status.json")["state"] == "paused":
        if runner.poll() is not None:
            raise RuntimeError("Runner exited before resume; inspect runner.log and journal")
        if time.monotonic() > deadline:
            raise TimeoutError("Runner did not resume within900seconds")
        time.sleep(1)
    assert read(out / "status.json")["state"] == "running", "Runner did not enter running state"
    with (out / "auditor.log").open("ab") as log:
        auditor = subprocess.Popen(audit, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    write(out / "audit_launch.json", {"pid": auditor.pid, "command": audit, "started_at": now(), "pause_journal": str(journal_path), "process_starttime_ticks": process(auditor.pid)["starttime_ticks"]})
    journal.update(state="resumed", resumed_at=now(), runner_pid=runner.pid, auditor_pid=auditor.pid)
    write(journal_path, journal)
    print(json.dumps({"state": "running", "runner_pid": runner.pid, "auditor_pid": auditor.pid, "completed_trials_at_resume": proof["records"], "journal": str(journal_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pause", action="store_true")
    mode.add_argument("--resume", type=Path, metavar="JOURNAL")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out == DEFAULT_OUT.resolve(), "This tool is scoped to the reviewed Radix paired batch"
    assert 30 <= args.timeout <= 1800
    lock = (out / "batch_control.lock").open("a+")
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.pause:
            pause(out, args.timeout)
        else:
            resume(out, args.resume.resolve())


if __name__ == "__main__":
    main()
