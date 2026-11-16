"""Run matched Live/Replay/Paused K2 trials with the current controller.

The historical Live files use the old kgua controller identity. Their specs are
the configuration source, but all three conditions are rerun together and the
historical results are never modified. Pending-effect tasks are Live-only.

Example (100 seeds per configuration, four concurrent controller processes):
  .venv/bin/python studies/k2_complete_triplets.py \
      --out runs/k2_triplets_20261026 --workers 4

Use --smoke for one seed per configuration; --resume continues the same batch.
Code and the CUDA library are snapshotted before execution. Each completed
triplet produces immutable traces, a result JSON, and a validated Replay pair.
summary.csv, intervals.csv, and status.json are refreshed after each triplet.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("c1", "c2", "c3", "k1c")
CONDITIONS = ("live", "replay", "paused")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize_spec(raw: dict) -> dict:
    d = copy.deepcopy(raw)
    if d["scenario"]["scenario_id"] not in ("moving_target", "jump_target", "balloon_select"):
        raise ValueError("Only the single-response K2 balloon tasks support this triplet protocol")
    c = d["controller"]
    if c["kind"] not in ("kgua", "chronotope") or c["planner"] != "detector":
        raise ValueError("Expected the K2 resident detector controller")
    c["kind"] = "chronotope"
    c.setdefault("ack_aware_pending", False)
    c.setdefault("receipt_timeout_ns", 0)
    d["task_clock_policy"] = "live"
    d["replay_source_live_trace_sha256"] = None
    return d


def prepare(out: Path, parts: list[str], seeds: list[int], workers: int) -> dict:
    jobs = []
    for part in parts:
        records = [json.loads(line) for line in (ROOT / f"runs/k2_{part}/trials.jsonl").read_text().splitlines()]
        for record in records:
            cell, seed = record["name"].rsplit("_s", 1)
            if int(seed) not in seeds:
                continue
            source = ROOT / f"runs/k2_{part}/{record['name']}.jsonl"
            with source.open() as f:
                header = json.loads(next(f))
            jobs.append({"part": part, "cell": cell, "seed": int(seed),
                         "source_trace": str(source.relative_to(ROOT)),
                         "source_sha256": digest(source), "spec": normalize_spec(header["spec"])})
    jobs.sort(key=lambda j: (j["seed"], parts.index(j["part"]), j["cell"]))
    if not jobs:
        raise ValueError("No matching source trials")
    out.mkdir(parents=True, exist_ok=False)
    code = out / "code"
    for rel in ("src", "chronotope/python"):
        shutil.copytree(ROOT / rel, code / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (code / "chronotope/build").mkdir(parents=True)
    shutil.copy2(ROOT / "chronotope/build/libchronotope.so", code / "chronotope/build/libchronotope.so")
    shutil.copy2(Path(__file__), code / "k2_complete_triplets.py")
    hashes = {str(p.relative_to(code)): digest(p) for p in sorted(code.rglob("*")) if p.is_file()}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "parts": parts,
                "seeds": seeds, "workers": workers, "jobs": jobs, "code_sha256": hashes,
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "protocol": "Fresh Live, Replay of that Live, independent Paused; original K2 specs with controller name migrated and inactive new fields set to defaults.",
                "excluded": {"r1": "Triplets already recorded", "p1": "Multi-action task is Live-only", "p1_kind": "Multi-action task is Live-only"}}
    write_json(out / "manifest.json", manifest)
    return manifest


def setup_worker(out: str) -> None:
    code = Path(out) / "code"
    sys.path[:0] = [str(code / "src"), str(code / "chronotope/python")]
    os.environ["CHRONOTOPE_LIB"] = str(code / "chronotope/build/libchronotope.so")
    from chronotope_core.glue import register
    register()


def job_path(out: Path, job: dict) -> Path:
    return out / job["part"] / f"{job['cell']}_s{job['seed']}"


def run_job(out_string: str, job: dict) -> dict:
    from chronotope.evaluate import evaluate_trace
    from chronotope.pair import validate_pair
    from chronotope.runner import PayloadStore, run_trial
    from chronotope.spec import TrialSpec
    from chronotope.trace import Trace
    from chronotope_core import version
    from chronotope_core.controller import build

    out = Path(out_string)
    directory = job_path(out, job)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / f"attempt_{1 + len(list(directory.glob('attempt_*'))):04d}"
    attempt.mkdir()
    start = time.monotonic()
    try:
        # Each worker has its own store: no partial-payload read/write races.
        store = PayloadStore(out / job["part"] / f"payloads_worker_{os.getpid()}")
        base = TrialSpec.from_dict(job["spec"])
        traces, results = {}, {}
        for mode in CONDITIONS:
            live = traces.get("live")
            spec = base.derive(mode, live.file_hash() if mode == "replay" else None)
            controller = build(spec)
            try:
                path = run_trial(spec, attempt / f"{mode}.jsonl", store,
                                 live_trace=live if mode == "replay" else None, controller=controller)
            finally:
                controller.close()
            trace = Trace.load(path)
            result = evaluate_trace(trace)
            if not result.valid_trial or result.contract_violations:
                raise RuntimeError(f"{mode}: invalid trial: {result.to_dict()}")
            if result.return_G is None:
                raise RuntimeError(f"{mode}: missing return on valid terminal trace")
            traces[mode] = trace
            results[mode] = result.to_dict()
        pair = validate_pair(traces["replay"], traces["live"])
        if not pair.valid:
            raise RuntimeError(f"Invalid Replay pair: {pair.reason}")
        record = {"part": job["part"], "cell": job["cell"], "seed": job["seed"],
                  "spec": base.raw, "source_trace": job["source_trace"],
                  "source_sha256": job["source_sha256"], "core_version": version(),
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_s": time.monotonic() - start, "pair": pair.__dict__, "results": results,
                  "payload_store": str(store.root.relative_to(out)),
                  "traces": {m: str(t.path.relative_to(out)) for m, t in traces.items()},
                  "trace_sha256": {m: t.file_hash() for m, t in traces.items()}}
        write_json(attempt / "result.json", record)
        write_json(directory / "result.json", record)
        return record
    except Exception:
        write_json(attempt / "error.json", {"job": job, "traceback": traceback.format_exc()})
        raise


def write_csv(path: Path, rows: list[list]) -> None:
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    atomic_text(path, buffer.getvalue())


def report(out: Path, manifest: dict, records: dict, started: float, state: str, error: str | None = None) -> None:
    grouped, planned = defaultdict(list), defaultdict(int)
    for job in manifest["jobs"]:
        planned[(job["part"], job["cell"])] += 1
    for r in records.values():
        grouped[(r["part"], r["cell"])].append(r)
    summary = [["Task", "Variant", "Study", "Configuration", "Paused SR", "Paused Return",
                "Replay SR", "Replay Return", "Live SR", "Live Return", "Completed seeds",
                "Planned seeds", "Valid Replay pairs", "Status", "Batch directory"]]
    intervals = [["Study", "Configuration", "Condition", "n", "SR", "Return"] +
                 [f"{k} {stat} (ms)" for k in ("E_O", "O_B", "B_D", "D_A", "E_A") for stat in ("mean", "median")]]
    for (part, cell), target in planned.items():
        rs = grouped[(part, cell)]
        six = []
        for mode in ("paused", "replay", "live"):
            values = [r["results"][mode] for r in rs]
            sr = f"{sum(bool(v['valid_at_receipt']) for v in values)}/{len(values)}" if values else "Pending"
            ret = mean(v["return_G"] for v in values) if values else "Pending"
            six.extend([sr, ret])
            timing = []
            for key in ("E_O", "O_B", "B_D", "D_A", "E_A"):
                xs = [v["intervals_ns"][key]/1e6 for v in values if v["intervals_ns"].get(key) is not None]
                timing.extend([mean(xs), median(xs)] if xs else ["", ""])
            intervals.append([part, cell, mode.title(), len(values), sr, ret, *timing])
        variant = {"c1": "Drift", "c2": "Jump", "c3": "Select", "k1c": "Drift"}[part]
        summary.append(["Moving Balloon", variant, part, cell, *six, len(rs), target,
                        f"{sum(r['pair']['valid'] for r in rs)}/{len(rs)}",
                        "Complete" if len(rs) == target else "Running" if rs else "Queued", str(out.relative_to(ROOT))])
    write_csv(out / "summary.csv", summary)
    write_csv(out / "intervals.csv", intervals)
    elapsed = time.monotonic() - started
    status = {"state": state, "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(),
              "completed_triplets": len(records), "planned_triplets": len(manifest["jobs"]),
              "completed_trials": 3*len(records), "planned_trials": 3*len(manifest["jobs"]),
              "workers": manifest["workers"], "elapsed_this_session_s": elapsed, "error": error}
    write_json(out / "status.json", status)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.seeds < 1 or a.workers < 1:
        ap.error("seeds and workers must be positive")
    out = a.out.resolve()
    if not out.is_relative_to(ROOT / "runs"):
        ap.error("output must be a new directory under runs/")
    if a.resume:
        manifest = json.loads((out / "manifest.json").read_text())
        for rel, expected in manifest["code_sha256"].items():
            if digest(out / "code" / rel) != expected:
                raise RuntimeError(f"Snapshotted code changed: {rel}")
    else:
        seeds = list(range(a.seed0, a.seed0 + (1 if a.smoke else a.seeds)))
        manifest = prepare(out, a.parts, seeds, a.workers)
    lock = (out / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another process is already running this batch: {out}")
    records, remaining = {}, []
    for job in manifest["jobs"]:
        result_path = job_path(out, job) / "result.json"
        key = (job["part"], job["cell"], job["seed"])
        if result_path.exists():
            r = json.loads(result_path.read_text())
            if r["source_sha256"] != job["source_sha256"] or r["spec"] != job["spec"] or not r["pair"]["valid"]:
                raise RuntimeError(f"Resume identity mismatch: {result_path}")
            for mode, rel in r["traces"].items():
                if digest(out / rel) != r["trace_sha256"][mode]:
                    raise RuntimeError(f"Completed trace changed: {rel}")
            records[key] = r
        else:
            remaining.append(job)
    start = time.monotonic()
    report(out, manifest, records, start, "running")
    print(f"Batch {out}: {len(remaining)} remaining triplets, {manifest['workers']} workers", flush=True)
    failure = None
    iterator = iter(remaining)
    with ProcessPoolExecutor(max_workers=manifest["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=setup_worker, initargs=(str(out),)) as pool:
        active = {}
        def submit_next():
            job = next(iterator, None)
            if job is not None:
                active[pool.submit(run_job, str(out), job)] = job
        for _ in range(manifest["workers"]):
            submit_next()
        while active:
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                job = active.pop(future)
                try:
                    result = future.result()
                    records[(job["part"], job["cell"], job["seed"])] = result
                    print(f"{len(records)}/{len(manifest['jobs'])} {job['part']}/{job['cell']} s{job['seed']} pair=valid ({result['elapsed_s']:.1f}s)", flush=True)
                except Exception:
                    failure = traceback.format_exc()
                    print(failure, file=sys.stderr, flush=True)
                report(out, manifest, records, start, "stopping_after_error" if failure else "running", failure)
                if failure is None:
                    submit_next()
    state = "failed" if failure else "complete"
    report(out, manifest, records, start, state, failure)
    print(f"{state}: {len(records)}/{len(manifest['jobs'])} matched triplets", flush=True)
    lock.close()
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
