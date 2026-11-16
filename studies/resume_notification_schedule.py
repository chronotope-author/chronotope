"""Explicit scheduling-only continuation of the frozen Notification experiment.

The original plan, execution manifest, runner and scored records remain immutable.
Only unfinished subsecond jobs may be deferred. Controller construction, model
settings, inference, scoring and evidence audits delegate to the frozen runner.
The run body below is an isolated copy of that runner's run function; its changes
are confined to scheduling, provenance and status counts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request

SCHEMA = 'notification-continuation-schedule-2'
SCHEDULE_NAME = 'continuation_schedule.json'
FROZEN_RUNNER = 'code/run_legacy_detector_full_agents3.py'
ORIGINAL_WINDOWS = {16., 8., 4., 2., 1., .5, .25, .125}
SUBSECOND_WINDOWS = frozenset((.5, .25, .125))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def job_id(job):
    return f"{job['arm']}_w{round(job['window_s']*1e9)}_{job['index']:03d}_{job['mode']}"


def _plan_jobs(plan):
    assert plan['planned_trials'] == len(plan['jobs']) == 4800
    assert plan['qualification_trials'] == 0
    assert {job['window_s'] for job in plan['jobs']} == ORIGINAL_WINDOWS
    assert {job['task'] for job in plan['jobs']} == {'notification'}
    assert {job['phase'] for job in plan['jobs']} == {'evaluation'}
    assert {job['arm'] for job in plan['jobs']} == {'legacy_cpp_detector', 'agents3_full'}
    assert {job['mode'] for job in plan['jobs']} == {'live', 'replay', 'paused'}
    jobs = {job_id(job): job for job in plan['jobs']}
    assert len(jobs) == 4800
    cells = defaultdict(set)
    for job in jobs.values():
        cells[job['arm'], job['window_s'], job['mode']].add(job['index'])
    assert len(cells) == 48 and all(indices == set(range(100)) for indices in cells.values())
    return jobs


def validate_schedule(out, schedule=None, *, verify_records=True, plan=None):
    out = Path(out)
    schedule = read(out / SCHEDULE_NAME) if schedule is None else schedule
    plan = read(out / 'plan.json') if plan is None else plan
    jobs = _plan_jobs(plan)
    assert schedule['schema'] == SCHEMA
    deferred_windows = schedule['deferred_windows_s']
    assert deferred_windows == sorted(set(deferred_windows), reverse=True)
    assert deferred_windows and set(deferred_windows) <= SUBSECOND_WINDOWS
    assert schedule['original_planned_trials'] == 4800
    assert schedule['qualification_trials'] == 0
    assert isinstance(schedule['user_authorization'], str) and schedule['user_authorization'].strip()
    for key, path in [('original_plan_sha256', out / 'plan.json'),
                      ('original_execution_sha256', out / 'execution_manifest.json'),
                      ('frozen_runner_sha256', out / FROZEN_RUNNER),
                      ('helper_sha256', Path(__file__))]:
        assert schedule[key] == digest(path), f'Scheduling provenance changed: {path}'
    retained = schedule['retained_completed_ids']
    assert len(retained) == len(set(retained)) and set(retained) <= jobs.keys()
    expected = [job_id(job) for job in plan['jobs']
                if job['window_s'] not in deferred_windows or job_id(job) in retained]
    expected_set = set(expected)
    deferred = [job_id(job) for job in plan['jobs'] if job_id(job) not in expected_set]
    assert schedule['scheduled_job_ids'] == expected
    assert schedule['deferred_job_ids'] == deferred
    assert schedule['planned_trials'] == len(expected)
    assert schedule['deferred_trials'] == len(deferred)
    assert len(expected) == 4800 - 600 * len(deferred_windows) + sum(
        jobs[ident]['window_s'] in deferred_windows for ident in retained)
    assert set(schedule['retained_record_sha256']) == set(retained)
    if verify_records:
        records = {path.stem: path for path in (out / 'records').glob('*.json')}
        assert set(retained) <= records.keys() <= set(expected), 'Unexpected or missing scored records'
        for ident in retained:
            assert digest(records[ident]) == schedule['retained_record_sha256'][ident], ident
        for ident, path in records.items():
            record = read(path)
            assert record['job'] == jobs[ident]
            assert digest(out / record['trace']) == record['trace_sha256']
            assert not record['outcome']['invalid_trial'] and not record['outcome']['contract_violations']
    return schedule


def load_execution_schedule(out, plan=None):
    """Read and validate scheduling/provenance only; full record audit is separate."""
    path = Path(out) / SCHEDULE_NAME
    if not path.exists():
        return None
    return validate_schedule(out, read(path), verify_records=False, plan=plan)


def scheduled_jobs(out, plan=None):
    out = Path(out)
    plan = read(out / 'plan.json') if plan is None else plan
    schedule = load_execution_schedule(out, plan)
    if schedule is None:
        return plan['jobs']
    selected = set(schedule['scheduled_job_ids'])
    return [job for job in plan['jobs'] if job_id(job) in selected]


def planned_count(out):
    schedule = load_execution_schedule(out)
    return schedule['planned_trials'] if schedule is not None else 4800


def completion_gate_for(out, status=None):
    out = Path(out)
    status = read(out / 'status.json') if status is None else status
    try:
        schedule = load_execution_schedule(out)
        count = schedule['planned_trials'] if schedule else 4800
        expected_state = 'complete_scheduled' if schedule else 'complete'
        if status.get('state') != expected_state:
            return False, 'predecessor_' + str(status.get('state', 'missing'))
        if status.get('completed_trials') != count or status.get('planned_trials') != count:
            return False, 'predecessor_incomplete_count'
        if schedule and status.get('schedule_sha256') != digest(out / SCHEDULE_NAME):
            return False, 'predecessor_schedule_mismatch'
        if status.get('qualification_trials') != 0 or status.get('error'):
            return False, 'predecessor_metadata_error'
        return True, expected_state
    except (AssertionError, KeyError, ValueError, FileNotFoundError) as error:
        return False, f'predecessor_invalid_schedule: {error}'


def _active_runner(out):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (state != 'Z' and str(out).encode() in argv and b'--run' in argv
                and any(arg.endswith((b'run_legacy_detector_full_agents3.py',
                                      b'resume_notification_schedule.py')) for arg in argv)):
            return int(proc.name)
    return None


def create_schedule(out, user_authorization, *, deferred_windows_s=SUBSECOND_WINDOWS):
    """Create once, only after an explicitly authorized controlled stop."""
    out = Path(out).resolve()
    assert isinstance(user_authorization, str) and user_authorization.strip()
    deferred_windows = sorted(set(deferred_windows_s), reverse=True)
    assert deferred_windows and set(deferred_windows) <= SUBSECOND_WINDOWS
    assert not (out / SCHEDULE_NAME).exists(), 'Never overwrite a continuation schedule'
    status = read(out / 'status.json')
    assert status['state'] in ('stopped', 'paused'), 'Controlled stop required before scheduling'
    assert _active_runner(out) is None, 'Original/continuation runner is still alive'
    plan = read(out / 'plan.json'); jobs = _plan_jobs(plan)
    paths = {path.stem: path for path in (out / 'records').glob('*.json')}
    assert len(paths) == status['completed_trials']
    retained = [job_id(job) for job in plan['jobs'] if job_id(job) in paths]
    assert set(paths) == set(retained) <= jobs.keys()
    selected = [job_id(job) for job in plan['jobs']
                if job['window_s'] not in deferred_windows or job_id(job) in paths]
    selected_set = set(selected)
    deferred = [job_id(job) for job in plan['jobs'] if job_id(job) not in selected_set]
    schedule = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
                'user_authorization': user_authorization, 'deferred_windows_s': deferred_windows,
                'original_planned_trials': 4800, 'planned_trials': len(selected),
                'deferred_trials': len(deferred), 'qualification_trials': 0,
                'retained_completed_ids': retained, 'scheduled_job_ids': selected,
                'deferred_job_ids': deferred,
                'retained_record_sha256': {ident: digest(paths[ident]) for ident in retained},
                'original_plan_sha256': digest(out / 'plan.json'),
                'original_execution_sha256': digest(out / 'execution_manifest.json'),
                'frozen_runner_sha256': digest(out / FROZEN_RUNNER),
                'helper_sha256': digest(__file__),
                'interrupted_unscored_traces': 'Preserved in place, excluded from scheduled jobs and denominators',
                'controller_changes': False, 'automatic_resume': False}
    validate_schedule(out, schedule, plan=plan)
    with (out / SCHEDULE_NAME).open('x') as handle:
        json.dump(schedule, handle, indent=2); handle.write('\n')
    return schedule


def _load_frozen_runner(out):
    path = Path(out) / FROZEN_RUNNER
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('_frozen_notification_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complete_existing_scope(out, schedule):
    """Close an already scored subset using files only; no model or controller loads."""
    active = _active_runner(out)
    assert active is None or active == os.getpid(), 'Another experiment runner is alive'
    records = {path.stem for path in (out / 'records').glob('*.json')}
    if records != set(schedule['scheduled_job_ids']):
        return False
    previous = read(out / 'status.json')
    assert previous['state'] in ('stopped', 'paused'), 'Controlled stop required to close existing scope'
    execution = read(out / 'execution_manifest.json')
    status = {'state': 'complete_scheduled',
              'started_at': previous.get('started_at'),
              'updated_at': datetime.now(timezone.utc).isoformat(),
              'completed_trials': len(records), 'planned_trials': schedule['planned_trials'],
              'original_planned_trials': 4800, 'deferred_trials': schedule['deferred_trials'],
              'deferred_windows_s': schedule['deferred_windows_s'],
              'schedule_sha256': digest(out / SCHEDULE_NAME),
              'pid': None, 'current_job': None, 'qualification_trials': 0,
              'model': execution.get('model', previous.get('model')),
              'error': None, 'automatic_resume': False,
              'completed_without_new_trials': True,
              'model_inference_calls_this_continuation': 0}
    temporary = out / 'status.json.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(out / 'status.json')
    print(json.dumps({'state': 'complete_scheduled', 'completed_trials': len(records),
                      'planned_trials': schedule['planned_trials'], 'new_trials': 0,
                      'model_inference_calls': 0}), flush=True)
    return True


def run(out, resume=False):
    assert resume is True, 'Continuation requires an explicit --resume flag'
    out = Path(out).resolve()
    schedule = validate_schedule(out)
    if _complete_existing_scope(out, schedule):
        return
    frozen = _load_frozen_runner(out)
    require_parent_stopped, setup = frozen.require_parent_stopped, frozen.setup
    validate_plan, same_cohort = frozen.validate_plan, frozen.same_cohort
    verify_hashes, write, now = frozen.verify_hashes, frozen.write, frozen.now
    ROOT, ARMS = frozen.ROOT, frozen.ARMS
    MODEL, REVISION = frozen.MODEL, frozen.REVISION
    PRECISION, RUNTIME_PRECISION = frozen.PRECISION, frozen.RUNTIME_PRECISION
    GROUNDER_REVISION = frozen.GROUNDER_REVISION
    schedule_sha256 = digest(out / SCHEDULE_NAME)
    execution_jobs = scheduled_jobs(out)
    require_parent_stopped(); setup(out / 'code')
    from agents3_full import FullAgentS3Adapter
    from audit_legacy_detector_full_agents3 import audit_trial_requests
    from chronotope.distributional.manifest import ManifestSet, build_manifest
    from chronotope.distributional.episode import ControllerSpec, trial_spec_for, make_controller, verify_trace_evidence, score_trace
    from chronotope.distributional.metrics import percentiles
    from chronotope.controller import CallBudget
    from chronotope.runner import run_trial, PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    plan = read(out / 'plan.json'); validate_plan(plan); same_cohort(plan)
    cfg = read(out / 'execution_manifest.json')
    verify_hashes(cfg['hashes'], out); verify_hashes(cfg['checkpoint_hashes']); verify_hashes(cfg['engine_hashes'])
    verify_hashes(cfg['grounder_checkpoint']['file_hashes'])
    assert digest(cfg['plugin']['path']) == cfg['plugin']['sha256']
    for name in ('CHRONOTOPE_AGENT_S3_POLICY', 'CHRONOTOPE_AGENT_S3_PROMPT', 'CHRONOTOPE_AGENT_S3_AUDIT'):
        os.environ.pop(name, None)
    os.environ['CHRONOTOPE_PLANNER_PROMPT'] = str(out / 'configs/legacy_compact_prompt.json')
    for component in cfg['agent_s3_components']['components'].values():
        with urllib.request.urlopen(component['base_url'].rstrip('/') + '/models', timeout=15) as response:
            assert component['model'] in [m['id'] for m in json.load(response)['data']]
    lock = (ROOT / 'runs/legacy_d1_gpu.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifests = {name: ManifestSet.load(out / 'manifests' / name) for name in plan['manifest_hashes']}
    configs = {arm: read(out / 'configs' / f'{arm}.json') for arm in ARMS}
    previous = read(out / 'status.json'); records = [read(p) for p in sorted((out / 'records').glob('*.json'))]
    if records and not resume: raise RuntimeError('Existing records require explicit --resume')
    planned = {job_id(j): j for j in plan['jobs']}; completed = set()
    for record in records:
        ident = job_id(record['job'])
        assert ident in planned and record['job'] == planned[ident] and ident not in completed
        assert digest(out / record['trace']) == record['trace_sha256']
        assert not record['outcome']['invalid_trial'] and not record['outcome']['contract_violations']
        completed.add(ident)
    started = previous.get('started_at', now()) if resume else now(); current = None
    def publish(state, error=None):
        groups = defaultdict(list)
        for r in records: groups[r['job']['arm'], r['job']['window_s'], r['job']['mode']].append(r)
        rows = []
        for (arm, window, mode), own in sorted(groups.items()):
            outcomes = [r['outcome'] for r in own]; native = arm == ARMS[0]
            row = {'Task': 'Disappearing Notification', 'Controller': cfg['arm_labels'][arm],
                'Model': MODEL, 'Revision': REVISION, 'Source precision': PRECISION, 'Precision': RUNTIME_PRECISION,
                'Runtime': 'TensorRT-Edge-LLM' if native else 'vLLM', 'KV cache': 'FP16',
                'Detector / grounder': 'GPU task-color detector; NCC fallback' if native else cfg['agent_s3_components']['components']['grounder']['model'],
                'Grounder revision': None if native else GROUNDER_REVISION,
                'Grounder source precision': None if native else 'FP32 safetensors; checkpoint config declares BF16',
                'Grounder precision': None if native else 'BF16',
                'Grounder runtime': None if native else 'vLLM',
                'Planner generation budget': 64 if native else 4096,
                'History': 'No model-consumed history; legacy GPU clock bookkeeping' if native else 'Upstream CPU text and screenshot history',
                'Prompt': 'Compact JSON, version4' if native else 'Upstream Agent S3 Worker prompt',
                'Window (s)': window, 'Condition': mode, 'n': len(own), 'Successes': sum(o['Y'] for o in outcomes),
                'SR': sum(o['Y'] for o in outcomes)/len(own), 'No receipt': sum(not o['R'] for o in outcomes),
                'Planner calls': sum(e['role'] == 'planner' for r in own for e in r['request_audit']),
                'Reflection calls': sum(e['role'] == 'reflection' for r in own for e in r['request_audit']),
                'Grounder calls': sum(e['role'] == 'grounder' for r in own for e in r['request_audit']),
                'Text grounder calls': sum(e['role'] == 'text_grounder' for r in own for e in r['request_audit']),
                'Invalid': sum(o['invalid_trial'] for o in outcomes), 'Replay pairs valid': sum(r['pair_valid'] is True for r in own)}
            for label, key in [('E-to-O', 'L_EO'), ('O-to-B', 'L_OB'), ('B-to-D', 'L_BD'), ('D-to-A', 'L_DA'), ('E-to-A', 'L_EA')]:
                for pc, value in percentiles([o[key] for o in outcomes]).items():
                    if pc in ('p50', 'p95'): row[f'{label} {pc} (ms)'] = None if value is None else value/1e6
            rows.append(row)
        if rows:
            with (out / 'summary.csv.tmp').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            (out / 'summary.csv.tmp').replace(out / 'summary.csv')
        write(out / 'status.json', {'state': state, 'started_at': started, 'updated_at': now(), 'completed_trials': len(records),
            'planned_trials': schedule['planned_trials'], 'original_planned_trials': 4800,
            'deferred_trials': schedule['deferred_trials'], 'deferred_windows_s': schedule['deferred_windows_s'],
            'schedule_sha256': schedule_sha256,
            'pid': os.getpid() if state == 'running' else None, 'current_job': current,
            'qualification_trials': 0, 'model': MODEL, 'error': error, 'automatic_resume': False})
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    publish('running')
    try:
        for current in execution_jobs:
            ident = job_id(current)
            if ident in completed: continue
            publish('running'); arm = current['arm']; mode = current['mode']
            manifest = manifests[current['manifest']]; row = manifest.rows[current['index']]; scenario = build_manifest(row)
            controller_spec = ControllerSpec(arm, configs[arm])
            folder = out / 'trials' / f"w{round(current['window_s']*1e9)}" / arm; folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{row.episode_id}_{mode}.jsonl'; assert not path.exists()
            live = Trace.load(folder / f'{row.episode_id}_live.jsonl') if mode == 'replay' else None
            spec = trial_spec_for(row, controller_spec, mode, live.file_hash() if live else None)
            call_log = out / 'calls' / f'{ident}.jsonl'
            adapter = FullAgentS3Adapter(cfg['agent_s3_components'], call_log) if arm == ARMS[1] else None
            os.environ['CHRONOTOPE_PLANNER_AUDIT'] = str(call_log)
            controller = make_controller(spec, scenario, controller_spec, budget=CallBudget(100), vlm_adapter=adapter)
            t0 = time.monotonic()
            try: run_trial(spec, path, PayloadStore(out / 'payloads'), live_trace=live, controller=controller, manifest=scenario)
            finally:
                close = getattr(controller, 'close', None)
                if close: close()
                if adapter: adapter.close()
            trace = Trace.load(path)
            if trace.all('decision_failed') or (adapter and adapter.errors):
                raise RuntimeError(f'Controller/runtime infrastructure error: {ident}')
            request_audit = audit_trial_requests(out, ident, arm, trace, scenario, cfg)
            verify_trace_evidence(row, scenario, trace, manifest.frozen_hash)
            assert trace.header()['spec']['controller'] == configs[arm]
            outcome = score_trace(row, trace, manifest.frozen_hash, controller=arm, repeat_index=0)
            pair = validate_pair(trace, live) if live else None
            record = {'job': current, 'request_audit': request_audit, 'outcome': outcome.to_dict(),
                'trace': str(path.relative_to(out)), 'trace_sha256': trace.file_hash(),
                'pair_valid': pair.valid if pair else None, 'pair_reason': pair.reason if pair else None,
                'wall_seconds': time.monotonic()-t0, 'completed_at': now()}
            write(out / 'records' / f'{ident}.json', record); records.append(record); completed.add(ident)
            if outcome.invalid_trial or outcome.contract_violations or (pair and not pair.valid):
                raise RuntimeError(f'Invalid trace/pair: {ident}')
            print(f"{len(records)}/{schedule['planned_trials']} {ident} success={outcome.Y}", flush=True); publish('running')
        assert len(records) == schedule['planned_trials']
        current = None; publish('complete_scheduled')
    except BaseException as error:
        publish('stopped' if isinstance(error, KeyboardInterrupt) else 'failed', f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ('create', 'validate', 'run'):
        actions.add_argument('--' + action, action='store_true')
    parser.add_argument('--authorization')
    parser.add_argument('--defer-windows', type=float, nargs='+', default=sorted(SUBSECOND_WINDOWS, reverse=True),
                        help='Authorized existing subsecond windows to defer (default: 0.5 0.25 0.125)')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.run and not args.resume:
        parser.error('--run requires explicit --resume')
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if args.create:
        if not args.authorization:
            parser.error('--create requires the actual explicit --authorization text')
        result = create_schedule(args.out, args.authorization, deferred_windows_s=args.defer_windows)
        print(json.dumps({'state': 'scheduled', 'planned_trials': result['planned_trials'],
                          'deferred_trials': result['deferred_trials'], 'inference_calls': 0}))
    elif args.validate:
        result = validate_schedule(args.out)
        print(json.dumps({'state': 'passed', 'planned_trials': result['planned_trials'],
                          'deferred_trials': result['deferred_trials'], 'inference_calls': 0}))
    else:
        run(args.out, resume=args.resume)
