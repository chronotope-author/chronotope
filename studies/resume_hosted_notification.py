"""Explicitly authorized continuation; the frozen hosted runner stays unchanged.

Audit and retain every scored result, including failures. Archive unscored
attempts before rerunning only missing jobs. Each recovery manifest runs once.
No model calls occur during preparation and no automatic retry is added.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

import run_hosted_notification as base


def trial_path(out, job):
    return out / 'trials' / job['cell_id'] / job['arm'] / f'{job["episode_id"]}_{job["mode"]}.jsonl'


def select_pending(plan, records):
    jobs = {job['id']: job for job in plan['jobs']}
    if len(jobs) != len(plan['jobs']) or not records.keys() <= jobs.keys():
        raise ValueError('Duplicate jobs or unexpected scored records')
    for ident, record in records.items():
        if record['job'] != jobs[ident]:
            raise ValueError('Scored job differs from frozen plan: ' + ident)
    # Scores never determine eligibility for continuation.
    pending = [job for job in plan['jobs'] if job['id'] not in records]
    seen = set(records)
    for job in pending:
        if job['mode'] == 'replay' and job['id'].removesuffix('_replay') + '_live' not in seen:
            raise ValueError('Replay has no completed or preceding Live job')
        seen.add(job['id'])
    return pending


def evidence_inventory(out, plan, records):
    retained, interrupted = {}, {}
    known = set()
    for job in plan['jobs']:
        paths = [trial_path(out, job), out / 'calls' / f'{job["id"]}.jsonl']
        known.update(paths)
        target = retained if job['id'] in records else interrupted
        if job['id'] in records:
            paths.append(out / 'records' / f'{job["id"]}.json')
        for path in paths:
            if path.exists():
                target[str(path.relative_to(out))] = base.digest(path)
            elif job['id'] in records:
                raise ValueError('Completed evidence is missing: ' + str(path))
    actual = set((out / 'trials').rglob('*.jsonl')) | set((out / 'calls').glob('*.jsonl'))
    if not actual <= known:
        raise ValueError('Unrecognized evidence files')
    return retained, interrupted


def verify_hashes(root, hashes):
    for relative, expected in hashes.items():
        if base.digest(root / relative) != expected:
            raise ValueError('Preserved evidence changed: ' + relative)


def audit_completed(out, plan, records):
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    from chronotope.distributional.manifest import EpisodeRow, build_manifest
    from chronotope.distributional.episode import score_trace, verify_trace_evidence
    cases = base.load_cases(out, plan)
    for job in plan['jobs']:
        if job['id'] not in records:
            continue
        record = records[job['id']]
        path = trial_path(out, job)
        trace = Trace.load(path)
        if record['trace'] != str(path.relative_to(out)) or trace.file_hash() != record['trace_sha256']:
            raise ValueError('Completed trace provenance changed')
        case = cases[job['manifest']][job['index']]
        row = EpisodeRow.from_dict(case['distributional_row']); scenario = build_manifest(row)
        verify_trace_evidence(row, scenario, trace, case['distributional_manifest_hash'])
        outcome = score_trace(row, trace, case['distributional_manifest_hash'], controller=job['arm'], repeat_index=0)
        if outcome.invalid_trial or outcome.contract_violations or outcome.to_dict() != record['outcome']:
            raise ValueError('Completed score does not pass evidence audit')
        calls = out / 'calls' / f'{job["id"]}.jsonl'
        if base.call_audit(calls, job['arm'], plan['model'], trace, scenario) != record['request_audit']:
            raise ValueError('Completed request audit differs')
        if job['mode'] == 'replay':
            live = Trace.load(trial_path(out, dict(job, mode='live')))
            if not validate_pair(trace, live).valid or record['pair_valid'] is not True:
                raise ValueError('Completed Replay pair is invalid')


def prepare(out, authorization):
    if not authorization.strip():
        raise ValueError('Explicit user authorization is required')
    status = base.read(out / 'status.json')
    if status['state'] not in ('failed', 'stopped', 'paused'):
        raise ValueError('Only an explicitly authorized interrupted cohort can resume')
    with (base.ROOT / 'runs/legacy_d1_gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.configure(out); base.validate(out)
        frozen = base.read(out / 'execution_manifest.json')
        for name, expected in frozen['hashes'].items():
            if base.digest(name) != expected:
                raise ValueError('Frozen artifact changed: ' + name)
        plan = base.read(out / 'plan.json')
        records = {p.stem: base.read(p) for p in (out / 'records').glob('*.json')}
        pending = select_pending(plan, records)
        if len(records) != status['completed_trials'] or not pending:
            raise ValueError('Status count differs or no jobs remain')
        audit_completed(out, plan, records)
        retained, interrupted = evidence_inventory(out, plan, records)
        folder = out / 'recoveries' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        folder.mkdir(parents=True)
        shutil.copy2(out / 'status.json', folder / 'status_before.json')
        if (out / 'summary.csv').exists():
            shutil.copy2(out / 'summary.csv', folder / 'summary_before.csv')
        shutil.copy2(__file__, folder / 'resume_hosted_notification.py')
        manifest = {'schema': 'hosted-explicit-resume-1', 'created_at': base.now(),
            'user_authorization': authorization, 'cohort': str(out),
            'plan_sha256': base.digest(out / 'plan.json'),
            'execution_manifest_sha256': base.digest(out / 'execution_manifest.json'),
            'helper_sha256': base.digest(__file__), 'prior_status_sha256': base.digest(out / 'status.json'),
            'completed_ids': [job['id'] for job in plan['jobs'] if job['id'] in records],
            'pending_ids': [job['id'] for job in pending],
            'retained_evidence_sha256': retained, 'interrupted_evidence_sha256': interrupted,
            'planned_trials': len(plan['jobs']), 'qualification_trials': 0,
            'automatic_resume': False, 'successor_jobs': [], 'preparation_model_calls': 0}
        base.write(folder / 'manifest.json', manifest)
        return folder / 'manifest.json'


def archive_interrupted(out, folder, hashes):
    verify_hashes(out, hashes)
    for relative in hashes:
        target = folder / 'interrupted' / relative
        if target.exists():
            raise FileExistsError(target)
    for relative in hashes:
        target = folder / 'interrupted' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        (out / relative).rename(target)
    verify_hashes(folder / 'interrupted', hashes)


def run(out, recovery):
    manifest = base.read(recovery); folder = recovery.parent
    if manifest['schema'] != 'hosted-explicit-resume-1' or manifest['cohort'] != str(out):
        raise ValueError('Recovery belongs to a different cohort')
    for path, key in [(out / 'plan.json', 'plan_sha256'),
                      (out / 'execution_manifest.json', 'execution_manifest_sha256'),
                      (Path(__file__), 'helper_sha256'), (out / 'status.json', 'prior_status_sha256')]:
        if base.digest(path) != manifest[key]:
            raise ValueError('Recovery provenance changed: ' + str(path))
    with (base.ROOT / 'runs/legacy_d1_gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Original full preflight: model predecessor, hashes, cases, grounder and credential.
        model = base.preflight(out)
        plan = base.read(out / 'plan.json')
        prior = {p.stem: base.read(p) for p in (out / 'records').glob('*.json')}
        pending = select_pending(plan, prior)
        if set(prior) != set(manifest['completed_ids']) or [j['id'] for j in pending] != manifest['pending_ids']:
            raise ValueError('Recovery schedule changed')
        verify_hashes(out, manifest['retained_evidence_sha256'])
        verify_hashes(out, manifest['interrupted_evidence_sha256'])
        # Exclusive creation makes a manifest single-use, even after another failure.
        with (folder / 'started.json').open('x') as stream:
            json.dump({'started_at': base.now(), 'pid': os.getpid(),
                       'manifest_sha256': base.digest(recovery)}, stream)
        archive_interrupted(out, folder, manifest['interrupted_evidence_sha256'])
        execute_pending(out, recovery, manifest, model, plan, prior, pending)


def execute_pending(out, recovery, manifest, model, plan, prior, pending):
    from chronotope.controller import CallBudget
    from chronotope.runner import run_trial, PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    from chronotope_core.controller import build as build_native
    from chronotope.distributional.manifest import EpisodeRow, build_manifest
    from chronotope.distributional.episode import verify_trace_evidence, score_trace
    from run_core5_detector_full_agents3 import make_host
    cases = base.load_cases(out, plan)
    full = base.read(out / 'configs/full_agent_components.json')
    records = [prior[job['id']] for job in plan['jobs'] if job['id'] in prior]
    current = None

    def publish(state, error=None):
        base.summarize(out, records)
        base.write(out / 'status.json', {'state': state, 'updated_at': base.now(), 'model': model,
            'completed_trials': len(records), 'planned_trials': len(plan['jobs']), 'qualification_trials': 0,
            'pid': os.getpid() if state == 'running' else None, 'current_job': current,
            'error': error, 'automatic_resume': False, 'retained_completed_trials': len(prior),
            'recovery_manifest': str(recovery.relative_to(out)), 'recovery_manifest_sha256': base.digest(recovery)})

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for current in pending:
            publish('running')
            ident, arm, mode = current['id'], current['arm'], current['mode']
            case = cases[current['manifest']][current['index']]
            config = base.read(out / 'configs' / f'notification_{arm}.json')
            path = trial_path(out, current); path.parent.mkdir(parents=True, exist_ok=True)
            calls = out / 'calls' / f'{ident}.jsonl'
            if path.exists() or calls.exists() or (out / 'records' / f'{ident}.json').exists():
                raise FileExistsError('Refusing to overwrite evidence for ' + ident)
            live = Trace.load(trial_path(out, dict(current, mode='live'))) if mode == 'replay' else None
            spec = base.make_trial_spec(case, config, mode, live.file_hash() if live else None)
            row = EpisodeRow.from_dict(case['distributional_row']); scenario = build_manifest(row)
            os.environ['CHRONOTOPE_PLANNER_AUDIT'] = str(calls)
            adapter = base.HostedAgentS3Adapter(full, calls) if arm == base.ARMS[1] else None
            controller = None; start = time.monotonic()
            try:
                controller = build_native(spec, budget=CallBudget(100)) if arm == base.ARMS[0] else make_host(spec, 'notification', adapter, {'code_action': 'defer_unsupported'})
                run_trial(spec, path, PayloadStore(out / 'payloads'), live_trace=live, controller=controller, manifest=scenario)
            finally:
                if controller and hasattr(controller, 'close'): controller.close()
                if adapter: adapter.close()
            trace = Trace.load(path)
            if trace.all('decision_failed') or (adapter and adapter.errors):
                raise RuntimeError('Infrastructure error; interrupted trace retained, no scored record')
            request_evidence = base.call_audit(calls, arm, model, trace, scenario)
            verify_trace_evidence(row, scenario, trace, case['distributional_manifest_hash'])
            outcome = score_trace(row, trace, case['distributional_manifest_hash'], controller=arm, repeat_index=0)
            pair = validate_pair(trace, live) if live else None
            if outcome.invalid_trial or outcome.contract_violations or (pair and not pair.valid):
                raise RuntimeError('Invalid trial contract or Replay pair; evidence retained unscored')
            record = {'job': current, 'outcome': outcome.to_dict(), 'request_audit': request_evidence,
                'trace': str(path.relative_to(out)), 'trace_sha256': trace.file_hash(),
                'pair_valid': pair.valid if pair else None, 'wall_seconds': time.monotonic() - start,
                'completed_at': base.now()}
            base.write(out / 'records' / f'{ident}.json', record); records.append(record)
            print(f'{len(records)}/3000 {ident} success={outcome.Y}', flush=True)
        current = None; publish('auditing')
        verify_hashes(out, manifest['retained_evidence_sha256'])
        verify_hashes(recovery.parent / 'interrupted', manifest['interrupted_evidence_sha256'])
        base.audit(out)
        base.write(recovery.parent / 'completion_audit.json', {'state': 'passed',
            'retained_trials_unchanged': len(prior), 'resumed_trials': len(pending),
            'completed_trials': len(records), 'recovery_manifest_sha256': base.digest(recovery),
            'original_audit_sha256': base.digest(out / 'completion_audit.json')})
        publish('complete')
    except BaseException as exc:
        publish('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--authorization')
    parser.add_argument('--recovery', type=Path)
    args = parser.parse_args(); out = args.out.resolve()
    if args.action == 'prepare':
        if not args.authorization: parser.error('--authorization is required')
        print(prepare(out, args.authorization))
    else:
        if not args.recovery: parser.error('--recovery is required')
        run(out, args.recovery.resolve())
