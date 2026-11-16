"""Pause only the active NVIDIA prompt-only batch for timing instrumentation.

This helper never launches inference, changes a service, resumes evaluation, or
edits frozen code. It preserves completed evidence and archives interrupted,
unscored files. Only the verified runner receives SIGTERM; an orphaned native
constructor may be closed through its own verified private Unix socket.
"""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from runtime_reversal.batch_control import (
    ROOT, PERFORMANCE_ENV, LAUNCH_ENV, checked_identity, digest, environment_subset,
    exclusive_gpu_lock, now, process, read, service_idle, still_same, write,
)
from run_nvidia_prompt_only import MODEL, REVISION, validate_plan as validate_job_plan

OUT = ROOT / 'runs/nvidia_prompt_only_20261029/experiment'
RUNNER = OUT / 'code/run_nvidia_prompt_only.py'
NATIVE = OUT / 'code/chronotope/build/chronotope'


def validate_plan(out):
    assert Path(out).resolve() == OUT
    plan = read(out / 'plan.json')
    validate_job_plan(plan)
    execution = read(out / 'execution_manifest.json')
    assert (execution['model'], execution['revision']) == (MODEL, REVISION)
    assert execution['planned_trials'] == 4800 and execution['qualification_trials'] == 0
    return plan, execution


def native_processes(out):
    return owned_native_processes(out, read(out / 'execution_manifest.json'))


def owned_native_processes(out, execution):
    """Match executable identity AND this run's explicit audit destination."""
    assert digest(NATIVE) == execution['hashes']['code/chronotope/build/chronotope']
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        identity = process(directory.name)
        if not identity:
            continue
        argv = identity['command']
        if not argv or argv[0] != str(NATIVE) or '--serve' not in argv:
            continue
        assert '--agents3-prompt-audit' in argv, 'Native run ownership is unclear'
        index = argv.index('--agents3-prompt-audit')
        audit = Path(argv[index + 1]).resolve()
        assert audit.parent == (out / 'calls').resolve(), 'Native audit belongs to another run'
        identity['socket_path'] = argv[argv.index('--serve') + 1]
        identity['audit_path'] = str(audit)
        found.append(identity)
    return found


def close_orphan(identity):
    """After runner exit, close only the recorded native process via its socket."""
    current = process(identity['pid'])
    expected = {k: identity[k] for k in ('pid', 'starttime_ticks', 'command')}
    if current is None:
        return False
    assert current == expected, 'Native identity changed before socket close'
    path = Path(identity['socket_path'])
    assert path.parent.parent == Path('/tmp') and path.parent.name.startswith('chronotope_sock_')
    assert path.name == 'agent.sock', 'Unexpected native socket path'
    if not path.exists():
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2)
        try:
            connection.connect(str(path))
            connection.sendall(b'{"op":"close"}\n')
            response = connection.recv(4096)
        except (ConnectionRefusedError, FileNotFoundError, TimeoutError):
            return False
    assert response and json.loads(response.splitlines()[0]).get('ok') is True
    return True


AUDIT = r'''
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);code=out/'code'
sys.path[:0]=[str(code),str(code/'src'),str(code/'chronotope/python')]
from legacy_d1_w8 import setup
setup(code)
from chronotope.distributional.manifest import ManifestSet,build_manifest
from chronotope.trace import Trace
from audit_nvidia_prompt_only import audit_trial_requests
from watch_same_model_deadline import audit_record
execution=json.loads((out/'execution_manifest.json').read_text())
plan=json.loads((out/'plan.json').read_text())
manifests={name:ManifestSet.load(out/'manifests'/name) for name in plan['manifest_hashes']}
for name,m in manifests.items():assert m.frozen_hash==plan['manifest_hashes'][name]
count=0
for path in sorted((out/'records').glob('*.json')):
    record=json.loads(path.read_text());job=record['job'];audit_record(out,record)
    scenario=build_manifest(manifests[job['manifest']].rows[job['index']])
    actual=audit_trial_requests(out,path.stem,job['arm'],Trace.load(out/record['trace']),scenario,execution)
    assert actual==record['request_audit'],path
    count+=1
print(json.dumps({'state':'passed','records':count,'request_audits_recomputed':True,'inference_calls':0}))
'''


def job_id(job):
    return f"{job['arm']}_w{round(job['window_s']*1e9)}_{job['index']:03d}_{job['mode']}"


def verify_artifacts(out, execution):
    for relative, sha in execution['hashes'].items():
        assert digest(out / relative) == sha, relative
    for path, sha in {**execution['checkpoint_hashes'], **execution['engine_hashes']}.items():
        assert digest(path) == sha, path
    assert digest(execution['plugin']['path']) == execution['plugin']['sha256']


def verify_records(out, plan, log):
    planned = {job_id(job): job for job in plan['jobs']}
    records = sorted((out / 'records').glob('*.json'))
    proof = {}
    for path in records:
        record = read(path)
        assert record['job'] == planned[path.stem]
        assert not record['outcome']['invalid_trial'] and not record['outcome']['contract_violations']
        if record['job']['mode'] == 'replay':
            assert record['pair_valid'] is True
        trace = out / record['trace']
        assert digest(trace) == record['trace_sha256']
        for item in (path, trace, out / 'calls' / (path.stem + '.jsonl')):
            proof[str(item.relative_to(out))] = digest(item)
    assert {p.stem for p in records} == {job_id(j) for j in plan['jobs'][:len(records)]}
    result = subprocess.run(
        [(_CHRONOTOPE_HOME + '/agents3-venv/bin/python'), '-B', '-c', AUDIT, str(out)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    Path(log).write_text(result.stdout)
    assert result.returncode == 0, f'Completed request audit failed: {log}'
    return {'records': len(records), 'hashes': proof, 'request_audits_recomputed': True}


def pause(out, timeout, reason='User requested missing per-request vLLM timing measurements; instrumentation-only rollout'):
    assert out == OUT, 'Helper is scoped to the active prompt-only batch'
    plan, execution = validate_plan(out)
    initial = read(out / 'status.json')
    assert initial['state'] == 'running'
    runner = checked_identity(read(out / 'launch.json'))
    assert str(RUNNER) in runner['command'] and initial['pid'] == runner['pid']
    instrumentation = out / 'timing_instrumentation.json'
    service_path = Path(read(instrumentation)['service_launch']) if instrumentation.exists() else out / 'service_launch.json'
    service_launch = read(service_path)
    service = checked_identity(service_launch)
    auditor = checked_identity(read(out / 'audit_launch.json'))
    assert str(out / 'code/watch_same_model_deadline.py') in auditor['command']
    folder = out / 'pauses' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True)
    journal = {
        'state': 'pause_requested', 'requested_at': now(), 'out': str(out),
        'reason': reason,
        'runner': runner, 'observers': {'auditor': auditor}, 'initial_status': initial,
        'service': service, 'service_command': service_launch['command'],
        'service_launch_path': str(service_path),
        'service_performance_environment': environment_subset(service['pid']),
        'runner_launch_environment': environment_subset(runner['pid'], LAUNCH_ENV),
        'completed_records_before_signal': {p.name: digest(p) for p in (out / 'records').glob('*.json')},
        'plan_sha256': digest(out / 'plan.json'),
        'execution_manifest_sha256': digest(out / 'execution_manifest.json'),
        'helper_sha256': digest(Path(__file__)),
        'shared_control_helpers_sha256': digest(ROOT / 'studies/runtime_reversal/batch_control.py'),
        'automatic_resume': False, 'model_service_changed': False,
        'signal_policy': 'SIGTERM to verified runner only; private socket close for its orphaned native constructor',
        'orphan_socket_closes': [],
    }
    for name in ('launch.json', 'audit_launch.json', 'status.json', 'audit_status.json',
                 'execution_manifest.json', 'service_launch.json'):
        if (out / name).exists():
            shutil.copy2(out / name, folder / ('before_' + name))
    write(folder / 'journal.json', journal)
    try:
        assert still_same(runner)
        os.kill(runner['pid'], signal.SIGTERM)
        deadline = time.monotonic() + timeout
        closed = set()
        while True:
            runner_alive = still_same(runner)
            natives = owned_native_processes(out, execution)
            if not runner_alive:
                for native in natives:
                    key = (native['pid'], native['starttime_ticks'])
                    if key not in closed and close_orphan(native):
                        closed.add(key)
                        journal['orphan_socket_closes'].append({**native, 'closed_at': now()})
                        write(folder / 'journal.json', journal)
            if not runner_alive and not natives and service_idle(service_launch['base_url']):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Runner/native/service did not drain; no additional signals sent')
            time.sleep(1)
        terminal = read(out / 'status.json')
        assert terminal['state'] in ('stopped', 'complete'), terminal
        while still_same(auditor):
            if time.monotonic() > deadline:
                raise TimeoutError('Auditor did not exit; no signals sent to it')
            time.sleep(1)
        with exclusive_gpu_lock():
            verify_artifacts(out, execution)
            proof = verify_records(out, plan, folder / 'completed_request_audit.log')
            for name, sha in journal['completed_records_before_signal'].items():
                assert digest(out / 'records' / name) == sha
            recorded = {p.stem for p in (out / 'records').glob('*.json')}
            traces = {read(p)['trace'] for p in (out / 'records').glob('*.json')}
            current = terminal.get('current_job')
            allowed_call = f"calls/{job_id(current)}.jsonl" if current else None
            allowed_trace = (f"trials/w{round(current['window_s']*1e9)}/{current['arm']}/{current['episode_id']}_{current['mode']}.jsonl" if current else None)
            partial = [p for p in (out / 'trials').rglob('*.jsonl') if str(p.relative_to(out)) not in traces]
            partial += [p for p in (out / 'calls').glob('*.jsonl') if p.stem not in recorded]
            archived = []
            for path in partial:
                relative = str(path.relative_to(out))
                assert relative in (allowed_trace, allowed_call), f'Unexpected unrecorded file: {relative}'
                dest = folder / 'incomplete' / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                item = {'source': relative, 'sha256': digest(path), 'archive': str(dest.relative_to(out)),
                        'reason': 'Interrupted unrecorded trial; excluded from denominators; retry on resume'}
                shutil.move(str(path), str(dest))
                archived.append(item)
                journal['archived_unscored_files'] = archived
                write(folder / 'journal.json', journal)
            journal.update(state='complete' if terminal['state'] == 'complete' else 'paused',
                paused_at=now(), terminal_status=terminal, completed_evidence=proof,
                archived_unscored_files=archived, native_processes_remaining=[], model_requests_idle=True)
            write(folder / 'journal.json', journal)
            if terminal['state'] != 'complete':
                write(out / 'status.json', {**terminal, 'state': 'paused', 'updated_at': now(), 'pid': None,
                    'completed_trials': proof['records'], 'error': None, 'pause_reason': journal['reason'],
                    'pause_journal': str(folder / 'journal.json'), 'automatic_resume': False})
        print(json.dumps({'state': journal['state'], 'completed_trials': proof['records'],
                          'journal': str(folder / 'journal.json'), 'archived_unscored_files': len(archived)}), flush=True)
    except BaseException as error:
        journal.update(state='pause_incomplete', error=f'{type(error).__name__}: {error}', updated_at=now())
        write(folder / 'journal.json', journal)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--pause', action='store_true', required=True)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--reason', default='User requested missing per-request vLLM timing measurements; instrumentation-only rollout')
    args = parser.parse_args()
    out = args.out.resolve()
    assert out == OUT and 30 <= args.timeout <= 1800
    with (out / 'batch_control.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pause(out, args.timeout, args.reason)
