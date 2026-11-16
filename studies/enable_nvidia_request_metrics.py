"""Enable stock vLLM response timings for the explicitly paused prompt-only run.

Adds exactly one server flag. No inference readiness calls, controller edits,
engine changes, or edits to frozen manifests/evidence. Resume is explicit.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

import nvidia_timing_rollout as ctl
from launch_nvidia_agents3_policy import stable_identity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/nvidia_prompt_only_20261029/experiment'
FLAG = '--enable-per-request-metrics'


def process_environment(pid):
    # Used only in memory; never serialize process credentials to a journal.
    return dict(pair.decode().split('=', 1) for pair in
                Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in pair)


def descendants(pid):
    parents = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(path.name)] = int(fields[1])
        except FileNotFoundError:
            continue
    selected = {pid}
    while True:
        children = {child for child, parent in parents.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    return [identity for child in selected - {pid} if (identity := ctl.process(child))]


def verify_paused(journal_path):
    j = ctl.read(journal_path)
    assert j['state'] == 'paused' and j['out'] == str(OUT)
    assert ctl.read(OUT / 'status.json')['state'] == 'paused'
    assert ctl.digest(OUT / 'plan.json') == j['plan_sha256']
    assert ctl.digest(OUT / 'execution_manifest.json') == j['execution_manifest_sha256']
    plan, execution = ctl.validate_plan(OUT)
    assert not ctl.native_processes(OUT)
    for rel, sha in j['completed_evidence']['hashes'].items():
        assert ctl.digest(OUT / rel) == sha, rel
    return j, plan, execution


def enable(journal_path):
    j, plan, execution = verify_paused(journal_path)
    directory = journal_path.parent / 'service_metrics'
    assert not directory.exists(), 'A service rollout already exists; inspect its status'
    old = ctl.process(j['service']['pid'])
    assert old == j['service'], 'Original service identity changed'
    assert ctl.service_idle(execution['model_service'])
    command = list(j['service_command'])
    assert command == old['command'][-len(command):]
    assert FLAG not in command and '--disable-log-stats' not in command
    command.append(FLAG)
    env = process_environment(old['pid'])
    owned_children = descendants(old['pid'])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    prior = ctl.read(OUT / 'service_launch.json')
    directory.mkdir()
    shutil.copy2(OUT / 'service_launch.json', directory / 'before_service_launch.json')
    revision = {'state': 'prepared', 'created_at': ctl.now(),
                'change': {'added_server_flags': [FLAG]},
                'old_service': old, 'old_service_descendants': owned_children,
                'command': command, 'completed_trials': j['completed_evidence']['records'],
                'request_body_changes': False, 'controller_code_changes': False,
                'checkpoint_engine_changes': False, 'inference_readiness_calls': 0,
                'source_helper_sha256': ctl.digest(Path(__file__)),
                'previous_execution_manifest_sha256': j['execution_manifest_sha256']}
    ctl.write(directory / 'revision.json', revision)
    with ctl.exclusive_gpu_lock():
        ctl.verify_artifacts(OUT, execution)
        assert ctl.still_same(old)
        # Signal only the recorded idle API parent; vLLM owns its child cleanup.
        os.kill(old['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 120
        while any(ctl.still_same(identity) for identity in [old, *owned_children]):
            if time.monotonic() > deadline:
                raise TimeoutError('Original service did not exit; no broader signal sent')
            time.sleep(1)
        import socket
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', 8012))
        with (directory / 'vllm.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = stable_identity(process, command)
        service = {**prior, **identity, 'launch_command': command,
                   'state': 'starting', 'started_at': ctl.now(), 'enable_per_request_metrics': True,
                   'instrumentation_revision': str(directory / 'revision.json'),
                   'supersedes_service_launch': str(directory / 'before_service_launch.json'),
                   'inference_readiness_requests': 0}
        ctl.write(directory / 'launch.json', service)
        revision.update(state='starting', new_service=identity)
        ctl.write(directory / 'revision.json', revision)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Metrics-enabled service exited; inspect its vllm.log')
            try:
                with urllib.request.urlopen(execution['model_service'] + '/models', timeout=5) as response:
                    ready = [m['id'] for m in json.load(response)['data']] == [execution['model']]
                if ready and ctl.service_idle(execution['model_service']):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError('Metrics-enabled model service not ready')
        log = (directory / 'vllm.log').read_text(errors='replace')
        lines = [s for s in log.splitlines() if 'Initializing a V1 LLM engine' in s]
        assert len(lines) == 1
        for value in ['dtype=torch.float16', 'quantization=modelopt_mixed', 'kv_cache_dtype=auto',
                      'max_seq_len=16384', 'enable_prefix_caching=False', 'speculative_config=None']:
            assert value in lines[0], value
        assert 'FlashInferCutlassNvFp4LinearKernel' in log
        assert ctl.environment_subset(process.pid) == j['service_performance_environment']
        service.update(state='ready', ready_at=ctl.now(), observed_engine_configuration=lines[0],
                       observed_quantized_linear_backend='FlashInferCutlassNvFp4LinearKernel')
        ctl.write(directory / 'launch.json', service)
        # Preserve the hash-tracked original service_launch.json and manifest.
        # The new instrumentation sidecar points to the replacement service.
        revision.update(state='ready', ready_at=ctl.now(), new_service=identity)
        ctl.write(directory / 'revision.json', revision)
        ctl.write(OUT / 'timing_instrumentation.json', {
            **revision, 'service_launch': str(directory / 'launch.json'),
            'pause_journal': str(journal_path),
            'historical_missing_metrics': 'Preserved as missing; no reconstructed timing values',
            'timing_boundary': 'vLLM TTFT = first scheduled to first generated token; excludes queue/API preprocessing'})
    print(json.dumps({'state': 'service_ready', 'pid': process.pid, 'revision': str(directory / 'revision.json')}), flush=True)


def resume(journal_path):
    j, plan, execution = verify_paused(journal_path)
    revision_path = journal_path.parent / 'service_metrics/revision.json'
    rev = ctl.read(revision_path)
    assert rev['state'] == 'ready'
    service = ctl.read(journal_path.parent / 'service_metrics/launch.json')
    found = ctl.process(service['pid'])
    assert found and found['starttime_ticks'] == service['process_starttime_ticks']
    assert found['command'][-len(service['command']):] == service['command']
    assert service['command'] == j['service_command'] + [FLAG]
    assert ctl.service_idle(execution['model_service'])
    base = [(_CHRONOTOPE_HOME + '/agents3-venv/bin/python'), '-B', str(OUT / 'code/run_nvidia_prompt_only.py'),
            '--out', str(OUT), '--run', '--resume']
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            candidate = ctl.process(path.name)
            assert not candidate or base[2] not in candidate['command'], 'An evaluation runner is already active'
    with ctl.exclusive_gpu_lock():
        ctl.verify_artifacts(OUT, execution)
        proof = ctl.verify_records(OUT, plan, journal_path.parent / 'resume_request_audit.log')
        assert proof == j['completed_evidence']
    env = dict(os.environ)
    for key in ctl.LAUNCH_ENV:
        if key in j['runner_launch_environment']:
            env[key] = j['runner_launch_environment'][key]
        else:
            env.pop(key, None)
    for name in ('launch.json', 'audit_launch.json'):
        source = OUT / name
        if source.exists():
            shutil.copy2(source, journal_path.parent / ('resume_before_' + name))
    with (OUT / 'runner.log').open('ab') as log:
        runner = subprocess.Popen(base, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ident = stable_identity(runner, base)
    ctl.write(OUT / 'launch.json', {**ident, 'launch_command': base, 'started_at': ctl.now(),
              'pause_journal': str(journal_path), 'automatic_resume': False,
              'instrumentation_revision': str(revision_path)})
    deadline = time.monotonic() + 1200
    while ctl.read(OUT / 'status.json')['state'] == 'paused':
        if runner.poll() is not None:
            raise RuntimeError('Runner exited before resuming; inspect runner.log')
        if time.monotonic() > deadline:
            raise TimeoutError('Runner did not enter running state')
        time.sleep(1)
    assert ctl.read(OUT / 'status.json')['state'] == 'running'
    audit = [str(ROOT / '.venv/bin/python'), '-B', str(OUT / 'code/watch_same_model_deadline.py'), '--out', str(OUT)]
    with (OUT / 'auditor.log').open('ab') as log:
        auditor = subprocess.Popen(audit, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ctl.write(OUT / 'audit_launch.json', {**stable_identity(auditor, audit), 'launch_command': audit,
              'started_at': ctl.now(), 'pause_journal': str(journal_path)})
    j.update(state='resumed', resumed_at=ctl.now(), runner_pid=runner.pid, auditor_pid=auditor.pid,
             instrumentation_revision=str(revision_path))
    ctl.write(journal_path, j)
    print(json.dumps({'state': 'running', 'runner_pid': runner.pid, 'auditor_pid': auditor.pid,
                      'completed_trials_at_resume': proof['records']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--journal', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--enable', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = p.parse_args()
    with (OUT / 'batch_control.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (enable if args.enable else resume)(args.journal.resolve())
