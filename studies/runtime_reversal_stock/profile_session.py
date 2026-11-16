"""Execute the reviewed stock-backend diagnostic plan under exclusive GPU use.

This orchestration is external to NVIDIA inference code. It preserves the main
evaluation's completed records and engines, restores its exact original service,
and resumes only when --resume-evaluation was explicitly supplied by the caller.
Preparation and dry review never change services or execute the plan.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/runtime_reversal_stock_20261029'
EXPERIMENT = ROOT / 'runs/agents3_policy_radixark27b_20261028/experiment'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LIFECYCLE = load('_stock_profile_lifecycle', ROOT / 'studies/runtime_reversal/profile_session.py')
LIFECYCLE.OUT = OUT  # All lifecycle journals belong to this new diagnostic.
CONTROL = LIFECYCLE.CONTROL


def read(path):
    return json.loads(Path(path).read_text())


def publish(state, **details):
    CONTROL.write(OUT / 'session_status.json', {
        'state': state, 'updated_at': CONTROL.now(), 'pid': os.getpid(),
        'application_episodes': 0, 'qualification_trials': 0, **details,
    })
    print(state, flush=True)


def stable_process(pid, expected=None):
    """Popen can return before exec; don't pin a transient empty argv."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        actual = CONTROL.process(pid)
        if actual and actual['command'] and (expected is None or CONTROL.matches(actual, expected)):
            return actual
        time.sleep(.02)
    raise RuntimeError(f'Could not identify launched process{pid}')


def validate_plan(plan):
    assert plan['task_family'] == 'notification'
    assert plan['application_episodes'] == plan['qualification_trials'] == 0
    assert plan['planned_model_requests'] == 4 and plan['repetition_penalty'] == 1.0
    assert plan['upstream_commit'] == 'bb291453bc10d9bd1142070c605e090d4e483ee2'
    assert len(plan['steps']) >= 2
    deferred = plan.get('continuation_plan')
    if deferred:
        assert Path(deferred).resolve() == OUT / 'runtime_continuation_plan.json'
    requests = 0
    names = set()
    for step in plan['steps']:
        assert step['name'] not in names
        names.add(step['name'])
        assert step['kind'] in ['export', 'export_audit', 'runtime_build', 'engine_build', 'native_capture']
        assert step.get('model') in [None, 'qwen08', 'radixark27b']
        assert isinstance(step['command'], list) and all(isinstance(v, str) for v in step['command'])
        assert step['command']
        if step['kind'] != 'export_audit':
            assert '@GPU_LOCK_FD@' in step['command']
        if step['kind'] == 'native_capture':
            requests += 1
    if deferred:
        assert requests == 0 and all(s['kind'] in ['export', 'export_audit'] for s in plan['steps'])
    else:
        assert requests == 2
        assert {step['model'] for step in plan['steps'] if step['kind'] == 'native_capture'} == {'qwen08', 'radixark27b'}
    for model in ['qwen08', 'radixark27b']:
        host = read(OUT / f'vllm_{model}_case7/launch_plan.json')
        body = read(OUT / f'vllm_{model}_case7/request.json')
        assert host['task_family'] == 'notification' and host['source_job']['index'] == 7
        assert host['expected_input_tokens'] == 998
        assert host['application_episodes'] == host['qualification_trials'] == 0
        assert body['repetition_penalty'] == 1.0


def run_step(step, lock, env):
    folder = OUT / 'steps'
    folder.mkdir(exist_ok=True)
    result_path = folder / (step['name'] + '.json')
    assert not result_path.exists(), 'Never implicitly repeat a diagnostic phase'
    command = [str(lock.fileno()) if item == '@GPU_LOCK_FD@' else item for item in step['command']]
    result = {'state': 'running', 'started_at': CONTROL.now(), 'command': command,
              'kind': step['kind'], 'model': step.get('model')}
    CONTROL.write(result_path, result)
    publish('running_' + step['kind'], step=step['name'], model=step.get('model'))
    effective_env = dict(env, **step.get('environment_overrides', {}))
    with (folder / (step['name'] + '.log')).open('wb') as log:
        process = subprocess.Popen(command, cwd=step.get('cwd', str(ROOT)), env=effective_env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            pass_fds=(lock.fileno(),), start_new_session=True)
        identity = stable_process(process.pid)
        result['process'] = identity
        CONTROL.write(result_path, result)
        try:
            rc = process.wait()
        except BaseException:
            # Only this owned diagnostic helper group receives a termination.
            # Native helpers must close their owned Nsight/engine subprocesses.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    result.update(state='cleanup_timeout', updated_at=CONTROL.now())
                    CONTROL.write(result_path, result)
                    raise
            raise
        result.update(state='complete' if rc == 0 else 'failed', returncode=rc, completed_at=CONTROL.now())
        CONTROL.write(result_path, result)
        if rc:
            raise subprocess.CalledProcessError(rc, command)


def restore_original(journal, env):
    publish('restoring_original_service')
    services = CONTROL.processes_matching(journal['service_command'])
    if services:
        assert len(services) == 1
        restored = services[0]
    else:
        log = OUT / 'restored_service.log'
        process = LIFECYCLE.launch_service(journal['service_command'], env, log)
        initial = stable_process(process.pid, journal['service_command'])
        CONTROL.write(OUT / 'restored_service_launch.json', initial)
        LIFECYCLE.wait_service(process, 'http://127.0.0.1:8012/v1', CONTROL.MODEL, log)
        restored = stable_process(process.pid, journal['service_command'])
    assert CONTROL.environment_subset(restored['pid']) == journal['service_performance_environment']
    assert CONTROL.service_idle('http://127.0.0.1:8012/v1')
    active = EXPERIMENT.parent / 'services/launch.json'
    previous = read(active)
    CONTROL.write(OUT / ('original_active_service_launch_' + str(previous['pid']) + '.json'), previous)
    CONTROL.write(active, {**previous, 'pid': restored['pid'],
        'process_starttime_ticks': restored['starttime_ticks'],
        'restarted_at': CONTROL.now(), 'profiling_session': str(OUT)})
    return restored


def run(plan_path, pause_journal, resume_evaluation):
    assert not (OUT / 'session_status.json').exists(), 'Existing session requires explicit review, not automatic retry'
    plan = read(plan_path)
    validate_plan(plan)
    prepared = read(OUT / 'session_preparation.json')
    for path, sha in prepared['helpers'].items():
        assert CONTROL.digest(path) == sha, path
    journal = read(pause_journal)
    assert journal['state'] == 'paused' and journal['out'] == str(EXPERIMENT)
    assert read(EXPERIMENT / 'status.json')['state'] == 'paused'
    assert journal['model_requests_idle'] and journal['native_processes_remaining'] == []
    env = LIFECYCLE.environment(journal)
    original = CONTROL.checked_identity(read(EXPERIMENT.parent / 'services/launch.json'))
    assert CONTROL.matches(original, journal['service_command'])
    assert CONTROL.environment_subset(original['pid']) == journal['service_performance_environment']
    CONTROL.write(OUT / 'reviewed_session_plan.json', {
        'plan': plan, 'plan_sha256': CONTROL.digest(plan_path),
        'coordinator_sha256': CONTROL.digest(Path(__file__)),
        'pause_journal': str(pause_journal), 'resume_evaluation': resume_evaluation,
        'started_at': CONTROL.now(), 'original_service': original,
    })
    diagnostic = None
    error = None
    restored = None
    with CONTROL.exclusive_gpu_lock() as lock:
        assert CONTROL.service_idle('http://127.0.0.1:8012/v1')
        try:
            publish('stopping_original_idle_service')
            LIFECYCLE.stop_group(original, f'original_service_stop_{original["pid"]}.json')
            for step in plan['steps']:
                run_step(step, lock, env)
            if plan.get('continuation_plan'):
                continuation_path = Path(plan['continuation_plan'])
                publish('waiting_for_reviewed_runtime_plan', path=str(continuation_path))
                deadline = time.monotonic() + 3600
                while not continuation_path.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Runtime continuation plan was not ready within one hour after exports')
                    time.sleep(2)
                continuation = read(continuation_path)
                assert not continuation.get('continuation_plan'), 'Only one reviewed continuation is allowed'
                validate_plan(continuation)
                assert all(s['kind'] in ['runtime_build', 'engine_build', 'native_capture'] for s in continuation['steps'])
                CONTROL.write(OUT / 'reviewed_runtime_continuation.json', {
                    'plan': continuation, 'sha256': CONTROL.digest(continuation_path),
                    'reviewed_at': CONTROL.now(),
                })
                for step in continuation['steps']:
                    run_step(step, lock, env)
            for model in ['qwen08', 'radixark27b']:
                folder = OUT / f'vllm_{model}_case7'
                host = read(folder / 'launch_plan.json')
                publish('starting_profiled_vllm', model=model)
                process = LIFECYCLE.launch_service(host['nsys_command'], env, folder / 'service.log')
                # nsys resolves its launcher path on exec; pin PID/starttime,
                # then refresh the command identity when the target is ready.
                diagnostic = {'process': process, 'identity': stable_process(process.pid),
                              'plan': host, 'folder': str(folder), 'target': None}
                CONTROL.write(folder / 'service_launch.json', diagnostic['identity'])
                LIFECYCLE.wait_diagnostic(diagnostic, folder / 'service.log')
                identity = CONTROL.process(process.pid)
                if identity:
                    assert identity['starttime_ticks'] == diagnostic['identity']['starttime_ticks']
                    diagnostic['identity'] = identity
                    CONTROL.write(folder / 'service_launch.json', identity)
                publish('profiling_vllm', model=model)
                command = ['/usr/bin/python3', '-B', str(ROOT / 'studies/runtime_reversal/profile_vllm.py'),
                    '--capture', '--out', str(folder), '--lock-fd', str(lock.fileno())]
                subprocess.run(command, cwd=ROOT, env=env, check=True, pass_fds=(lock.fileno(),))
                LIFECYCLE.stop_diagnostic(diagnostic, f'vllm_{model}_stop.json', require_report=True)
                diagnostic = None
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            publish('diagnostic_failed', error=error)
        finally:
            if diagnostic is not None:
                try:
                    LIFECYCLE.stop_diagnostic(diagnostic, 'failed_diagnostic_stop.json')
                except BaseException as cleanup_error:
                    remaining = LIFECYCLE.diagnostic_members(diagnostic)
                    publish('diagnostic_cleanup_failed', error=error,
                        cleanup_error=f'{type(cleanup_error).__name__}: {cleanup_error}', remaining=remaining)
                    if remaining:
                        raise RuntimeError('Cannot restore service while diagnostic GPU processes remain') from cleanup_error
            restored = restore_original(journal, env)
            publish('profiles_complete_service_restored' if error is None else 'profile_failed_service_restored',
                error=error, restored_service=restored, pause_journal=str(pause_journal))
    if resume_evaluation:
        publish('resuming_original_evaluation', profiling_error=error)
        CONTROL.resume(EXPERIMENT, pause_journal)
        publish('profiles_complete_evaluation_resumed' if error is None else 'profile_failed_evaluation_resumed',
            error=error, restored_service=restored, evaluation_status=read(EXPERIMENT / 'status.json'),
            pause_journal=str(pause_journal))
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Stock coordinator received signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pause-journal', type=Path, required=True)
    parser.add_argument('--resume-evaluation', action='store_true')
    args = parser.parse_args()
    run(args.plan.resolve(), args.pause_journal.resolve(), args.resume_evaluation)
