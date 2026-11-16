"""Run the four approved, isolated request profiles and restore the service.

The batch must already be paused through batch_control.py. This coordinator
holds the evaluation GPU lock throughout builds, service changes and captures.
It restores the original service in finally, including when a profile fails.
Resuming evaluation is a separate explicit batch_control.py invocation.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/runtime_reversal_20261029'
EXPERIMENT = ROOT / 'runs/agents3_policy_radixark27b_20261028/experiment'
SPEC = importlib.util.spec_from_file_location('batch_control', Path(__file__).with_name('batch_control.py'))
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


def read(path):
    return json.loads(Path(path).read_text())


def publish(state, **details):
    CONTROL.write(OUT / 'session_status.json', {'state': state, 'updated_at': CONTROL.now(),
        'pid': os.getpid(), 'application_episodes': 0, 'qualification_trials': 0, **details})
    print(state, flush=True)


def environment(journal):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'studies'),
        PATH=(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin:/usr/local/cuda/bin:') + os.environ.get('PATH', ''),
        LD_LIBRARY_PATH=(_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:') + os.environ.get('LD_LIBRARY_PATH', ''),
        EDGELLM_PLUGIN_PATH=(_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    restore = read(OUT / 'original_vllm_restore.json')
    assert restore['restore_command'] == journal['service_command']
    env.update(restore['environment_overrides'])
    for key in CONTROL.PERFORMANCE_ENV:
        if key in journal['service_performance_environment']:
            env[key] = journal['service_performance_environment'][key]
        else:
            env.pop(key, None)
    return env


def group_members(pgid):
    members = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            identity = CONTROL.process(p.name)
            if identity:
                try:
                    if os.getpgid(identity['pid']) == pgid:
                        members.append(identity)
                except ProcessLookupError:
                    pass
    return members


def stop_group(identity, journal_name):
    before = group_members(identity['pid'])
    if before:
        leader = CONTROL.process(identity['pid'])
        assert leader is None or leader == identity, 'Process group leader PID reused'
        assert all(os.getsid(p['pid']) == identity['pid'] for p in before), 'Group is not the recorded dedicated session'
        os.killpg(identity['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 180
        while group_members(identity['pid']):
            if time.monotonic() > deadline:
                raise TimeoutError('Dedicated service group did not exit after SIGTERM')
            time.sleep(1)
        CONTROL.write(OUT / journal_name, {'stopped_at': CONTROL.now(), 'identity': identity,
            'group_members_before': before, 'group_members_remaining': []})


def session_members(identity):
    members = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit() and (actual := CONTROL.process(p.name)):
            try:
                if os.getsid(actual['pid']) == identity['pid']:
                    members.append(actual)
            except ProcessLookupError:
                pass
    return members


def diagnostic_target(diagnostic):
    command = diagnostic['plan']['service_command']
    # Nsight starts its target in a separate session. Establish ownership from
    # the parent chain while the profiler lives, then retain the exact target.
    candidates = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or not (p := CONTROL.process(path.name)):
            continue
        if p['command'] != command and p['command'][1:] != command:
            continue
        if p == diagnostic.get('target'):
            candidates.append(p)
            continue
        ancestor = p['pid']
        visited = set()
        while ancestor > 1 and ancestor not in visited:
            if ancestor == diagnostic['identity']['pid']:
                candidates.append(p)
                break
            visited.add(ancestor)
            try:
                fields = Path(f'/proc/{ancestor}/stat').read_text().rsplit(')', 1)[1].split()
                ancestor = int(fields[1])
            except FileNotFoundError:
                break
    assert len(candidates) <= 1, 'Ambiguous diagnostic vLLM server'
    if candidates:
        diagnostic['target'] = candidates[0]
    return candidates[0] if candidates else None


def diagnostic_members(diagnostic):
    members = session_members(diagnostic['identity'])
    target = diagnostic.get('target')
    if target and target['pid'] != diagnostic['identity']['pid']:
        # Retained server identity established ownership of its dedicated SID.
        leader = CONTROL.process(target['pid'])
        assert leader is None or leader == target, 'Diagnostic target PID reused'
        members.extend(session_members(target))
    return list({p['pid']: p for p in members}.values())


def wait_diagnostic(diagnostic, log):
    plan = diagnostic['plan']
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        target = diagnostic_target(diagnostic)
        if diagnostic['process'].poll() is not None and target is None:
            raise RuntimeError(f'Nsight and its diagnostic service exited: {log}')
        if target:
            try:
                with urllib.request.urlopen(plan['base_url'] + '/models', timeout=3) as response:
                    data = json.load(response)
                assert [item['id'] for item in data['data']] == [plan['model']]
                CONTROL.write(log.parent / 'target_service_launch.json', target)
                return
            except (OSError, TimeoutError):
                pass
        time.sleep(2)
    raise TimeoutError(f'Diagnostic service startup exceeded 20 minutes: {log}')


def stop_diagnostic(diagnostic, journal_name, require_report=False):
    """Stop the model server first, allowing Nsight to finish its export."""
    identity = diagnostic['identity']
    target = diagnostic_target(diagnostic) or diagnostic.get('target')
    before = diagnostic_members(diagnostic)
    if target and CONTROL.still_same(target):
        os.kill(target['pid'], signal.SIGTERM)
    # vLLM's API server shuts down its workers. Nsight owns only instrumentation
    # and export; never send a group signal while it may be writing the report.
    try:
        diagnostic['process'].wait(timeout=240)
    except subprocess.TimeoutExpired:
        remaining = diagnostic_members(diagnostic)
        CONTROL.write(OUT / journal_name, {'state': 'cleanup_timeout', 'remaining': remaining,
                      'identity': identity, 'target': target})
        raise TimeoutError('Diagnostic session has not stopped; preserve exporter and do not start another GPU service')
    remaining = diagnostic_members(diagnostic)
    if remaining:
        # A completed CLI may leave its target alive with --kill=none. Every
        # member of this start_new_session session belongs to this invocation.
        for member in remaining:
            if CONTROL.still_same(member):
                os.kill(member['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 180
        while diagnostic_members(diagnostic):
            if time.monotonic() > deadline:
                raise TimeoutError('Owned diagnostic descendants remain after SIGTERM')
            time.sleep(1)
    CONTROL.write(OUT / journal_name, {'state': 'stopped', 'stopped_at': CONTROL.now(),
                  'identity': identity, 'target': target, 'session_members_before': before,
                  'session_members_remaining': [], 'nsys_returncode': diagnostic['process'].returncode})
    if require_report:
        folder = Path(diagnostic['folder'])
        report = folder / 'vllm_timeline.nsys-rep'
        database = folder / 'vllm_timeline.sqlite'
        assert report.is_file() and report.stat().st_size > 0, 'Nsight report missing'
        assert database.is_file() and database.stat().st_size > 0, 'Nsight SQLite export missing'
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
            assert connection.execute('PRAGMA quick_check').fetchone() == ('ok',), 'Incomplete SQLite export'


def wait_service(process, url, model, log):
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Service exited {process.returncode}: {log}')
        try:
            with urllib.request.urlopen(url + '/models', timeout=3) as response:
                data = json.load(response)
            assert [item['id'] for item in data['data']] == [model]
            return
        except (OSError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f'Service startup exceeded 20 minutes: {log}')


def launch_service(command, env, log):
    with log.open('ab') as stream:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def helper(name, action, folder, lock, env):
    command = ['/usr/bin/python3', '-B', str(ROOT / 'studies/runtime_reversal' / name),
               action, '--out', str(folder), '--lock-fd', str(lock.fileno())]
    subprocess.run(command, cwd=ROOT, env=env, check=True, pass_fds=(lock.fileno(),))


class AdoptedProcess:
    """Observe an identity-verified non-child; exit status is not available."""
    def __init__(self, identity):
        self.identity = identity
        self.pid = identity['pid']
        self.returncode = None

    def poll(self):
        actual = CONTROL.process(self.pid)
        if actual is None:
            self.returncode = 'unavailable_non_child'
        else:
            assert actual == self.identity, 'Adopted profiler PID identity changed'
        return self.returncode

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(self.identity['command'], timeout)
            time.sleep(1)
        return self.returncode


def run(journal_path, continue_session=False, adopt_session=None):
    adoption = read(adopt_session) if adopt_session else None
    if (OUT / 'session_status.json').exists():
        assert continue_session or adoption, 'Session already attempted; inspect evidence before retry'
        previous = read(OUT / 'session_status.json')
        if adoption:
            assert previous['state'] == 'starting_profiled_vllm' and previous['model'] == 'qwen08'
            assert previous['pid'] == adoption['coordinator']['pid']
            assert not CONTROL.still_same(adoption['coordinator']), 'Prior coordinator is still live'
            assert CONTROL.still_same(adoption['nsys']) and CONTROL.still_same(adoption['target'])
            assert not (OUT / 'vllm_qwen08_case7/capture_result.json').exists(), 'Never repeat a request during adoption'
        else:
            assert previous['state'] == 'profile_failed_service_restored', previous
        CONTROL.write(OUT / ('previous_session_status_' + str(previous['pid']) + '.json'), previous)
    else:
        assert not continue_session and not adoption
    journal = read(journal_path)
    assert journal['state'] == 'paused' and read(EXPERIMENT / 'status.json')['state'] == 'paused'
    assert journal['native_processes_remaining'] == [] and journal['model_requests_idle']
    for backend in ['native', 'vllm']:
        for model in ['qwen08', 'radixark27b']:
            plan = read(OUT / f'{backend}_{model}_case7/launch_plan.json')
            assert plan['task_family'] == 'notification' and plan['application_episodes'] == 0
            assert plan['qualification_trials'] == 0 and plan['source_job']['index'] == 7
            assert plan['expected_input_tokens'] == 998
    env = environment(journal)
    original = (CONTROL.checked_identity(read(EXPERIMENT.parent / 'services/launch.json'))
                if continue_session and not adoption else journal['service'])
    if adoption:
        assert not CONTROL.processes_matching(journal['service_command']), 'Original service unexpectedly running during adoption'
    else:
        assert CONTROL.still_same(original), 'Original service identity changed'
        assert CONTROL.matches(original, journal['service_command'])
        assert CONTROL.environment_subset(original['pid']) == journal['service_performance_environment']
    diagnostic = None
    if adoption:
        folder = OUT / 'vllm_qwen08_case7'
        diagnostic = {'process': AdoptedProcess(adoption['nsys']), 'identity': adoption['nsys'],
                      'plan': read(folder / 'launch_plan.json'), 'folder': str(folder),
                      'target': adoption['target']}
        assert diagnostic_target(diagnostic) == adoption['target']
    error = None
    restored = None
    with CONTROL.exclusive_gpu_lock() as lock:
        if adoption:
            publish('adopting_ready_diagnostic_service', adoption_journal=str(adopt_session))
        else:
            publish('stopping_original_idle_service', pause_journal=str(journal_path))
            assert CONTROL.service_idle('http://127.0.0.1:8012/v1')
        try:
            if not adoption:
                stop_group(original, f'original_service_stop_{original["pid"]}.json')
            for model in ['qwen08', 'radixark27b']:
                folder = OUT / f'native_{model}_case7'
                if adoption:
                    completed = read(folder / 'capture_result.json')
                    assert completed['model_requests_executed'] == 1 and completed['answer']['input_tokens'] == 998
                    assert (folder / 'native_timeline.sqlite').is_file()
                    assert read(folder / 'analysis/summary.json')['cuda_graph_launch_calls'] == 32
                    publish('retaining_validated_prior_capture', model=model)
                    continue
                if (folder / 'recovery_audit.json').exists():
                    recovered = read(folder / 'recovery_audit.json')
                    assert model == 'qwen08' and continue_session
                    assert recovered['state'] == 'valid_capture_with_shutdown_error'
                    assert recovered['cuda_graph_launch_calls'] == 32
                    assert recovered['full_response_matches_original'] is True
                    publish('retaining_validated_prior_capture', model=model)
                    continue
                if not (folder / 'build_result.json').exists():
                    publish('building_native_instrumentation', model=model)
                    helper('profile_native.py', '--build', folder, lock, env)
                publish('profiling_native', model=model)
                helper('profile_native.py', '--capture', folder, lock, env)
            for model in ['qwen08', 'radixark27b']:
                folder = OUT / f'vllm_{model}_case7'
                plan = read(folder / 'launch_plan.json')
                if adoption and model == 'qwen08':
                    diagnostic = {'process': AdoptedProcess(adoption['nsys']), 'identity': adoption['nsys'],
                                  'plan': plan, 'folder': str(folder), 'target': adoption['target']}
                    assert diagnostic_target(diagnostic) == adoption['target']
                    publish('adopted_profiled_vllm', model=model)
                else:
                    publish('starting_profiled_vllm', model=model)
                    process = launch_service(plan['nsys_command'], env, folder / 'service.log')
                    diagnostic = {'process': process, 'identity': CONTROL.process(process.pid),
                                  'plan': plan, 'folder': str(folder), 'target': None}
                    CONTROL.write(folder / 'service_launch.json', diagnostic['identity'])
                wait_diagnostic(diagnostic, folder / 'service.log')
                publish('profiling_vllm', model=model)
                helper('profile_vllm.py', '--capture', folder, lock, env)
                stop_diagnostic(diagnostic, f'vllm_{model}_stop.json', require_report=True)
                diagnostic = None
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            publish('profile_error_restoring', error=error)
        finally:
            if diagnostic is not None:
                try:
                    stop_diagnostic(diagnostic, 'failed_diagnostic_stop.json')
                except BaseException as cleanup_error:
                    remaining = diagnostic_members(diagnostic)
                    publish('diagnostic_cleanup_failed', error=error,
                            cleanup_error=f'{type(cleanup_error).__name__}: {cleanup_error}',
                            remaining=remaining, restore_safe=not remaining)
                    if remaining:
                        raise RuntimeError('Cannot restore model service while owned diagnostic processes remain') from cleanup_error
            publish('restoring_original_service', profiling_error=error)
            services = CONTROL.processes_matching(journal['service_command'])
            if services:
                assert len(services) == 1 and services[0] == original
                restored = original
            else:
                log = OUT / 'restored_service.log'
                process = launch_service(journal['service_command'], env, log)
                restored = CONTROL.process(process.pid)
                CONTROL.write(OUT / 'restored_service_launch.json', restored)
                wait_service(process, 'http://127.0.0.1:8012/v1', CONTROL.MODEL, log)
            assert CONTROL.environment_subset(restored['pid']) == journal['service_performance_environment']
            assert CONTROL.service_idle('http://127.0.0.1:8012/v1')
            active_launch = EXPERIMENT.parent / 'services/launch.json'
            previous = read(active_launch)
            CONTROL.write(OUT / ('original_active_service_launch_' + str(previous['pid']) + '.json'), previous)
            CONTROL.write(active_launch, {**previous, 'pid': restored['pid'],
                'process_starttime_ticks': restored['starttime_ticks'],
                'restarted_at': CONTROL.now(), 'profiling_session': str(OUT)})
            publish('profiles_complete_service_restored' if error is None else 'profile_failed_service_restored',
                error=error, restored_service=restored, pause_journal=str(journal_path))
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Coordinator received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pause-journal', type=Path, required=True)
    parser.add_argument('--continue-session', action='store_true')
    parser.add_argument('--adopt-session', type=Path, help='Explicit identity-verified handover journal; old coordinator must be stopped')
    args = parser.parse_args()
    run(args.pause_journal.resolve(), args.continue_session, args.adopt_session)
