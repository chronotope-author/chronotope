"""Durable core-five chain; waiting never builds, infers, or changes services.

Current Qwen Notification -> remaining four Qwen tasks -> official Gemma core five.
Every handoff requires successful completion, a CPU evidence audit and the GPU
lock. A reviewed conservative default can defer unsupported Authenticator
actions while the ready tasks proceed. Deferred trials are not completion and
cannot advance the next model. Stopped predecessors are never automatically resumed.
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
import signal
import subprocess
import sys
import time

from resume_notification_schedule import (completion_gate_for, load_execution_schedule,
                                          planned_count, scheduled_jobs, validate_schedule)

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'runs/legacy_detector_full_agents3_20261029/experiment'
QWEN = ROOT / 'runs/legacy_detector_full_agents3_core5_qwen_20261029/experiment'
GEMMA = ROOT / 'runs/legacy_detector_full_agents3_gemma4_26b_20261029/experiment'
LOCK = ROOT / 'runs/legacy_d1_gpu.lock'
AGENT_PYTHON = (_CHRONOTOPE_HOME + '/agents3-venv/bin/python')
MODEL_PYTHON = (_CHRONOTOPE_HOME + '/edgellm-venv/bin/python')
TOKENIZER_PYTHON = (_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/python')
RUNNER = ROOT / 'studies/run_core5_detector_full_agents3.py'
NATIVE = ROOT / 'studies/prepare_ledger_native.py'
ASSETS = ROOT / 'studies/gemma_ledger_assets.py'
CONTINUATION = ROOT / 'studies/resume_notification_schedule.py'
SUBSECOND_WINDOWS = {0.5, 0.25, 0.125}
QWEN_PHASES = ('qwen_verify_native', 'qwen_native_build', 'qwen_verify_agent', 'qwen_freeze')
GEMMA_PHASES = ('gemma_download', 'gemma_stop_services', 'gemma_export', 'gemma_build',
                'gemma_verify_native', 'gemma_native_build', 'gemma_verify_agent',
                'gemma_services', 'gemma_freeze')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def verify_hashes(mapping, base=ROOT):
    for name, expected in mapping.items():
        path = Path(name)
        assert digest(path if path.is_absolute() else base / path) == expected, name


def process_alive(pid):
    try:
        text = Path(f'/proc/{int(pid)}/stat').read_text()
        return text[text.rfind(')') + 2:].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def study_processes(out):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(out).encode() in args and any(b'run_' in arg and arg.endswith(b'.py') for arg in args) and process_alive(path.name):
            found.append(int(path.name))
    return found


def completion_gate(status, count):
    if status.get('state') != 'complete':
        return False, 'predecessor_' + str(status.get('state', 'missing'))
    if status.get('completed_trials') != count or status.get('planned_trials') != count:
        return False, 'predecessor_incomplete_count'
    if status.get('qualification_trials') != 0 or status.get('error'):
        return False, 'predecessor_metadata_error'
    return True, 'complete'


def action_choice(root):
    path = root / 'decisions.json'
    return read(path).get('typed_action_support', {}) if path.exists() else {}


def decisions_gate(root):
    choice = action_choice(root)
    approved = (choice.get('state') == 'approved' and choice.get('selection') in
            ('legacy_unsupported_report_failures', 'native_type_value_and_host_bridge') and
            bool(choice.get('user_authorization')))
    deferred = (choice.get('state') == 'conservative_default' and
                choice.get('selection') == 'defer_unsupported' and
                choice.get('user_authorization') is None and bool(choice.get('basis')))
    return approved or deferred


def action_status(root):
    choice = action_choice(root)
    deferred = decisions_gate(root) and choice.get('selection') == 'defer_unsupported'
    windows = window_selection(root)
    return {'action_choice_state': choice.get('state', 'missing'),
            'action_selection': choice.get('selection'),
            'typing_change_authorized': (decisions_gate(root) and
                                         choice.get('selection') == 'native_type_value_and_host_bridge'),
            'deferred_task': 'code' if deferred else None,
            'deferred_trials_per_successor': 2400 if deferred else 0,
            'user_deferred_windows_s': windows['deferred_windows_s'],
            'deferred_reason': ('Authenticator typed-action transport is unresolved; no trials scored or '
                                'controller changes authorized by this default.') if deferred else None}


def window_selection(root):
    path = root / 'decisions.json'
    selected = read(path).get('window_selection', {}) if path.exists() else {}
    windows = selected.get('deferred_windows_s', [])
    assert isinstance(windows, list) and all(type(value) in (int, float) for value in windows)
    assert len(windows) == len(set(windows)) and set(windows) <= SUBSECOND_WINDOWS, 'Unreviewed window deferral'
    authorization = selected.get('user_authorization')
    if windows:
        assert isinstance(authorization, str) and authorization.strip(), 'Window deferral requires user direction'
    return {'deferred_windows_s': windows, 'window_selection_authorization': authorization}


def study_count(root, out):
    if out == CURRENT:
        return planned_count(out)
    if out == GEMMA:
        return 15000 - 600 * len(window_selection(root)['deferred_windows_s'])
    assert out == QWEN
    return 10200


def study_completion_gate(root, out, status):
    if out == CURRENT:
        return completion_gate_for(out, status)
    count = study_count(root, out)
    if out == GEMMA and count != 15000:
        windows = sorted(window_selection(root)['deferred_windows_s'])
        deferred_count = 600 * len(windows)
        if status.get('state') != 'complete_scoped':
            return False, 'predecessor_' + str(status.get('state', 'missing'))
        if (status.get('completed_trials'), status.get('planned_trials'), status.get('runnable_trials')) != (count, 15000, count):
            return False, 'predecessor_incomplete_count'
        if (status.get('deferred_windows_s') != windows or
                status.get('user_deferred_window_trials') != deferred_count or
                status.get('unsupported_action_deferred_trials') != 0 or
                status.get('deferred_trials') != deferred_count):
            return False, 'predecessor_unexpected_deferral'
        if status.get('qualification_trials') != 0 or status.get('error'):
            return False, 'predecessor_metadata_error'
        return True, 'complete_scoped'
    return completion_gate(status, count)


def seal_action_support(root):
    """Materialize the reviewed choice without treating a default as user approval."""
    assert decisions_gate(root), 'An approved choice or reviewed conservative default is required'
    choice = action_choice(root)
    support = {'configuration_reviewed': True,
               'user_authorized': choice['state'] == 'approved',
               'code_action': choice['selection']}
    if choice['state'] == 'conservative_default':
        support['basis'] = choice['basis']
    else:
        support['user_authorization'] = choice['user_authorization']
    selected = window_selection(root)
    if selected['deferred_windows_s']:
        support.update(selected)
    paths = []
    for out in (QWEN, GEMMA):
        path = out.parent / 'inputs/action_support.json'
        if path.exists():
            assert read(path) == support, f'Conflicting action-support configuration: {path}'
        else:
            write(path, support)
        paths.append(path)
    return paths


def default_commands():
    native = lambda root: [TOKENIZER_PYTHON, '-B', str(NATIVE), '--root', str(root)]
    verify_agent = lambda root: [AGENT_PYTHON, '-B', str(ROOT / 'studies/verify_full_agents3.py'),
        '--config', str(root / 'experiment/configs/full_agent_components.json'),
        '--out', str(root / 'full_agents3_cpu_verification.json')]
    runner = lambda out: [AGENT_PYTHON, '-B', str(RUNNER), '--out', str(out)]
    asset = lambda phase: [MODEL_PYTHON, '-B', str(ASSETS), '--phase', phase, '--lock-fd', '{lock_fd}']
    commands = {
        'qwen_verify_native': native(QWEN.parent) + ['--verify-cpu', '--checkpoint', str(ROOT / 'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f')],
        'qwen_native_build': native(QWEN.parent) + ['--build', '--lock-fd', '{lock_fd}'],
        'qwen_verify_agent': verify_agent(QWEN.parent),
        'qwen_freeze': runner(QWEN) + ['--freeze', '--services', str(CURRENT.parent / 'services/launch.json')],
        'qwen_run': [AGENT_PYTHON, '-B', str(QWEN / 'code' / RUNNER.name), '--out', str(QWEN), '--run'],
        'gemma_download': asset('download'), 'gemma_stop_services': asset('stop-services'),
        'gemma_export': asset('export'), 'gemma_build': asset('build'),
        'gemma_verify_native': native(GEMMA.parent) + ['--verify-cpu', '--checkpoint', str(ROOT / 'runs/model_cache/nvidia_gemma4_26b_a4b_nvfp4_a19cfe0')],
        'gemma_native_build': native(GEMMA.parent) + ['--build', '--lock-fd', '{lock_fd}'],
        'gemma_verify_agent': verify_agent(GEMMA.parent), 'gemma_services': asset('services'),
        'gemma_freeze': runner(GEMMA) + ['--freeze', '--services', str(GEMMA.parent / 'services/launch.json')],
        'gemma_run': [AGENT_PYTHON, '-B', str(GEMMA / 'code' / RUNNER.name), '--out', str(GEMMA), '--run'],
    }
    return commands


def validate_scope():
    from core5_ledger_plan import validate_plan, load_cases
    old = read(CURRENT / 'plan.json')
    assert len(old['jobs']) == old['planned_trials'] == 4800 and old['qualification_trials'] == 0
    assert {j['task'] for j in old['jobs']} == {'notification'}
    assert {j['phase'] for j in old['jobs']} == {'evaluation'}
    assert {j['arm'] for j in old['jobs']} == {'legacy_cpp_detector', 'agents3_full'}
    assert {j['mode'] for j in old['jobs']} == {'live', 'replay', 'paused'}
    assert {j['window_s'] for j in old['jobs']} == {16., 8., 4., 2., 1., .5, .25, .125}
    assert {j['index'] for j in old['jobs']} == set(range(100))
    assert len({(j['arm'], j['window_s'], j['index'], j['mode']) for j in old['jobs']}) == 4800
    for path, count in ((QWEN, 10200), (GEMMA, 15000)):
        plan = read(path / 'plan.json'); assert validate_plan(plan) == count; load_cases(path, plan)
        assert plan['successor_jobs'] == []
    qwen, gemma = read(QWEN / 'plan.json'), read(GEMMA / 'plan.json')
    assert (qwen['model'], qwen['revision']) == ('nvidia/Qwen3.8-27B-NVFP4', '482ca0f3832238542f8f5295dde86b5f22711d80')
    assert (gemma['model'], gemma['revision']) == ('nvidia/Gemma-4-26B-A4B-NVFP4', 'a19cfe00be84568a6867111c9a68c9c44fdcffe6')
    for name, value in qwen['case_file_sha256'].items():
        assert gemma['case_file_sha256'][name] == value, 'Other-four cases must match between models'


def prepare(root):
    validate_scope(); root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'queue_manifest.json').exists()
    manifest = {'schema': 'core5-qwen-gemma-queue-1', 'created_at': now(), 'root': str(root),
        'studies': [{'path': str(out), 'count': study_count(root, out), 'original_count': count}
                    for out, count in ((CURRENT, 4800), (QWEN, 10200), (GEMMA, 15000))],
        'plan_hashes': {str(path / 'plan.json'): digest(path / 'plan.json') for path in (CURRENT, QWEN, GEMMA)},
        'predecessor_execution_sha256': digest(CURRENT / 'execution_manifest.json'),
        'predecessor_schedule_sha256': (digest(CURRENT / 'continuation_schedule.json')
                                       if (CURRENT / 'continuation_schedule.json').exists() else None),
        'queue_source': str(Path(__file__).resolve()), 'queue_source_sha256': digest(__file__),
        'qualification_trials': 0, 'on_stopped_predecessor': 'Wait; never resume automatically',
        'on_handoff_error': 'Stop and preserve evidence; no automatic retry',
        'reviewed_pipeline_required': True, 'typed_action_choice_required': False,
        'reviewed_action_support_required': True,
        'unresolved_action_default': 'Defer Authenticator only; never authorize typing changes or score unsupported trials',
        'subsecond_window_policy': 'Reserve response windows below 1s for models with at most 4B total parameters; no additional model run authorized',
        'additional_models_queued': []}
    write(root / 'queue_manifest.json', manifest)
    if not (root / 'decisions.json').exists():
        write(root / 'decisions.json', {'typed_action_support': {'state': 'pending', 'selection': None,
            'user_authorization': None}, 'additional_gemma_31b': 'not_queued'})
    write(root / 'queue_status.json', {'state': 'prepared_waiting_predecessor', 'created_at': now(),
        'pid': None, 'predecessor': str(CURRENT), 'queued_studies': [str(QWEN), str(GEMMA)],
        'queued_trials': study_count(root, QWEN) + study_count(root, GEMMA),
        'original_queued_trials': 25200, 'qualification_trials': 0, 'automatic_resume': False,
        'decision_ready': decisions_gate(root), 'pipeline_ready': False, **action_status(root)})
    return manifest


def seal(root):
    """Pin actual reviewed helpers. Never execute placeholders or infer approval."""
    assert not (root / 'handoff_started.json').exists()
    assert not (root / 'pipeline_manifest.json').exists(), 'Pipeline already sealed; preserve provenance'
    validate_scope()
    sources = [Path(__file__).resolve(), RUNNER, NATIVE, ASSETS, CONTINUATION, root / 'decisions.json']
    if (CURRENT / 'continuation_schedule.json').exists():
        load_execution_schedule(CURRENT)
        sources.append(CURRENT / 'continuation_schedule.json')
    sources += seal_action_support(root)
    sources += [ROOT / 'studies' / name for name in ('core5_ledger_plan.py', 'verify_full_agents3.py',
                'agents3_full.py', 'audit_legacy_detector_full_agents3.py', 'legacy_d1_w8.py')]
    for out in (QWEN, GEMMA):
        assert read(out / 'preparation_report.json')['state'] == 'passed'
        for folder in (out / 'configs', out.parent / 'build_source/chronotope'):
            assert folder.is_dir(), str(folder)
            sources += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
        for name in ('engine_build_result.json', 'grounder_checkpoint_verification.json'):
            # Gemma engines are deliberately absent until the deferred build.
            path = out.parent / name
            if path.exists(): sources.append(path)
    proof = {'schema': 'reviewed-core5-pipeline-1', 'created_at': now(), 'commands': default_commands(),
             'source_hashes': {str(p): digest(p) for p in sorted(set(sources))},
             'reviewed': True, 'qualification_trials': 0, 'inference_probes': 0}
    assert set(proof['commands']) == set(QWEN_PHASES + GEMMA_PHASES + ('qwen_run', 'gemma_run'))
    write(root / 'pipeline_manifest.json', proof)
    return proof


def validate_pins(root, manifest, pipeline=None):
    assert digest(manifest['queue_source']) == manifest['queue_source_sha256']
    verify_hashes(manifest['plan_hashes'])
    assert digest(CURRENT / 'execution_manifest.json') == manifest['predecessor_execution_sha256']
    schedule = CURRENT / 'continuation_schedule.json'
    assert (digest(schedule) if schedule.exists() else None) == manifest.get('predecessor_schedule_sha256'), 'Predecessor schedule changed; review and reseal queue'
    if pipeline is not None:
        assert pipeline['reviewed'] and pipeline['qualification_trials'] == pipeline['inference_probes'] == 0
        assert pipeline['commands'] == default_commands(), 'Unreviewed pipeline command change'
        verify_hashes(pipeline['source_hashes'])


def pipeline_readiness(root, manifest):
    path = root / 'pipeline_manifest.json'
    if not path.exists():
        return 'not_sealed', None
    try:
        validate_pins(root, manifest, read(path))
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        return 'needs_refresh', f'{type(exc).__name__}: {exc}'
    return 'ready', None


def audit_notification(out):
    """Recompute the completed original Notification evidence without inference."""
    assert completion_gate_for(out)[0]
    code = out / 'code'; sys.path[:0] = [str(code), str(code / 'src'), str(code / 'chronotope/python')]
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    from chronotope.distributional.manifest import ManifestSet, build_manifest
    from chronotope.distributional.episode import score_trace, verify_trace_evidence
    from audit_legacy_detector_full_agents3 import audit_trial_requests
    plan = read(out / 'plan.json'); execution = read(out / 'execution_manifest.json')
    schedule = load_execution_schedule(out, plan)
    if schedule is not None:
        validate_schedule(out, schedule, verify_records=True, plan=plan)
    jobs = scheduled_jobs(out, plan); count = len(jobs)
    verify_hashes(execution['hashes'], out)
    manifests = {name: ManifestSet.load(out / 'manifests' / name) for name in plan['manifest_hashes']}
    paths = list((out / 'records').glob('*.json')); assert len(paths) == count
    records = {p.stem: read(p) for p in paths}; failures = 0
    expected_ids = {f"{j['arm']}_w{round(j['window_s'] * 1e9)}_{j['index']:03d}_{j['mode']}" for j in jobs}
    assert set(records) == expected_ids, 'Scored records differ from scheduled jobs'
    for job in jobs:
        ident = f"{job['arm']}_w{round(job['window_s'] * 1e9)}_{job['index']:03d}_{job['mode']}"
        record = records[ident]; assert record['job'] == job
        assert digest(out / record['trace']) == record['trace_sha256']
        assert not record['outcome']['invalid_trial'] and not record['outcome']['contract_violations']
        trace = Trace.load(out / record['trace']); manifest = manifests[job['manifest']]
        assert manifest.frozen_hash == plan['manifest_hashes'][job['manifest']]
        row = manifest.rows[job['index']]; scenario = build_manifest(row)
        verify_trace_evidence(row, scenario, trace, manifest.frozen_hash)
        assert score_trace(row, trace, manifest.frozen_hash, controller=job['arm'], repeat_index=0).to_dict() == record['outcome']
        assert audit_trial_requests(out, ident, job['arm'], trace, scenario, execution) == record['request_audit']
        if job['mode'] == 'replay':
            live_id = ident.removesuffix('_replay') + '_live'
            assert record['pair_valid'] is True and validate_pair(trace, Trace.load(out / records[live_id]['trace'])).valid
        failures += int(record['outcome']['Y'] == 0)
    return {'state': 'passed', 'completed_trials': count, 'planned_trials': count,
            'original_planned_trials': 4800, 'user_deferred_trials': 4800-count,
            'schedule_sha256': digest(out / 'continuation_schedule.json') if schedule else None,
            'task_failures_retained': failures,
            'plan_sha256': digest(out / 'plan.json'), 'execution_manifest_sha256': digest(out / 'execution_manifest.json'),
            'model_inference_calls': 0, 'experimental_trials': 0, 'qualification_trials': 0}


def environment():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'studies'))
    for name in ('CUDA_LAUNCH_BLOCKING', 'VLLM_NSYS_PROFILING', 'VLLM_PROFILER_CONFIG', 'VLLM_TORCH_PROFILER_DIR'):
        env.pop(name, None)
    build = ROOT / 'runs/runtime_reversal_stock_20261029/build'
    env.update(EDGELLM_PLUGIN_PATH=str(build / 'libNvInfer_edgellm_plugin.so'),
               LD_LIBRARY_PATH=':'.join([str(build), (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'),
                                       '/usr/local/cuda/targets/sbsa-linux/lib']),
               PATH=(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin:/usr/local/cuda/bin:') + env.get('PATH', '/usr/bin:/bin'))
    return env


def command(root, name, argv, lock=None):
    folder = root / 'queue_phases'; folder.mkdir(exist_ok=True)
    path = folder / f'{name}.json'; assert not path.exists(), 'No automatic phase retry'
    argv = [arg.replace('{lock_fd}', str(lock.fileno())) if '{lock_fd}' in arg and lock is not None else arg for arg in argv]
    assert not any('{lock_fd}' in arg for arg in argv)
    with (folder / f'{name}.log').open('xb') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=(lock.fileno(),) if lock is not None else ())
        proof = {'state': 'running', 'started_at': now(), 'pid': proc.pid, 'command': argv}
        write(path, proof)
        try:
            code = proc.wait()
            proof.update(state='complete' if code == 0 else 'failed', exit_code=code, completed_at=now())
            write(path, proof)
            if code: raise RuntimeError(f'{name} failed ({code}); see {folder / (name + ".log")}')
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=20)
                except subprocess.TimeoutExpired: pass
                proof.update(state='interrupted', process_still_alive=proc.poll() is None)
                write(path, proof)
            raise
    return folder / f'{name}.log'


def run(root, poll=15):
    manifest = read(root / 'queue_manifest.json'); assert manifest['root'] == str(root)
    with (root / 'queue.lock').open('a+') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / 'handoff_started.json').exists(), 'Handoff already attempted; no automatic pipeline restart'
        validate_pins(root, manifest)
        gpu = None; active_runner = None
        def publish(state, **detail):
            write(root / 'queue_status.json', {'state': state, 'updated_at': now(), 'pid': os.getpid(),
                'queued_studies': [str(QWEN), str(GEMMA)],
                'queued_trials': study_count(root, QWEN) + study_count(root, GEMMA),
                'original_queued_trials': 25200, 'qualification_trials': 0,
                'automatic_resume': False, 'active_runner_pid': active_runner,
                **action_status(root), **detail})
        def wait_complete(out, count):
            while True:
                status = read(out / 'status.json'); ready, reason = study_completion_gate(root, out, status)
                if ready and not study_processes(out): return
                if status.get('state') == 'running' and status.get('pid') and not process_alive(status['pid']):
                    raise RuntimeError(f'{out}: recorded running process disappeared; reconcile before continuing')
                pipeline_state, pipeline_error = pipeline_readiness(root, manifest)
                publish('queued_waiting_predecessor', predecessor=str(out), reason=reason,
                        predecessor_completed=status.get('completed_trials'), predecessor_planned=count,
                        predecessor_deferred_trials=status.get('deferred_trials', 0),
                        decision_ready=decisions_gate(root), pipeline_ready=pipeline_state == 'ready',
                        pipeline_state=pipeline_state, pipeline_error=pipeline_error)
                time.sleep(poll)
        def acquire_gpu():
            nonlocal gpu
            while True:
                gpu = LOCK.open('a+')
                try: fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB); return
                except BlockingIOError:
                    gpu.close(); gpu = None; publish('waiting_for_gpu_lock'); time.sleep(poll)
        def release_gpu():
            nonlocal gpu
            if gpu is not None:
                fcntl.flock(gpu, fcntl.LOCK_UN); gpu.close(); gpu = None
        def audit(out, name, count):
            if out == CURRENT:
                argv = [AGENT_PYTHON, '-B', str(Path(__file__).resolve()), '--out', str(root), '--audit-notification']
            else:
                argv = [AGENT_PYTHON, '-B', str(out / 'code' / RUNNER.name), '--out', str(out), '--audit-completed']
            publish('auditing_completed_study', predecessor=str(out))
            log = command(root, name, argv)
            proof = json.loads([line for line in log.read_text().splitlines() if line.strip()][-1])
            assert proof['state'] == 'passed' and proof['completed_trials'] == proof['planned_trials'] == count
            assert proof['model_inference_calls'] == proof['experimental_trials'] == proof['qualification_trials'] == 0
            assert proof['plan_sha256'] == digest(out / 'plan.json')
            write(root / f'{name}.json', proof)
        def launch(out, key):
            nonlocal active_runner
            validate_pins(root, manifest, pipeline)
            cfg = read(out / 'execution_manifest.json'); plan = read(out / 'plan.json')
            assert cfg['planned_trials'] == plan['planned_trials'] and cfg['qualification_trials'] == 0
            assert read(out / 'status.json')['state'] == 'frozen_ready'
            verify_hashes(cfg['hashes'], out)
            assert digest(out / 'code' / RUNNER.name) == pipeline['source_hashes'][str(RUNNER)]
            assert cfg['action_support']['code_action'] == choice['selection']
            release_gpu()
            argv = pipeline['commands'][key]
            with (out / 'runner.log').open('ab') as log:
                proc = subprocess.Popen(argv, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active_runner = proc.pid
            write(out / 'launch.json', {'pid': proc.pid, 'command': argv, 'started_at': now(),
                'queue_manifest_sha256': digest(root / 'queue_manifest.json'),
                'pipeline_manifest_sha256': digest(root / 'pipeline_manifest.json')})
            publish('starting_successor', successor=str(out), runner_pid=proc.pid)
            limit = time.monotonic() + 1800
            while read(out / 'status.json')['state'] == 'frozen_ready':
                if proc.poll() is not None: raise RuntimeError(f'{out}: runner exited before entering running state')
                if time.monotonic() > limit: raise TimeoutError(f'{out}: runner startup exceeded30minutes')
                time.sleep(1)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            wait_complete(CURRENT, study_count(root, CURRENT))
            while True:
                pipeline_state, pipeline_error = pipeline_readiness(root, manifest)
                decision_ready = decisions_gate(root)
                if decision_ready and pipeline_state == 'ready':
                    break
                publish('waiting_for_action_choice' if not decision_ready else
                        ('pipeline_needs_refresh' if pipeline_state == 'needs_refresh' else 'waiting_for_reviewed_pipeline'),
                        decision_ready=decision_ready, pipeline_ready=pipeline_state == 'ready',
                        pipeline_state=pipeline_state, pipeline_error=pipeline_error)
                time.sleep(poll)
            pipeline = read(root / 'pipeline_manifest.json'); validate_pins(root, manifest, pipeline)
            choice = read(root / 'decisions.json')['typed_action_support']; choice_sha = digest(root / 'decisions.json')
            acquire_gpu(); assert study_completion_gate(root, CURRENT, read(CURRENT / 'status.json'))[0]
            audit(CURRENT, 'current_qwen_completion_audit', study_count(root, CURRENT))
            write(root / 'handoff_started.json', {'at': now(), 'pid': os.getpid(), 'decision_sha256': choice_sha,
                'pipeline_sha256': digest(root / 'pipeline_manifest.json')})
            for out, phases, key in ((QWEN, QWEN_PHASES, 'qwen_run'),
                                    (GEMMA, GEMMA_PHASES, 'gemma_run')):
                count = study_count(root, out)
                if gpu is None: acquire_gpu()
                assert digest(root / 'decisions.json') == choice_sha, 'Choice changed during handoff'
                for phase in phases:
                    assert digest(root / 'decisions.json') == choice_sha
                    validate_pins(root, manifest, pipeline)
                    publish('preparing_successor', successor=str(out), phase=phase)
                    command(root, phase, pipeline['commands'][phase], gpu)
                launch(out, key)
                wait_complete(out, count); active_runner = None
                acquire_gpu(); audit(out, key + '_completion_audit', count)
                release_gpu()
            selected = study_count(root, QWEN) + study_count(root, GEMMA)
            publish('complete_scoped' if selected != 25200 else 'complete',
                    completed_successor_trials=selected, user_deferred_successor_trials=25200-selected)
        except BaseException as exc:
            publish('queue_stopped' if isinstance(exc, KeyboardInterrupt) else 'queue_failed',
                    error=f'{type(exc).__name__}: {exc}', automatic_pipeline_retry=False)
            raise
        finally:
            release_gpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'seal', 'run', 'audit-notification'): actions.add_argument('--' + name, action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args(); root = args.out.resolve()
    if not 1 <= args.poll_seconds <= 60: parser.error('--poll-seconds must be1..60')
    if args.prepare:
        prepare(root); print(json.dumps({'state': 'prepared_waiting_predecessor',
            'queued_trials': study_count(root, QWEN) + study_count(root, GEMMA), 'inference_calls': 0}))
    elif args.seal:
        seal(root); print(json.dumps({'state': 'pipeline_sealed', 'inference_calls': 0}))
    elif args.audit_notification:
        print(json.dumps(audit_notification(CURRENT)))
    else: run(root, args.poll_seconds)
