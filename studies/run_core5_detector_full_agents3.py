"""Frozen core-five ledger evaluation: legacy native detector versus full Agent S3.

Preparation and contract validation are CPU-only. Build/freeze/run are separate
steps; the successor queue owns predecessor completion and exclusive GPU handoff.
No screening episodes, policy rewrites, or competence gates are introduced.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import urllib.request

from legacy_d1_w8 import digest, now, write, setup as legacy_setup

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'runs/legacy_detector_full_agents3_20261029/experiment'
PLUGIN = ROOT / 'runs/runtime_reversal_stock_20261029/build/libNvInfer_edgellm_plugin.so'
APPROVED_BACKENDS = {
    'bb291453bc10d9bd1142070c605e090d4e483ee2': 'v0.10.0',
    '358de11c7c5f30fdaac75f0726a3b2df7e842778': 'v0.11.1',
}
ARMS = ('legacy_cpp_detector', 'agents3_full')
TASK_NAMES = {'notification': 'Disappearing Notification', 'drift': 'Moving Balloon',
              'code': 'Expiring Authenticator Code', 'button': 'Displaced Button',
              'trading': 'Reversing Trading Signal'}
PROMPT_VERSIONS = {'notification': 4, 'drift': 5, 'code': 11, 'button': 12, 'trading': 10}
MODEL_SETTINGS = {
    'qwen35_0p8b': {
        'id': 'Qwen/Qwen3.5-0.8B', 'revision': '2fc06364715b967f1860aea9cf38778875588b17',
        'source_precision': 'BF16 checkpoint; no quantization',
        'runtime_precision': 'FP16 LLM and vision floating tensors; stock BF16-to-FP16 conversion in TensorRT-Edge-LLM, matching vLLM dtype half',
        'kv_cache': 'FP16',
        'engines': str(ROOT / 'runs/notification_qwen08_edgellm0111_20261031/engines'),
    },
    'nvidia_27b': {
        'id': 'nvidia/Qwen3.8-27B-NVFP4', 'revision': '482ca0f3832238542f8f5295dde86b5f22711d80',
        'source_precision': 'NVFP4 MLP/lm_head; FP8 attention; BF16 floating components',
        'runtime_precision': 'NVFP4 MLP/lm_head; FP8 attention projections; FP16 vision and unquantized LLM tensors',
        'kv_cache': 'FP16',
        'engines': str(ROOT / 'runs/agents3_policy_nvidia27b_20261029/engines'),
    },
    'gemma4_26b_a4b': {
        'id': 'nvidia/Gemma-4-26B-A4B-NVFP4', 'revision': 'a19cfe00be84568a6867111c9a68c9c44fdcffe6',
        'source_precision': 'NVFP4 routed expert weights and activation scales; BF16 floating components; FP8 KV scales',
        'runtime_precision': 'NVFP4 routed expert weights with FP16 activations (W4A16); FP16 attention, shared MLP, router, embeddings and vision',
        'kv_cache': 'FP8 E4M3',
        'engines': str(ROOT / 'runs/legacy_detector_full_agents3_gemma4_26b_20261029/engines'),
    },
    'nemotron35_bolt_e4b': {
        'id': 'nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16',
        'revision': '49843eed8db1f1c616c47af75cfd36e07aec0667',
        'source_precision': 'BF16 checkpoint; approximately 8B total parameters and 4B effective',
        'runtime_precision': 'FP16 LLM and vision floating tensors; stock BF16-to-FP16 conversion in TensorRT-Edge-LLM, matching vLLM dtype half',
        'kv_cache': 'FP16',
        'engines': str(ROOT / 'runs/notification_nemotron_bolt_20261030/engines'),
    },
}


def read(path):
    return json.loads(Path(path).read_text())


def setup(code):
    legacy_setup(code)
    sys.path.insert(0, str(code))
    frozen = code.parent / 'execution_manifest.json'
    plugin = read(frozen)['plugin']['path'] if frozen.exists() else str(PLUGIN)
    os.environ['EDGELLM_PLUGIN_PATH'] = plugin
    from chronotope_core.glue import register
    register()


def verify_hashes(mapping, root=None):
    assert mapping, 'Missing provenance hashes'
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = (root or ROOT) / path
        assert digest(path) == expected, f'Artifact changed: {path}'


def execution_jobs(plan, action_support):
    """Keep unsupported typing and explicitly deferred windows pending.

    Deferral is a conservative scheduling choice, not permission to change either
    controller or to score an unsupported action as a model failure. The complete
    plan and its denominators remain intact.
    """
    assert action_support['configuration_reviewed'] is True
    choice = action_support['code_action']
    assert choice in ('defer_unsupported', 'legacy_unsupported_report_failures',
                      'native_type_value_and_host_bridge')
    if choice == 'defer_unsupported':
        assert action_support['user_authorized'] is False, 'Deferral is not a typing authorization'
    else:
        assert action_support['user_authorized'] is True
    windows = action_support.get('deferred_windows_s', [])
    assert isinstance(windows, list) and all(type(window) in (int, float) for window in windows)
    assert len(windows) == len(set(windows)) and set(windows) <= {0.5, 0.25, 0.125}, 'Unreviewed window deferral'
    if windows:
        authorization = action_support.get('window_selection_authorization')
        assert isinstance(authorization, str) and authorization.strip(), 'Window deferral requires user direction'

    def pending(job):
        return ((choice == 'defer_unsupported' and job['task'] == 'code') or
                (job['task'] == 'notification' and job.get('window_s') in windows))

    deferred = [job for job in plan['jobs'] if pending(job)]
    ready = [job for job in plan['jobs'] if not pending(job)]
    return ready, deferred


def deferral_status(ready, deferred):
    """Report action capability and user-selected window deferrals separately."""
    code_count = sum(job['task'] == 'code' for job in deferred)
    window_jobs = [job for job in deferred if job['task'] == 'notification']
    assert code_count + len(window_jobs) == len(deferred)
    assert all(job['window_s'] in (0.5, 0.25, 0.125) for job in window_jobs)
    reasons = []
    if code_count:
        reasons.append('Typed-action transport is unsupported by the unchanged configuration; '
                       'no Authenticator trials have run or been scored.')
    if window_jobs:
        windows = ', '.join(f"{window:g}s" for window in sorted({job['window_s'] for job in window_jobs}, reverse=True))
        reasons.append(f'Notification trials at {windows} are deferred at the user\'s request; '
                       'other response windows remain scheduled.')
    return {'runnable_trials': len(ready), 'deferred_trials': len(deferred),
            'unsupported_action_deferred_trials': code_count,
            'user_deferred_window_trials': len(window_jobs),
            'deferred_task': 'code' if code_count else None,
            'deferred_window_task': 'notification' if window_jobs else None,
            'deferred_windows_s': sorted({job['window_s'] for job in window_jobs}),
            'deferred_reason': ' '.join(reasons) or None}


def completion_state(plan, completed, deferred):
    """A finished runnable subset must never be presented as a complete study."""
    remaining = {job['id'] for job in plan['jobs']} - set(completed)
    deferred_ids = {job['id'] for job in deferred}
    assert not (set(completed) & deferred_ids), 'Deferred jobs must not have scored records'
    assert remaining <= deferred_ids, 'Runnable jobs remain unfinished'
    if any(job['task'] == 'code' and job['id'] in remaining for job in deferred):
        return 'blocked_action_support'
    return 'complete_scoped' if remaining else 'complete'


def audit_jobs(plan, action_support, status):
    """Accept a complete study or an explicitly selected window-only scope."""
    ready, deferred = execution_jobs(plan, action_support)
    assert status['error'] is None
    assert status['planned_trials'] == plan['planned_trials'] == len(plan['jobs'])
    if status['state'] == 'complete':
        assert not deferred, 'Complete study still contains deferred jobs'
    else:
        assert status['state'] == 'complete_scoped', 'Study is not complete'
        assert deferred and all(job['task'] == 'notification' and
                                job['window_s'] in action_support['deferred_windows_s']
                                for job in deferred), 'Only authorized window deferrals permit a scoped audit'
        detail = deferral_status(ready, deferred)
        assert detail['unsupported_action_deferred_trials'] == 0
        for key in ('runnable_trials', 'deferred_trials', 'unsupported_action_deferred_trials',
                    'user_deferred_window_trials', 'deferred_task', 'deferred_window_task', 'deferred_windows_s'):
            assert status[key] == detail[key], f'Scoped completion metadata mismatch: {key}'
    assert status['completed_trials'] == len(ready)
    return ready


def task_instruction(task):
    from chronotope.controller import UITARS_INSTRUCTION
    if task == 'code':
        return ('Type the four-digit code displayed in the yellow panel into the text field below it. '
                'The field is already focused. Preserve leading zeros. If no code is visible, wait.')
    if task == 'button':
        return ('Click the center of the purple Continue button on this page. '
                'A banner may move the button. If the button is absent, wait.')
    if task == 'trading':
        return ('When a buy signal is active the order button in the right-hand panel turns green; '
                'otherwise it is grey. Click the center of the green order button when visible. '
                'If no green order button is visible, wait.')
    return UITARS_INSTRUCTION['balloon' if task == 'drift' else 'notification']


def validate_components(full, model):
    reference = read(REFERENCE / 'configs/full_agent_components.json')
    assert full['schema'] == reference['schema'] == 'full-agents3-components-1'
    assert full['policy'] == reference['policy']
    assert full['generation'] == reference['generation']
    assert full['grounder_generation'] == reference['grounder_generation']
    assert set(full['components']) == set(reference['components'])
    for role, component in full['components'].items():
        if role == 'grounder':
            for key in ('model', 'revision', 'precision', 'kv_cache', 'source_precision'):
                assert component[key] == reference['components'][role][key], (role, key)
        else:
            assert (component['model'], component['revision']) == (model['id'], model['revision'])
            assert component['precision'] == model['runtime_precision']
            assert component['kv_cache'] == model['kv_cache']
            assert component['max_images_per_request'] >= 9
        assert component['runtime'] == 'vLLM' and component['base_url'].startswith('http://127.0.0.1:')


def prepare(out, model_key, include_notification, components=None, *, tasks=None, notification_windows=None,
            engine_directory=None):
    from core5_ledger_plan import prepare_cohort
    model = dict(MODEL_SETTINGS[model_key])
    if engine_directory is not None:
        assert Path(engine_directory).resolve() == (out.parent / 'engines').resolve(), 'Engine override must use this run root'
        model['engines'] = str(Path(engine_directory).resolve())
    if not (out / 'plan.json').exists():
        prepare_cohort(out, model_key, model['id'], model['revision'], include_notification,
                       tasks=tasks, notification_windows=notification_windows)
    assert not (out / 'execution_manifest.json').exists()
    assert not list((out / 'records').glob('*.json'))
    code = out / 'code'
    for sub in ('src', 'specs', 'chronotope/python', 'audit_deps', 'gui_agents'):
        source = REFERENCE / 'code' / sub
        if source.exists() and not (code / sub).exists():
            shutil.copytree(source, code / sub, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('legacy_d1_w8.py', 'agents3_full.py'):
        shutil.copy2(REFERENCE / 'code' / name, code / name)
    for name in ('run_core5_detector_full_agents3.py', 'core5_ledger_plan.py', 'audit_legacy_detector_full_agents3.py'):
        shutil.copy2(ROOT / 'studies' / name, code / name)
    # Historical browser skin files are data used by the permitted Displaced Button task.
    if not (code / 'assets/skin_browser').exists():
        shutil.copytree(ROOT / 'assets/skin_browser', code / 'assets/skin_browser')
    setup(code)
    full = read(components) if components else read(REFERENCE / 'configs/full_agent_components.json')
    for role, component in full['components'].items():
        if role != 'grounder':
            component.update(model=model['id'], revision=model['revision'], precision=model['runtime_precision'],
                             source_precision=model['source_precision'], kv_cache=model['kv_cache'])
    validate_components(full, model)
    (out / 'configs').mkdir(exist_ok=True)
    write(out / 'configs/full_agent_components.json', full)
    tasks = {job['task'] for job in read(out / 'plan.json')['jobs']}
    for task in tasks:
        native = read(REFERENCE / 'configs/legacy_cpp_detector.json')
        native.update(model_id=model['id'], base_url=model['engines'], prompt_version=PROMPT_VERSIONS[task],
                      image_scale=1 if task == 'button' else 2)
        host = read(REFERENCE / 'configs/agents3_full.json')
        host.update(model_id=model['id'], base_url=full['components']['planner']['base_url'],
                    image_scale=1 if task == 'button' else 2)
        write(out / 'configs' / f'{task}_legacy_cpp_detector.json', native)
        write(out / 'configs' / f'{task}_agents3_full.json', host)
    write(out / 'configs/model.json', dict(model, key=model_key))
    write(out / 'configs/task_instructions.json', {task: task_instruction(task) for task in tasks})
    checks = validate(out)
    write(out / 'preparation_report.json', {'state': 'passed', 'created_at': now(), 'spec_checks': checks,
          'model_inference_calls': 0, 'qualification_trials': 0, 'experimental_trials': 0,
          'task_families': sorted(tasks), 'native_type_action': 'pending_explicit_configuration' if 'code' in tasks else 'not_required',
          'historical_notification_scoring_preserved': True})
    write(out / 'status.json', {'state': 'prepared', 'completed_trials': 0,
          'planned_trials': checks, 'pid': None, 'automatic_resume': False,
          'model': model['id'], 'qualification_trials': 0})
    print(json.dumps({'state': 'prepared', 'out': str(out), 'spec_checks': checks}), flush=True)


def load_cases(out, plan):
    from core5_ledger_plan import load_cases as verified_load_cases
    return verified_load_cases(out, plan)


def validate(out):
    setup(out / 'code')
    from core5_ledger_plan import make_trial_spec, validate_plan
    plan = read(out / 'plan.json')
    validate_plan(plan)
    model = read(out / 'configs/model.json')
    assert {job['model'] for job in plan['jobs']} == {model['key']}
    expected_model = dict(MODEL_SETTINGS[model['key']], key=model['key'])
    backend_path = out.parent / 'backend.json'
    if backend_path.exists() and read(backend_path)['commit'] == '358de11c7c5f30fdaac75f0726a3b2df7e842778' and model['key'] == 'gemma4_26b_a4b':
        # Stock 0.11.1 detects LLM KV scales from checkpoint tensors; this
        # checkpoint has none. Its global quantization metadata was misleading.
        expected_model.update(kv_cache='FP16', source_precision='NVFP4 routed expert weights and activation scales; BF16 floating components; no LLM KV quantization scales')
    assert model['engines'] in (expected_model['engines'], str((out.parent / 'engines').resolve()))
    expected_model['engines'] = model['engines']
    assert model == expected_model
    assert plan['qualification_trials'] == 0
    assert plan['planned_trials'] == len(plan['jobs']) > 0
    assert set(job['task'] for job in plan['jobs']) <= set(TASK_NAMES)
    assert len({job['id'] for job in plan['jobs']}) == len(plan['jobs'])
    validate_components(read(out / 'configs/full_agent_components.json'), model)
    cases = load_cases(out, plan)
    seen = set(); groups = defaultdict(set)
    for job in plan['jobs']:
        assert job['phase'] == 'evaluation' and job['arm'] in ARMS
        assert job['mode'] in ('live', 'replay', 'paused')
        key = (job['cell_id'], job['arm'], job['index'], job['mode'])
        if job['mode'] == 'replay':
            assert key[:3] + ('live',) in seen
        assert key not in seen
        seen.add(key); groups[job['cell_id'], job['arm'], job['mode']].add(job['index'])
        case = cases[job['manifest']][job['index']]
        assert case['episode_id'] == job['episode_id']
        config = read(out / 'configs' / f"{job['task']}_{job['arm']}.json")
        make_trial_spec(case, config, job['mode'], '0' * 64 if job['mode'] == 'replay' else None)
    assert all(indices == set(range(100)) for indices in groups.values())
    return len(plan['jobs'])


def freeze(out, services):
    assert not (out / 'execution_manifest.json').exists()
    assert not list((out / 'records').glob('*.json'))
    model = read(out / 'configs/model.json'); root = out.parent; code = out / 'code'
    native = read(root / 'native_build_result.json')
    assert native['state'] == 'complete' and native['source_patches'] == []
    backend_path = root / 'backend.json'
    backend = read(backend_path) if backend_path.exists() else {
        'commit': 'bb291453bc10d9bd1142070c605e090d4e483ee2',
        'tag': 'v0.10.0', 'plugin': str(PLUGIN)}
    assert APPROVED_BACKENDS.get(backend['commit']) == backend['tag'], 'Unreviewed backend revision'
    assert native['upstream_commit'] == backend['commit']
    plugin = Path(backend['plugin'])
    if backend_path.exists():
        assert plugin.resolve() == (Path(backend['build']) / 'libNvInfer_edgellm_plugin.so').resolve()
        build_proof = read(Path(backend['build']).parent / 'build_result.json')
        assert build_proof['state'] == 'complete' and build_proof['upstream_commit'] == backend['commit']
        assert build_proof['backend_source_patches'] == []
        verify_hashes(build_proof['artifact_hashes'])
        tokenizer_proof = read(backend['tokenizer_verification'])
        assert tokenizer_proof['state'] == 'passed' and tokenizer_proof['all_token_ids_match']
        assert tokenizer_proof['upstream_commit'] == backend['commit']
    assert native['model_inference_calls'] == native['qualification_trials'] == 0
    verify_hashes(native['source_hashes']); verify_hashes(native['binary_hashes'])
    verification = read(root / 'native_compact_verification.json')
    assert verification['state'] == 'passed'
    assert verification['model_inference_calls'] == verification['experimental_trials'] == 0
    verify_hashes(verification['source_file_sha256'])
    verify_hashes(verification['prompt_files_sha256'])
    assets = read(root / 'engine_build_result.json')
    assert (assets['model'], assets['revision'], assets['state']) == (model['id'], model['revision'], 'built')
    assert assets['source_patches'] == []
    assert assets['upstream_commit'] == backend['commit']
    assert assets['kv_cache']['dtype'] == model['kv_cache']
    verify_hashes(assets['hashes']); verify_hashes(assets['source_checkpoint_hashes'])
    for name in ('run_core5_detector_full_agents3.py', 'core5_ledger_plan.py', 'audit_legacy_detector_full_agents3.py'):
        shutil.copy2(ROOT / 'studies' / name, code / name)
    source = root / 'build_source/chronotope'
    assert not (source / 'src/agents3_policy.h').exists()
    for sub in ('src', 'python/chronotope_core'):
        shutil.copytree(source / sub, code / 'chronotope' / sub, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(source / 'CMakeLists.txt', code / 'chronotope/CMakeLists.txt')
    (code / 'chronotope/build').mkdir(exist_ok=True)
    for name in ('chronotope', 'libchronotope.so'):
        shutil.copy2(root / 'native_build' / name, code / 'chronotope/build' / name)
    prompts = read(root / 'inputs/task_prompts.json')
    references = read(root / 'inputs/task_prompt_references.json')
    tasks = {job['task'] for job in read(out / 'plan.json')['jobs']}
    assert tasks <= prompts.keys() and tasks <= references.keys()
    from audit_legacy_detector_full_agents3 import prompt_fingerprint
    for task in tasks:
        prompt = prompts[task]
        assert prompt_fingerprint(prompt) == prompt['prompt_sha256']
        assert (prompt['model'], prompt['revision']) == (model['id'], model['revision'])
        assert prompt['prompt_version'] == PROMPT_VERSIONS[task]
        assert prompt['history'] == 'none' and prompt['reflection'] is False
        assert prompt['generation_settings']['max_tokens'] == 64
        write(out / 'configs' / f'{task}_compact_prompt.json', prompt)
    full = read(out / 'configs/full_agent_components.json'); validate_components(full, model)
    service = read(services)
    assert service['state'] == 'ready' and service['qualification_trials'] == service['inference_readiness_requests'] == 0
    checkpoint_configs = [Path(path) for path in assets['source_checkpoint_hashes'] if Path(path).name == 'config.json']
    assert len(checkpoint_configs) == 1
    source_checkpoint = checkpoint_configs[0].parent
    for role, component in full['components'].items():
        observed = service['components'][role]
        assert observed['state'] == 'ready' and observed['inference_readiness_requests'] == 0
        assert (observed['model'], observed['revision'], observed['base_url']) == (component['model'], component['revision'], component['base_url'])
        command = observed['command']
        assert command[command.index('--served-model-name') + 1] == component['model']
        if role != 'grounder':
            assert str(source_checkpoint) in command, 'Served weights do not match engine source checkpoint'
            assert command[command.index('--dtype') + 1] in ('half', 'float16')
            assert '--no-enable-prefix-caching' in command
            engine = observed['observed_engine_configuration']
            assert 'dtype=torch.float16' in engine and 'enable_prefix_caching=False' in engine
            if model['key'] == 'gemma4_26b_a4b':
                kv_arg = 'float16' if model['kv_cache'] == 'FP16' else 'fp8_e4m3'
                assert command[command.index('--kv-cache-dtype') + 1] == kv_arg
                assert observed['effective_kv_cache_dtype'] == model['kv_cache']
                assert command[command.index('--moe-backend') + 1] == 'marlin'
                assert 'kv_cache_dtype=' + kv_arg in engine and 'quantization=modelopt' in engine
            else:
                assert command[command.index('--kv-cache-dtype') + 1] in ('auto', 'float16')
                if model['key'] in ('nemotron35_bolt_e4b', 'qwen35_0p8b'):
                    assert 'quantization=None' in engine
                else:
                    assert 'quantization=modelopt_mixed' in engine
        else:
            assert command[command.index('--dtype') + 1] == 'bfloat16'
    full_verification = read(root / 'full_agents3_cpu_verification.json')
    assert full_verification['state'] == 'passed'
    assert full_verification['model_inference_calls'] == full_verification['experimental_trials'] == 0
    verify_hashes(full_verification['source_file_sha256'])
    for name, expected in full_verification['source_file_sha256'].items():
        source_path = Path(name)
        if source_path.name == 'agents3_full.py':
            assert digest(code / 'agents3_full.py') == expected
        elif '/site-packages/gui_agents/' in str(source_path):
            relative = str(source_path).split('/site-packages/gui_agents/', 1)[1]
            assert digest(code / 'gui_agents' / relative) == expected
    sha = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert full_verification['component_config_sha256'] == sha
    grounder = read(root / 'grounder_checkpoint_verification.json')
    assert grounder['state'] == 'passed'; verify_hashes(grounder['file_hashes'])
    assert (grounder['model'], grounder['revision']) == (full['components']['grounder']['model'], full['components']['grounder']['revision'])
    assert grounder['checkpoint_path'] in service['components']['grounder']['command']
    action_support = read(root / 'inputs/action_support.json')
    ready_jobs, deferred_jobs = execution_jobs(read(out / 'plan.json'), action_support)
    if 'code' in tasks:
        if action_support['code_action'] == 'native_type_value_and_host_bridge':
            assert verification['typed_action_transport_passed'] is True
    checks = validate(out)
    write(out / 'configs/action_support.json', action_support)
    write(out / 'service_launch.json', service)
    shutil.copy2(ROOT / 'AGENTS.md', out / 'AGENTS.md')
    hashes = {str(p.relative_to(out)): digest(p) for sub in ('code', 'configs', 'cases')
              for p in (out / sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    for name in ('plan.json', 'AGENTS.md', 'service_launch.json', 'preparation_report.json'):
        hashes[name] = digest(out / name)
    execution = {'created_at': now(), 'model': model['id'], 'revision': model['revision'],
        'model_settings': model, 'planned_trials': checks, 'qualification_trials': 0,
        **deferral_status(ready_jobs, deferred_jobs),
        'spec_checks': checks, 'hashes': hashes, 'checkpoint_hashes': assets['source_checkpoint_hashes'],
        'engine_hashes': assets['hashes'], 'plugin': {'path': str(plugin), 'sha256': digest(plugin)},
        'native_backend': backend,
        'native_build': native, 'native_verification': verification, 'source_patches': [],
        'agent_s3_components': full, 'full_agent_source_verification': full_verification,
        'grounder_checkpoint': grounder, 'native_task_prompts': prompts,
        'native_task_prompt_references': references, 'action_support': action_support,
        'arm_labels': {ARMS[0]: 'Legacy C++ + device detector (compact JSON; TensorRT-Edge-LLM)',
                       ARMS[1]: 'Agent S3 (upstream GUI policy; separate grounder)'},
        'interpretation': ['Whole-configuration comparison; prompts and grounding differ.',
                           'Native path is legacy, with stock CPU image staging; no GPU-history claim.',
                           'No qualification trials or competence gates; all scored failures retained.']}
    write(out / 'execution_manifest.json', execution)
    for name in ('records', 'calls'):
        (out / name).mkdir(exist_ok=True)
    write(out / 'status.json', {'state': 'frozen_ready', 'completed_trials': 0, 'planned_trials': checks,
          **deferral_status(ready_jobs, deferred_jobs),
          'qualification_trials': 0, 'pid': None, 'model': model['id'], 'automatic_resume': False})
    print(json.dumps({'state': 'frozen_ready', 'spec_checks': checks}), flush=True)


def parse_historical_type_bridge(text):
    """Reuse the historical matched-Agent-S3 atomic typing grammar, without eval.

    A focus click and select-all preceding a single typed value are part of the
    benchmark's focused-field action. Conflicting values or arbitrary expressions
    never become a typed action. Leading zeros are retained as text.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, TypeError):
        return None
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ''
        if not ((owner == 'pyautogui' and node.func.attr in ('write', 'typewrite')) or
                (owner == 'pyperclip' and node.func.attr == 'copy')):
            continue
        try:
            value = ast.literal_eval(node.args[0]) if node.args else None
        except (ValueError, TypeError):
            continue
        if isinstance(value, str):
            values.append(value)
    if values and len(set(values)) == 1:
        return {'action': 'type', 'value': values[0]}
    return None


def make_host(spec, task, adapter, action_support):
    from chronotope.controller import CallBudget, Decision, VLMDirectController
    from chronotope.png import encode_png

    class TypedController(VLMDirectController):
        def decide(self):
            obs = self._newest
            self._baseline_hash = obs.payload_hash
            self.budget.take()
            response = self.adapter.invoke(encode_png(obs.payload, self.w, self.h, self.scale),
                                           self.system, 'Screenshot attached.')
            parsed = response.parsed
            if action_support['code_action'] == 'native_type_value_and_host_bridge':
                parsed = parse_historical_type_bridge(response.detail['raw_text']) or parsed
            parsed = parsed or {}
            value = parsed.get('value')
            valid = parsed.get('action') == 'type' and isinstance(value, str) and len(value) == 4 and value.isdigit()
            if valid:
                self._done = True
            return Decision({'kind': 'type', 'value': value} if valid else None, None,
                invocation_id=response.invocation_id,
                invocation_detail={**response.detail, 'observation_id': obs.observation_id,
                    'action_bridge': action_support['code_action'],
                    'outcome': 'type' if valid else 'wait' if parsed.get('action') == 'wait' else 'unparseable'})

    m = spec.manifest(); c = spec.raw['controller']
    # Choose an existing supported constructor prompt; the original historical
    # task instruction is then passed to upstream Worker unchanged.
    cls = TypedController if task == 'code' else VLMDirectController
    ctl = cls(adapter=adapter, budget=CallBudget(100), prompt_version=17,
           image_scale=c['image_scale'], invoke_on='frame_change', canvas_w_px=m.canvas_w_px,
           canvas_h_px=m.canvas_h_px, skin='balloon' if task == 'drift' else 'notification',
           target_color=m.target_color())
    ctl.system = task_instruction(task)
    return ctl


def audit_completed(out):
    """Recompute a completed batch's provenance, request audits and scores on CPU."""
    out = Path(out).resolve(); setup(out / 'code')
    from audit_legacy_detector_full_agents3 import audit_trial_requests
    from chronotope.evaluate import evaluate_trace
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    from chronotope.distributional.manifest import EpisodeRow, build_manifest
    from chronotope.distributional.episode import score_trace, verify_trace_evidence
    plan = read(out / 'plan.json'); cfg = read(out / 'execution_manifest.json'); status = read(out / 'status.json')
    jobs = audit_jobs(plan, cfg['action_support'], status)
    verify_hashes(cfg['hashes'], out)
    validate(out); cases = load_cases(out, plan)
    files = list((out / 'records').glob('*.json'))
    assert len(files) == len(jobs)
    records = {p.stem: read(p) for p in files}; checks = 0
    assert set(records) == {job['id'] for job in jobs}, 'Audit record IDs do not match the selected scope'
    for job in jobs:
        record = records[job['id']]
        assert record['job'] == job
        assert digest(out / record['trace']) == record['trace_sha256']
        assert not record['outcome']['invalid_trial'] and not record['outcome']['contract_violations']
        case = cases[job['manifest']][job['index']]
        trace = Trace.load(out / record['trace'])
        from core5_ledger_plan import make_trial_spec
        controller = read(out / 'configs' / f"{job['task']}_{job['arm']}.json")
        assert trace.header()['spec']['controller'] == controller
        spec = make_trial_spec(case, controller, 'live')
        scenario = spec.manifest()
        if case.get('distributional_row') is not None:
            row = EpisodeRow.from_dict(case['distributional_row']); scenario = build_manifest(row)
            verify_trace_evidence(row, scenario, trace, case['distributional_manifest_hash'])
            assert score_trace(row, trace, case['distributional_manifest_hash'], controller=job['arm'], repeat_index=0).to_dict() == record['outcome']
        evaluated = evaluate_trace(trace)
        assert evaluated.valid_trial and not evaluated.contract_violations
        assert evaluated.to_dict() == record['evaluation']
        local_cfg = dict(cfg, native_prompt_contract=cfg['native_task_prompts'][job['task']],
                         native_prompt_reference=cfg['native_task_prompt_references'][job['task']])
        assert audit_trial_requests(out, job['id'], job['arm'], trace, scenario, local_cfg) == record['request_audit']
        if job['mode'] == 'replay':
            live_path = out / 'trials' / job['cell_id'] / job['arm'] / f"{job['episode_id']}_live.jsonl"
            assert record['pair_valid'] is True and validate_pair(trace, Trace.load(live_path)).valid
        checks += 1
    return {'state': 'passed', 'completed_trials': checks, 'planned_trials': checks,
            'original_planned_trials': plan['planned_trials'],
            'user_deferred_window_trials': plan['planned_trials'] - checks,
            'plan_sha256': digest(out / 'plan.json'), 'execution_manifest_sha256': digest(out / 'execution_manifest.json'),
            'model_inference_calls': 0, 'experimental_trials': 0, 'qualification_trials': 0}


def run(out, resume=False):
    setup(out / 'code')
    from core5_ledger_plan import make_trial_spec
    from agents3_full import FullAgentS3Adapter
    from audit_legacy_detector_full_agents3 import audit_trial_requests
    from chronotope.controller import CallBudget
    from chronotope.evaluate import evaluate_trace
    from chronotope.runner import run_trial, PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    from chronotope_core.controller import build as build_native
    from chronotope.distributional.metrics import percentiles
    from chronotope.distributional.manifest import EpisodeRow, build_manifest
    from chronotope.distributional.episode import score_trace, verify_trace_evidence
    plan = read(out / 'plan.json'); cfg = read(out / 'execution_manifest.json')
    ready_jobs, deferred_jobs = execution_jobs(plan, cfg['action_support'])
    verify_hashes(cfg['hashes'], out); verify_hashes(cfg['checkpoint_hashes']); verify_hashes(cfg['engine_hashes'])
    verify_hashes(cfg['grounder_checkpoint']['file_hashes'])
    assert digest(cfg['plugin']['path']) == cfg['plugin']['sha256']
    validate(out)
    for name in ('CHRONOTOPE_AGENT_S3_POLICY', 'CHRONOTOPE_AGENT_S3_PROMPT', 'CHRONOTOPE_AGENT_S3_AUDIT'):
        os.environ.pop(name, None)
    for component in cfg['agent_s3_components']['components'].values():
        with urllib.request.urlopen(component['base_url'].rstrip('/') + '/models', timeout=15) as response:
            assert component['model'] in [m['id'] for m in json.load(response)['data']]
    lock = (ROOT / 'runs/legacy_d1_gpu.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cases = load_cases(out, plan); records = []
    for p in sorted((out / 'records').glob('*.json')):
        r = read(p); assert digest(out / r['trace']) == r['trace_sha256']; records.append(r)
    if records and not resume:
        raise RuntimeError('Existing results require explicit --resume')
    completed = {r['job']['id'] for r in records}; planned = {j['id']: j for j in plan['jobs']}
    assert len(completed) == len(records)
    for record in records:
        assert record['job'] == planned[record['job']['id']]
        assert not record['outcome']['invalid_trial'] and not record['outcome']['contract_violations']
    started = read(out / 'status.json').get('started_at', now()) if resume else now()
    current = None

    def publish(state, error=None):
        groups = defaultdict(list)
        for record in records:
            job = record['job']; groups[job['cell_id'], job['task'], job['arm'], job['mode']].append(record)
        rows = []
        model = cfg['model_settings']
        for (cell, task, arm, mode), own in sorted(groups.items()):
            outcomes = [r['outcome'] for r in own]; native = arm == ARMS[0]
            row = {'Task': TASK_NAMES[task], 'Setting': cell, 'Controller': cfg['arm_labels'][arm],
                   'Model': cfg['model'], 'Revision': cfg['revision'], 'Source precision': model['source_precision'],
                   'Precision': model['runtime_precision'], 'KV cache': model['kv_cache'],
                   'Runtime': 'TensorRT-Edge-LLM' if native else 'vLLM',
                   'Detector / grounder': 'GPU task-color detector; NCC fallback' if native else 'ByteDance-Seed/UI-TARS-1.5-7B',
                   'Grounder precision': '' if native else 'BF16',
                   'Window (s)': own[0]['job']['window_s'], 'Condition': mode, 'n': len(own),
                   'Successes': sum(o['Y'] for o in outcomes), 'SR': sum(o['Y'] for o in outcomes)/len(own),
                   'No receipt': sum(not o['R'] for o in outcomes), 'Invalid': sum(o['invalid_trial'] for o in outcomes),
                   'Return': (sum(o['G_return'] for o in outcomes)/len(own)
                              if all(o['G_return'] is not None for o in outcomes) else None),
                   'Replay pairs valid': sum(r['pair_valid'] is True for r in own),
                   **{role.title() + ' calls': sum(e['role'] == role for r in own for e in r['request_audit'])
                      for role in ('planner', 'reflection', 'grounder', 'text_grounder')}}
            for label, key in [('E-to-O', 'L_EO'), ('O-to-B', 'L_OB'), ('B-to-D', 'L_BD'), ('D-to-A', 'L_DA'), ('E-to-A', 'L_EA')]:
                for pc, value in percentiles([o[key] for o in outcomes]).items():
                    if pc in ('p50', 'p95'): row[f'{label} {pc} (ms)'] = None if value is None else value/1e6
            rows.append(row)
        if rows:
            with (out / 'summary.csv.tmp').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            (out / 'summary.csv.tmp').replace(out / 'summary.csv')
        write(out / 'status.json', {'state': state, 'started_at': started, 'updated_at': now(),
              'completed_trials': len(records), 'planned_trials': plan['planned_trials'],
              **deferral_status(ready_jobs, deferred_jobs),
              'pid': os.getpid() if state == 'running' else None, 'current_job': current,
              'qualification_trials': 0, 'model': cfg['model'], 'error': error, 'automatic_resume': False})

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    publish('running')
    try:
        for current in ready_jobs:
            ident = current['id']
            if ident in completed:
                continue
            publish('running'); task, arm, mode = current['task'], current['arm'], current['mode']
            case = cases[current['manifest']][current['index']]
            controller_config = read(out / 'configs' / f'{task}_{arm}.json')
            folder = out / 'trials' / current['cell_id'] / arm; folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{current['episode_id']}_{mode}.jsonl"
            assert not path.exists(), f'Incomplete trace requires explicit recovery: {path}'
            live = Trace.load(folder / f"{current['episode_id']}_live.jsonl") if mode == 'replay' else None
            spec = make_trial_spec(case, controller_config, mode, live.file_hash() if live else None)
            scenario = spec.manifest()
            distributional_row = None
            if case.get('distributional_row') is not None:
                distributional_row = EpisodeRow.from_dict(case['distributional_row'])
                scenario = build_manifest(distributional_row)
            call_log = out / 'calls' / f'{ident}.jsonl'
            adapter = FullAgentS3Adapter(cfg['agent_s3_components'], call_log) if arm == ARMS[1] else None
            os.environ['CHRONOTOPE_PLANNER_PROMPT'] = str(out / 'configs' / f'{task}_compact_prompt.json')
            os.environ['CHRONOTOPE_PLANNER_AUDIT'] = str(call_log)
            controller = build_native(spec, budget=CallBudget(100)) if arm == ARMS[0] else make_host(spec, task, adapter, cfg['action_support'])
            start = time.monotonic()
            try:
                run_trial(spec, path, PayloadStore(out / 'payloads'), live_trace=live,
                          controller=controller, manifest=scenario)
            finally:
                close = getattr(controller, 'close', None)
                if close: close()
                if adapter: adapter.close()
            trace = Trace.load(path)
            if trace.all('decision_failed') or (adapter and adapter.errors):
                raise RuntimeError(f'Controller/runtime infrastructure error: {ident}')
            audit_cfg = dict(cfg, native_prompt_contract=cfg['native_task_prompts'][task],
                             native_prompt_reference=cfg['native_task_prompt_references'][task])
            audit = audit_trial_requests(out, ident, arm, trace, scenario, audit_cfg)
            evaluated = evaluate_trace(trace)
            intervals = evaluated.intervals_ns
            outcome = {'Y': int(bool(evaluated.valid_at_receipt)), 'C': int(bool(evaluated.semantic_correct)),
                       'R': int(bool(trace.all('receipt'))), 'invalid_trial': not evaluated.valid_trial,
                       'contract_violations': evaluated.contract_violations, 'G_return': evaluated.return_G,
                       'failure_mode': 'success' if evaluated.valid_at_receipt else 'no_action' if not evaluated.action_committed else ','.join(evaluated.labels),
                       **{f'L_{key.replace("_", "")}': intervals.get(key) for key in ('E_O', 'O_B', 'B_D', 'D_A', 'E_A')}}
            if distributional_row is not None:
                verify_trace_evidence(distributional_row, scenario, trace, case['distributional_manifest_hash'])
                outcome = score_trace(distributional_row, trace, case['distributional_manifest_hash'],
                                      controller=arm, repeat_index=0).to_dict()
            pair = validate_pair(trace, live) if live else None
            record = {'job': current, 'request_audit': audit, 'outcome': outcome,
                      'evaluation': evaluated.to_dict(), 'trace': str(path.relative_to(out)), 'trace_sha256': trace.file_hash(),
                      'pair_valid': pair.valid if pair else None, 'pair_reason': pair.reason if pair else None,
                      'wall_seconds': time.monotonic()-start, 'completed_at': now()}
            write(out / 'records' / f'{ident}.json', record); records.append(record); completed.add(ident)
            if outcome['invalid_trial'] or outcome['contract_violations'] or (pair and not pair.valid):
                raise RuntimeError(f'Invalid trace/pair: {ident}')
            print(f"{len(records)}/{plan['planned_trials']} {ident} success={outcome['Y']}", flush=True)
            publish('running')
        current = None; publish(completion_state(plan, completed, deferred_jobs))
    except BaseException as error:
        publish('stopped' if isinstance(error, KeyboardInterrupt) else 'failed', f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare', 'validate', 'freeze', 'run', 'audit-completed'):
        group.add_argument('--' + action, action='store_true')
    parser.add_argument('--model-key', choices=MODEL_SETTINGS)
    parser.add_argument('--include-notification', action='store_true')
    parser.add_argument('--tasks', nargs='+', choices=tuple(TASK_NAMES))
    parser.add_argument('--notification-windows', nargs='+', type=float,
                        choices=(16., 8., 4., 2., 1., .5, .25, .125))
    parser.add_argument('--engine-directory', type=Path)
    parser.add_argument('--components', type=Path)
    parser.add_argument('--services', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.out.resolve()
    if args.prepare:
        if not args.model_key: parser.error('--prepare requires --model-key')
        prepare(out, args.model_key, args.include_notification, args.components, tasks=args.tasks,
                notification_windows=args.notification_windows, engine_directory=args.engine_directory)
    elif args.validate:
        print(json.dumps({'spec_checks': validate(out), 'model_inference_calls': 0}))
    elif args.freeze:
        if not args.services: parser.error('--freeze requires --services')
        freeze(out, args.services)
    elif args.audit_completed:
        print(json.dumps(audit_completed(out)))
    else:
        run(out, args.resume)
