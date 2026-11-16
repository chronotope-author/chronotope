"""Fresh compact legacy C++/detector versus full upstream Agent S3 evaluation.

Only the existing 100 Notification cases, eight windows and three conditions.
All competence failures remain in the table; no screening or warmup episodes.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from collections import defaultdict
import csv
import fcntl
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
PARENT = ROOT / 'runs/nvidia_prompt_only_20261029/experiment'
ASSETS = ROOT / 'runs/agents3_policy_nvidia27b_20261029'
MODEL = 'nvidia/Qwen3.8-27B-NVFP4'
REVISION = '482ca0f3832238542f8f5295dde86b5f22711d80'
PRECISION = 'NVFP4 MLP/lm_head; FP8 attention; BF16 vision'
RUNTIME_PRECISION = 'NVFP4 MLP/lm_head; FP8 attention projections; FP16 vision and unquantized LLM tensors'
CHECKPOINT = ROOT / 'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f'
PLUGIN = ROOT / 'runs/runtime_reversal_stock_20261029/build/libNvInfer_edgellm_plugin.so'
ARMS = ('legacy_cpp_detector', 'agents3_full')
ARM_RENAME = {'legacy_cpp': ARMS[0], 'agents3_stripped': ARMS[1]}
WINDOWS = (.125, .25, .5, 1., 2., 4., 8., 16.)
UPSTREAM = Path((_CHRONOTOPE_HOME + '/agents3-venv/lib/python3.12/site-packages/gui_agents'))
GROUNDER = 'ByteDance-Seed/UI-TARS-1.5-7B'
GROUNDER_REVISION = '683d002dd99d8f95104d31e70391a39348857f4e'


def read(path):
    return json.loads(Path(path).read_text())


def setup(code):
    legacy_setup(code)
    sys.path.insert(0, str(code))
    os.environ['EDGELLM_PLUGIN_PATH'] = str(PLUGIN)


def job_id(job):
    return f"{job['arm']}_w{round(job['window_s']*1e9)}_{job['index']:03d}_{job['mode']}"


def validate_plan(plan):
    assert plan['arms'] == list(ARMS)
    assert sorted(plan['windows_s']) == list(WINDOWS)
    assert set(plan['modes']) == {'live', 'replay', 'paused'}
    assert plan['trials_per_cell'] == 100
    assert set(plan['models']) == {'nvidia_27b'}
    model = plan['models']['nvidia_27b']
    assert (model['id'], model['revision'], model['precision']) == (MODEL, REVISION, PRECISION)
    assert plan['planned_trials'] == 4800 and plan['qualification_trials'] == 0
    assert len(plan['jobs']) == 4800
    seen = set(); groups = defaultdict(set)
    for job in plan['jobs']:
        assert (job['phase'], job['task'], job['family']) == ('evaluation', 'notification', 'closing_window')
        assert job['arm'] in ARMS and job['model'] == 'nvidia_27b' and job['window_s'] in WINDOWS
        assert job['mode'] in ('live', 'replay', 'paused') and 0 <= job['index'] < 100
        key = (job['arm'], job['window_s'], job['index'], job['mode'])
        assert key not in seen
        if job['mode'] == 'replay':
            assert key[:3] + ('live',) in seen
        seen.add(key); groups[job['arm'], job['window_s'], job['mode']].add(job['index'])
    assert len(groups) == 48 and all(x == set(range(100)) for x in groups.values())


def same_cohort(plan):
    parent = read(PARENT / 'plan.json')
    expected = [dict(j, arm=ARM_RENAME[j['arm']]) for j in parent['jobs']]
    assert plan['jobs'] == expected, 'Case/window/condition/order changed beyond controller labels'
    assert plan['manifest_hashes'] == parent['manifest_hashes']


def validate_components(config):
    assert config['schema'] == 'full-agents3-components-1'
    components = config['components']
    assert set(components) == {'planner', 'reflection', 'grounder', 'text_grounder'}
    for role, component in components.items():
        for key in ('model', 'revision', 'precision', 'runtime', 'base_url'):
            assert component.get(key), (role, key)
        assert component['runtime'] == 'vLLM'
        if role != 'grounder':
            assert (component['model'], component['revision']) == (MODEL, REVISION)
            assert component['precision'] == RUNTIME_PRECISION
            assert component['kv_cache'] == 'FP16'
            assert component['max_images_per_request'] >= 9
        else:
            assert (component['model'], component['revision']) == (GROUNDER, GROUNDER_REVISION)
            assert component['precision'] == component['kv_cache'] == 'BF16'
            assert component['source_precision'] == 'FP32'
            assert component['config_declared_dtype'] == component['runtime_precision'] == 'BF16'
    assert config['policy']['upstream_worker_prompt'] is True
    assert config['policy']['upstream_aci'] is True
    assert config['policy']['history_images_retained'] is True
    assert config['policy']['reflection'] is True
    assert config['policy']['direct_pixel_prompt_rewrite'] is False
    assert config['policy']['history_image_stripping'] is False
    assert config['policy']['history_n'] == 8
    assert config['policy']['code_agent_enabled'] is False
    assert config['policy']['code_agent_disabled_reason'] == 'upstream_default'
    assert config['generation'] == {'max_tokens': 4096, 'temperature': 0.0, 'top_p': 0.8,
                                    'repetition_penalty': 1.0, 'enable_thinking': False}
    assert config['grounder_generation'] == {'max_tokens': 4096, 'temperature': 0.0, 'top_p': 0.8,
                                             'repetition_penalty': 1.05}


def require_parent_stopped():
    status = read(PARENT / 'status.json')
    assert status['state'] != 'running', 'Pause/supersede the preceding diagnostic before launch'
    if status.get('pid'):
        proc = Path(f"/proc/{status['pid']}/cmdline")
        assert not proc.exists() or str(PARENT).encode() not in proc.read_bytes()


def verify_hashes(mapping, out=None):
    assert mapping
    for path, expected in mapping.items():
        path = Path(path)
        if not path.is_absolute():
            path = (out or ROOT) / path
        assert digest(path) == expected, str(path)


def validate_specs(out):
    plan = read(out / 'plan.json'); validate_plan(plan); same_cohort(plan)
    setup(out / 'code')
    from chronotope.distributional.manifest import ManifestSet, build_manifest, verify_row_against_manifest
    from chronotope.distributional.episode import ControllerSpec, trial_spec_for
    configs = {arm: read(out / 'configs' / f'{arm}.json') for arm in ARMS}
    native, host = (configs[arm] for arm in ARMS)
    assert native['model_id'] == host['model_id'] == MODEL
    assert native['planner'] == 'trt' and native['server'] == 'cpp_socket'
    assert (native['prompt_version'], native['max_tokens'], native['image_scale']) == (4, 64, 2)
    assert native['binder'] == 'detector' and native['at_commit_policy'] == 'late_bind'
    assert native['memory'] == 'none'
    assert host['server'] == 'agents3_service' and host['max_tokens'] == 4096
    validate_components(read(out / 'configs/full_agent_components.json'))
    manifests = {name: ManifestSet.load(out / 'manifests' / name) for name in plan['manifest_hashes']}
    for name, manifest in manifests.items():
        assert manifest.frozen_hash == plan['manifest_hashes'][name] and len(manifest.rows) == 100
        for row in manifest.rows:
            verify_row_against_manifest(row, build_manifest(row))
    for job in plan['jobs']:
        row = manifests[job['manifest']].rows[job['index']]
        assert row.task_family == 'closing_window' and row.deadline_s == job['window_s']
        assert row.episode_id == job['episode_id']
        trial_spec_for(row, ControllerSpec(job['arm'], configs[job['arm']]), job['mode'],
                       '0' * 64 if job['mode'] == 'replay' else None)
    return len(plan['jobs'])


def prepare(out, components):
    assert not out.exists(), 'Never replace an existing experiment'
    full = read(components); validate_components(full)
    parent = read(PARENT / 'plan.json')
    plan = dict(parent, jobs=[dict(j, arm=ARM_RENAME[j['arm']]) for j in parent['jobs']], arms=list(ARMS),
                created_at=now(), state='prepared', qualification_trials=0,
                supersedes=str(PARENT), predecessor_plan_sha256=digest(PARENT / 'plan.json'),
                comparison='Legacy compact JSON + device detector versus upstream Agent S3 GUI policy',
                policy_difference={'matching_scope': 'Same planner checkpoint; deliberately different controller policies and grounding',
                    'legacy_cpp_detector': 'Stateless compact JSON; device detector at commitment, NCC fallback',
                    'agents3_full': 'Upstream Worker, OSWorldACI, original prompt, reflection, text and screenshot history; separate grounding model'})
    plan['models'] = {key: dict(value, legacy_tensorrt_engine_status='Existing official NVIDIA stock export/build; hashes verified before freeze')
                      for key, value in parent['models'].items()}
    validate_plan(plan); same_cohort(plan)
    out.mkdir(parents=True)
    shutil.copytree(PARENT / 'manifests', out / 'manifests')
    (out / 'configs').mkdir()
    code = out / 'code'
    for sub in ('src', 'specs', 'chronotope/python', 'audit_deps'):
        source = PARENT / 'code' / sub
        if source.exists():
            shutil.copytree(source, code / sub, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(UPSTREAM, code / 'gui_agents', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('legacy_d1_w8.py', 'prepare_deadline_envelope.py'):
        shutil.copy2(PARENT / 'code' / name, code / name)
    for name in ('run_legacy_detector_full_agents3.py', 'audit_legacy_detector_full_agents3.py'):
        shutil.copy2(ROOT / 'studies' / name, code / name)
    native = read(PARENT / 'configs/legacy_cpp.json')
    native.update(prompt_version=4, max_tokens=64, binder='detector', at_commit_policy='late_bind',
                  base_url=str(ASSETS / 'engines'), memory='none')
    host = read(PARENT / 'configs/agents3_stripped.json')
    host['base_url'] = full['components']['planner']['base_url']
    write(out / 'configs/legacy_cpp_detector.json', native)
    write(out / 'configs/agents3_full.json', host)
    write(out / 'configs/full_agent_components.json', full)
    write(out / 'plan.json', plan)
    checks = validate_specs(out)
    write(out / 'preparation_report.json', {'state': 'passed', 'created_at': now(), 'spec_checks': checks,
        'qualification_trials': 0, 'model_inference_calls': 0, 'experimental_trials': 0,
        'same_ordered_cohort_except_arm_labels': True, 'matched_prompt_claim': False,
        'source_upstream_directory': str(UPSTREAM),
        'upstream_source_hashes': {str(p.relative_to(code / 'gui_agents')): digest(p) for p in (code / 'gui_agents').rglob('*') if p.is_file()}})
    write(out / 'status.json', {'state': 'prepared', 'completed_trials': 0, 'planned_trials': 4800,
        'qualification_trials': 0, 'pid': None, 'model': MODEL, 'automatic_resume': False})
    write(out.parent / 'preparation_status.json', {'state': 'prepared_pending_native_adapter_and_services',
        'updated_at': now(), 'automatic_launch': False, 'experimental_trials': 0,
        'spec_checks': checks, 'requested_native': plan['policy_difference'][ARMS[0]],
        'requested_host': plan['policy_difference'][ARMS[1]], 'grounder_enabled': True,
        'user_choice': 'Keep the grounder', 'model': MODEL, 'revision': REVISION,
        'previous_experiment': str(PARENT), 'previous_automatic_resume': False})
    print(json.dumps({'state': 'prepared', 'spec_checks': checks, 'out': str(out)}), flush=True)


def freeze(out, services):
    require_parent_stopped()
    assert not (out / 'execution_manifest.json').exists()
    root = out.parent; code = out / 'code'
    native = read(root / 'native_build_result.json')
    assert native['state'] == 'complete' and native['source_patches'] == []
    assert native['upstream_commit'] == 'bb291453bc10d9bd1142070c605e090d4e483ee2'
    assert native['model_inference_calls'] == native['qualification_trials'] == 0
    verify_hashes(native['source_hashes']); verify_hashes(native['binary_hashes'])
    verification = read(root / 'native_compact_verification.json')
    assert verification['state'] == 'passed'
    assert verification['model_inference_calls'] == verification['experimental_trials'] == 0
    verify_hashes(verification['source_file_sha256'])
    full_verification = read(root / 'full_agents3_cpu_verification.json')
    assert full_verification['state'] == 'passed'
    assert full_verification['model_inference_calls'] == full_verification['experimental_trials'] == 0
    verify_hashes(full_verification['source_file_sha256'])
    assert all(value is True for key, value in full_verification['policy_guarantees'].items()
               if key != 'optional_code_execution')
    assert full_verification['policy_guarantees']['optional_code_execution'] == 'disabled upstream default'
    grounder_checkpoint = read(root / 'grounder_checkpoint_verification.json')
    assert grounder_checkpoint['state'] == 'passed'
    assert (grounder_checkpoint['model'], grounder_checkpoint['revision']) == (GROUNDER, GROUNDER_REVISION)
    assert grounder_checkpoint['source_tensor_precision'] == 'FP32'
    assert grounder_checkpoint['configured_runtime_dtype'] == grounder_checkpoint['configured_kv_cache_dtype'] == 'BF16'
    assert grounder_checkpoint['model_inference_calls'] == grounder_checkpoint['experimental_trials'] == 0
    verify_hashes(grounder_checkpoint['file_hashes'])
    assets = read(ASSETS / 'engine_build_result.json')
    assert (assets['model'], assets['revision'], assets['state']) == (MODEL, REVISION, 'built')
    assert assets['source_patches'] == [] and assets['kv_cache']['dtype'] == 'FP16'
    verify_hashes(assets['hashes']); verify_hashes(assets['source_checkpoint_hashes'])
    source = root / 'build_source/chronotope'
    assert not (source / 'src/agents3_policy.h').exists(), 'Do not add Worker history to native legacy control'
    shutil.copytree(source / 'src', code / 'chronotope/src')
    shutil.copy2(source / 'CMakeLists.txt', code / 'chronotope/CMakeLists.txt')
    shutil.copy2(source / 'python/chronotope_core/remote.py', code / 'chronotope/python/chronotope_core/remote.py')
    (code / 'chronotope/build').mkdir(parents=True)
    for name in ('chronotope', 'libchronotope.so'):
        shutil.copy2(root / 'native_build' / name, code / 'chronotope/build' / name)
    for name in ('run_legacy_detector_full_agents3.py', 'audit_legacy_detector_full_agents3.py', 'agents3_full.py'):
        shutil.copy2(ROOT / 'studies' / name, code / name)
    for path, expected in full_verification['source_file_sha256'].items():
        path = Path(path)
        if path.is_relative_to(UPSTREAM):
            assert digest(code / 'gui_agents' / path.relative_to(UPSTREAM)) == expected
        elif path.name == 'agents3_full.py':
            assert digest(code / 'agents3_full.py') == expected
    preparation = read(out / 'preparation_report.json')
    verify_hashes(preparation['upstream_source_hashes'], code / 'gui_agents')
    prompt = read(root / 'inputs/legacy_compact_prompt.json')
    reference = read(root / 'inputs/native_compact_reference.json')
    from audit_legacy_detector_full_agents3 import prompt_fingerprint
    assert prompt_fingerprint(prompt) == prompt['prompt_sha256']
    assert prompt['schema'] == 'legacy-compact-prompt-1' and prompt['response_format'] == 'legacy_json'
    write(out / 'configs/legacy_compact_prompt.json', prompt)
    service_config = read(services)
    full = read(out / 'configs/full_agent_components.json'); validate_components(full)
    from audit_legacy_detector_full_agents3 import sha
    assert sha(json.dumps(full, sort_keys=True, separators=(',', ':')).encode()) == full_verification['component_config_sha256']
    assert service_config['state'] == 'ready'
    assert service_config['qualification_trials'] == service_config['inference_readiness_requests'] == 0
    for role, component in full['components'].items():
        observed = service_config['components'][role]
        assert observed['state'] == 'ready' and observed['inference_readiness_requests'] == 0
        assert (observed['model'], observed['revision'], observed['base_url']) == (component['model'], component['revision'], component['base_url'])
        command = observed['command']
        assert command[command.index('--served-model-name') + 1] == component['model']
        assert command[command.index('--dtype') + 1] == observed['dtype']
        assert command[command.index('--kv-cache-dtype') + 1] == observed['kv_cache_dtype']
        assert json.loads(command[command.index('--limit-mm-per-prompt') + 1])['image'] >= component['max_images_per_request']
        aliases = {'half': 'FP16', 'float16': 'FP16', 'fp16': 'FP16', 'bfloat16': 'BF16', 'bf16': 'BF16'}
        kv = observed.get('kv_cache', observed.get('effective_kv_cache_dtype'))
        assert aliases.get(str(kv).lower()) == component['kv_cache']
        engine_config = observed['observed_engine_configuration']
        assert 'enable_prefix_caching=False' in engine_config
        if role == 'grounder':
            assert aliases.get(observed['dtype'].lower()) == 'BF16'
            assert 'dtype=torch.bfloat16' in engine_config and 'quantization=None' in engine_config
            assert observed['checkpoint_path'] == grounder_checkpoint['checkpoint_path']
            assert observed['checkpoint_path'] in command
        else:
            assert aliases.get(observed['dtype'].lower()) == 'FP16'
            assert observed['quantization'] == 'modelopt_mixed'
            assert 'dtype=torch.float16' in engine_config and 'quantization=modelopt_mixed' in engine_config
            assert str(CHECKPOINT) in command
        assert observed.get('max_images_per_request', observed.get('max_images_per_prompt', 0)) >= component['max_images_per_request']
    for name in ('records', 'calls'):
        (out / name).mkdir(exist_ok=True)
    checks = validate_specs(out)
    shutil.copy2(ROOT / 'AGENTS.md', out / 'AGENTS.md')
    for name in ('native_build_result.json', 'native_compact_verification.json', 'full_agents3_cpu_verification.json', 'grounder_checkpoint_verification.json'):
        shutil.copy2(root / name, out / name)
    write(out / 'service_launch.json', service_config)
    artifacts = {str(p.relative_to(out)): digest(p) for sub in ('code', 'configs', 'manifests') for p in (out / sub).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    for name in ('plan.json', 'preparation_report.json', 'AGENTS.md', 'native_build_result.json', 'native_compact_verification.json', 'full_agents3_cpu_verification.json', 'grounder_checkpoint_verification.json', 'service_launch.json'):
        artifacts[name] = digest(out / name)
    execution = {'created_at': now(), 'model': MODEL, 'revision': REVISION,
        'precision': PRECISION, 'runtime_precision': RUNTIME_PRECISION, 'kv_cache': assets['kv_cache'],
        'arms': ARMS, 'planned_trials': 4800, 'qualification_trials': 0, 'spec_checks': checks,
        'checkpoint_path': str(CHECKPOINT), 'checkpoint_hashes': assets['source_checkpoint_hashes'],
        'engine_hashes': assets['hashes'], 'hashes': artifacts,
        'plugin': {'path': str(PLUGIN), 'sha256': digest(PLUGIN)},
        'model_service': full['components']['planner']['base_url'], 'agent_s3_components': full,
        'service_provenance': service_config, 'native_prompt_contract': prompt,
        'full_agent_source_verification': full_verification,
        'grounder_checkpoint': grounder_checkpoint,
        'native_prompt_reference': reference, 'native_build': native,
        'upstream_commit': native['upstream_commit'], 'source_patches': [],
        'arm_labels': {ARMS[0]: 'Legacy C++ + device detector (compact JSON; TensorRT-Edge-LLM)', ARMS[1]: 'Agent S3 (upstream GUI policy; separate grounder)'},
        'interpretation': [
            'Whole-system comparison: same planner checkpoint, deliberately different policies and grounding.',
            'Legacy uses a compact JSON answer and device detector/NCC fallback; Agent S3 retains original Worker and OSWorldACI.',
            'Agent S3 keeps upstream text/screenshot history, reflection and retries. Optional code-agent state is explicit in component config.',
            'Do not attribute all latency or success differences solely to orchestration overhead.',
            'No qualification trials or competence gates; all completed scored cases retained.',
            'Native remains legacy, with stock-runtime CPU image staging; no full GPU-history claim.',
        ]}
    write(out / 'execution_manifest.json', execution)
    write(out / 'status.json', {'state': 'frozen_ready', 'completed_trials': 0, 'planned_trials': 4800,
        'qualification_trials': 0, 'pid': None, 'model': MODEL, 'automatic_resume': False})
    write(root / 'preparation_status.json', {'state': 'frozen_ready', 'updated_at': now(),
        'automatic_launch': False, 'experimental_trials': 0, 'spec_checks': checks,
        'grounder_enabled': True, 'user_choice': 'Keep the grounder',
        'execution_manifest': str(out / 'execution_manifest.json'), 'model': MODEL, 'revision': REVISION})
    print(json.dumps({'state': 'frozen_ready', 'spec_checks': checks, 'out': str(out)}), flush=True)


def run(out, resume=False):
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
            'planned_trials': 4800, 'pid': os.getpid() if state == 'running' else None, 'current_job': current,
            'qualification_trials': 0, 'model': MODEL, 'error': error, 'automatic_resume': False})
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    publish('running')
    try:
        for current in plan['jobs']:
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
            print(f'{len(records)}/4800 {ident} success={outcome.Y}', flush=True); publish('running')
        current = None; publish('complete')
    except BaseException as error:
        publish('stopped' if isinstance(error, KeyboardInterrupt) else 'failed', f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare', 'validate', 'freeze', 'run'): group.add_argument('--' + action, action='store_true')
    parser.add_argument('--components', type=Path)
    parser.add_argument('--services', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.out.resolve()
    if args.resume and not args.run: parser.error('--resume requires --run')
    if args.prepare:
        if not args.components: parser.error('--prepare requires --components')
        prepare(out, args.components)
    elif args.validate: print(json.dumps({'spec_checks': validate_specs(out), 'model_inference_calls': 0}))
    elif args.freeze:
        if not args.services: parser.error('--freeze requires --services')
        freeze(out, args.services)
    else: run(out, args.resume)
