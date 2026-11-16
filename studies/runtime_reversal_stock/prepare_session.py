"""CPU-only preparation for the pristine Edge-LLM backend diagnostic.

Restore the already scored Notification request, then make the explicitly
authorized repetition-penalty change to match pristine Edge-LLM's behavior.
No service calls, model requests, exports, builds, or application episodes run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/runtime_reversal_stock_20261029'
EXPERIMENT = ROOT / 'runs/agents3_policy_radixark27b_20261028/experiment'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL = load('_stock_batch_control', ROOT / 'studies/runtime_reversal/batch_control.py')
VLLM = load('_stock_vllm_profile', ROOT / 'studies/runtime_reversal/profile_vllm.py')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'session_preparation.json').exists(), 'Preparation already recorded'
    active_launch = CONTROL.read(EXPERIMENT.parent / 'services/launch.json')
    current = CONTROL.checked_identity(active_launch)
    command = CONTROL.read(EXPERIMENT / 'service_launch.json')['command']
    assert CONTROL.matches(current, command)
    keys = set(CONTROL.PERFORMANCE_ENV) | {
        'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'EDGELLM_PLUGIN_PATH',
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
    }
    raw = Path(f'/proc/{current["pid"]}/environ').read_bytes().split(b'\0')
    env = {}
    for pair in raw:
        if b'=' in pair:
            key, value = pair.split(b'=', 1)
            if key.decode() in keys:
                env[key.decode()] = value.decode()
    restore = {
        'state': 'cpu_only_restore_plan', 'prepared_at': CONTROL.now(),
        'original_identity': current, 'original_pid': current['pid'],
        'original_starttime_ticks': current['starttime_ticks'],
        'restore_command': command, 'cwd': str(ROOT),
        'source_launch': str(EXPERIMENT / 'service_launch.json'),
        'source_launch_sha256': CONTROL.digest(EXPERIMENT / 'service_launch.json'),
        'environment_overrides': env, 'base_url': 'http://127.0.0.1:8012/v1',
        'model_requests_executed': 0, 'application_episodes': 0,
        'instructions': [
            'Restore exact original command on port8012 without a profiler config.',
            'Restore captured performance environment before resuming evaluation.',
            'Only readiness GETs are permitted; no extra warmup or qualification request.',
            'Keep exclusive GPU flock until diagnostics exit and original service is ready.',
        ],
    }
    CONTROL.write(OUT / 'original_vllm_restore.json', restore)
    plans = []
    for model in ['qwen08', 'radixark27b']:
        folder = OUT / f'vllm_{model}_case7'
        plan = VLLM.prepare(model, folder, index=7, port=8014,
                            delay_iterations=6, max_iterations=32)
        body = VLLM.read(folder / 'request.json')
        assert body['repetition_penalty'] == 1.05
        # Only this field changes; native upstream has no penalty API and uses1.
        before = json.dumps(body).encode()
        assert VLLM.sha(before) == plan['original_request_sha256']
        body['repetition_penalty'] = 1.0
        VLLM.write(folder / 'request.json', body)
        plan.update(
            diagnostic='stock_backend_model_request_profile',
            evaluation_policy_equivalent=False,
            comparison_label='Pristine NVIDIA Edge-LLM vs vLLM; repetition penalty1.0',
            original_evaluation_request_sha256=plan['original_request_sha256'],
            diagnostic_request_sha256=VLLM.sha(json.dumps(body).encode()),
            request_file_sha256=VLLM.sha((folder / 'request.json').read_bytes()),
            request_changes=[{'path': 'repetition_penalty', 'old': 1.05, 'new': 1.0,
                'reason': 'User requested removal of local backend code; pristine Edge-LLM has no repetition-penalty request field.'}],
        )
        plan['source_hashes'][str(Path(__file__).resolve())] = CONTROL.digest(Path(__file__))
        plan['limits'] += [
            'This is a new stock-backend diagnostic policy, not the active1.05 evaluation policy.',
            'Old request bytes are preserved by original_evaluation_request_sha256; only penalty differs.',
        ]
        CONTROL.write(folder / 'launch_plan.json', plan)
        plans.append(str(folder / 'launch_plan.json'))
    CONTROL.write(OUT / 'session_preparation.json', {
        'state': 'prepared_cpu_only', 'prepared_at': CONTROL.now(),
        'task_family': 'notification', 'case_index': 7,
        'planned_model_requests': 4, 'application_episodes': 0,
        'qualification_trials': 0, 'model_requests_executed': 0,
        'repetition_penalty': 1.0, 'vllm_plans': plans,
        'upstream_source': str(OUT / 'source/TensorRT-Edge-LLM'),
        'helpers': {str(p): CONTROL.digest(p) for p in [
            Path(__file__).resolve(), ROOT / 'studies/runtime_reversal/profile_vllm.py',
            ROOT / 'studies/runtime_reversal/batch_control.py',
            ROOT / 'studies/runtime_reversal/profile_session.py']},
    })
    print(json.dumps({'state': 'prepared_cpu_only', 'vllm_plans': plans,
                      'model_requests_executed': 0, 'repetition_penalty': 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
