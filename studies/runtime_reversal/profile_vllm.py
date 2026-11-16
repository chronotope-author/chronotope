"""Restore a scored Notification request and profile it on an isolated vLLM service.

Preparation is CPU-only. Capture sends one diagnostic model request, never starts
an application episode, never scores competence, and never changes evaluation
records. The caller must first pause/finish the evaluation and launch the exact
service command in launch_plan.json under Nsight Systems.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
import base64
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'runs/legacy_d1_gpu.lock'
MODELS = {
    'qwen08': ROOT / 'runs/agents3_policy_qwen08_20261028/experiment',
    'radixark27b': ROOT / 'runs/agents3_policy_radixark27b_20261028/experiment',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(model, out, index=7, port=8014, delay_iterations=6, max_iterations=32):
    assert port in (8014, 8015), 'Use a dedicated diagnostic port, never an evaluation service'
    assert 0 <= index < 100
    assert delay_iterations >= 0 and max_iterations >= 0
    assert not out.exists(), 'Refuse to overwrite prepared diagnostic inputs'
    experiment = MODELS[model]
    execution = read(experiment / 'execution_manifest.json')
    ident = f'agents3_stripped_w16000000000_{index:03d}_live'
    record_path = experiment / 'records' / (ident + '.json')
    record = read(record_path)
    assert record['job']['task'] == 'notification'
    assert record['job']['phase'] == 'evaluation' and record['job']['mode'] == 'live'
    assert record['job']['family'] == 'closing_window'
    call_path = experiment / 'calls' / (ident + '.jsonl')
    calls = [json.loads(line) for line in call_path.read_text().splitlines()]
    call = calls[0]
    assert call['role'] == 'planner' and call['model'] == execution['model']
    assert call['http_status'] == 200
    assert call['policy_contract_sha256'] == execution['policy_sha256']
    body = copy.deepcopy(call['request'])
    trace_path = experiment / record['trace']
    assert sha(trace_path.read_bytes()) == record['trace_sha256']
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    decision = next(e for e in events if e['record'] == 'decision_invocation')
    observation = next(e for e in events if e['record'] == 'observation'
                       and e['observation_id'] == decision['observation_id'])
    payload_path = experiment / 'payloads' / (observation['payload_hash'] + '.bin')
    payload = payload_path.read_bytes()
    assert sha(payload) == observation['payload_hash']
    # Use the exact frozen encoder that produced the original HTTP request.
    encoder_path = experiment / 'code/src/chronotope/png.py'
    spec = importlib.util.spec_from_file_location('_frozen_profile_png', encoder_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    cfg = read(experiment / 'configs/agents3_stripped.json')
    assert cfg['image_scale'] == 2 and len(payload) == 320 * 240 * 3
    png = encoder.encode_png(payload, 320, 240, cfg['image_scale'])
    url = 'data:image;base64,' + base64.b64encode(png).decode()
    restored = 0
    for message in body['messages']:
        for part in message['content']:
            if part['type'] == 'image_url':
                assert part['image_url']['sha256'] == sha(url.encode())
                assert part['image_url']['encoded_bytes'] == len(url)
                part['image_url'] = {'url': url}
                restored += 1
    assert restored == 1
    assert sha(json.dumps(body).encode()) == call['request_sha256'], 'Restored HTTP request differs'
    assert body['model'] == execution['model']
    assert body['max_tokens'] == 4096 and body['temperature'] == 0
    assert body['repetition_penalty'] == 1.05
    paired = read(experiment / 'records' / (ident.replace('agents3_stripped', 'legacy_cpp') + '.json'))
    assert record['request_audit'][0]['input_token_ids_sha256'] == paired['request_audit'][0]['input_token_ids_sha256']
    original_launch = read(experiment / 'service_launch.json')
    command = list(original_launch['command'])
    assert '--profiler-config' not in command
    command[command.index('--port') + 1] = str(port)
    profile_config = {'profiler': 'cuda', 'delay_iterations': delay_iterations,
                      'max_iterations': max_iterations}
    command += ['--profiler-config', json.dumps(profile_config, separators=(',', ':'))]
    out.mkdir(parents=True)
    write(out / 'request.json', body)
    (out / 'frame.png').write_bytes(png)
    (out / 'frame.rgb').write_bytes(payload)
    nsys_command = ['/usr/local/cuda/bin/nsys', 'profile', '--trace=cuda,nvtx,osrt',
                    '--sample=none', '--cpuctxsw=none', '--cuda-graph-trace=node',
                    '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                    '--kill=none', '--export=sqlite',
                    '--force-overwrite=false', '--output', str(out / 'vllm_timeline'), *command]
    manifest = {
        'state': 'prepared', 'created_at': now(), 'diagnostic': 'model_request_profile',
        'task': 'Disappearing Notification', 'task_family': 'notification',
        'application_episodes': 0, 'qualification_trials': 0, 'model_requests_executed': 0,
        'source_experiment': str(experiment), 'source_job': record['job'],
        'model': execution['model'], 'revision': execution['revision'],
        'precision': execution['precision'], 'kv_cache': original_launch['kv_cache_dtype'],
        'original_request_sha256': call['request_sha256'],
        'request_file_sha256': sha((out / 'request.json').read_bytes()),
        'initial_input_token_ids_sha256': record['request_audit'][0]['input_token_ids_sha256'],
        'expected_input_tokens': record['request_audit'][0]['input_tokens'],
        'policy_sha256': execution['policy_sha256'],
        'source_hashes': {str(p): sha(p.read_bytes()) for p in
                          [Path(__file__).resolve(), record_path, call_path, trace_path, payload_path, encoder_path,
                           experiment / 'execution_manifest.json', experiment / 'service_launch.json']},
        'base_url': f'http://127.0.0.1:{port}/v1', 'profiler_config': profile_config,
        'capture_scope': {
            'first_generated_token_1based': delay_iterations,
            'last_generated_token_1based': delay_iterations + max_iterations - 1,
            'skipped_generated_tokens': delay_iterations - 1,
            'model_iterations': max_iterations,
            'assumptions': 'One prefill iteration generates token1; one token per subsequent decode iteration; verify NVTX ranges in trace.',
            'semantics': 'vLLM profiler.step() runs before execute_model; delay_iterations=6 starts before iteration6 after tokens1-5; max_iterations=32 stops before iteration38.',
            'source_references': [
                (_CHRONOTOPE_HOME + '/qwen-serve/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py:906'),
                (_CHRONOTOPE_HOME + '/qwen-serve/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py:1078'),
                (_CHRONOTOPE_HOME + '/qwen-serve/.venv/lib/python3.12/site-packages/vllm/profiler/wrapper.py:83')],
        },
        'service_command': command, 'nsys_command': nsys_command,
        'limits': ['Separate diagnostic; never add these requests to evaluation denominators.',
                   'GPU node tracing perturbs runtime; use ordinary evaluation for wall-clock performance.',
                   'Full request capture includes first-request compilation if it occurs; distinguish it in analysis.',
                   'No warmup request, changed output budget, forced output tokens, or competence gate is inserted.',
                   'Launch and stop service only after evaluation is paused/complete and GPU access is exclusive.'],
    }
    write(out / 'launch_plan.json', manifest)
    return manifest


def request(url, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return response.status, json.loads(raw) if raw else None


@contextmanager
def gpu_lock(inherited_fd=None):
    if inherited_fd is None:
        with LOCK.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        return
    expected = LOCK.stat()
    inherited = os.fstat(inherited_fd)
    assert (inherited.st_dev, inherited.st_ino) == (expected.st_dev, expected.st_ino), 'Inherited fd is not the experiment GPU lock'
    # The parent retains this same open file description throughout service
    # switching and restoration. Do not unlock or close its inherited fd.
    # A separate open description must fail to acquire: the lock must already
    # be held, and the inherited description must be the one that owns it.
    with LOCK.open('a+') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise RuntimeError('Parent did not already hold the inherited GPU lock')
    fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    yield


def capture(out, inherited_lock_fd=None):
    plan = read(out / 'launch_plan.json')
    parsed = urllib.parse.urlparse(plan['base_url'])
    assert parsed.hostname == '127.0.0.1' and parsed.port in (8014, 8015)
    assert plan['task_family'] == 'notification' and plan['application_episodes'] == 0
    for path, expected in plan['source_hashes'].items():
        assert sha(Path(path).read_bytes()) == expected, path
    assert sha((out / 'request.json').read_bytes()) == plan['request_file_sha256']
    assert not (out / 'capture_result.json').exists(), 'Never silently repeat a diagnostic request'
    body = read(out / 'request.json')
    # The evaluation runner holds this lock for its entire run. This fails fast
    # unless it has released the GPU; it does not pause or resume anything.
    with gpu_lock(inherited_lock_fd):
        _, models = request(plan['base_url'] + '/models')
        assert [m['id'] for m in models['data']] == [plan['model']]
        origin = plan['base_url'].removesuffix('/v1')
        started = False
        result = {'state': 'capturing', 'started_at': now(), 'diagnostic': 'model_request_profile',
                  'model': plan['model'], 'revision': plan['revision'],
                  'qualification_trials': 0, 'application_episodes': 0, 'model_requests_executed': 0}
        write(out / 'capture_result.json', result)
        try:
            request(origin + '/start_profile', {})
            started = True
            begin = time.monotonic_ns()
            result['model_requests_executed'] = 1
            status, response = request(plan['base_url'] + '/chat/completions', body)
            end = time.monotonic_ns()
            assert status == 200 and response['model'] == plan['model']
            assert response['usage']['prompt_tokens'] == plan['expected_input_tokens']
            write(out / 'response.json', response)
            result.update(state='captured_request', elapsed_ns=end-begin, usage=response['usage'],
                          response_sha256=sha((out / 'response.json').read_bytes()),
                          finish_reason=response['choices'][0]['finish_reason'], completed_at=now())
        except BaseException as error:
            result.update(state='failed', error=f'{type(error).__name__}: {error}', completed_at=now())
            raise
        finally:
            if started:
                try:
                    request(origin + '/stop_profile', {})
                except Exception as error:
                    result['stop_profile_error'] = f'{type(error).__name__}: {error}'
            write(out / 'capture_result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--capture', action='store_true')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int, default=7)
    parser.add_argument('--port', type=int, default=8014)
    parser.add_argument('--delay-iterations', type=int, default=6)
    parser.add_argument('--max-iterations', type=int, default=32)
    parser.add_argument('--lock-fd', type=int, help='GPU flock fd inherited with subprocess pass_fds; parent retains ownership')
    args = parser.parse_args()
    if args.prepare:
        if args.model is None:
            parser.error('--prepare requires --model')
        result = prepare(args.model, args.out.resolve(), args.index, args.port,
                         args.delay_iterations, args.max_iterations)
    else:
        result = capture(args.out.resolve(), args.lock_fd)
    print(json.dumps({k: result[k] for k in ['state', 'model', 'model_requests_executed']}))


if __name__ == '__main__':
    main()
