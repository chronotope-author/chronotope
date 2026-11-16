"""Prepare/build/profile one frozen legacy C++ Notification model request.

Preparation is CPU-only. Build and capture require exclusive experiment GPU
access and must be invoked only after the evaluation is paused or complete.
The diagnostic never applies an action or contributes an evaluation denominator.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

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


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, f'Frozen source anchor not unique: {old[:80]}'
    return text.replace(old, new, 1)


def instrument(source, delay, steps):
    """Only mark a middle decode slice; preserve policy, stopping and outputs."""
    source = replace_once(source, '#include <cuda_runtime.h>',
                          '#include <cuda_runtime.h>\n#include <cuda_profiler_api.h>')
    source = replace_once(source, '            auto respond = [&]',
                          '            size_t profile_request_index = 0;\n            auto respond = [&]')
    old = '''                    req.onTokenGenerated = [&](rt::TokenCallbackInfo const&) { if (!first) first = now_ns(); ++impl_->steps; };
                    rt::LLMGenerationResponse resp;
                    bool ok = impl_->runtime->handleRequest(req, resp, impl_->stream);'''
    new = '''                    // Diagnostic-only copy: trace a fixed middle decode slice of
                    // the first model request, then finish the unchanged Worker turn.
                    const bool profile_this_request = profile_request_index++ == 0;
                    size_t profile_tokens = 0;
                    bool profile_started = false, profile_stopped = false;
                    int profile_start_status = -1, profile_stop_status = -1;
                    req.onTokenGenerated = [&](rt::TokenCallbackInfo const&) {
                        if (!first) first = now_ns();
                        ++impl_->steps;
                        ++profile_tokens;
                        if (profile_this_request && profile_tokens == PROFILE_DELAY) {
                            profile_start_status = static_cast<int>(cudaProfilerStart());
                            profile_started = profile_start_status == static_cast<int>(cudaSuccess);
                        }
                        if (profile_this_request && profile_started && profile_tokens == PROFILE_END) {
                            profile_stop_status = static_cast<int>(cudaProfilerStop());
                            profile_stopped = true;
                        }
                    };
                    rt::LLMGenerationResponse resp;
                    bool ok = impl_->runtime->handleRequest(req, resp, impl_->stream);
                    if (profile_started && !profile_stopped) {
                        profile_stop_status = static_cast<int>(cudaProfilerStop());
                        profile_stopped = true;
                    }
                    audit["diagnostic_profile"] = {{"first_request", profile_this_request},
                        {"delay_tokens", PROFILE_DELAY}, {"requested_decode_steps", PROFILE_STEPS},
                        {"observed_output_callbacks", profile_tokens}, {"started", profile_started},
                        {"stopped", profile_stopped}, {"cuda_start_status", profile_start_status},
                        {"cuda_stop_status", profile_stop_status}};'''
    new = new.replace('PROFILE_DELAY', str(delay)).replace('PROFILE_END', str(delay + steps))
    new = new.replace('PROFILE_STEPS', str(steps))
    return replace_once(source, old, new)


def prepare(model, out, index=7, delay=5, steps=32):
    assert 0 <= index < 100 and delay > 0 and steps > 0
    assert not out.exists(), 'Refuse to overwrite diagnostic provenance'
    experiment = MODELS[model]
    execution = read(experiment / 'execution_manifest.json')
    name = f'legacy_cpp_w16000000000_{index:03d}_live'
    record_path = experiment / 'records' / (name + '.json')
    call_path = experiment / 'calls' / (name + '.jsonl')
    record = read(record_path)
    assert record['job']['task'] == 'notification' and record['job']['phase'] == 'evaluation'
    assert record['job']['family'] == 'closing_window' and record['job']['mode'] == 'live'
    calls = [json.loads(line) for line in call_path.read_text().splitlines()]
    first = calls[0]
    assert first['turn'] == 0 and first['request_kind'] == 'planner' and first['ok']
    trace_path = experiment / record['trace']
    assert digest(trace_path) == record['trace_sha256']
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    decision = next(e for e in events if e['record'] == 'decision_invocation')
    observation = next(e for e in events if e['record'] == 'observation'
                       and e['observation_id'] == decision['observation_id'])
    payload = experiment / 'payloads' / (observation['payload_hash'] + '.bin')
    assert digest(payload) == record['request_audit'][0]['source_payload_hash']
    assert payload.stat().st_size == 320 * 240 * 3
    cfg = read(experiment / 'configs/legacy_cpp.json')
    assert cfg['planner'] == 'trt' and cfg['image_scale'] == 2 and cfg['max_tokens'] == 4096
    assert cfg['at_commit_policy'] == 'frozen' and cfg['temperature'] == 0
    assert cfg['pending_radius_px'] == cfg['pending_timeout_ns'] == cfg['receipt_timeout_ns'] == 0
    assert cfg['max_evidence_age_ns'] == 0 and not cfg['ack_aware_pending'] and not cfg['validate_kind']
    frozen = experiment / 'code/chronotope'
    source = out / 'source/chronotope'
    source.mkdir(parents=True)
    shutil.copytree(frozen / 'src', source / 'src')
    # The frozen evaluation snapshot omits tests/test_main.cpp. Remove only the
    # unused test target so CMake can configure the unchanged application target.
    cmake_original = (frozen / 'CMakeLists.txt').read_text()
    test_start = cmake_original.index('if(CHRONOTOPE_EDGELLM_BUILD)\n  add_executable(chronotope_tests')
    test_end = cmake_original.index('\n', cmake_original.index('target_link_libraries(chronotope_tests', test_start))
    cmake_profile = cmake_original[:test_start] + '# Frozen tests were not archived; diagnostic builds only chronotope_main.\n' + cmake_original[test_end:]
    (source / 'CMakeLists.txt').write_text(cmake_profile)
    original = (source / 'src/planner_trt.cpp').read_text()
    instrumented = instrument(original, delay, steps)
    (source / 'src/planner_trt.cpp').write_text(instrumented)
    (out / 'instrumentation.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), instrumented.splitlines(keepends=True),
        fromfile=str(frozen / 'src/planner_trt.cpp'), tofile=str(source / 'src/planner_trt.cpp'))) + ''.join(difflib.unified_diff(
        cmake_original.splitlines(keepends=True), cmake_profile.splitlines(keepends=True),
        fromfile=str(frozen / 'CMakeLists.txt'), tofile=str(source / 'CMakeLists.txt'))))
    shutil.copy2(payload, out / 'frame.rgb')
    write(out / 'expected_first_request.json', first)
    runtime_build = experiment.parent / 'runtime/build'
    build = out / 'build'
    policy = experiment / 'configs/agents3_policy.json'
    native_command = [str(build / 'chronotope'), '--serve', '@SOCKET@', '--width', '320', '--height', '240',
        '--pixfmt', 'RGB24', '--planner', 'trt', '--policy', cfg['at_commit_policy'],
        '--radius', str(cfg['template_radius_px']), '--min-score', str(cfg['min_match_score']),
        '--restart-threshold', str(cfg['restart_threshold_px'] if cfg['restart_threshold_px'] is not None else 6),
        '--image-scale', str(cfg['image_scale']), '--prompt-version', str(cfg['prompt_version']),
        '--max-tokens', str(cfg['max_tokens']), '--temperature', str(cfg['temperature']),
        '--detector-at-d', '0', '--max-evidence-age-ms', '0', '--pending-radius', '0',
        '--pending-timeout-ms', '0', '--receipt-timeout-ms', '0',
        '--engine-dir', str(Path(cfg['base_url']) / 'llm'), '--multimodal-engine-dir', cfg['base_url'],
        '--agents3-policy', str(policy), '--agents3-policy-audit', str(out / 'calls.jsonl')]
    build_commands = [['cmake', '-S', str(source), '-B', str(build),
        '-DCHRONOTOPE_EDGELLM_BUILD=' + str(runtime_build),
        ('-DCHRONOTOPE_TRT_PREFIX=' + _CHRONOTOPE_HOME + '/trt-prefix'), '-DCMAKE_BUILD_TYPE=Release'],
        ['cmake', '--build', str(build), '--target', 'chronotope_main', '-j6']]
    nsys_command = ['/usr/local/cuda/bin/nsys', 'profile', '--trace=cuda,nvtx,osrt',
        '--sample=none', '--cpuctxsw=none', '--cuda-graph-trace=node',
        '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--kill=none',
        '--force-overwrite=false', '--export=sqlite', '--output', str(out / 'native_timeline'), *native_command]
    files = [record_path, call_path, trace_path, payload, policy, frozen / 'CMakeLists.txt',
             experiment / 'configs/legacy_cpp.json', experiment / 'execution_manifest.json',
             Path(__file__).resolve(), *sorted((frozen / 'src').glob('*'))]
    files = [p for p in files if p.is_file()]
    manifest = {
        'state': 'prepared', 'created_at': now(), 'diagnostic': 'model_request_profile',
        'task': 'Disappearing Notification', 'task_family': 'notification',
        'application_episodes': 0, 'qualification_trials': 0, 'model_requests_executed': 0,
        'source_experiment': str(experiment), 'source_job': record['job'], 'source_observation': observation,
        'model': execution['model'], 'revision': execution['revision'], 'precision': execution['precision'],
        'kv_cache': execution.get('kv_cache', {'dtype': read(Path(cfg['base_url']) / 'llm/config.json')['kv_cache_dtype']}),
        'runtime': 'legacy C++/TensorRT-Edge-LLM; CPU text policy/history',
        'runtime_build': str(runtime_build), 'frozen_base_binary': str(frozen / 'build/chronotope'),
        'frozen_base_binary_expected_sha256': execution['hashes']['code/chronotope/build/chronotope'],
        'engine_hashes_expected': execution['engine_hashes'],
        'initial_input_token_ids_sha256': record['request_audit'][0]['input_token_ids_sha256'],
        'expected_input_tokens': record['request_audit'][0]['input_tokens'],
        'capture_scope': {'first_model_request_only': True, 'after_output_token': delay,
                          'decode_steps': steps, 'through_output_token': delay + steps,
                          'full_generation_preserved': True},
        'source_hashes': {str(p): digest(p) for p in files},
        'prepared_hashes': {str(p): digest(p) for p in [out / 'frame.rgb', out / 'expected_first_request.json',
            out / 'instrumentation.patch', source / 'CMakeLists.txt', *sorted((source / 'src').glob('*'))] if p.is_file()},
        'build_commands': build_commands, 'native_command': native_command, 'nsys_command': nsys_command,
        'environment': {'EDGELLM_PLUGIN_PATH': (_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so'),
                        'LD_LIBRARY_PATH_prefix': (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib')},
        'limits': ['Diagnostic request replay, zero application episodes; never add to evaluation denominators.',
                   'No competence gate, warmup request, changed generation budget, or forced output tokens.',
                   'CUDA node tracing perturbs runtime. Use unprofiled evaluation for wall-clock comparisons.',
                   'Only frozen-source copies are instrumented; runtime archive, engines and policy stay unchanged.',
                   'GPU/model service changes and builds require the running evaluation to be paused or complete.'],
    }
    write(out / 'launch_plan.json', manifest)
    return manifest


@contextmanager
def exclusive(lock_fd=None):
    if lock_fd is not None:
        assert Path(f'/proc/self/fd/{lock_fd}').resolve() == LOCK.resolve(), 'Inherited descriptor is not the experiment GPU lock'
        with LOCK.open('a+') as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(probe, fcntl.LOCK_UN)
                raise RuntimeError('The inherited GPU descriptor is not already locked')
        # A descriptor inherited with pass_fds shares the parent's open-file
        # description. Do not close or unlock it: the coordinator retains it
        # through all profiles and restoration of the evaluation service.
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
        return
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_prepared(plan):
    for section in ['source_hashes', 'prepared_hashes']:
        for path, expected in plan[section].items():
            assert digest(path) == expected, f'Changed diagnostic input: {path}'
    assert plan['task_family'] == 'notification' and plan['application_episodes'] == 0


def build(out, lock_fd=None):
    plan = read(out / 'launch_plan.json')
    verify_prepared(plan)
    assert not (out / 'build_result.json').exists(), 'Refuse an implicit repeat build'
    with exclusive(lock_fd):
        assert digest(plan['frozen_base_binary']) == plan['frozen_base_binary_expected_sha256']
        runtime = Path(plan['runtime_build']) / 'cpp/libedgellmCore.a'
        runtime_hash = digest(runtime)
        with (out / 'build.log').open('w') as log:
            for command in plan['build_commands']:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = {'state': 'built', 'model': plan['model'], 'model_requests_executed': 0,
                  'completed_at': now(), 'instrumented_binary_sha256': digest(out / 'build/chronotope'),
                  'runtime_archive_sha256': runtime_hash, 'runtime_archive': str(runtime)}
        assert digest(runtime) == runtime_hash, 'Runtime archive changed during isolated build'
        write(out / 'build_result.json', result)
    return result


def capture(out, lock_fd=None):
    plan = read(out / 'launch_plan.json')
    built = read(out / 'build_result.json')
    verify_prepared(plan)
    assert not (out / 'capture_result.json').exists(), 'Never silently repeat a diagnostic request'
    assert digest(out / 'build/chronotope') == built['instrumented_binary_sha256']
    with exclusive(lock_fd), tempfile.TemporaryDirectory(prefix='native_profile_', dir='/tmp') as temporary:
        for path, expected in plan['engine_hashes_expected'].items():
            assert digest(path) == expected, f'Engine provenance changed: {path}'
        assert digest(built['runtime_archive']) == built['runtime_archive_sha256']
        sock_path = str(Path(temporary) / 'agent.sock')
        command = [sock_path if s == '@SOCKET@' else s for s in plan['nsys_command']]
        env = os.environ.copy()
        env['EDGELLM_PLUGIN_PATH'] = plan['environment']['EDGELLM_PLUGIN_PATH']
        env['LD_LIBRARY_PATH'] = plan['environment']['LD_LIBRARY_PATH_prefix'] + ':' + env.get('LD_LIBRARY_PATH', '')
        result = {'state': 'capturing', 'started_at': now(), 'model': plan['model'],
                  'qualification_trials': 0, 'application_episodes': 0, 'model_requests_executed': 0,
                  'actual_command': command}
        write(out / 'capture_result.json', result)
        process = None
        connection = None
        reader = None
        try:
            with (out / 'capture.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                deadline = time.monotonic() + 600
                while not Path(sock_path).exists():
                    if process.poll() is not None:
                        raise RuntimeError(f'Native profiler exited {process.returncode}; see capture.log')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Native profiler did not create its socket')
                    time.sleep(0.1)
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.settimeout(900)
                connection.connect(sock_path)
                reader = connection.makefile('rb')

                def call(body, payload=b''):
                    connection.sendall((json.dumps(body) + '\n').encode() + payload)
                    response = json.loads(reader.readline())
                    if not response.get('ok'):
                        raise RuntimeError(f'Native {body["op"]} failed: {response}')
                    return response

                result['hello'] = call({'op': 'hello'})
                observation = plan['source_observation']
                payload = (out / 'frame.rgb').read_bytes()
                result['admission'] = call({'op': 'admit', 'len': len(payload), **{
                    key: observation[key] for key in ['observation_id', 'source_event_id',
                                                       'source_time_ns', 'publication_time_ns']}}, payload)
                result['begin_decision'] = call({'op': 'begin_decision', 'observation_id': observation['observation_id']})
                start = time.monotonic_ns()
                result['model_requests_executed'] = 1
                result['answer'] = call({'op': 'decide_trt'})
                result['instrumented_turn_elapsed_ns'] = time.monotonic_ns() - start
                # Never resolve/commit/apply the proposal to an application.
                call({'op': 'abort_decision'})
                # The server acknowledges close before exiting. Read that reply
                # before disconnecting so its write cannot terminate via SIGPIPE.
                call({'op': 'close'})
                reader.close(); reader = None
                connection.close(); connection = None
                process.wait(timeout=180)
                assert process.returncode == 0, f'Profiler exited {process.returncode}'
            calls = [json.loads(line) for line in (out / 'calls.jsonl').read_text().splitlines()]
            result['model_requests_executed'] = len(calls)
            expected = read(out / 'expected_first_request.json')
            first = calls[0]
            checked = ['policy_sha256', 'turn', 'request_kind', 'format_attempt', 'transport_attempt',
                       'messages', 'transport_messages', 'generation_settings', 'screenshot_width',
                       'screenshot_height', 'input_token_ids', 'input_tokens']
            for key in checked:
                assert first[key] == expected[key], f'Model request differs: {key}'
            assert first['ok'], first
            captured = first['diagnostic_profile']
            captured['captured_decode_steps'] = max(0, min(captured['requested_decode_steps'],
                captured['observed_output_callbacks'] - captured['delay_tokens']))
            assert captured['started'] and captured['stopped'], captured
            assert captured['cuda_start_status'] == captured['cuda_stop_status'] == 0, captured
            result.update(state='captured_request', completed_at=now(), capture_scope=captured,
                          verified_model_input_fields=checked, first_output_tokens=first['output_tokens'],
                          original_first_output_tokens=expected['output_tokens'],
                          first_response_matches_original=first['response'] == expected['response'])
        except BaseException as error:
            result.update(state='failed', completed_at=now(), error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if connection is not None:
                try:
                    connection.sendall(b'{"op":"close"}\n')
                except OSError:
                    pass
                if reader is not None:
                    reader.close()
                connection.close()
            if process is not None and process.poll() is None:
                # Only the dedicated diagnostic process group is stopped on failure.
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            write(out / 'capture_result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--build', action='store_true')
    action.add_argument('--capture', action='store_true')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int, default=7)
    parser.add_argument('--delay-tokens', type=int, default=5)
    parser.add_argument('--decode-steps', type=int, default=32)
    parser.add_argument('--lock-fd', type=int, help='Inherited GPU flock descriptor owned by the coordinator')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        if args.model is None:
            parser.error('--prepare requires --model')
        result = prepare(args.model, out, args.index, args.delay_tokens, args.decode_steps)
    elif args.build:
        result = build(out, args.lock_fd)
    else:
        result = capture(out, args.lock_fd)
    print(json.dumps({key: result[key] for key in ['state', 'model', 'model_requests_executed']}))


if __name__ == '__main__':
    main()
