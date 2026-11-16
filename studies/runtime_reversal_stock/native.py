"""Prepare/build/profile unmodified upstream TensorRT-Edge-LLM.

Only ``prepare`` is CPU-only. All other commands require the coordinator's
already-held GPU lock and a paused/completed evaluation. No application episodes
are created; only the stored Notification case 7 first request is replayed.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

from prepare_export import ROOT, EXPERIMENT, MODELS, exclusive, now, read, sha, verify_source, write

OUT = ROOT / 'runs/runtime_reversal_stock_20261029'
SOURCE = OUT / 'source/TensorRT-Edge-LLM'
BUILD = OUT / 'build'
DRIVER = ROOT / 'studies/runtime_reversal_stock/profile_runtime.cpp'
HELPER = Path(__file__).resolve()


def environment():
    env = dict(os.environ)
    # Intentionally replace inherited paths: never load the old locally patched plugin.
    env.update(EDGELLM_PLUGIN_PATH=str(BUILD / 'libNvInfer_edgellm_plugin.so'),
        LD_LIBRARY_PATH=':'.join([str(BUILD), (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'),
                                 '/usr/local/cuda/targets/sbsa-linux/lib']),
        PYTHONDONTWRITEBYTECODE='1')
    return env


def prepare():
    assert not (OUT / 'native_plan.json').exists(), 'Refuse to overwrite prepared provenance'
    manifest = read(OUT / 'upstream_source_manifest.json')
    verify_source(manifest)
    build_commands = [
        ['cmake', '-S', str(SOURCE), '-B', str(BUILD), '-DCMAKE_BUILD_TYPE=Release',
         ('-DTRT_PACKAGE_DIR=' + _CHRONOTOPE_HOME + '/trt-prefix/usr'),
         '-DCMAKE_TOOLCHAIN_FILE=' + str(SOURCE / 'cmake/aarch64_linux_toolchain.cmake'),
         '-DEMBEDDED_TARGET=gb10', '-DCUDA_CTK_VERSION=13.0', '-DENABLE_CUTE_DSL=ALL',
         '-DBUILD_PYTHON_BINDINGS=OFF', '-DBUILD_UNIT_TESTS=OFF', '-DENABLE_NVTX_PROFILING=OFF',
         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
        ['cmake', '--build', str(BUILD), '--target', 'llm_build', 'visual_build',
         'llm_inference', 'NvInfer_edgellm_plugin', '-j6'],
    ]
    models = {}
    for model, experiment in MODELS.items():
        dest = OUT / f'native_{model}_case7'
        assert not dest.exists(), dest
        dest.mkdir()
        call = experiment / 'calls/legacy_cpp_w16000000000_007_live.jsonl'
        record = read(experiment / 'records/legacy_cpp_w16000000000_007_live.json')
        assert record['job']['task'] == 'notification' and record['job']['phase'] == 'evaluation'
        first = json.loads(call.read_text().splitlines()[0])
        assert first['turn'] == 0 and first['request_kind'] == 'planner' and first['ok']
        assert first['input_tokens'] == 998
        assert first['screenshot_width'] == 640 and first['screenshot_height'] == 480
        # Host preparation already audited the exact recorded RGB payload and PNG.
        host = OUT / f'vllm_{model}_case7'
        host_plan = read(host / 'launch_plan.json')
        for path, expected in host_plan['source_hashes'].items():
            assert sha(path) == expected, path
        host_request = read(host / 'request.json')
        assert sha(host / 'request.json') == host_plan['request_file_sha256']
        import base64
        image_payloads = [item['image_url']['url'].split(',', 1)[1]
            for message in host_request['messages'] if isinstance(message['content'], list)
            for item in message['content'] if item['type'] == 'image_url']
        assert len(image_payloads) == 1
        assert base64.b64decode(image_payloads[0]) == (host / 'frame.png').read_bytes()
        shutil.copy2(host / 'frame.png', dest / 'frame.png')
        messages = copy.deepcopy(first['transport_messages'])
        image_count = 0
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if item['type'] == 'image':
                        item['image'] = str(dest / 'frame.png')
                        image_count += 1
        assert image_count == 1
        settings = first['generation_settings']
        assert settings == {'chat_template_kwargs': {'enable_thinking': False}, 'max_tokens': 4096,
            'mm_processor_kwargs': {'max_pixels': 1048576, 'min_pixels': 65536},
            'repetition_penalty': 1.05, 'temperature': 0.0, 'top_p': 0.8}
        request = {'batch_size': 1, 'temperature': 0.0, 'top_p': 0.8, 'top_k': 1,
            'max_generate_length': 4096, 'apply_chat_template': True,
            'add_generation_prompt': True, 'enable_thinking': False,
            'requests': [{'messages': messages}]}
        write(dest / 'request.json', request)
        write(dest / 'expected_first_request.json', first)
        engines = OUT / 'engines' / model
        engine_commands = [
            [str(BUILD / 'examples/llm/llm_build'), '--onnxDir', str(OUT / 'exports' / model / 'llm'),
             '--engineDir', str(engines / 'llm'), '--maxInputLen', '16384',
             '--maxKVCacheCapacity', '16384', '--maxBatchSize', '1'],
            [str(BUILD / 'examples/multimodal/visual_build'), '--onnxDir', str(OUT / 'exports' / model / 'visual'),
             '--engineDir', str(engines), '--minImageTokens', '64', '--maxImageTokens', '1024',
             '--maxImageTokensPerImage', '1024'],
        ]
        native = [str(BUILD / 'stock_profile_runtime'), str(engines / 'llm'), str(engines),
                  str(dest / 'request.json'), str(dest / 'response.json')]
        capture = ['/usr/local/cuda/bin/nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
            '--cpuctxsw=none', '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi',
            '--capture-range-end=stop', '--kill=none', '--force-overwrite=false', '--export=sqlite',
            '--output', str(dest / 'native_timeline'), *native]
        models[model] = {'source_experiment': str(experiment), 'source_call': str(call),
            'source_call_sha256': sha(call), 'source_job': record['job'], 'output': str(dest),
            'engine_commands': engine_commands, 'capture_command': capture,
            'prepared_hashes': {str(p): sha(p) for p in dest.iterdir() if p.is_file()},
            'original_input_tokens': 998, 'original_generation_settings': settings,
            'request': request, 'repetition_penalty': 1.0,
            'input_matching': 'Exact original structured text/image; stock template/preprocessor retained. Audit actual token IDs.',
            'native_top_k': 1, 'top_k_note': 'Temperature zero selects greedy decoding; top-k does not change argmax.',
            'precision_note': 'Read new stock export/engine config; do not assume old patched-export precision.'}
    result = {'state': 'prepared', 'prepared_at': now(), 'upstream_commit': manifest['commit'],
        'source_manifest_sha256': sha(OUT / 'upstream_source_manifest.json'),
        'helper_sha256': sha(HELPER), 'driver_sha256': sha(DRIVER), 'source_patches': [],
        'build_commands': build_commands, 'models': models,
        'environment_overrides': {k: environment()[k] for k in ['EDGELLM_PLUGIN_PATH','LD_LIBRARY_PATH']},
        'external_measurement_only': ['Public requestFileParser loads stored messages/image.',
            'Public onTokenGenerated callback marks tokens 5 through 37 with cudaProfilerStart/Stop.',
            'Public response and getBaseModelInputTokenIds are saved after full generation.',
            'No model-policy, source patch, forced tokens, action parsing/application, or warmup request.'],
        'application_episodes': 0, 'qualification_trials': 0, 'model_requests_executed': 0}
    write(OUT / 'native_plan.json', result)
    return result


def verify(plan):
    assert sha(HELPER) == plan['helper_sha256'], 'Prepared helper changed'
    assert sha(DRIVER) == plan['driver_sha256'], 'Prepared measurement driver changed'
    assert sha(OUT / 'upstream_source_manifest.json') == plan['source_manifest_sha256']
    verify_source(read(OUT / 'upstream_source_manifest.json'))
    assert read(EXPERIMENT / 'status.json')['state'] in ('paused', 'complete')


def run(command, log, cwd=ROOT):
    with Path(log).open('wb') as stream:
        subprocess.run(command, cwd=cwd, env=environment(), stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def driver_commands():
    # Compile with the stock example's generated compiler/includes, then link the
    # same pristine libraries/device-link object with only our example main swapped.
    entries = read(BUILD / 'compile_commands.json')
    matches = [e for e in entries if Path(e['file']) == SOURCE / 'examples/llm/llm_inference.cpp']
    assert len(matches) == 1
    entry = matches[0]
    compile_command = shlex.split(entry['command'])
    object_path = str(BUILD / 'stock_profile_runtime.o')
    compile_command[compile_command.index('-o') + 1] = object_path
    compile_command[compile_command.index(str(SOURCE / 'examples/llm/llm_inference.cpp'))] = str(DRIVER)
    link_file = BUILD / 'examples/llm/CMakeFiles/llm_inference.dir/link.txt'
    link_command = shlex.split(link_file.read_text())
    stock_object = 'CMakeFiles/llm_inference.dir/llm_inference.cpp.o'
    assert link_command.count(stock_object) == 1
    link_command[link_command.index(stock_object)] = object_path
    link_command[link_command.index('-o') + 1] = str(BUILD / 'stock_profile_runtime')
    return [(compile_command, Path(entry['directory'])), (link_command, BUILD / 'examples/llm')]


def build(plan):
    for index, command in enumerate(plan['build_commands']):
        run(command, OUT / f'native_build_{index}.log')
    commands = driver_commands()
    write(OUT / 'driver_build_commands.json', [{'command': c, 'cwd': str(d)} for c,d in commands])
    for index, (command, cwd) in enumerate(commands):
        run(command, OUT / f'native_driver_build_{index}.log', cwd)
    artifacts = [BUILD / 'stock_profile_runtime', BUILD / 'libNvInfer_edgellm_plugin.so',
        BUILD / 'cpp/libedgellmCore.a', BUILD / 'examples/utils/libexampleUtils.a',
        BUILD / 'examples/llm/llm_build', BUILD / 'examples/multimodal/visual_build']
    return {'artifact_hashes': {str(p): sha(p) for p in artifacts}}


def verify_build():
    built = read(OUT / 'native_build_result.json')
    assert built['state'] == 'complete'
    for path, expected in built['artifact_hashes'].items():
        assert sha(path) == expected, path


def engine(plan, model):
    verify_build()
    exported = read(OUT / f'export_{model}_result.json')
    assert exported['state'] == 'exported'
    for path, expected in exported['output_hashes'].items():
        assert sha(OUT / 'exports' / model / path) == expected, path
    assert not (OUT / 'engines' / model).exists(), 'Refuse engine overwrite'
    for index, command in enumerate(plan['models'][model]['engine_commands']):
        run(command, OUT / f'engine_{model}_{index}.log')
    verify_source(read(OUT / 'upstream_source_manifest.json'))
    directory = OUT / 'engines' / model
    return {'artifact_hashes': {str(p): sha(p) for p in directory.rglob('*') if p.is_file()},
        'llm_config': read(directory / 'llm/config.json'),
        'visual_config': read(directory / 'visual/config.json')}


def capture(plan, model):
    verify_build()
    built = read(OUT / f'native_engine_{model}_result.json')
    assert built['state'] == 'complete'
    for path, expected in built['artifact_hashes'].items():
        assert sha(path) == expected, path
    selected = plan['models'][model]
    for path, expected in selected['prepared_hashes'].items():
        assert sha(path) == expected, path
    dest = Path(selected['output'])
    assert not (dest / 'response.json').exists() and not (dest / 'native_timeline.nsys-rep').exists()
    run(selected['capture_command'], dest / 'capture.log')
    response = read(dest / 'response.json')
    assert response['model_requests_executed'] == 1 and response['ok']
    original = read(dest / 'expected_first_request.json')
    # Stock getter is populated only when optional thinker embeddings are
    # requested. Do not enable additional inference work just to obtain an audit.
    observed = response['input_token_ids'][0] if response['input_token_ids'] else None
    expected = original['input_token_ids']
    mismatch = next((i for i,(a,b) in enumerate(zip(expected, observed)) if a != b), None) if observed is not None else None
    if observed is not None and mismatch is None and len(expected) != len(observed):
        mismatch = min(len(expected), len(observed))
    assert len(response['input_token_counts']) == 1
    observed_count = response['input_token_counts'][0]
    assert len(response['formatted_requests']) == 1
    formatted_sha = hashlib.sha256(response['formatted_requests'][0]['complete'].encode()).hexdigest()
    expected_formatted_sha = selected['raw_prompt_audit']['formatted_text_sha256']
    audit = {'expected_input_tokens': len(expected), 'observed_input_tokens': observed_count,
        'input_token_ids_available_from_stock_runtime': observed is not None,
        'input_token_ids_equal_to_recorded_patched_request': observed == expected if observed is not None else None,
        'input_token_ids_unavailable_reason': ('Stock getter only populates when outputThinkerEmbeddings=true; '
            'that additional computation remains disabled.' if observed is None else None),
        'runtime_formatted_text_sha256': formatted_sha,
        'expected_checkpoint_formatted_text_sha256': expected_formatted_sha,
        'runtime_formatted_text_matches_checkpoint_render': formatted_sha == expected_formatted_sha,
        'cpu_audit_input_ids_match_recorded': selected['raw_prompt_audit']['checkpoint_rendered_input_ids_match_recorded'],
        'runtime_token_id_equality_directly_verified': observed == expected if observed is not None else False,
        'first_differing_token_index': mismatch,
        'audit_scope': 'Public runtime prompt text/count plus CPU token audit; no forced tokens or extra hidden-state computation.',
        'output_tokens': [len(ids) for ids in response['output_token_ids']],
        'full_generation_preserved': True, 'model_requests_executed': 1,
        'application_episodes': 0, 'qualification_trials': 0}
    write(dest / 'input_audit.json', audit)
    assert observed_count == len(expected) == 998, 'Runtime prompt token count differs from CPU audit'
    assert formatted_sha == expected_formatted_sha, 'Runtime formatted text differs from checkpoint render'
    return audit


def main():
    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt('Coordinator requested shutdown')
    signal.signal(signal.SIGTERM, stop_on_signal)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'build', 'engine', 'capture'])
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.operation == 'prepare':
        result = prepare()
    else:
        assert args.lock_fd is not None, 'Parent-held GPU lock is mandatory'
        if args.operation != 'build':
            assert args.model is not None
        result_path = OUT / ('native_' + args.operation +
            ('' if args.operation == 'build' else '_' + args.model) + '_result.json')
        assert not result_path.exists(), 'Refuse implicit repeat/overwrite; preserve failed evidence'
        with exclusive(args.lock_fd):
            plan = read(OUT / 'native_plan.json')
            verify(plan)
            result = {'state': 'running', 'started_at': now(), 'operation': args.operation,
                'model': args.model, 'application_episodes': 0, 'qualification_trials': 0}
            write(result_path, result)
            start = time.monotonic()
            try:
                extra = build(plan) if args.operation == 'build' else (
                    engine(plan, args.model) if args.operation == 'engine' else capture(plan, args.model))
                verify_source(read(OUT / 'upstream_source_manifest.json'))
                result.update(extra, state='complete')
            except BaseException as error:
                result.update(state='failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                result.update(completed_at=now(), elapsed_seconds=time.monotonic()-start)
                write(result_path, result)
    print(json.dumps({'state': result['state'], 'application_episodes': 0, 'qualification_trials': 0}))


if __name__ == '__main__':
    main()
