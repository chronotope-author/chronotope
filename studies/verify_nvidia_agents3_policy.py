"""Verify Agent S3 policy and input parity on CPU, with canned responses only.

Compiles the real native policy header, compares its transcripts and parsing
against fixtures differential-tested against the installed Agent S3 Worker,
then compares each rendered request and expanded token sequence with the actual
Hugging Face processor. This is not an inference or benchmark runner.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents3_policy_contract import policy_fingerprint


HARNESS = r'''#include "agents3_policy.h"
#include <fstream>
#include <iostream>
using namespace chronotope::agents3_policy;
int main(int argc, char** argv) {
  try {
    if(argc != 3) throw std::runtime_error("contract and fixture paths required");
    Json c, f; std::ifstream(argv[1]) >> c; std::ifstream(argv[2]) >> f;
    Json output = {{"cases", Json::array()}, {"parser_cases", Json::array()}};
    for(const auto& fixture : f.at("cases")) {
      Policy policy(c);
      Json requests=Json::array(), turns=Json::array(), sleeps=Json::array();
      for(const auto& responses : fixture.at("responses_by_turn")) {
        size_t i=0;
        auto result=policy.run_turn(
          [&](const std::string& role,const Json& messages,int format_attempt,int transport_attempt) {
            if(i >= responses.size()) throw std::runtime_error("Fixture response queue exhausted");
            requests.push_back({{"role",role}, {"messages",messages},
              {"generation_settings",c.at("generation_settings")},
              {"transport_messages",normalize_transport_messages(messages, c.value("native_chat_template", Json::object()).value("strip_historical_thinking", true))},
              {"raw_text",stock_raw_render(messages)},
              {"format_attempt",format_attempt}, {"transport_attempt",transport_attempt}});
            return responses.at(i++).get<std::string>();
          },[&](double seconds){sleeps.push_back(seconds);});
        if(i != responses.size()) throw std::runtime_error("Unused fixture responses");
        turns.push_back(result);
      }
      output["cases"].push_back({{"name",fixture.at("name")}, {"requests",requests},
        {"turns",turns}, {"sleep_seconds",sleeps},
        {"final_planner_messages",policy.planner_messages()},
        {"final_reflection_messages",policy.reflection_messages()},
        {"turn_count",policy.turn_count()}});
    }
    for(const auto& test : f.at("parser_cases")) output["parser_cases"].push_back(
      validate_plan_response(test.at("response"),c.at("screenshot_width"),c.at("screenshot_height")));
    std::cout << output.dump();
  } catch(const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def equal(actual, expected, path):
    """Give a bounded, useful first difference instead of dumping transcripts."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(actual) != set(expected):
            raise AssertionError(f'{path}: keys differ: {set(actual) ^ set(expected)}')
        for key in expected:
            equal(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            raise AssertionError(f'{path}: length {len(actual)} != {len(expected)}')
        for i, (a, e) in enumerate(zip(actual, expected)):
            equal(a, e, f'{path}[{i}]')
    elif actual != expected:
        raise AssertionError(f'{path}: actual {repr(actual)[:500]} != expected {repr(expected)[:500]}')


def native_render(messages, template):
    assert template['trim_content'] is False
    out = []
    for message in messages:
        role = template['roles'][message['role']]
        out.append(role['prefix'])
        for part in message['content']:
            out.append(part['text'] if part['type'] == 'text' else template['content_types'][part['type']]['format'])
        out.append(role['suffix'])
    out.append(template['generation_prompt'])
    return ''.join(out)


def verify(args):
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from tokenizers import Tokenizer
    import transformers
    import tokenizers
    import torch

    stable_inputs = {path: digest(path) for path in [args.contract, args.fixtures,
        args.native_dir / 'agents3_policy.h', args.checkpoint / 'chat_template.jinja']}
    contract = json.loads(args.contract.read_text())
    fixtures = json.loads(args.fixtures.read_text())
    equal(policy_fingerprint(contract), contract['policy_sha256'], 'contract SHA256')
    equal(fixtures['policy_sha256'], contract['policy_sha256'], 'fixture contract SHA256')
    build = args.out.parent / 'cpu_policy_verification'
    build.mkdir(parents=True, exist_ok=True)

    # These tests use a mocked urllib transport and cannot request model output.
    worker_test = subprocess.run([str(args.agents3_python), '-B', '-m', 'unittest', 'discover',
        '-s', str(args.code / 'nvidia_cpu_tests'), '-p', 'test_agents3_policy_contract.py', '-q'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1'})
    (build / 'worker_differential_tests.log').write_text(worker_test.stdout)
    if worker_test.returncode:
        raise RuntimeError('Installed Worker differential tests failed; see cpu_policy_verification/worker_differential_tests.log')
    import re
    count = re.search(r'Ran (\d+) tests', worker_test.stdout)
    assert count, 'Could not confirm actual Worker differential test count'

    harness = build / 'native_policy_harness.cpp'
    harness.write_text(HARNESS)
    binary = build / 'native_policy_harness'
    compile_command = ['g++', '-std=c++20', '-O2', '-Wall', '-Wextra',
        '-I' + str(args.native_dir), '-I' + str(args.json_include), str(harness), '-o', str(binary)]
    compiled = subprocess.run(compile_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (build / 'compile.log').write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError('Native CPU harness compilation failed; see cpu_policy_verification/compile.log')
    output = subprocess.run([str(binary), str(args.contract), str(args.fixtures)],
        check=True, text=True, capture_output=True)
    native = json.loads(output.stdout)
    (build / 'native_result.json').write_text(json.dumps(native, indent=2, ensure_ascii=False) + '\n')
    equal(len(native['cases']), len(fixtures['cases']), 'case count')
    all_requests = []
    for actual, expected in zip(native['cases'], fixtures['cases']):
        name = expected['name']
        equal(actual['name'], name, f'{name}.name')
        equal(actual['turns'], expected['expected_turns'], f'{name}.turns')
        equal(actual['turn_count'], len(expected['expected_turns']), f'{name}.turn_count')
        for field in ['sleep_seconds', 'final_planner_messages', 'final_reflection_messages']:
            equal(actual[field], expected[field], f'{name}.{field}')
        equal([{k: value for k, value in request.items() if k in ('role', 'messages', 'generation_settings')}
            for request in actual['requests']], expected['expected_requests'], f'{name}.requests')
        all_requests.extend((name, i, r) for i, r in enumerate(actual['requests']))
    equal(len(native['parser_cases']), len(fixtures['parser_cases']), 'parser case count')
    for i, (actual, expected) in enumerate(zip(native['parser_cases'], fixtures['parser_cases'])):
        equal(actual, expected['expected'], f'parser_cases[{i}] {expected["response"]}')

    processor = AutoProcessor.from_pretrained(str(args.checkpoint), local_files_only=True)
    exported = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
    template_path = args.checkpoint / 'chat_template.jinja'
    # Reuse a stored evaluation frame; no application episode is launched.
    old_trace = args.reference_run / 'trials/w16000000000/legacy_cpp/closing_window-8400000_live.jsonl'
    events = [json.loads(line) for line in old_trace.read_text().splitlines()]
    begin = next(event for event in events if event['record'] == 'decision_begin')
    obs = next(event for event in events if event['record'] == 'observation' and event['observation_id'] == begin['observation_id'])
    payload = args.reference_run / 'payloads' / (obs['payload_hash'] + '.bin')
    raw = np.frombuffer(payload.read_bytes(), dtype=np.uint8).reshape(240, 320, 3)
    rgb = raw.repeat(2, axis=0).repeat(2, axis=1)
    image = Image.fromarray(rgb)
    mm_kwargs = contract['generation_settings']['mm_processor_kwargs']
    summaries = []
    for name, index, request in all_requests:
        host_text = processor.tokenizer.apply_chat_template(request['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        native_text = request['raw_text']
        equal(native_text, host_text, f'{name}.request[{index}].chat_text')
        equal(exported.encode(native_text, add_special_tokens=False).ids,
            processor.tokenizer.encode(host_text, add_special_tokens=False), f'{name}.request[{index}].text_token_ids')
        inputs = processor(text=[host_text], images=[image], return_tensors='pt', **mm_kwargs)
        equal(inputs.image_grid_thw.tolist(), [[1, 30, 40]], f'{name}.request[{index}].image_grid')
        image_tokens = int((inputs.input_ids == 248056).sum())
        equal(image_tokens, 300, f'{name}.request[{index}].image_tokens')
        ids = exported.encode(native_text.replace('<|image_pad|>', '<|image_pad|>' * image_tokens), add_special_tokens=False).ids
        equal(ids, inputs.input_ids[0].tolist(), f'{name}.request[{index}].expanded_input_ids')
        summaries.append({'case': name, 'request_index': index, 'role': request['role'],
            'format_attempt': request['format_attempt'], 'transport_attempt': request['transport_attempt'],
            'input_tokens': len(ids), 'chat_text_sha256': hashlib.sha256(host_text.encode()).hexdigest(),
            'input_ids_sha256_int64_le': hashlib.sha256(np.array(ids, dtype='<i8').tobytes()).hexdigest()})
    normalized = ((rgb.astype(np.float32) / np.float32(255) - np.float32(.5)) / np.float32(.5)).astype(np.float16)
    patches = np.stack([normalized, normalized]).transpose(0, 3, 1, 2)
    patches = patches.reshape(1, 2, 3, 15, 2, 16, 20, 2, 16).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(1200, 1536)
    host_patches = inputs.pixel_values.numpy().astype(np.float16)
    assert np.array_equal(patches, host_patches), 'Image FP16 patches differ'
    sources = [Path(__file__).resolve(), args.code / 'agents3_policy_contract.py',
        args.code / 'agents3_stripped.py', args.code / 'nvidia_cpu_tests/test_agents3_policy_contract.py',
        args.native_dir / 'agents3_policy.h', args.contract, args.fixtures, harness, binary,
        template_path, args.checkpoint / 'tokenizer.json', args.checkpoint / 'tokenizer.json',
        args.checkpoint / 'chat_template.jinja', args.checkpoint / 'preprocessor_config.json',
        ROOT/'runs/runtime_reversal_stock_20261029/source/TensorRT-Edge-LLM/cpp/tokenizer/tokenizer.cpp', old_trace, payload]
    for path, original_sha in stable_inputs.items():
        equal(digest(path), original_sha, f'Verification input changed while running: {path}')
    return {'state': 'passed', 'created_at': datetime.now(timezone.utc).isoformat(),
        'policy_sha256': contract['policy_sha256'], 'experimental_trials': 0, 'model_inference_calls': 0,
        'worker_differential_tests_passed': int(count[1]), 'native_policy_cases_passed': len(native['cases']),
        'native_parser_cases_passed': len(native['parser_cases']), 'native_request_transcripts_checked': len(all_requests),
        'request_roles_messages_settings_equal': True, 'retry_sleeps_and_persistent_histories_equal': True,
        'native_hf_chat_text_equal': True, 'native_hf_expanded_input_ids_equal': True,
        'first_request_input_tokens': summaries[0]['input_tokens'], 'request_checks': summaries,
        'image_tokens': 300, 'image_grid_thw': [[1, 30, 40]], 'patch_fp16_equal': True,
        'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'patch_sha256_fp16_le': hashlib.sha256(host_patches.astype('<f2').tobytes()).hexdigest(),
        'compile_command': compile_command,
        'versions': {'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__, 'torch': torch.__version__},
        'source_file_sha256': {str(path): digest(path) for path in sources},
        'limits': ['Canned response/CPU verification; no model inference or benchmark trials.',
            'Actual C++ application raw renderer matches checkpoint Jinja across fixture transcripts; checkpoint tokenizer IDs checked;freeze additionally verifies exported tokenizer identity. Upstream runtime applies raw API and later verifies actual formatted text/count.',
            'Image patches are CPU emulation of native preprocessing, not GPU-buffer capture.',
            'Runtime kernels, numerical accumulation, and generated model output can still differ.',
            'Policy coverage is the listed fixtures and parser cases, not every possible malformed Python string.']}



if __name__=='__main__':
    from types import SimpleNamespace
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();code=root/'experiment/code'
    tests=code/'nvidia_cpu_tests';tests.mkdir(exist_ok=True)
    text=(ROOT/'tests/test_agents3_policy_contract.py').read_text()
    text=text.replace("sys.path[:0]=[str(ROOT/'studies'),str(ROOT/'src')]", "sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'chronotope/python')]")
    text=text.replace("self.assertEqual(c['repetition_penalty'],1.05)", "self.assertEqual(c['repetition_penalty'],1.0)")
    (tests/'test_agents3_policy_contract.py').write_text(text)
    args=SimpleNamespace(contract=root/'policy_inputs/agents3_policy.json',fixtures=root/'policy_inputs/policy_fixtures.json',
        native_dir=root/'build_source/chronotope/src',engine_root=root/'engines',out=root/'policy_verification.json',code=code,
        agents3_python=Path((_CHRONOTOPE_HOME + '/agents3-venv/bin/python')),
        json_include=ROOT/'runs/runtime_reversal_stock_20261029/source/TensorRT-Edge-LLM/3rdParty/nlohmannJson/include',
        checkpoint=ROOT/'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f',
        reference_run=ROOT/'runs/agents3_policy_qwen08_20261028/experiment')
    sys.path[:0]=[str(code),str(code/'src'),str(code/'chronotope/python')]
    try:
        report=verify(args)
        report.update(model='nvidia/Qwen3.8-27B-NVFP4',revision='482ca0f3832238542f8f5295dde86b5f22711d80',
            native_template_mode='public raw API;C++checkpointformatting;unmodifiedbackend',repetition_penalty=1.0,
            qualification_trials=0,native_direct_token_ids_available=False,
            runtime_precision='NVFP4 MLP/head;FP8 attentionprojections;FP16vision/dense;FP16KV')
    except Exception as error:
        args.out.write_text(json.dumps({'state':'failed','experimental_trials':0,'model_inference_calls':0,
            'error':f'{type(error).__name__}: {error}'},indent=2)+'\n')
        raise
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['state','policy_sha256','native_policy_cases_passed',
        'native_parser_cases_passed','native_request_transcripts_checked','first_request_input_tokens','patch_fp16_equal']}))
