#!/usr/bin/env python3
"""CPU prompt preparation and deferred native application build for ledger jobs.

The control, detector, history policy and action parser are preserved. Only the
checkpoint-specific chat template and task-specific existing compact prompt are
adapted. This helper never calls a model. --build must run after the predecessor.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'runs/legacy_detector_full_agents3_20261029'
STOCK = ROOT / 'runs/runtime_reversal_stock_20261029'
TASK_VERSIONS = {'notification': 4, 'drift': 5, 'code': 11, 'button': 12, 'trading': 10}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n'); tmp.replace(path)


def source_hashes(path):
    return {str(p): sha(p.read_bytes()) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and 'build' not in p.parts}


def render_contract(base_contract, checkpoint):
    """Render existing compact task text with the pinned checkpoint's template.

    Requires tokenizer assets only; torch/GPU/model weights are not used. The
    returned reference matches the per-invocation audit's existing field names.
    """
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import _compile_jinja_template
    checkpoint = Path(checkpoint)
    cfg = read(checkpoint / 'config.json'); tc = read(checkpoint / 'tokenizer_config.json')
    contract = dict(base_contract)
    task = contract['task']; assert task in TASK_VERSIONS
    assert contract['prompt_version'] == TASK_VERSIONS[task]
    sys.path.insert(0, str(ROOT / 'src'))
    from chronotope.controller import SYSTEM_PROMPTS
    expected = SYSTEM_PROMPTS[TASK_VERSIONS[task]]
    if task == 'code': expected = expected.format(n=contract.get('code_length', 4))
    assert contract['planner_system'] == expected, 'Task text must remain the established compact prompt'
    assert contract['schema'] == 'legacy-compact-prompt-1'
    assert contract['response_format'] == 'legacy_json'
    assert contract['coordinate_convention'] == ('text' if task == 'code' else 'norm1000')
    assert contract['history'] == 'none' and contract['reflection'] is False
    assert contract['formatting_retries'] == 0 and contract['text_buffer_suffix'] == ''
    generation = contract['generation_settings']
    assert generation == {'temperature': 0.0, 'max_tokens': 64, 'top_p': 1.0,
                          'repetition_penalty': 1.0, 'chat_template_kwargs': {'enable_thinking': False}, 'stop': ['}']}
    width, height = contract['screenshot_width'], contract['screenshot_height']
    assert width > 0 and height > 0
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': contract['planner_system']}]},
                {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': contract['initial_user_text']}]}]
    template_path = checkpoint / 'chat_template.jinja'
    template_text = template_path.read_text() if template_path.exists() else tc['chat_template']
    if isinstance(template_text, list):
        template_text = next(x['template'] for x in template_text if x['name'] == 'default')
    template = _compile_jinja_template(template_text)
    specials = {k: (v['content'] if isinstance(v, dict) else v) for k, v in tc.items() if k.endswith('_token')}
    formatted = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False, **specials)
    model_type = cfg['model_type']
    if model_type == 'gemma4':
        placeholder = '<|image|>'
        pp = read(checkpoint / 'processor_config.json')['image_processor']
        limit = pp['max_soft_tokens']; patch = pp['patch_size']; pool = pp['pooling_kernel_size']
        factor = math.sqrt(limit * pool * pool * patch * patch / (width * height))
        side = patch * pool
        resized_w = math.floor(width * factor / side) * side
        resized_h = math.floor(height * factor / side) * side
        # Current core five task frame geometry is nondegenerate; reject new
        # extreme-aspect configurations rather than guess the stock fallback.
        assert resized_w > 0 and resized_h > 0
        image_tokens = resized_w * resized_h // (side * side)
        expanded_image = specials['boi_token'] + placeholder * image_tokens + specials['eoi_token']
        image_format = placeholder
        preprocessing = {'max_soft_tokens': limit, 'resized_width': resized_w, 'resized_height': resized_h,
                         'patch_size': patch, 'pooling_kernel_size': pool}
    elif model_type in ('qwen3_5', 'qwen3_8', 'qwen3_vl'):
        # This frozen legacy Qwen configuration has a 32-pixel spatial stride.
        # The original 640x480 screenshot gives exactly 300 visual tokens.
        assert (width, height) == (640, 480), 'Qwen geometry needs fresh stock-preprocessor validation'
        placeholder = '<|image_pad|>'; image_tokens = 300
        expanded_image = placeholder * image_tokens
        image_format = '<|vision_start|>' + placeholder + '<|vision_end|>'
        preprocessing = {'spatial_stride': 32, 'resized_width': width, 'resized_height': height}
    else:
        raise ValueError(f'Unreviewed checkpoint model type {model_type}')
    assert formatted.count(image_format) == 1, 'Expected exactly one checkpoint image marker'
    before, after = formatted.split(image_format)
    # All rendered text before the image is a raw system fragment. Public
    # applyChatTemplate=false concatenates it with user(image, suffix) unchanged.
    tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
    expanded = formatted.replace(placeholder, expanded_image)
    ids = tokenizer.encode(expanded, add_special_tokens=False).ids
    assert len(ids) + 64 <= 16384
    contract.update(model_template='checkpoint_rendered',
                    rendered_messages=messages, rendered_system_text=before,
                    rendered_user_prefix='', rendered_user_suffix=after,
                    formatted_text=formatted, image_placeholder=placeholder,
                    runtime_image_format=image_format, expanded_image_text=expanded_image,
                    image_tokens=image_tokens, preflight_input_token_ids=ids,
                    image_preprocessing=preprocessing,
                    checkpoint_template_sha256=sha(template_text.encode()))
    contract.pop('prompt_sha256', None)
    contract['prompt_sha256'] = sha(canonical(contract))
    reference = {'messages': messages, 'formatted_text': formatted, 'input_tokens': len(ids),
                 'input_token_ids': ids, 'image_tokens': image_tokens,
                 'checkpoint_template_sha256': contract['checkpoint_template_sha256'],
                 'inference_calls': 0, 'qualification_trials': 0}
    return contract, reference


def prepare_native(root):
    root = Path(root).resolve(); dst = root / 'build_source/chronotope'
    if dst.exists():
        result = read(root / 'native_source_preparation.json')
        assert source_hashes(dst) == result['source_hashes'], 'Prepared source changed'
        # The old build snapshot retained only remote.py; freeze needs the
        # complete, unchanged Python benchmark bridge alongside native sources.
        for old in (PARENT / 'experiment/code/chronotope/python/chronotope_core').glob('*.py'):
            new = dst / 'python/chronotope_core' / old.name
            if not new.exists(): shutil.copy2(old, new)
        result['source_hashes'] = source_hashes(dst)
        write(root / 'native_source_preparation.json', result)
        return result
    shutil.copytree(PARENT / 'build_source/chronotope', dst,
                    ignore=shutil.ignore_patterns('build', '__pycache__', '*.pyc'))
    for old in (PARENT / 'experiment/code/chronotope/python/chronotope_core').glob('*.py'):
        new = dst / 'python/chronotope_core' / old.name
        if not new.exists(): shutil.copy2(old, new)
    header = dst / 'src/agents3_prompt.h'
    text = header.read_text()
    needle = 'inline StaticPrompt prepare(const Json& c) {\n'
    assert text.count(needle) == 1
    branch = '''    if (c.value("model_template", std::string()) == "checkpoint_rendered") {
        const auto& g = c.at("generation_settings");
        if (c.at("schema") != "legacy-compact-prompt-1" || c.at("response_format") != "legacy_json" ||
            (c.at("coordinate_convention") != "norm1000" && c.at("coordinate_convention") != "text") || c.at("history") != "none" ||
            c.at("reflection") != false || c.at("formatting_retries") != 0 ||
            g.at("temperature") != 0.0 || g.at("max_tokens") != 64 || g.at("top_p") != 1.0 ||
            g.at("repetition_penalty") != 1.0 || g.at("stop") != Json::array({"}"}) ||
            g.at("chat_template_kwargs").at("enable_thinking") != false)
            throw std::runtime_error("Unsupported checkpoint-rendered legacy prompt");
        StaticPrompt result;
        result.messages = c.at("rendered_messages");
        result.system_text = c.at("rendered_system_text");
        result.user_prefix = c.at("rendered_user_prefix");
        result.user_suffix = c.at("rendered_user_suffix");
        result.formatted_text = c.at("formatted_text");
        if (result.system_text + result.user_prefix + c.at("runtime_image_format").get<std::string>() +
                result.user_suffix != result.formatted_text)
            throw std::runtime_error("Checkpoint prompt fragments do not reconstruct rendered text");
        return result;
    }
'''
    header.write_text(text.replace(needle, needle + branch))
    planner = dst / 'src/planner_trt.cpp'; text = planner.read_text()
    old = '''        if (impl_->compact_json && (impl_->cfg.prompt_version != 4 ||
            impl_->prompt_contract.at("planner_system").get<std::string>() != impl_->system))
            throw std::runtime_error("Compact prompt differs from the original legacy v4 system prompt");'''
    new = '''        const bool checkpoint_rendered = impl_->prompt_contract.value("model_template", std::string()) == "checkpoint_rendered";
        if (impl_->compact_json && checkpoint_rendered &&
            impl_->cfg.prompt_version != impl_->prompt_contract.at("prompt_version").get<int>())
            throw std::runtime_error("Native task prompt version differs from its frozen contract");
        if (impl_->compact_json && !checkpoint_rendered && (impl_->cfg.prompt_version != 4 ||
            impl_->prompt_contract.at("planner_system").get<std::string>() != impl_->system))
            throw std::runtime_error("Compact prompt differs from the original legacy v4 system prompt");'''
    assert text.count(old) == 1; text = text.replace(old, new)
    old = '''        std::string expanded = impl_->prompt.formatted_text, pads;
        for (int i = 0; i < 300; ++i) pads += "<|image_pad|>";
        const auto marker = expanded.find("<|image_pad|>");
        if (marker == std::string::npos || expanded.find("<|image_pad|>", marker + 13) != std::string::npos)
            throw std::runtime_error("Fixed prompt must contain exactly one image");
        expanded.replace(marker, 13, pads);
        impl_->prompt_input_ids = tokenizer.encode(expanded, false);'''
    new = '''        std::string expanded = impl_->prompt.formatted_text;
        std::string image_marker = "<|image_pad|>", image_expansion;
        if (checkpoint_rendered) {
            image_marker = impl_->prompt_contract.at("image_placeholder");
            image_expansion = impl_->prompt_contract.at("expanded_image_text");
        } else for (int i = 0; i < 300; ++i) image_expansion += image_marker;
        const auto marker = expanded.find(image_marker);
        if (image_marker.empty() || marker == std::string::npos ||
            expanded.find(image_marker, marker + image_marker.size()) != std::string::npos)
            throw std::runtime_error("Fixed prompt must contain exactly one image");
        expanded.replace(marker, image_marker.size(), image_expansion);
        impl_->prompt_input_ids = tokenizer.encode(expanded, false);
        if (checkpoint_rendered && impl_->prompt_input_ids !=
            impl_->prompt_contract.at("preflight_input_token_ids").get<std::vector<int32_t>>())
            throw std::runtime_error("Stock tokenizer differs from checkpoint CPU fixture");'''
    assert text.count(old) == 1; text = text.replace(old, new)
    old = '{"coordinate_convention", impl_->compact_json ? "norm1000" : "screenshot_pixels"}'
    new = '{"coordinate_convention", impl_->prompt_contract.value("coordinate_convention", impl_->compact_json ? "norm1000" : "screenshot_pixels")}'
    assert text.count(old) == 1; planner.write_text(text.replace(old, new))
    changed = []
    for path in dst.rglob('*'):
        if not path.is_file(): continue
        source = PARENT / 'build_source/chronotope' / path.relative_to(dst)
        if not source.exists(): source = PARENT / 'experiment/code/chronotope' / path.relative_to(dst)
        if path.read_bytes() != source.read_bytes(): changed.append(str(path.relative_to(dst)))
    assert sorted(changed) == ['src/agents3_prompt.h', 'src/planner_trt.cpp']
    result = {'state': 'prepared', 'created_at': datetime.now(timezone.utc).isoformat(),
              'source_hashes': source_hashes(dst), 'application_changed_files': sorted(changed),
              'backend_source_patches': [], 'policy_changes': [],
              'scope': 'Existing compact task prompts, checkpoint chat format and tokenizer audit only',
              'inference_calls': 0, 'model_inference_calls': 0, 'qualification_trials': 0}
    write(root / 'native_source_preparation.json', result)
    return result


def build_native(root):
    root = Path(root).resolve()
    from resume_notification_schedule import completion_gate_for
    ready, reason = completion_gate_for(PARENT / 'experiment')
    assert ready, reason
    prepared = prepare_native(root)
    model = read(root / 'experiment/configs/model.json')
    runtime_template = read(Path(model['engines']) / 'llm/processed_chat_template.json')
    for prompt in read(root / 'inputs/task_prompts.json').values():
        assert runtime_template['content_types']['image']['format'] == prompt['runtime_image_format'], 'Stock image marker changed'
    out = root / 'native_build'
    commands = [
        ['cmake', '-S', str(root / 'build_source/chronotope'), '-B', str(out),
         f'-DCHRONOTOPE_EDGELLM_BUILD={STOCK / "build"}',
         f'-DCHRONOTOPE_EDGELLM_SOURCE={STOCK / "source/TensorRT-Edge-LLM"}',
         ('-DCHRONOTOPE_TRT_PREFIX=' + _CHRONOTOPE_HOME + '/trt-prefix'), '-DCMAKE_BUILD_TYPE=Release'],
        ['cmake', '--build', str(out), '--target', 'chronotope_main', 'chronotope', '-j4']]
    result = {'state': 'building', 'started_at': datetime.now(timezone.utc).isoformat(),
              'commands': commands, 'source_hashes': prepared['source_hashes'], 'source_patches': [],
              'upstream_commit': 'bb291453bc10d9bd1142070c605e090d4e483ee2',
              'inference_calls': 0, 'model_inference_calls': 0, 'qualification_trials': 0}
    write(root / 'native_build_result.json', result)
    try:
        for i, cmd in enumerate(commands):
            with (root / f'native_build_{i}.log').open('xb') as log:
                subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert source_hashes(root / 'build_source/chronotope') == prepared['source_hashes']
        result.update(state='complete', binary_hashes={str(p): sha(p.read_bytes()) for p in
                      [out / 'chronotope', out / 'libchronotope.so']})
    except BaseException as exc:
        result.update(state='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        result['completed_at'] = datetime.now(timezone.utc).isoformat(); write(root / 'native_build_result.json', result)
    return result


def prepare_bundles(root, checkpoint):
    root = Path(root).resolve()
    plan = read(root / 'experiment/plan.json')
    model = read(root / 'experiment/configs/model.json')
    tasks = sorted({job['task'] for job in plan['jobs']})
    sys.path.insert(0, str(ROOT / 'src'))
    from chronotope.controller import SYSTEM_PROMPTS
    base = read(PARENT / 'inputs/legacy_compact_prompt.json')
    prompts, references = {}, {}
    for task in tasks:
        contract = json.loads(json.dumps(base))
        system = SYSTEM_PROMPTS[TASK_VERSIONS[task]]
        if task == 'code': system = system.format(n=4)
        contract.update(task=task, model=model['id'], revision=model['revision'],
                        prompt_version=TASK_VERSIONS[task], planner_system=system,
                        coordinate_convention='text' if task == 'code' else 'norm1000',
                        source_policy='Established compact task prompt and original JSON brace stop',
                        initial_user_text=('Screenshot attached. Type the code currently shown, or wait.'
                                           if task == 'code' else base['initial_user_text']))
        action_file = root / 'inputs/action_support.json'
        if task == 'code' and action_file.exists():
            support = read(action_file)
            if support.get('code_action') == 'native_type_value_and_host_bridge':
                assert support.get('user_authorized') is True and support.get('configuration_reviewed') is True
                assert 'parse_type4(' in (root / 'build_source/chronotope/src/agents3_prompt.h').read_text()
                contract['allow_type4'] = True
        prompts[task], references[task] = render_contract(contract, checkpoint)
    write(root / 'inputs/task_prompts.json', prompts)
    write(root / 'inputs/task_prompt_references.json', references)
    return prompts, references


def verify_cpu(root, checkpoint):
    root = Path(root).resolve()
    prepared = prepare_native(root)
    prompts, references = prepare_bundles(root, checkpoint)
    fixture = root / 'inputs/native_prompt_fixture.cpp'
    fixture.write_text('''#include "agents3_prompt.h"
#include <iostream>
int main() {
    chronotope::agents3_prompt::Json c; std::cin >> c;
    const auto p = chronotope::agents3_prompt::prepare(c);
    std::cout << chronotope::agents3_prompt::Json({{"messages", p.messages},
        {"formatted_text", p.formatted_text}, {"system_text", p.system_text},
        {"user_prefix", p.user_prefix}, {"user_suffix", p.user_suffix}}).dump();
}
''')
    exe = root / 'inputs/native_prompt_fixture'
    subprocess.run(['g++', '-std=c++17', '-O0', '-I' + str(root / 'build_source/chronotope/src'),
                    '-I' + str(STOCK / 'source/TensorRT-Edge-LLM/3rdParty/nlohmannJson/include'),
                    str(fixture), '-o', str(exe)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    results = {}
    for task, contract in prompts.items():
        result = subprocess.run([str(exe)], input=json.dumps(contract).encode(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True)
        got = json.loads(result.stdout)
        assert got['messages'] == references[task]['messages']
        assert got['formatted_text'] == references[task]['formatted_text']
        assert got['system_text'] + got['user_prefix'] + contract['runtime_image_format'] + got['user_suffix'] == got['formatted_text']
        assert len(contract['preflight_input_token_ids']) == references[task]['input_tokens']
        results[task] = {'input_tokens': references[task]['input_tokens'], 'image_tokens': contract['image_tokens'],
                         'prompt_sha256': contract['prompt_sha256'], 'checkpoint_rendering_equal': True}
    proof = {'state': 'passed', 'created_at': datetime.now(timezone.utc).isoformat(),
             'model_inference_calls': 0, 'experimental_trials': 0, 'qualification_trials': 0,
             'source_file_sha256': {str(p): sha(p.read_bytes()) for p in [
                 Path(__file__).resolve(), root / 'build_source/chronotope/src/agents3_prompt.h',
                 root / 'build_source/chronotope/src/planner_trt.cpp', fixture]},
             'prompt_files_sha256': {str(p): sha(p.read_bytes()) for p in [
                 root / 'inputs/task_prompts.json', root / 'inputs/task_prompt_references.json']},
             'task_checks': results, 'application_changed_files': prepared['application_changed_files'],
             'action_parser_changed': bool(prompts.get('code', {}).get('allow_type4')),
             'type_action_support': ('authorized four-digit type transport' if prompts.get('code', {}).get('allow_type4')
                                     else 'unchanged; requires explicit separate choice'),
             'backend_source_patches': []}
    write(root / 'native_compact_verification.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--build', action='store_true'); parser.add_argument('--verify-cpu', action='store_true')
    parser.add_argument('--checkpoint', type=Path); parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.build:
        assert args.lock_fd is not None, 'Deferred build requires the queue GPU lock descriptor'
        expected = os.stat(ROOT / 'runs/legacy_d1_gpu.lock')
        observed = os.fstat(args.lock_fd)
        assert (expected.st_dev, expected.st_ino) == (observed.st_dev, observed.st_ino)
        result = build_native(args.root)
    elif args.verify_cpu:
        assert args.checkpoint is not None
        result = verify_cpu(args.root, args.checkpoint)
    else:
        result = prepare_native(args.root)
    print(json.dumps({'state': result['state'], 'root': str(args.root)}))


if __name__ == '__main__':
    main()
