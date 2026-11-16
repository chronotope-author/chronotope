"""Audit saved Qwen/Nemotron requests on CPU; no model or trial execution."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/nemotron_prompt_transition_20261030'
RUNS = {
    'qwen_010': ROOT / 'runs/legacy_detector_full_agents3_20261029/experiment',
    'qwen_0111': ROOT / 'runs/notification_qwen27b_edgellm0111_20261030/experiment',
    'nemotron_0111': ROOT / 'runs/notification_nemotron_bolt_edgellm0111_20261030/experiment',
}
CHECKPOINTS = {
    'qwen_010': ROOT / 'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f',
    'qwen_0111': ROOT / 'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f',
    'nemotron_0111': ROOT / 'runs/model_cache/nvidia_nemotron35_bolt_bf16_49843ee',
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def calls(run, arm, index):
    matches = list((RUNS[run] / 'calls').glob(f'{arm}*w16000000000_{index:03d}_live.jsonl'))
    assert len(matches) == 1, matches
    return matches[0], [json.loads(line) for line in matches[0].read_text().splitlines() if line.strip()]


def completed_indices(run, arm):
    indices = set()
    for path in (RUNS[run] / 'records').glob(f'{arm}*.json'):
        job = read(path)['job']
        if job['mode'] == 'live' and float(job['window_s']) == 16:
            indices.add(job['index'])
    return indices


def normalize_request(call, native):
    if native:
        return {key: call[key] for key in ('messages', 'generation_settings',
                'coordinate_convention', 'response_format', 'screenshot_width', 'screenshot_height')}
    return {key: value for key, value in call['request'].items() if key != 'model'}


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    # This is the frozen Transformers dependency used by the Nemotron service.
    sys.path.insert(0, str(ROOT / 'runs/notification_gemma4_26b_20261030/serving_dependencies'))
    from transformers.utils.chat_template_utils import _compile_jinja_template
    from tokenizers import Tokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    checks, first, sources = {}, {}, {}
    for arm in ('legacy_cpp_detector', 'agents3_full'):
        native = arm == 'legacy_cpp_detector'
        cohorts = {}
        for before in ('qwen_010', 'qwen_0111'):
            shared = sorted(completed_indices(before, arm) & completed_indices('nemotron_0111', arm))
            mismatches, responses = [], Counter()
            for index in shared:
                old_path, old_calls = calls(before, arm, index)
                new_path, new_calls = calls('nemotron_0111', arm, index)
                if normalize_request(old_calls[0], native) != normalize_request(new_calls[0], native):
                    mismatches.append(index)
                if native:
                    responses['first_wait' if '"wait"' in new_calls[0]['response'] else 'first_other'] += 1
                    if all('"wait"' in call['response'] for call in new_calls):
                        responses['all_calls_wait'] += 1
                sources[str(old_path)] = digest(old_path)
                sources[str(new_path)] = digest(new_path)
            cohorts[before] = {'matched_completed_live_cases': len(shared),
                               'indices': shared, 'request_mismatch_indices': mismatches,
                               'nemotron_response_counts': dict(responses)}
        checks[arm] = cohorts

        for run in RUNS:
            source, recorded = calls(run, arm, 0)
            call = recorded[0]
            messages = deepcopy(call['messages'] if native else call['request']['messages'])
            checkpoint = CHECKPOINTS[run]
            tokenizer_config = read(checkpoint / 'tokenizer_config.json')
            template_path = checkpoint / 'chat_template.jinja'
            template_source = template_path.read_text()
            download_path = (ROOT / 'runs/agents3_policy_nvidia27b_20261029/checkpoint_download.json'
                             if run.startswith('qwen') else RUNS[run].parent / 'checkpoint_download.json')
            checkpoint_hashes = read(download_path)['hashes']
            assert checkpoint_hashes[str(template_path)] == digest(template_path)
            for name in ('tokenizer.json', 'tokenizer_config.json', 'processor_config.json'):
                asset = checkpoint / name
                assert checkpoint_hashes[str(asset)] == digest(asset)
            specials = {k: (v['content'] if isinstance(v, dict) else v)
                        for k, v in tokenizer_config.items() if k.endswith('_token')}
            rendered = _compile_jinja_template(template_source).render(
                messages=messages, add_generation_prompt=True, enable_thinking=False, **specials)
            if native:
                assert rendered == call['formatted_text'], (run, 'native rendered prompt mismatch')
            if run.startswith('qwen'):
                marker, image_tokens = '<|image_pad|>', 300
                expansion = marker * image_tokens
                image_preprocessing = {'resized_width': 640, 'resized_height': 480, 'spatial_stride': 32}
            else:
                contract = read(RUNS[run] / 'configs/notification_compact_prompt.json')
                marker, expansion = contract['image_placeholder'], contract['expanded_image_text']
                image_tokens = contract['image_tokens']
                image_preprocessing = contract['image_preprocessing']
            assert rendered.count(marker) == 1
            expanded = rendered.replace(marker, expansion)
            ids = Tokenizer.from_file(str(checkpoint / 'tokenizer.json')).encode(expanded, add_special_tokens=False).ids
            recorded_input = call['input_tokens'] if native else call['response']['usage']['prompt_tokens']
            assert len(ids) == recorded_input, (run, arm, len(ids), recorded_input)
            output = call['response'] if native else call['response']['choices'][0]['message']['content']
            output_tokens = call['output_tokens'] if native else call['response']['usage']['completion_tokens']
            folder = OUT / arm
            folder.mkdir(exist_ok=True)
            (folder / f'{run}.rendered.txt').write_text(rendered)
            write(folder / f'{run}.messages.json', messages)
            (folder / f'{run}.answer.txt').write_text(output + '\n')
            first[f'{arm}/{run}'] = {
                'source_call_log': str(source), 'source_call_sha256': digest(source),
                'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                'template_sha256': digest(template_path), 'template_matches_pinned_checkpoint': True,
                'recorded_input_tokens': recorded_input, 'reconstructed_input_tokens': len(ids),
                'recorded_output_tokens': output_tokens, 'response': output,
                'calls_in_case0': len(recorded), 'image_tokens': image_tokens,
                'image_preprocessing': image_preprocessing,
                'native_rendered_text_matches_recorded': True if native else None,
                'agent_rendering_provenance': None if native else 'CPU reconstruction from recorded OpenAI-format messages and pinned checkpoint template; token count matches server usage, server token IDs were not logged',
            }
        for before in ('qwen_010', 'qwen_0111'):
            folder = OUT / arm
            for suffix in ('rendered.txt', 'messages.json', 'answer.txt'):
                a, b = folder / f'{before}.{suffix}', folder / f'nemotron_0111.{suffix}'
                diff = ''.join(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True),
                                                   fromfile=a.name, tofile=b.name))
                (folder / f'{before}_to_nemotron.{suffix}.diff').write_text(diff)

    summary = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'question': 'Did switching Qwen to Nemotron change the prompts in legacy C++ or Agent S3?',
        'runs': {k: str(v) for k, v in RUNS.items()},
        'first_request_audits': checks, 'first_case': first,
        'scope': 'Compare first requests for the same completed Live cases before model outputs can enter history or grounding queries.',
        'source_call_hashes': sources,
        'limitations': ['Checkpoint-specific image preprocessing and tokenizers differ across models.',
                        'Matching Agent S3 reconstructed token count does not prove server token-ID identity.',
                        'This source/request audit alone does not identify which phrase causes Nemotron to wait.'],
        'model_inference_calls': 0, 'experimental_trials': 0,
        'evaluation_controllers_or_prompts_modified': False,
    }
    write(OUT / 'comparison.json', summary)
    print(json.dumps({'checks': checks, 'first_case_counts': {
        k: {f: v[f] for f in ('recorded_input_tokens', 'recorded_output_tokens', 'image_tokens')}
        for k, v in first.items()}, 'output': str(OUT)}))


if __name__ == '__main__':
    main()
