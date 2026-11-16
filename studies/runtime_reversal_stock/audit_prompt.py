"""CPU-only audit of the stock template against the recorded Notification request.

This does not modify templates, model requests, sources or engines.  The stock
formatted text below follows tokenizer.cpp applyChatTemplate for this specific
two-message, one-image request. Runtime-captured input IDs remain the definitive
check once a stock engine is built.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/runtime_reversal_stock_20261029'
MODELS = {
    'qwen08': ROOT / 'runs/model_cache/qwen35_0p8b_legacy_source',
    'radixark27b': ROOT / 'runs/model_cache/radixark_qwen38_nvfp4_319f741c',
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_stock(template, messages):
    """Mirror stock tokenizer.cpp:1108-1263 for these explicit messages."""
    assert [m['role'] for m in messages] == ['system', 'user']
    assert template.get('default_system_prompt', '') == ''
    rendered = template.get('prompt_prefix', '')
    for message in messages:
        role = template['roles'][message['role']]
        rendered += role['prefix']
        contents = message['content']
        if isinstance(contents, str):
            contents = [{'type': 'text', 'text': contents}]
        for content in contents:
            if content['type'] == 'text':
                text = content['text']
                if role.get('trim_content', False) or template.get('trim_content', False):
                    text = text.strip(' \t\n\r\f\v')
                rendered += text
            else:
                rendered += template['content_types'][content['type']]['format']
        rendered += role['suffix']
    return rendered + template['generation_prompt']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()
    destination = OUT / f'prompt_{args.model}_cpu_audit.json'
    assert not destination.exists(), 'Preserve previous evidence; do not overwrite'
    exported = OUT / 'exports' / args.model / 'llm'
    template_path = exported / 'processed_chat_template.json'
    template = read(template_path)
    recorded_path = ROOT / 'runs/runtime_reversal_20261029' / f'native_{args.model}_case7' / 'expected_first_request.json'
    recorded = read(recorded_path)
    messages = recorded['transport_messages']
    assert recorded['turn'] == 0 and recorded['input_tokens'] == 998
    assert recorded['screenshot_width'] == 640 and recorded['screenshot_height'] == 480
    tokenizer = Tokenizer.from_file(str(exported / 'tokenizer.json'))
    hf = AutoTokenizer.from_pretrained(str(MODELS[args.model]), local_files_only=True)
    native_text = render_stock(template, messages)
    hf_text = hf.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    image_id = tokenizer.token_to_id('<|image_pad|>')
    expected = recorded['input_token_ids']
    image_tokens = (640 // 32) * (480 // 32)
    assert image_tokens == 300 == expected.count(image_id)

    def encode(text):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        assert ids.count(image_id) == 1
        return [v for i in ids for v in ([i] * image_tokens if i == image_id else [i])]

    native_ids, hf_ids = encode(native_text), encode(hf_text)
    assert hf_ids == expected, 'Checkpoint template must reproduce the original input exactly'
    mismatch = next((i for i, (a, b) in enumerate(zip(expected, native_ids)) if a != b), None)
    if mismatch is None and len(expected) != len(native_ids):
        mismatch = min(len(expected), len(native_ids))
    placeholders = [p for p in ('<placeholder_system_prompt>', '<placeholder_user_text>') if p in native_text]
    result = {
        'state': 'cpu_audited',
        'model': args.model,
        'source_commit': read(OUT / 'upstream_source_manifest.json')['commit'],
        'stock_template_path': str(template_path),
        'stock_template_sha256': digest(template_path),
        'recorded_request_path': str(recorded_path),
        'recorded_request_sha256': digest(recorded_path),
        'source_checkpoint': str(MODELS[args.model]),
        'recorded_input_tokens': len(expected),
        'hf_template_input_tokens': len(hf_ids),
        'hf_template_input_ids_match_recorded': hf_ids == expected,
        'stock_structured_request_input_tokens_predicted': len(native_ids),
        'stock_structured_request_input_ids_match_recorded': native_ids == expected,
        'first_differing_token_index': mismatch,
        'recorded_tokens_near_difference': expected[max(0, mismatch - 3):mismatch + 6] if mismatch is not None else [],
        'stock_tokens_near_difference': native_ids[max(0, mismatch - 3):mismatch + 6] if mismatch is not None else [],
        'stock_trim_content': template.get('trim_content', False),
        'literal_exporter_placeholders_in_stock_prompt': placeholders,
        'injected_reasoning_instruction_count': native_text.count('Reasoning effort is set to xhigh.'),
        'image_tokens': image_tokens,
        'image_dimensions': [640, 480],
        'image_resize_required': False,
        'raw_prompt_api_option': {
            'supported_upstream_flag': 'apply_chat_template=false',
            'description': 'Caller can provide the checkpoint-rendered text split around the original image content item. Stock API concatenates those text pieces without trimming and renders the same image marker. This is a distinct raw-prompt diagnostic interface, not the default structured-message path.',
            'expected_input_tokens': len(hf_ids),
            'executed': False,
        },
        'stock_formatted_text': native_text,
        'checkpoint_formatted_text': hf_text,
        'stock_predicted_input_token_ids': native_ids,
        'recorded_input_token_ids': expected,
        'model_requests_executed': 0,
        'application_episodes': 0,
        'qualification_trials': 0,
        'limitation': 'C++ formatting mirrored for this fixed request; actual engine runtime IDs still must be captured. No image tensors or model forward pass executed.',
    }
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in (
        'model', 'recorded_input_tokens', 'hf_template_input_ids_match_recorded',
        'stock_structured_request_input_tokens_predicted',
        'first_differing_token_index', 'literal_exporter_placeholders_in_stock_prompt',
        'injected_reasoning_instruction_count')}, indent=2))


if __name__ == '__main__':
    main()
