"""Freeze the approved public raw-prompt API revision; CPU-only, no model calls.

Preserve the unexecuted structured-request draft. Use checkpoint-rendered text
through the stock apply_chat_template=false API while retaining ordinary stock
image loading/preprocessing. This changes request data only, never upstream code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

from native import OUT, HELPER, DRIVER
from prepare_export import read, write, sha, now, verify_source


def main():
    plan_path = OUT / 'native_plan.json'
    archive = OUT / 'native_plan_structured_revision1.json'
    assert not archive.exists(), 'Refuse an implicit second revision'
    assert not (OUT / 'native_build_result.json').exists(), 'Adopt before native execution'
    plan = read(plan_path)
    assert sha(HELPER) == plan['helper_sha256'] and sha(DRIVER) == plan['driver_sha256']
    verify_source(read(OUT / 'upstream_source_manifest.json'))
    additions = {}
    for model, selected in plan['models'].items():
        for path, expected in selected['prepared_hashes'].items():
            assert sha(path) == expected, path
        candidate_path = OUT / f'raw_prompt_{model}_cpu_audit_request.json'
        evidence_path = OUT / f'raw_prompt_{model}_cpu_audit.json'
        prompt_evidence_path = OUT / f'prompt_{model}_cpu_audit.json'
        candidate, evidence, prompt_evidence = map(read, (candidate_path, evidence_path, prompt_evidence_path))
        assert sha(candidate_path) == evidence['request_sha256']
        assert evidence['stock_raw_api_formatted_text_matches_checkpoint_rendered_text']
        assert evidence['checkpoint_rendered_input_ids_match_recorded']
        assert evidence['predicted_expanded_input_tokens'] == 998
        assert evidence['image_marker_count'] == 1 and evidence['image_tokens_after_stock_expansion'] == 300
        assert not candidate['apply_chat_template'] and not candidate['add_generation_prompt']
        assert not candidate['enable_thinking'] and candidate['max_generate_length'] == 4096
        assert candidate['temperature'] == 0 and candidate['top_p'] == .8 and candidate['top_k'] == 1
        assert len(candidate['requests']) == 1
        messages = candidate['requests'][0]['messages']
        assert [m['role'] for m in messages] == ['system', 'user']
        template_path = Path(prompt_evidence['stock_template_path'])
        assert sha(template_path) == prompt_evidence['stock_template_sha256']
        image_marker = read(template_path)['content_types']['image']['format']
        chunks = []
        images = []
        for message in messages:
            contents = message['content']
            if isinstance(contents, str):
                contents = [{'type': 'text', 'text': contents}]
            for item in contents:
                if item['type'] == 'text':
                    chunks.append(item['text'])
                else:
                    assert item['type'] == 'image'
                    images.append(Path(item['image']))
                    chunks.append(image_marker)
        assert images == [Path(selected['output']) / 'frame.png']
        rendered = ''.join(chunks)
        assert rendered == prompt_evidence['checkpoint_formatted_text']
        assert hashlib.sha256(rendered.encode()).hexdigest() == evidence['formatted_text_sha256']
        expected = read(Path(selected['output']) / 'expected_first_request.json')
        assert prompt_evidence['recorded_input_token_ids'] == expected['input_token_ids']
        checkpoint = Path(prompt_evidence['source_checkpoint'])
        extra_paths = [candidate_path, evidence_path, prompt_evidence_path,
            checkpoint / 'chat_template.jinja', checkpoint / 'tokenizer_config.json',
            checkpoint / 'tokenizer.json', template_path, Path(__file__).resolve()]
        additions[model] = (candidate, evidence, extra_paths)
    shutil.copy2(plan_path, archive)
    for model, selected in plan['models'].items():
        candidate, evidence, extra_paths = additions[model]
        dest = Path(selected['output'])
        shutil.copy2(dest / 'request.json', dest / 'request_structured_revision1.json')
        write(dest / 'request.json', candidate)
        selected['request'] = candidate
        selected['input_matching'] = (
            'Original checkpoint Jinja renders the recorded prompt offline; the public stock '
            'apply_chat_template=false API concatenates that text around the unchanged image item. '
            'CPU token audit matches all 998 recorded IDs; runtime IDs remain independently audited.')
        selected['raw_prompt_audit'] = evidence
        selected['prepared_hashes'].update({str(p): sha(p) for p in [dest / 'request.json',
            dest / 'request_structured_revision1.json', *extra_paths]})
    plan.update(request_revision=2, revised_at=now(),
        superseded_unexecuted_plan=str(archive), superseded_unexecuted_plan_sha256=sha(archive),
        request_revision_reason='Use the documented preformatted-text API to match original checkpoint prompts; '
            'stock extracted structured templates trim a newline (0.8B) or insert placeholders (Radix).',
        source_patches=[], model_requests_executed=0)
    plan['external_measurement_only'][0] = (
        'Public requestFileParser loads preformatted checkpoint-rendered text and the stored PNG; '
        'stock apply_chat_template=false retains standard image preprocessing and token expansion.')
    write(plan_path, plan)
    print('Prepared native request revision 2; both CPU audits match all 998 recorded IDs; no GPU/model requests.')


if __name__ == '__main__':
    main()
