"""Evidence audit for compact legacy C++ + detector versus upstream Agent S3.

The policies deliberately differ. This audit verifies provenance and request
accounting, never model competence or equality of generated answers.
"""
import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def prompt_fingerprint(prompt):
    return sha(canonical({k: v for k, v in prompt.items() if k != 'prompt_sha256'}))


def audit_trial_requests(out, ident, arm, trace, scenario, execution):
    path = out / 'calls' / f'{ident}.jsonl'
    events = trace.all('decision_invocation')
    if not path.exists():
        assert not events, 'Missing request log'
        return []
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    evidence = []
    if arm == 'legacy_cpp_detector':
        assert len(calls) == len(events), 'Native must have one request per decision'
        contract = execution['native_prompt_contract']
        reference = execution['native_prompt_reference']
        assert prompt_fingerprint(contract) == contract['prompt_sha256']
        for index, (call, event) in enumerate(zip(calls, events)):
            assert call['ok'], call.get('error')
            assert call['request_kind'] == 'planner'
            assert call['response_format'] == 'legacy_json'
            assert call['prompt_sha256'] == contract['prompt_sha256']
            assert call['generation_settings'] == contract['generation_settings']
            assert call['messages'] == reference['messages']
            assert call['formatted_text'] == reference['formatted_text']
            assert call['input_tokens'] == reference['input_tokens']
            assert 0 <= call['output_tokens'] <= 64
            assert (event['input_tokens'], event['output_tokens']) == (call['input_tokens'], call['output_tokens'])
            assert event['model_reported'] == execution['model']
            # Detector may intentionally replace model coordinates after this
            # invocation. Neither model misses nor malformed answers gate trials.
            evidence.append({'request_index': index, 'decision_id': event['decision_id'],
                'role': 'planner', 'model': execution['model'], 'revision': execution['revision'],
                'input_tokens': call['input_tokens'], 'output_tokens': call['output_tokens'],
                'messages_sha256': sha(canonical(call['messages'])),
                'formatted_text_sha256': sha(call['formatted_text'].encode()),
                'response_sha256': sha(call['response'].encode()),
                'detector_correction_enabled': True, 'matched_prompt_claim': False})
        return evidence

    assert arm == 'agents3_full'
    full = execution['agent_s3_components']
    components = full['components']
    config_sha256 = sha(json.dumps(full, sort_keys=True, separators=(',', ':')).encode())
    verified = execution['full_agent_source_verification']
    assert config_sha256 == verified['component_config_sha256']
    cursor = 0
    for event in events:
        assert event['component_config_sha256'] == config_sha256
        assert event['upstream_worker_template_sha256'] == verified['upstream_worker_template_sha256']
        assert event['component_models'] == {role: component['model'] for role, component in components.items()}
        assert event['optional_code_execution'] == 'disabled upstream default'
        # The adapter records every planner/reflection/grounding request in the
        # invocation, with no assumption that exactly one model call occurred.
        count = event.get('model_calls')
        if count is None:
            count = sum(event.get(k, 0) for k in
                        ('planner_calls', 'reflection_calls', 'grounder_calls', 'text_grounder_calls', 'code_agent_calls'))
        assert isinstance(count, int) and count >= 0
        own = calls[cursor:cursor + count]
        assert len(own) == count
        assert event['model_call_request_sha256'] == [call['request_sha256'] for call in own]
        for role in components:
            assert event[role + '_calls'] == sum(call['role'] == role for call in own)
        ins = outs = 0
        for call in own:
            role = call['role']
            assert role in components, f'Unrecorded model component: {role}'
            component = components[role]
            assert call['model'] == component['model']
            assert call['revision'] == component['revision']
            assert call['component_config_sha256'] == config_sha256
            assert not call.get('error'), call.get('error')
            assert call.get('http_status') == 200
            result = call['response']
            assert result['model'] == component['model']
            usage = result['usage']
            assert usage['prompt_tokens'] >= 0 and usage['completion_tokens'] >= 0
            ins += usage['prompt_tokens']; outs += usage['completion_tokens']
            # Preserve the upstream messages, including past screenshots. No
            # direct-pixel substitutions, history stripping, or paired-policy
            # replay are applied by this audit.
            body = call['request']
            assert body['model'] == component['model']
            generation = full['grounder_generation' if role == 'grounder' else 'generation']
            for key in ('max_tokens', 'temperature', 'top_p', 'repetition_penalty'):
                assert body[key] == generation[key], (role, key)
            if 'enable_thinking' in generation:
                assert body['chat_template_kwargs']['enable_thinking'] == generation['enable_thinking']
            images = sum(part.get('type') == 'image_url' for message in body['messages']
                         if isinstance(message.get('content'), list) for part in message['content'])
            assert images <= component['max_images_per_request']
            if role == 'planner':
                assert images >= 1
                first = body['messages'][0]
                assert first['role'] == 'system'
                content = first['content']
                system_text = content if isinstance(content, str) else ''.join(
                    part['text'] for part in content if part.get('type') == 'text')
                assert sha(system_text.encode()) == event['upstream_worker_prompt_sha256']
            evidence.append({'request_index': cursor, 'decision_id': event['decision_id'],
                'role': role, 'model': component['model'], 'revision': component['revision'],
                'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens'],
                'request_sha256': call['request_sha256'], 'messages_sha256': sha(canonical(body['messages'])),
                'response_sha256': sha(canonical(result)),
                'component_config_sha256': config_sha256, 'image_count': images,
                'metrics_present': result.get('metrics') is not None,
                'matched_prompt_claim': False})
            cursor += 1
        assert (ins, outs) == (event.get('input_tokens'), event.get('output_tokens'))
    assert cursor == len(calls), 'Requests outside measured Agent S3 invocations'
    return evidence
