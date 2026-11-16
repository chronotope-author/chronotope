"""Read-only request audit for a stateless native prompt and unchanged Python Worker.

Only the first planner request is matched across arms. Native later decisions
remain stateless; the host retains its original history/reflection/retry policy.
This module performs no inference and does not screen task competence.
"""
import base64
import hashlib
import io
import json


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def prompt_fingerprint(prompt):
    return sha(canonical({k: v for k, v in prompt.items() if k != 'prompt_sha256'}))


def initial_messages(prompt):
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': prompt['planner_system']}]},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt['initial_user_text'] + prompt['text_buffer_suffix']},
            {'type': 'image'}]},
    ]


def audit_trial_requests(out, ident, arm, trace, scenario, execution):
    from agents3_policy_contract import normalize_messages, generation_settings, PolicyReference, validate_plan_response
    from worker_policy_audit import model_input
    from chronotope.controller import parse_agents3
    from chronotope.runner import PayloadStore
    from chronotope.png import encode_png
    from PIL import Image

    path = out / 'calls' / f'{ident}.jsonl'
    events = trace.all('decision_invocation')
    if not path.exists():
        assert not events, 'Missing model audit log'
        return []
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    observations = {r['observation_id']: r for r in trace.all('observation')}
    store = PayloadStore(out / 'payloads')
    evidence = []
    cursor = 0
    prompt = execution['native_prompt_contract']
    assert prompt_fingerprint(prompt) == prompt['prompt_sha256']
    expected_native_messages = initial_messages(prompt)
    reference = PolicyReference(execution['host_worker_contract'])

    for turn, event in enumerate(events):
        assert event['model_reported'] == execution['model']
        obs = observations[event['observation_id']]
        png = encode_png(store.get(obs['payload_hash']), scenario.canvas_w_px, scenario.canvas_h_px, 2)
        with Image.open(io.BytesIO(png)) as image:
            assert image.size == (640, 480)
            rgb_sha = sha(image.convert('RGB').tobytes())
        source_url = 'data:image;base64,' + base64.b64encode(png).decode()

        if arm == 'legacy_cpp':
            assert cursor < len(calls), 'Native decision has no request'
            own = [calls[cursor]]
            cursor += 1
            call = own[0]
            assert call['request_kind'] == 'planner'
            assert call['messages'] == expected_native_messages, 'Native prompt acquired history or changed'
            checked = validate_plan_response(call['response'], 640, 480, png)
            # Native performs one parse with no formatting retry; invalid output
            # is a scored no-action result, not an infrastructure failure.
            executable = checked['executable'] if not checked['errors'] and checked['executable'] is not None else 'WAIT'
            expected_action = parse_agents3([executable])
            if expected_action and expected_action.get('action') == 'click':
                assert (event['image_x'], event['image_y']) == (expected_action['x'], expected_action['y'])
                assert event['outcome'] == 'click'
            else:
                assert event['outcome'] == 'wait', 'Native emitted an unexpected action'
        else:
            count = event['planner_calls'] + event['reflection_calls']
            own = calls[cursor:cursor + count]
            cursor += count
            assert own, 'Host decision has no audited calls'
            replay_index = 0

            def replay_model(role, messages):
                nonlocal replay_index
                assert replay_index < len(own), 'Missing host Worker calls'
                call = own[replay_index]
                replay_index += 1
                assert call['role'] == role
                assert normalize_messages(call['request']['messages']) == messages, 'Host Worker behavior changed'
                return call['response']['choices'][0]['message']['content']

            reference.run_turn(replay_model, png)
            assert replay_index == len(own), 'Additional host calls outside Worker policy'
            if turn == 0:
                assert normalize_messages(own[0]['request']['messages']) == expected_native_messages

        total_in = total_out = 0
        for index, call in enumerate(own):
            if arm == 'legacy_cpp':
                assert call['ok'], call.get('error')
                assert call['prompt_sha256'] == prompt['prompt_sha256']
                messages = call['messages']
                settings = call['generation_settings']
                count_in, count_out = call['input_tokens'], call['output_tokens']
            else:
                assert call['model'] == execution['model']
                assert call['policy_contract_sha256'] == execution['host_worker_contract']['policy_sha256']
                body = call['request']
                messages = normalize_messages(body['messages'])
                settings = {key: body[key] for key in generation_settings()}
                for message in body['messages']:
                    for part in message['content']:
                        if part['type'] == 'image_url':
                            assert part['image_url']['sha256'] == sha(source_url.encode())
                usage = call['response']['usage']
                count_in, count_out = usage['prompt_tokens'], usage['completion_tokens']
            assert settings == prompt['generation_settings'] == generation_settings()
            assert sum(p['type'] == 'image' for m in messages for p in m['content']) == 1
            expected = model_input(messages, execution['checkpoint_path'], execution['image_tokens'])
            assert count_in == expected['input_tokens'], 'Input token count differs from checkpoint rendering'
            assert 0 <= count_out <= prompt['generation_settings']['max_tokens']
            if arm == 'legacy_cpp':
                assert call['formatted_text'] == expected['formatted_text']
                assert call['input_token_ids_available'] is False
                predicted = call['preflight_input_token_ids']
                assert len(predicted) == count_in
                assert sha(json.dumps(predicted, separators=(',', ':')).encode()) == expected['input_token_ids_sha256']
            expected.pop('formatted_text')
            total_in += count_in
            total_out += count_out
            evidence.append({
                'turn': turn, 'request_index': index,
                'request_kind': call.get('request_kind', call.get('role')),
                'source_payload_hash': obs['payload_hash'], 'source_rgb_sha256': rgb_sha,
                'messages_sha256': sha(canonical(messages)), **expected,
                'output_tokens': count_out, 'native_direct_token_ids_verified': False,
                'input_id_evidence': 'CPU checkpoint tokenization; runtime prompt/count matched',
                'first_request_match_scope': turn == 0 and index == 0,
                'native_prompt_sha256': prompt['prompt_sha256'],
            })
        assert (total_in, total_out) == (event['input_tokens'], event['output_tokens'])
    assert cursor == len(calls), 'Unaffiliated model calls or more than one native call per decision'
    if evidence:
        peer = ident.replace(arm, 'agents3_stripped' if arm == 'legacy_cpp' else 'legacy_cpp', 1)
        other = out / 'records' / f'{peer}.json'
        if other.exists():
            counterpart = json.loads(other.read_text()).get('request_audit', [])
            if counterpart:
                for key in ('source_payload_hash', 'source_rgb_sha256', 'messages_sha256',
                            'text_sha256', 'input_token_ids_sha256', 'input_tokens'):
                    assert evidence[0][key] == counterpart[0][key], f'First paired request differs: {key}'
    return evidence
