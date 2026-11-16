"""Provider transport only; upstream Agent S3 Worker/history/reflection stay intact.

The existing OpenAI-shaped engine wrapper still owns SDK/backoff retries. This
transport converts its request into a provider-native request and records both.
No provider tools, agent loops, persistent conversation IDs or response caches.
"""
from __future__ import annotations

import copy
import base64
import json
import os
import time

import httpx

from agents3_full import FullAgentS3Adapter

MODELS = {
    'gpt-6.1-sol': {'provider': 'openai', 'runtime': 'OpenAI API',
                  'endpoint': 'https://api.openai.com/v1/responses', 'key_env': 'OPENAI_API_KEY'},
    'claude-fable-5-1': {'provider': 'anthropic', 'runtime': 'Anthropic API',
                       'endpoint': 'https://api.anthropic.com/v1/messages', 'key_env': 'ANTHROPIC_API_KEY'},
}
GENERATION = {'effort': 'low', 'max_output_tokens': 4096}


def provider_request(model, messages):
    """Preserve text, roles, image ordering and history; change wire schema only."""
    provider = MODELS[model]['provider']
    converted = []
    system = []
    for message in messages:
        role = message['role']
        if role not in ('system', 'user', 'assistant'):
            raise ValueError(f'Unsupported Agent S3 message role: {role}')
        parts = message['content']
        if isinstance(parts, str):
            parts = [{'type': 'text', 'text': parts}]
        content = []
        for part in parts:
            if part['type'] == 'text':
                kind = ('output_text' if role == 'assistant' else 'input_text') if provider == 'openai' else 'text'
                content.append({'type': kind, 'text': part['text']})
            elif part['type'] == 'image_url':
                url = part['image_url']['url']
                # Upstream's vLLM message builder omits the PNG subtype even
                # though encode_image supplies PNG bytes. Fix MIME, not pixels.
                if url.startswith('data:image;base64,'):
                    encoded = url.split(',', 1)[1]
                    if not base64.b64decode(encoded, validate=True).startswith(b'\x89PNG\r\n\x1a\n'):
                        raise ValueError('Upstream screenshot is not PNG')
                    url = 'data:image/png;base64,' + encoded
                if not url.startswith('data:image/png;base64,'):
                    raise ValueError('Expected embedded PNG screenshot')
                if provider == 'openai':
                    content.append({'type': 'input_image', 'image_url': url, 'detail': 'high'})
                else:
                    content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                             'data': url.split(',', 1)[1]}})
            else:
                raise ValueError(f'Unsupported content type: {part["type"]}')
        if provider == 'anthropic' and role == 'system':
            if converted or system or any(p['type'] != 'text' for p in content):
                raise ValueError('Expected one leading text system message')
            system = content
        else:
            converted.append({'role': role, 'content': content})
    if provider == 'openai':
        return {'model': model, 'input': converted, 'store': False, 'service_tier': 'default',
                'reasoning': {'effort': GENERATION['effort']},
                'max_output_tokens': GENERATION['max_output_tokens']}
    return {'model': model, 'system': system, 'messages': converted, 'service_tier': 'standard_only',
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': GENERATION['effort']},
            'max_tokens': GENERATION['max_output_tokens']}


def normalize_response(model, response):
    """Expose only visible response text to the unchanged Worker; retain raw usage."""
    if response['model'] != model:
        raise ValueError('Provider returned a different model identifier; explicit review required')
    usage = response['usage']
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('input_tokens', 'output_tokens')):
        raise ValueError('Provider omitted measured token usage')
    if MODELS[model]['provider'] == 'openai':
        if response.get('status') not in ('completed', 'incomplete'):
            raise ValueError('Provider response did not complete')
        parts = [part for item in response['output'] if item['type'] == 'message' for part in item['content']]
        text = ''.join(p['text'] for p in parts if p['type'] == 'output_text')
        reason = 'length' if response['status'] == 'incomplete' else 'stop'
    else:
        text = ''.join(p['text'] for p in response['content'] if p['type'] == 'text')
        reason = 'length' if response['stop_reason'] == 'max_tokens' else 'stop'
    return {'id': response['id'], 'object': 'chat.completion', 'created': int(time.time()), 'model': model,
            'choices': [{'index': 0, 'finish_reason': reason, 'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': usage['input_tokens'], 'completion_tokens': usage['output_tokens'],
                      'total_tokens': usage['input_tokens'] + usage['output_tokens']},
            'provider_response': response}


class HostedTransport(httpx.BaseTransport):
    def __init__(self, model, inner=None):
        self.model = model
        self.config = MODELS[model]
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request):
        incoming = json.loads(request.read())
        if incoming['model'] != self.model:
            raise ValueError('Engine and provider model differ')
        body = provider_request(self.model, incoming['messages'])
        key = os.environ.get(self.config['key_env'])
        if not key:
            raise RuntimeError('Missing ' + self.config['key_env'])
        headers = {'content-type': 'application/json'}
        if self.config['provider'] == 'openai':
            headers['authorization'] = 'Bearer ' + key
        else:
            headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'})
        actual = httpx.Request('POST', self.config['endpoint'], headers=headers, json=body,
                              extensions={'timeout': {'connect': 30, 'read': 600, 'write': 600, 'pool': 600}})
        start = time.monotonic_ns()
        response = self.inner.handle_request(actual)
        response.read()
        try:
            if response.is_error:
                # Preserve HTTP status/Retry-After for the existing SDK retry policy.
                return httpx.Response(response.status_code, headers=response.headers, content=response.content, request=request)
            result = normalize_response(self.model, response.json())
            result.update(provider_request=body, provider_request_start_ns=start,
                          provider_response_end_ns=time.monotonic_ns(),
                          provider_request_id=response.headers.get('x-request-id') or response.headers.get('request-id'))
            return httpx.Response(200, json=result, request=request)
        finally:
            response.close()

    def close(self):
        self.inner.close()


class HostedAgentS3Adapter(FullAgentS3Adapter):
    def _validate_runtime(self, role, component):
        if role == 'grounder':
            return super()._validate_runtime(role, component)
        expected = MODELS[component['model']]
        if component['runtime'] != expected['runtime'] or component['base_url'] != expected['endpoint']:
            raise ValueError('Hosted component provider configuration differs')

    def __init__(self, config, log, *, _transport_factory=None):
        def transport(role):
            inner = _transport_factory(role) if _transport_factory else None
            return (inner or httpx.HTTPTransport()) if role == 'grounder' else HostedTransport(config['components'][role]['model'], inner)
        super().__init__(config, log, _transport_factory=transport)

    def spec_fields(self):
        return {**super().spec_fields(), 'inference_runtime': MODELS[self.model_id]['runtime'],
                'precision': 'provider undisclosed', 'provider_generation': copy.deepcopy(GENERATION)}
