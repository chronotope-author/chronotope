"""Agent S3-derived host, matched request; no Worker policy/history/reflection.

Uses Agent S3's LMMAgent/get_response and vLLM engine instance, replacing the
transport method because the upstream engine drops stop/top_k/thinking options.
Every decision is exactly one stateless request matching legacy native prompt 4.
This is not the original Agent S3 Worker policy and must not be labelled as such.
"""
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import urllib.request

MODEL = 'Qwen/Qwen3.5-0.8B'
SYSTEM_PROMPT = ('You control a desktop by clicking. You are shown one screenshot. '
    'At some point a notification card with a blue button appears at an unknown position and expires shortly after. '
    'If the card is visible, answer with action "click" and the coordinates of the center of its blue button on a 0-1000 grid '
    '(x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). '
    'If no card is visible, answer with action "wait". Answer with the JSON object only.')
USER_TEXT = "Screenshot attached. Click the target's center or wait."
REQUEST_CONTRACT = {
    'model': MODEL, 'prompt_version': 4, 'system': SYSTEM_PROMPT, 'user_text': USER_TEXT,
    'message_order': ['system:text', 'user:image', 'user:text'],
    'coordinate_space': 'norm1000', 'max_tokens': 64, 'temperature': 0.0,
    'top_p': 1.0, 'top_k': 1, 'repetition_penalty': 1.0,
    'enable_thinking': False, 'stop': ['}'], 'history': 'none', 'reflection': False,
    'format_retries': 0, 'mm_processor_kwargs': {'min_pixels': 65536, 'max_pixels': 1048576},
}


def contract_fingerprint():
    return hashlib.sha256(json.dumps(REQUEST_CONTRACT, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def native_canvas_coordinates(x, y, width, height):
    """Match C++ signed integer division (truncates toward zero, unlike //)."""
    def trunc(n, d):
        return (1 if n >= 0 else -1) * (abs(n) // d)
    return {'kind': 'click', 'x': trunc(int(x) * width, 1000), 'y': trunc(int(y) * height, 1000)}


def _after_key(text, key):
    pos = text.find('"' + key + '"')
    if pos < 0:
        return None
    colon = text.find(':', pos)
    return None if colon < 0 else text[colon + 1:].lstrip(' \t\r\n')


def _native_string(text, key):
    rest = _after_key(text, key)
    if not rest or rest[0] != '"':
        return None
    chars = []; i = 1
    while i < len(rest):
        ch = rest[i]
        if ch == '"':
            return ''.join(chars)
        if ch == '\\' and i + 1 < len(rest):
            i += 1; escape = rest[i]
            if escape in {'n', 't', 'r', 'b', 'f'}:
                chars.append({'n': '\n', 't': '\t', 'r': '\r', 'b': '\b', 'f': '\f'}[escape])
            elif escape == 'u' and i + 4 < len(rest):
                match = re.match(r'[0-9a-fA-F]+', rest[i + 1:i + 5])
                codepoint = int(match[0], 16) if match else 0
                chars.append(chr(codepoint) if codepoint < 128 else '?')
                i += 4
            else:
                chars.append(escape)
        else:
            chars.append(ch)
        i += 1
    return None


def _native_int(text, key):
    rest = _after_key(text, key)
    if not rest or rest[0] not in '-0123456789':
        return None
    match = re.match(r'-?[0-9]+', rest)
    value = int(match[0]) if match else 0  # strtol("-") returns zero.
    return max(-(1 << 63), min((1 << 63) - 1, value))


def _native_float_prefix(text):
    # C++ strtod accepts leading whitespace, signs, decimals and exponents.
    match = re.match(r'\s*[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)', text)
    if not match:
        return None
    value = float(match[0])
    return (value, match.end()) if math.isfinite(value) else None


def parse_native_action(text):
    """Match native click extraction; nonclick/malformed model output means wait.

    Like the legacy parser, this accepts fenced/list-wrapped and stop-trimmed
    JSON, and accepts integer x/y or the checkpoint's coordinate:[x,y] form.
    It does not enforce full JSON grammar or add model formatting retries.
    """
    action_value = _native_string(text, 'action')
    if action_value != 'click':
        return {'action': 'wait'}
    x, y = _native_int(text, 'x'), _native_int(text, 'y')
    if x is None or y is None:
        pos = text.find('"coordinate"')
        start = text.find('[', pos) if pos >= 0 else -1
        if start < 0:
            return {'action': 'wait'}
        pair = text[start + 1:]
        first = _native_float_prefix(pair)
        if first is None:
            return {'action': 'wait'}
        fx, end = first
        second = _native_float_prefix(pair[end:].lstrip(' ,'))
        if second is None:
            return {'action': 'wait'}
        fy, _ = second
        x, y = int(fx + 0.5), int(fy + 0.5)
    return {'action': 'click', 'x': x, 'y': y}


def _audit_request(body):
    audit = copy.deepcopy(body)
    for message in audit['messages']:
        for item in message['content']:
            if item['type'] == 'image_url':
                url = item['image_url']['url']
                item['image_url'] = {'encoded_bytes': len(url), 'sha256': hashlib.sha256(url.encode()).hexdigest()}
    return audit


class MatchedRequestAgentS3Adapter:
    def __init__(self, base_url, log, *, model_id=MODEL, max_tokens=64):
        if model_id != MODEL or max_tokens != REQUEST_CONTRACT['max_tokens']:
            raise ValueError('Matched request requires Qwen3.5-0.8B and max_tokens=64')
        self.base_url = base_url.rstrip('/')
        self.model_id, self.max_tokens = model_id, max_tokens
        self.log = Path(log)
        self.agent = None
        self.calls, self.errors = [], []
        self._image_audit = None

    def spec_fields(self):
        return {'server': 'agents3_service', 'model_id': self.model_id,
                'implementation': 'Agent S3-derived host, matched request; no Worker policy/history/reflection',
                'request_contract_sha256': contract_fingerprint(), 'reflection': False,
                'history': 'none', 'grounder': None, 'updater': None, 'thinking': False}

    def _generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        if temperature != 0.0 or max_new_tokens != self.max_tokens or kwargs:
            raise ValueError('LMMAgent generation settings changed from matched contract')
        if len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']:
            raise ValueError('Matched request must have no historical messages')
        if messages[0]['content'] != [{'type': 'text', 'text': SYSTEM_PROMPT}]:
            raise ValueError('System prompt does not match native prompt 4')
        if [x['type'] for x in messages[1]['content']] != ['image_url', 'text']:
            raise ValueError('Native user content order must be image then text')
        if messages[1]['content'][1]['text'] != USER_TEXT:
            raise ValueError('User prompt does not match native request')
        body = {'model': self.model_id, 'messages': messages, 'max_tokens': self.max_tokens,
                'temperature': 0.0, 'top_p': 1.0, 'top_k': 1, 'repetition_penalty': 1.0,
                'stop': ['}'], 'chat_template_kwargs': {'enable_thinking': False},
                'mm_processor_kwargs': copy.deepcopy(REQUEST_CONTRACT['mm_processor_kwargs'])}
        encoded = json.dumps(body, allow_nan=False).encode()
        record = {'role': 'planner', 'model': self.model_id, 'start_ns': time.monotonic_ns(),
                  'request_contract_sha256': contract_fingerprint(),
                  'request_sha256': hashlib.sha256(encoded).hexdigest(),
                  'request': _audit_request(body), **self._image_audit}
        try:
            req = urllib.request.Request(self.base_url + '/chat/completions', data=encoded,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
            record['response'] = result
            if result.get('model') != self.model_id or not result.get('usage'):
                raise RuntimeError('Model identity or usage missing from response')
            answer = result['choices'][0]['message'].get('content')
            if not isinstance(answer, str):
                raise RuntimeError('Model returned no text')
            record['http_status'] = 200
            return answer
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            self.errors.append(record['error'])
            raise
        finally:
            record['end_ns'] = time.monotonic_ns()
            self.calls.append(record)
            with self.log.open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')

    def invoke(self, png, system, user_text, structured=False, schema=None, parser=None):
        from PIL import Image
        from chronotope.controller import RawInvocation
        if system != SYSTEM_PROMPT or user_text != USER_TEXT:
            raise ValueError('Controller prompt differs from the frozen native request contract')
        if structured or schema is not None or parser is not None:
            raise ValueError('Native contract uses no guided decoding or alternate parser')
        with Image.open(io.BytesIO(png)) as image:
            rgb = image.convert('RGB')
            self._image_audit = {'source_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                                 'source_image_wh': list(rgb.size),
                                 'source_png_sha256': hashlib.sha256(png).hexdigest()}
        if self.agent is None:
            from gui_agents.s3.core.mllm import LMMAgent
            self.agent = LMMAgent(engine_params={'engine_type': 'vllm', 'model': self.model_id,
                                  'base_url': self.base_url, 'api_key': 'dummy', 'temperature': 0.0},
                                  system_prompt=SYSTEM_PROMPT)
            # Upstream LMMEnginevLLM drops stop/top_k/thinking kwargs; one audited
            # transport call is required to preserve the native request contract.
            self.agent.engine.generate = self._generate
        self.agent.reset()
        self.agent.messages.append({'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}},
            {'type': 'text', 'text': user_text}]})
        before = len(self.calls)
        answer = self.agent.get_response(temperature=0.0, max_new_tokens=self.max_tokens)
        if len(self.calls) != before + 1:
            raise RuntimeError('Expected exactly one model request per decision')
        call = self.calls[-1]
        if '{' in answer and '}' not in answer:
            answer += '}'  # Native restores the trimmed stop string too.
        choice = call['response']['choices'][0]
        return RawInvocation(parsed=parse_native_action(answer),
            invocation_id=f"agents3-matched-{call['start_ns']}", detail={
                'model_reported': self.model_id, 'finish_reason': choice.get('finish_reason'),
                'input_tokens': call['response']['usage']['prompt_tokens'],
                'output_tokens': call['response']['usage']['completion_tokens'],
                'raw_text': answer, 'planner_calls': 1, 'reflection_calls': 0,
                'grounder_calls': 0, 'updater_calls': 0, 'history_turns': 0,
                'history_representation': 'none', 'retained_history_images': 0,
                'request_contract_sha256': contract_fingerprint(),
                'model_call_request_sha256': [call['request_sha256']], **self._image_audit})
