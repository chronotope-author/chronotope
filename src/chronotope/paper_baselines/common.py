"""Pixel-only baseline boundary and audited local chat-completions transport."""
from __future__ import annotations

import ast
import base64
import copy
from dataclasses import dataclass, asdict
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.request

from ..controller import ControlUpdate, Decision
from ..hashing import hash_obj
from ..png import to_rgb

TASK_NAMES = {
    'notification': 'Disappearing Notification', 'drift': 'Moving Balloon',
    'code': 'Expiring Authenticator Code', 'button': 'Displaced Button',
    'trading': 'Reversing Trading Signal',
}


class SetupError(RuntimeError):
    """An unavailable dependency/transport, never a scored competence failure."""


class InvalidOutput(ValueError):
    """A model-format failure, retained as a no-action decision."""


def strict_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise InvalidOutput(f'Duplicate key: {k}')
            result[k] = v
        return result
    def invalid_constant(value):
        raise InvalidOutput(f'Non-finite JSON: {value}')
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        raise InvalidOutput(str(exc)) from exc


def validate_action(action, width, height):
    if not isinstance(action, dict):
        raise InvalidOutput('Expected an action object')
    if action == {'kind': 'wait'}:
        return None
    if set(action) == {'kind', 'x', 'y'} and action['kind'] == 'click':
        x, y = action['x'], action['y']
        if type(x) is int and type(y) is int and 0 <= x < width and 0 <= y < height:
            return dict(action)
    if set(action) == {'kind', 'value'} and action['kind'] == 'type':
        if isinstance(action['value'], str) and 0 < len(action['value']) <= 4096:
            return dict(action)
    raise InvalidOutput('Unsupported action or invalid pixel coordinates')


def action_schema(width, height):
    """A discriminated pixel action; no example coordinates or target metadata."""
    def branch(properties):
        return dict(type='object', properties=properties, required=list(properties),
                    additionalProperties=False)
    return {'oneOf': [
        branch({'kind': {'const': 'wait'}}),
        branch({'kind': {'const': 'click'},
                'x': {'type': 'integer', 'minimum': 0, 'maximum': width - 1},
                'y': {'type': 'integer', 'minimum': 0, 'maximum': height - 1}}),
        branch({'kind': {'const': 'type'},
                'value': {'type': 'string', 'minLength': 1, 'maxLength': 256}})]}


def parse_command(command, width, height):
    """Parse scalar paper actions without executing generated code."""
    if command in ('WAIT', 'wait', 'wait()', 'done'):
        return None
    try:
        node = ast.parse(command, mode='eval').body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError('Not a scalar call')
        args = [ast.literal_eval(a) for a in node.args]
        if node.func.id == 'click' and len(args) == 2:
            return validate_action(dict(kind='click', x=args[0], y=args[1]), width, height)
        if node.func.id == 'type' and len(args) == 1:
            return validate_action(dict(kind='type', value=args[0]), width, height)
    except InvalidOutput:
        raise
    except (ValueError, SyntaxError, TypeError) as exc:
        raise InvalidOutput('Invalid action command') from exc
    raise InvalidOutput('Only click, type, and wait are supported')


def pil_image(obs, width, height):
    from PIL import Image
    raw = to_rgb(obs.payload, width, height)
    mode = 'L' if len(raw) == width * height else 'RGB'
    return Image.frombytes(mode, (width, height), raw).convert('RGB')


def user_message(text, images=()):
    content = [{'type': 'text', 'text': text}]
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}})
    return {'role': 'user', 'content': content}


@dataclass(frozen=True)
class ModelIdentity:
    model: str
    revision: str
    source_precision: str
    runtime: str
    precision: str
    kv_cache_precision: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise SetupError('Every model identity and precision field must be explicit')


class ChatModel:
    """No automatic retries, model substitutions, or service management.

    Identity describes the configured endpoint; service launch evidence must verify
    the revision/precision before an evaluation. HTTP cannot prove those values.
    """
    def __init__(self, identity, base_url, *, timeout_s=120, api_key_env=None,
                 extra_body=None, transport=None):
        self.identity = identity
        self.base_url = base_url.rstrip('/')
        self.timeout_s = timeout_s
        self.api_key_env = api_key_env
        self.extra_body = copy.deepcopy(extra_body or {})
        if set(self.extra_body) & {'model', 'messages', 'max_tokens', 'temperature', 'tools', 'response_format'}:
            raise SetupError('extra_body cannot override the model request')
        self.transport = transport or self._post
        self.calls = []
        self.errors = []

    def spec_fields(self):
        return dict(identity=asdict(self.identity), base_url=self.base_url,
                    timeout_s=self.timeout_s, extra_body=self.extra_body)

    def _post(self, body):
        headers = {'Content-Type': 'application/json'}
        if self.api_key_env:
            headers['Authorization'] = 'Bearer ' + os.environ[self.api_key_env]
        req = urllib.request.Request(self.base_url + '/chat/completions',
                                     json.dumps(body, allow_nan=False).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(8192).decode('utf-8', errors='replace')
            raise SetupError(f'HTTP {exc.code}: {detail}') from exc

    def complete(self, messages, *, max_tokens, temperature, schema=None, tools=None):
        body = dict(self.extra_body, model=self.identity.model, messages=copy.deepcopy(messages),
                    max_tokens=max_tokens, temperature=temperature)
        if schema is not None:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'baseline_output', 'strict': True, 'schema': schema}}
            # Stock vLLM per-request setting: prevent unbounded whitespace from
            # consuming the token budget while still obeying the JSON grammar.
            body['structured_outputs'] = dict(body.get('structured_outputs', {}),
                                              json=schema, disable_any_whitespace=True)
        if tools is not None:
            body['tools'] = tools
            # Endpoints using this path must provide a tool parser. MobileDreamer
            # uses the JSON-schema bridge below its upstream algorithm instead,
            # preserving the existing planner endpoint without a tool parser.
            body['tool_choice'] = 'required'
            body['parallel_tool_calls'] = False
        start = time.monotonic_ns()
        record = {'request': body, 'identity': asdict(self.identity)}
        try:
            response = self.transport(body)
            message = response['choices'][0]['message']
            record.update(response=response)
            return message
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            self.errors.append(record['error'])
            raise SetupError(record['error']) from exc
        finally:
            record['wall_ns'] = time.monotonic_ns() - start
            # Keep text byte-for-byte; images are already in the harness payload store.
            record = copy.deepcopy(record)
            for message in record['request']['messages']:
                if isinstance(message.get('content'), list):
                    for part in message['content']:
                        if part.get('type') == 'image_url':
                            url = part['image_url'].pop('url')
                            part['image_url']['url_sha256'] = hashlib.sha256(url.encode()).hexdigest()
                            part['image_url']['encoded_bytes'] = len(url)
            self.calls.append(record)


class PixelController:
    def __init__(self, *, task, instruction, width, height, config, model):
        if task not in TASK_NAMES or not instruction.strip() or min(width, height) <= 0:
            raise SetupError('Expected a core-five task, public instruction, and valid geometry')
        self.task, self.instruction = task, instruction
        self.width, self.height = width, height
        self.config, self.model = copy.deepcopy(config), model
        self.newest = None
        self.last_decided_id = None
        self.clock = time.monotonic_ns

    def bind_clock(self, clock):
        self.clock = clock

    def spec_hash(self):
        return hash_obj(dict(kind=type(self).__name__, task=self.task, instruction=self.instruction,
                             width=self.width, height=self.height, config=self.config,
                             model=self.model.spec_fields()))

    def admit(self, obs):
        changed = self.newest is None or self.newest.payload_hash != obs.payload_hash
        self.newest = obs
        return ControlUpdate('action' if changed else 'none', hash_obj({'pixels': obs.payload_hash}))

    def ready_to_decide(self):
        return self.newest is not None and self.newest.observation_id != self.last_decided_id

    def result(self, action, *, outcome, calls_start, **detail):
        return Decision(action, 0, invocation_id=type(self).__name__, invocation_detail={
            'baseline': type(self).__name__, 'outcome': outcome,
            'model_calls': copy.deepcopy(self.model.calls[calls_start:]), **detail})
