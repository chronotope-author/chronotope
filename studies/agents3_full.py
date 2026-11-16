"""Historical Agent S3 GUI policy with measured local model transports.

The installed AgentS3, Worker, OSWorldACI, prompts, image history, reflection,
action formatting and retry loops are used unchanged. As in agents3_service.py,
env=None disables the optional code-execution tool. Returned PyAutoGUI strings
are parsed by the existing ChronotopeGym adapter; this module never executes
those strings on the desktop. Component generation overrides are explicit in
the experiment configuration and every HTTP attempt is recorded.
"""
import copy
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time


UPSTREAM_MODULES = (
    'gui_agents.s3.agents.agent_s', 'gui_agents.s3.agents.worker',
    'gui_agents.s3.agents.grounding', 'gui_agents.s3.agents.code_agent',
    'gui_agents.s3.core.module', 'gui_agents.s3.core.engine',
    'gui_agents.s3.core.mllm', 'gui_agents.s3.memory.procedural_memory',
    'gui_agents.s3.utils.common_utils', 'gui_agents.s3.utils.formatters',
)


def source_fingerprints():
    paths = [Path(__file__).resolve()]
    paths.extend(Path(importlib.import_module(name).__file__).resolve()
                 for name in UPSTREAM_MODULES)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def request_audit(body):
    """Retain all model text and image metadata without duplicating PNG bytes."""
    audit = copy.deepcopy(body)
    for message in audit.get('messages', []):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get('type') == 'image_url':
                url = part['image_url'].pop('url')
                part['image_url'].update(
                    sha256=hashlib.sha256(url.encode()).hexdigest(), encoded_bytes=len(url))
    return audit


def image_count(messages):
    return sum(part.get('type') == 'image_url' for message in messages
               if isinstance(message.get('content'), list)
               for part in message['content'])


def grounding_geometry(height, width):
    """The UI-TARS factor-28 smart resize used by historical agents3_service."""
    factor, min_pixels, max_pixels = 28, 3136, 12845056
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class FullAgentS3Adapter:
    def __init__(self, config, log, *, _transport_factory=None):
        self.config = copy.deepcopy(config)
        self.component_config_sha256 = hashlib.sha256(json.dumps(
            self.config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if config.get('schema') != 'full-agents3-components-1':
            raise ValueError('Expected full-agents3-components-1')
        self.components = self.config['components']
        if set(self.components) != {'planner', 'reflection', 'grounder', 'text_grounder'}:
            raise ValueError('Expected planner, reflection, grounder and text_grounder components')
        policy = self.config['policy']
        expected = {'upstream_worker_prompt': True, 'upstream_aci': True,
                    'history_images_retained': True, 'reflection': True,
                    'direct_pixel_prompt_rewrite': False, 'history_image_stripping': False,
                    'code_agent_enabled': False, 'code_agent_disabled_reason': 'upstream_default',
                    'history_n': 8}
        if any(policy.get(key) != value for key, value in expected.items()):
            raise ValueError('Configuration must retain the historical upstream GUI policy')
        planner = self.components['planner']
        for role, component in self.components.items():
            self._validate_runtime(role, component)
            for key in ('model', 'revision', 'precision', 'base_url'):
                if not component.get(key):
                    raise ValueError(f'Missing {role}.{key}')
            if role != 'grounder' and component['model'] != planner['model']:
                raise ValueError('Reflection and text grounding must use the planner checkpoint')
        self.model_id = planner['model']
        self.log = Path(log)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.agent = None
        self.aci = None
        self.calls = []
        self.errors = []
        self._clients = []
        self._transport_factory = _transport_factory

    def _validate_runtime(self, role, component):
        if component.get('runtime') != 'vLLM':
            raise ValueError(f'Unsupported component runtime: {role}')

    def spec_fields(self):
        return {'server': 'agents3_service', 'model_id': self.model_id,
                'implementation': 'upstream AgentS3 Worker and OSWorldACI; historical GUI configuration',
                'reflection': True, 'grounder': self.components['grounder']['model'],
                'history_representation': 'CPU text and screenshots', 'history_n': 8,
                'optional_code_execution': 'disabled upstream default'}

    def _finish_call(self, record):
        record['end_ns'] = time.monotonic_ns()
        if record.get('error'):
            self.errors.append(record['error'])
        self.calls.append(record)
        with self.log.open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')

    def _attach_transport(self, role, agent):
        import httpx
        from openai import OpenAI

        owner = self
        component = self.components[role]
        if agent.engine.model != component['model']:
            raise ValueError(f'Upstream engine model differs from configured {role}')
        generation = self.config['grounder_generation' if role == 'grounder' else 'generation']
        inner = self._transport_factory(role) if self._transport_factory else httpx.HTTPTransport()

        class RecordingTransport(httpx.BaseTransport):
            def handle_request(self, request):
                payload = request.read()
                body = json.loads(payload)
                record = {'role': role, 'model': component['model'],
                          'revision': component['revision'], 'start_ns': time.monotonic_ns(),
                          'component_config_sha256': owner.component_config_sha256,
                          'request_sha256': hashlib.sha256(payload).hexdigest(),
                          'request': request_audit(body)}
                try:
                    response = inner.handle_request(request)
                    response.read()
                    record['http_status'] = response.status_code
                    try:
                        record['response'] = response.json()
                    except ValueError:
                        record['response_text'] = response.text
                        raise ValueError('Model service returned non-JSON content')
                    if response.is_error:
                        record['error'] = f'HTTP {response.status_code} from {role} model service'
                    else:
                        result = record['response']
                        usage = result.get('usage')
                        if result.get('model') != component['model']:
                            raise ValueError(f'{role} service returned a different model identity')
                        if not isinstance(usage, dict) or any(
                            type(usage.get(key)) is not int or usage[key] < 0
                            for key in ('prompt_tokens', 'completion_tokens')
                        ):
                            raise ValueError(f'{role} response lacks measured token usage')
                        choices = result.get('choices') or []
                        if not choices or not isinstance(choices[0].get('message', {}).get('content'), str):
                            raise ValueError(f'{role} response has no text content')
                    return response
                except Exception as exc:
                    record['error'] = f'{type(exc).__name__}: {exc}'
                    raise
                finally:
                    owner._finish_call(record)

            def close(self):
                inner.close()

        # Preserve OpenAI SDK retries and the upstream engine's backoff wrapper.
        # The transport sees every individual HTTP attempt, including retries.
        client = OpenAI(base_url=component['base_url'], api_key='dummy',
                        http_client=httpx.Client(transport=RecordingTransport()))
        self._clients.append(client)

        def create(**kwargs):
            # The only generation changes are the explicit experiment settings.
            for key in ('max_tokens', 'temperature', 'top_p'):
                if key in generation:
                    kwargs[key] = generation[key]
            extra = dict(kwargs.get('extra_body') or {})
            if 'repetition_penalty' in generation:
                extra['repetition_penalty'] = generation['repetition_penalty']
            if 'enable_thinking' in generation:
                extra['chat_template_kwargs'] = {
                    **extra.get('chat_template_kwargs', {}),
                    'enable_thinking': generation['enable_thinking']}
            kwargs['extra_body'] = extra
            return client.chat.completions.create(**kwargs)

        agent.engine.llm_client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))

    def _initialize(self, png):
        from PIL import Image
        from gui_agents.s3.agents.agent_s import AgentS3
        from gui_agents.s3.agents.grounding import OSWorldACI

        with Image.open(io.BytesIO(png)) as image:
            self.width, self.height = image.size
        gh, gw = grounding_geometry(self.height, self.width)
        planner = self.components['planner']
        grounder = self.components['grounder']
        gen = {'engine_type': 'vllm', 'model': planner['model'],
               'base_url': planner['base_url'], 'api_key': 'dummy'}
        ground = {'engine_type': 'vllm', 'model': grounder['model'],
                  'base_url': grounder['base_url'], 'api_key': 'dummy',
                  'grounding_width': gw, 'grounding_height': gh}
        self.aci = OSWorldACI(env=None, platform='linux', engine_params_for_generation=gen,
                              engine_params_for_grounding=ground,
                              width=self.width, height=self.height)
        self.agent = AgentS3(gen, self.aci, platform='linux',
                             max_trajectory_length=8, enable_reflection=True)
        self.upstream_worker_template_sha256 = hashlib.sha256(
            self.agent.executor.generator_agent.system_prompt.encode()).hexdigest()
        self.grounding_wh = [gw, gh]
        for role, agent in (
            ('planner', self.agent.executor.generator_agent),
            ('reflection', self.agent.executor.reflection_agent),
            ('grounder', self.aci.grounding_model),
            ('text_grounder', self.aci.text_span_agent),
        ):
            self._attach_transport(role, agent)

    def invoke(self, png, system, user_text, structured=False, schema=None, parser=None):
        from PIL import Image
        from chronotope.controller import RawInvocation, parse_agents3

        if self.agent is None:
            self._initialize(png)
        with Image.open(io.BytesIO(png)) as image:
            if image.size != (self.width, self.height):
                raise ValueError('Screenshot geometry changed during the trial')
        start = time.monotonic_ns()
        before = len(self.calls)
        # Exact historical service instruction and upstream observation shape.
        info, actions = self.agent.predict(instruction=system, observation={'screenshot': png})
        own = self.calls[before:]
        if self.errors:
            raise RuntimeError('Inference service error: ' + self.errors[-1])
        if not own:
            raise RuntimeError('Agent S3 invocation made no measured model calls')
        worker = self.agent.executor
        details = {
            'model_reported': self.model_id, 'finish_reason': 'stop',
            'component_config_sha256': self.component_config_sha256,
            'upstream_worker_template_sha256': self.upstream_worker_template_sha256,
            'upstream_worker_prompt_sha256': hashlib.sha256(
                worker.generator_agent.system_prompt.encode()).hexdigest(),
            'input_tokens': sum(call['response']['usage']['prompt_tokens'] for call in own),
            'output_tokens': sum(call['response']['usage']['completion_tokens'] for call in own),
            'model_calls': len(own),
            **{role + '_calls': sum(call['role'] == role for call in own)
               for role in self.components},
            'code_agent_calls': 0, 'updater_calls': 0,
            'component_models': {role: comp['model'] for role, comp in self.components.items()},
            'history_representation': 'CPU text and screenshots',
            'history_turns': worker.turn_count,
            'retained_history_images': image_count(worker.generator_agent.messages),
            'retained_reflection_images': image_count(worker.reflection_agent.messages),
            'optional_code_execution': 'disabled upstream default',
            'model_call_request_sha256': [call['request_sha256'] for call in own],
            'agent_elapsed_ns': time.monotonic_ns() - start,
            'plan_code': info.get('plan_code'), 'raw_text': '\n'.join(actions),
            'screen_wh': [self.width, self.height], 'grounding_wh': self.grounding_wh,
        }
        # This is the existing benchmark action bridge, including its historical
        # click/wait grammar. It never executes the returned desktop code.
        return RawInvocation(parsed=parse_agents3(actions),
                             invocation_id=f'agents3-full-{start}', detail=details)

    def close(self):
        for client in self._clients:
            client.close()
        self._clients.clear()
