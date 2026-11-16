"""Released MobileDreamer candidate/one-step world-model selection, host adapted.

The release's Android superclass is absent. We load its unchanged helper methods
from hash-locked source and supply pixel/task transport plumbing. The release
does one-step selection; this is not the paper's depth-two rollout experiment.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
import logging
import re
import time

import numpy as np
from PIL import Image

from .common import (InvalidOutput, PixelController, SetupError, pil_image,
                     strict_json, user_message, validate_action)
from .upstream import load_methods, load_module

PREFIX = 'rollout_imagination_for_GUI_agent/'
METHODS = ('_generate_candidate_actions', '_action_to_string',
           '_select_best_action_with_world_model', '_select_with_reasoner',
           '_build_multimodal_message', '_predict_single_action_outcome')

# Harness action vocabulary; the algorithm itself is loaded from the release.
ACTION_TOOLS = [{'type': 'function', 'function': {'name': 'computer_use',
    'description': 'Interact with the supplied screenshot using its pixel coordinates.',
    'parameters': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['click', 'type', 'wait']},
        'coordinate': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2, 'maxItems': 2},
        'text': {'type': 'string'}}, 'required': ['action'], 'additionalProperties': False}}}]


def action_tools(width, height):
    def branch(properties):
        return dict(type='object', properties=properties, required=list(properties),
                    additionalProperties=False)
    result = deepcopy(ACTION_TOOLS)
    result[0]['function']['parameters'] = {'oneOf': [
        branch({'action': {'const': 'wait'}}),
        branch({'action': {'const': 'click'}, 'coordinate': {
            'type': 'array', 'prefixItems': [
                {'type': 'integer', 'minimum': 0, 'maximum': width - 1},
                {'type': 'integer', 'minimum': 0, 'maximum': height - 1}],
            'items': False, 'minItems': 2, 'maxItems': 2}}),
        branch({'action': {'const': 'type'},
                'text': {'type': 'string', 'minLength': 1, 'maxLength': 256}})]}
    return result


class ModelBridge:
    def __init__(self, model, config):
        self.model, self.config = model, config

    def call_llm(self, chat, tools=None):
        if tools:
            if len(tools) != 1 or tools[0]['function']['name'] != 'computer_use':
                raise SetupError('MobileDreamer bridge expects the single harness action tool')
            # The unchanged accepted planner has no tool-call parser. Request
            # exactly the same action arguments via stock JSON-schema decoding,
            # then adapt the result to the upstream helper's tool-call envelope.
            messages = deepcopy(chat)
            messages.append({'role': 'user', 'content':
                'Return only the JSON arguments for computer_use, following the supplied schema. '
                'Use pixel coordinates from the screenshot. Choose wait when no action is appropriate.'})
            message = self.model.complete(messages, max_tokens=self.config.get('max_tokens', 1024),
                temperature=self.config.get('temperature', 0.0), schema=tools[0]['function']['parameters'])
            if 'tool_calls' not in message:
                try:
                    arguments = strict_json(message.get('content'))
                    if not isinstance(arguments, dict):
                        raise InvalidOutput('Expected tool argument object')
                    message = {'role': 'assistant', 'content': '', 'tool_calls': [{
                        'id': 'harness_action', 'type': 'function', 'function': {
                            'name': 'computer_use', 'arguments': json.dumps(arguments)}}]}
                except InvalidOutput:
                    # The upstream algorithm sees no candidate; this remains a
                    # scored model-format failure, never a transport exception.
                    pass
        else:
            message = self.model.complete(chat, max_tokens=self.config.get('max_tokens', 1024),
                temperature=self.config.get('temperature', 0.0))
        return message, True, message


class WorldBridge:
    def __init__(self, model, source_root, config):
        self.model, self.config = model, config
        cls = load_methods(source_root, 'mobiledreamer', PREFIX + 'world_model_api/vllm_client.py',
            'WorldModelVLLMClient', ('_build_prompt',), {}, ('WORLD_MODEL_PROMPT_TEMPLATE',))
        self.prompt_builder = cls()

    def predict_next_state(self, current_screen_ocr, action):
        prompt = self.prompt_builder._build_prompt(current_screen_ocr, action)
        message = self.model.complete([{'role': 'user', 'content': prompt}],
            max_tokens=self.config.get('max_tokens', 4096), temperature=self.config.get('temperature', 0.0))
        value = message.get('content')
        if not isinstance(value, str):
            return '[Invalid world-model output]'
        return value


class CheckedOCR:
    """Upstream logs and swallows some inference errors; surface them as setup errors."""
    def __init__(self, delegate):
        self.delegate = delegate

    def image_to_text_description(self, screenshot):
        errors = []
        class Capture(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    errors.append(record.getMessage())
        handler = Capture()
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            result = self.delegate.image_to_text_description(screenshot)
        finally:
            logger.removeHandler(handler)
        if errors:
            raise SetupError('Upstream OCR error: ' + '; '.join(errors))
        if not isinstance(result, str):
            raise SetupError('Upstream OCR did not return a textual sketch')
        return result


def mobile_action(message, width, height):
    calls = message.get('tool_calls', [])
    if len(calls) != 1 or calls[0].get('function', {}).get('name') != 'computer_use':
        raise InvalidOutput('Expected exactly one computer_use call')
    action = strict_json(calls[0]['function']['arguments'])
    if not isinstance(action, dict):
        raise InvalidOutput('Expected a mobile action object')
    # The released tool vocabulary permits irrelevant coordinate/text fields.
    # A wait remains a wait; these fields never become an executable action.
    if action.get('action') == 'wait' and set(action) <= {'action', 'coordinate', 'text'}:
        return None
    if set(action) == {'action', 'coordinate'} and action['action'] == 'click':
        xy = action['coordinate']
        if isinstance(xy, list) and len(xy) == 2:
            return validate_action(dict(kind='click', x=xy[0], y=xy[1]), width, height)
    if set(action) == {'action', 'text'} and action['action'] == 'type':
        return validate_action(dict(kind='type', value=action['text']), width, height)
    raise InvalidOutput('Unsupported mobile action')


class MobileDreamerController(PixelController):
    def __init__(self, *, source_root, world_model, ocr=None, **kwargs):
        super().__init__(**kwargs)
        self.world_model = world_model
        cls = load_methods(source_root, 'mobiledreamer',
            PREFIX + 'world_model_guided/world_model_guided_agent.py', 'WorldModelGuidedAgent', METHODS,
            dict(deepcopy=deepcopy, logging=logging, json=json, re=re, time=time,
                 Image=Image, BytesIO=BytesIO, base64=base64, np=np))
        self.engine = cls()
        self.engine.num_candidate_actions = 3
        self.engine.actor_model = ModelBridge(self.model, self.config)
        self.engine.reasoenr_model = self.engine.actor_model  # spelling in upstream
        self.engine.world_model = WorldBridge(world_model, source_root, self.config.get('world_generation', {}))
        if ocr is None:
            from .ocr import PaddleOCRBridge
            ocr = PaddleOCRBridge(source_root, self.config.get('ocr', {}), self.config.get('ocr_runtime'))
            ocr.initialize()  # load assets before the trial clock; no image inference
        self.engine.ocr_module = CheckedOCR(ocr)
        self.engine.chat = []
        self.engine.actor_chat = []

    def spec_hash(self):
        from ..hashing import hash_obj
        return hash_obj({'controller': super().spec_hash(), 'world_model': self.world_model.spec_fields()})

    def decide(self):
        obs = self.newest
        self.last_decided_id = obs.observation_id
        calls_start, world_start = len(self.model.calls), len(self.world_model.calls)
        errors_start = (len(self.model.errors), len(self.world_model.errors))
        image = pil_image(obs, self.width, self.height)
        screenshot = np.asarray(image)
        prompt = (f'Task: {self.instruction}\nScreenshot size: {self.width}x{self.height}. '
                  'Propose one computer_use action toward this task. Use only the screenshot.')
        self.engine.actor_chat = [user_message(prompt, [image])]
        self.engine.chat = []
        data = {'action_prompt': self.instruction}
        # Extract outside upstream catch-and-fallback: infrastructure errors must
        # not become scored failures. Cache exactly this sketch for its helpers.
        sketch = self.engine.ocr_module.image_to_text_description(screenshot)
        original_ocr = self.engine.ocr_module
        class CachedSketch:
            def image_to_text_description(self, image):
                return sketch
        self.engine.ocr_module = CachedSketch()
        candidates = self.engine._generate_candidate_actions(self.instruction, self.instruction,
            screenshot, (self.width, self.height), action_tools(self.width, self.height), data)
        chosen = None
        try:
            if len(candidates) > 1:
                chosen, _ = self.engine._select_best_action_with_world_model(
                    self.instruction, screenshot, screenshot, candidates, data)
            elif candidates:
                chosen = candidates[0]
                _, prediction = self.engine._predict_single_action_outcome(self.instruction, screenshot,
                    self.engine._action_to_string(chosen), data)
                data.update(wm_predictions=[prediction], selected_action_idx=0)
        finally:
            self.engine.ocr_module = original_ocr
        # Upstream catches transport errors and can fall back to candidate zero.
        # Preserve algorithm output, but never score a case with failed infrastructure.
        if (len(self.model.errors), len(self.world_model.errors)) != errors_start:
            raise SetupError('MobileDreamer transport failure; retain this trial unscored')
        try:
            if chosen is None:
                raise InvalidOutput('No candidate action')
            action = mobile_action(chosen, self.width, self.height)
            outcome = 'action' if action else 'wait'
        except InvalidOutput as exc:
            action, outcome = None, f'invalid_output: {exc}'
        return self.result(action, outcome=outcome, calls_start=calls_start,
            observation_id=obs.observation_id, rollout=data,
            world_model_calls=deepcopy(self.world_model.calls[world_start:]),
            adaptation='Released one-step M=3 selection; harness tool schema; direct task instead of Android reasoner')
