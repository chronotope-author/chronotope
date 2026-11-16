"""AOI visual adaptation using the authors' extractor and observation records.

The harness buffers 3 Hz published images during blocking model calls. CLIP
processing runs at the next decision and its cost remains on the measured path.
This is explicitly buffered processing, not concurrent GPU perception. The core
five tasks have no audio stream. No DOM or target metadata enters the model.
"""
from __future__ import annotations

import json

from .common import (InvalidOutput, PixelController, action_schema, parse_command,
                     pil_image, strict_json, user_message, validate_action)
from .upstream import load_module


class AOIController(PixelController):
    def __init__(self, *, source_root, **kwargs):
        super().__init__(**kwargs)
        keyframes = load_module(source_root, 'aoi', 'aoi/keyframe_extractor.py')
        self.records = load_module(source_root, 'aoi', 'aoi/observation_record.py')
        self.cu = load_module(source_root, 'aoi', 'aoi/cu_model.py')
        self.extractor = keyframes.KeyframeExtractor(
            device=self.config.get('clip_device', 'cpu'), theta=0.04,
            pixel_threshold=0.01, pixel_capture_threshold=0.03, max_keyframes=5)
        # Model loading is setup, like the already-running planner services.
        # Keyframe encoding itself remains inside the measured trial.
        if hasattr(self.extractor, '_load_clip'):
            self.extractor._load_clip()
        self.trajectory = self.records.TrajectoryStore(context_depth=3, screenshot_history=5)
        self.sample_period_ns = 333_333_333
        self.next_sample_ns = None
        self.buffer = []
        self.step = 0

    def buffer_observation(self, obs):
        """Opt-in runner hook, before latest-frame coalescing. No model work here."""
        if self.next_sample_ns is None:
            self.next_sample_ns = obs.publication_time_ns
        if obs.publication_time_ns < self.next_sample_ns:
            return False
        self.buffer.append(obs)
        missed = (obs.publication_time_ns - self.next_sample_ns) // self.sample_period_ns
        self.next_sample_ns += (missed + 1) * self.sample_period_ns
        return True

    def decide(self):
        obs = self.newest
        self.last_decided_id = obs.observation_id
        calls_start = len(self.model.calls)
        start = self.clock()
        self.step += 1
        buffered, self.buffer = self.buffer, []
        for sample in buffered:
            self.extractor.on_sample(pil_image(sample, self.width, self.height),
                                     sample.publication_time_ns / 1e9)
        keyframes = self.extractor.get_and_reset()
        current = pil_image(obs, self.width, self.height)
        record = self.records.ObservationRecord(
            step_id=self.step, context_steps=self.trajectory.get_context(self.step),
            current_audio_text='', current_audio_context='', keyframes=keyframes,
            post_action_screenshot=current, task_instruction=self.instruction)
        message = user_message(record.to_prompt_text(), record.get_images())
        structured = self.config.get('structured_actions', False)
        instruction = self.cu.NARRATION_INSTRUCTION
        if structured:
            instruction = instruction.split('Format your response exactly as:')[0]
            instruction = instruction.replace('(e.g. click(640, 360))', '')
            instruction += (
                f'\nThe actual screenshot is {self.width}x{self.height} pixels. '
                f'Valid x is 0..{self.width-1}; valid y is 0..{self.height-1}. '
                'Ground coordinates in the current image. If no target is visible, wait. '
                'The transport requires JSON with narration and action fields. '
                'Action uses kind=click with x,y, kind=wait, or kind=type with value. '
                'Do not copy example coordinates. Keep narration to one short sentence.')
        message['content'].append({'type': 'text', 'text': f'Task: {self.instruction}{instruction}'})
        schema = None
        if structured:
            schema = {'type': 'object', 'properties': {
                'narration': {'type': 'string', 'maxLength': 160},
                'action': action_schema(self.width, self.height)},
                'required': ['narration', 'action'], 'additionalProperties': False}
        output = self.model.complete([message], max_tokens=self.config.get('max_tokens', 1024),
                                     temperature=self.config.get('temperature', 0.0), schema=schema)
        raw = output.get('content')
        if not isinstance(raw, str):
            raw = ''
        if structured:
            try:
                value = strict_json(raw)
                if (not isinstance(value, dict) or set(value) != {'narration', 'action'}
                        or not isinstance(value['narration'], str)):
                    raise InvalidOutput('Invalid narration/action object')
                action = validate_action(value['action'], self.width, self.height)
                command = ('wait()' if action is None else
                           f"click({action['x']}, {action['y']})" if action['kind'] == 'click' else
                           f"type({json.dumps(action['value'])})")
                # Keep the authors' narration parser and trajectory representation.
                raw = f"NARRATION: {value['narration']}\nACTION: {command}"
            except InvalidOutput as exc:
                return self.result(None, outcome=f'invalid_output: {exc}', calls_start=calls_start,
                                   observation_id=obs.observation_id)
        parsed = self.cu._parse_output(raw)
        self.trajectory.append(self.step, start / 1e9, self.clock() / 1e9,
                               '', parsed.narration, parsed.action, len(keyframes), False,
                               screenshot=current)
        try:
            action = parse_command(parsed.action, self.width, self.height)
            outcome = 'action' if action else 'wait'
        except InvalidOutput as exc:
            action, outcome = None, f'invalid_output: {exc}'
        return self.result(action, outcome=outcome, calls_start=calls_start,
            observation_id=obs.observation_id,
            sampled_observation_ids=[o.observation_id for o in buffered],
            keyframes=len(keyframes), keyframe_stats=self.extractor.get_stats(),
            adaptation='AOI visual; buffered CLIP processing; no audio or DOM')
