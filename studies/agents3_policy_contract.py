"""CPU-only contract/export and reference state machine for stripped Agent S3.

The exported prompts are produced from the installed Worker procedural memory
and the existing DirectPixelACI adapter. PolicyReference never runs inference:
its caller supplies deterministic fixture responses. Native C++ consumes the
JSON artifact and owns the equivalent orchestration; it must not call this
Python state machine at runtime.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import re
import textwrap

POLICY_SCHEMA = 'agents3-stripped-worker-policy-1'
INITIAL_USER_TEXT = 'The initial screen is provided. No action has been taken yet.'
TEXT_BUFFER_SUFFIX = '\nCurrent Text Buffer = []\n'
REFLECTION_PLANNER_PREFIX = 'REFLECTION: You may use this reflection on the previous action and overall trajectory:\n'
FEEDBACK_SINGLE_ACTION = 'Incorrect code: There must be a single agent action in the code response.'
FEEDBACK_INVALID_CODE = 'Incorrect code: The agent action must be a valid function and use valid parameters from the docstring list.'
CODE_PATTERN = r'```(?:\w+\s+)?(.*?)```'
FUNCTION_PATTERN = r'(agent\.\w+\(\s*.*\))'


def generation_settings(max_tokens=4096):
    """Actual request fields; no JSON stop string and no guided decoding."""
    return {'temperature': 0.0, 'max_tokens': max_tokens, 'top_p': 0.8,
            'repetition_penalty': 1.05, 'chat_template_kwargs': {'enable_thinking': False},
            'mm_processor_kwargs': {'min_pixels': 65536, 'max_pixels': 1048576}}


def policy_fingerprint(contract):
    body = {k: v for k, v in contract.items() if k != 'policy_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def build_policy_contract(width=640, height=480, *, base_instruction=None,
                          model_id='Qwen/Qwen3.5-0.8B', max_tokens=4096):
    from agents3_stripped import DirectPixelACI, direct_pixel_prompt
    from chronotope.controller import UITARS_INSTRUCTION
    from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
    from gui_agents.s3.agents import worker as worker_module
    from gui_agents.s3.utils import common_utils, formatters
    from gui_agents.s3.core import mllm
    if width <= 0 or height <= 0 or max_tokens != 4096:
        raise ValueError('Positive screenshot geometry and original 4096-token policy required')
    if base_instruction is None:
        base_instruction = UITARS_INSTRUCTION['notification']
    instruction = base_instruction + f' The screenshot is {width} pixels wide and {height} pixels high. Use absolute screenshot pixel coordinates.'
    planner_template = direct_pixel_prompt(PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
        DirectPixelACI, skipped_actions=['call_code_agent']).replace('CURRENT_OS', 'linux'))
    reflection_task = textwrap.dedent(f'''
                    Task Description: {instruction}
                    Current Trajectory below:
                    ''')
    source_hashes = {}
    for module in [worker_module, common_utils, formatters, mllm]:
        source = Path(inspect.getfile(module))
        source_hashes[module.__name__] = hashlib.sha256(source.read_bytes()).hexdigest()
    contract = {
        'schema': POLICY_SCHEMA, 'model': model_id, 'agent_s3_version': importlib.metadata.version('gui-agents'),
        'base_instruction': base_instruction, 'instruction': instruction,
        'screenshot_width': width, 'screenshot_height': height, 'platform': 'linux',
        'planner_system_template': planner_template,
        'planner_system': planner_template.replace('TASK_DESCRIPTION', instruction),
        'reflection_system_template': PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY,
        'reflection_system': PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY + '\n' + reflection_task,
        'initial_user_text': INITIAL_USER_TEXT, 'initial_reflection_user_text': INITIAL_USER_TEXT,
        'text_buffer_suffix': TEXT_BUFFER_SUFFIX, 'reflection_planner_prefix': REFLECTION_PLANNER_PREFIX,
        'feedback_single_action': FEEDBACK_SINGLE_ACTION, 'feedback_invalid_code': FEEDBACK_INVALID_CODE,
        'feedback_template': PROCEDURAL_MEMORY.FORMATTING_FEEDBACK_PROMPT,
        'history_limit': 8, 'planner_message_limit': 17, 'reflection_message_limit': 9,
        'history_flush': 'After each turn: if planner >17 remove messages[1:3] once; if reflection >9 remove messages[1] once',
        'history_images': 'Strip every previous image before and after each invocation; one current image per request',
        'message_content_order': ['text', 'image'], 'reflection_on_first_turn': False,
        'reflection_response_persisted_as_assistant': False,
        'retry_limit': 3, 'transport_retry_limit': 3, 'retry_sleep_seconds': 1.0,
        'format_retry_messages_persist': False,
        'max_tokens': max_tokens, 'temperature': 0.0, 'top_p': 0.8, 'repetition_penalty': 1.05,
        'enable_thinking': False, 'stop': [], 'generation_settings': generation_settings(max_tokens),
        'coordinate_space': 'screenshot_pixels', 'coordinate_rounding': 'Python round, ties to even; reject outside screenshot',
        'code_pattern': CODE_PATTERN, 'code_pattern_flags': ['DOTALL'], 'code_block_choice': 'last',
        'single_action_pattern': FUNCTION_PATTERN, 'single_action_pattern_flags': [],
        'allowed_actions': ['click', 'wait', 'done', 'fail'],
        'invalid_final_code': 'WAIT', 'done_fail': 'No application action; does not terminate the harness',
        'source_sha256': source_hashes,
    }
    if model_id == 'RadixArk/Qwen3.8-27B-NVFP4':
        # The exact Radix checkpoint's HF template retains historical thinking
        # blocks by default. This is a tokenizer/template difference, not a
        # change to Worker history or its enable_thinking=False generation.
        contract['native_chat_template'] = {
            'strip_historical_thinking': False,
            'assistant_prefix': '<|im_start|>assistant\n<think>\n\n</think>\n\n',
        }
    contract['policy_sha256'] = policy_fingerprint(contract)
    return contract


def normalize_messages(messages):
    """Keep exact text/order and replace image bytes with a semantic image marker."""
    normalized = []
    for message in messages:
        content = []
        for part in message['content']:
            if part['type'] == 'text':
                content.append({'type': 'text', 'text': part['text']})
            elif part['type'] in ('image_url', 'image'):
                content.append({'type': 'image'})
            else:
                raise ValueError('Unsupported content type ' + part['type'])
        normalized.append({'role': message['role'], 'content': content})
    return normalized


def parse_code(text):
    matches = re.findall(CODE_PATTERN, text.strip(), re.DOTALL)
    return matches[-1] if matches else ''


def split_reflection(text):
    thoughts = text.split('<thoughts>')[-1].split('</thoughts>')[0].strip()
    answer = text.split('<answer>')[-1].split('</answer>')[0].strip()
    return answer, thoughts


def validate_plan_response(text, width, height, png=None):
    """Return exact formatter errors and executable action using the safe adapter."""
    from PIL import Image
    from agents3_stripped import DirectPixelACI, safe_action_code
    if png is None:
        stream = io.BytesIO(); Image.new('RGB', (width, height)).save(stream, format='PNG'); png = stream.getvalue()
    code = parse_code(text)
    errors = []
    if len(re.findall(FUNCTION_PATTERN, code)) != 1:
        errors.append(FEEDBACK_SINGLE_ACTION)
    try:
        executable = safe_action_code(DirectPixelACI(width, height), code, {'screenshot': png})
    except (ValueError, TypeError, SyntaxError, AssertionError):
        executable = None
    if executable is None:
        errors.append(FEEDBACK_INVALID_CODE)
    return {'code': code, 'executable': executable, 'errors': errors}


def _message(role, text, image=False):
    parts = [{'type': 'text', 'text': text}]
    if image:
        parts.append({'type': 'image'})
    return {'role': role, 'content': parts}


class PolicyReference:
    """Deterministic CPU reference; responses supplied by callback, never a model."""
    def __init__(self, contract, sleep=lambda _: None):
        if contract['schema'] != POLICY_SCHEMA or policy_fingerprint(contract) != contract['policy_sha256']:
            raise ValueError('Policy contract mismatch')
        self.contract = contract
        self.sleep = sleep
        self.planner_messages = [_message('system', contract['planner_system'])]
        self.reflection_messages = [_message('system', contract['reflection_system'])]
        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.requests = []
        self.transport_errors = []

    def _strip_images(self):
        for messages in [self.planner_messages, self.reflection_messages]:
            for message in messages:
                message['content'] = [p for p in message['content'] if p['type'] == 'text']

    def _safe_call(self, respond, role, messages):
        response = ''
        for _ in range(self.contract['transport_retry_limit']):
            request = {'role': role, 'messages': copy.deepcopy(messages),
                       'generation_settings': copy.deepcopy(self.contract['generation_settings'])}
            self.requests.append(request)
            try:
                response = respond(role, copy.deepcopy(messages))
                if response is None:
                    raise AssertionError('Response from agent should not be None')
                break
            except Exception as exc:
                self.transport_errors.append(f'{type(exc).__name__}: {exc}')
            self.sleep(self.contract['retry_sleep_seconds'])
        return response if response is not None else ''

    def run_turn(self, respond, png=None):
        self._strip_images()
        c = self.contract
        reflection = reflection_thoughts = None
        if self.turn_count == 0:
            self.reflection_messages.append(_message('user', c['initial_reflection_user_text'], True))
            generator_text = c['initial_user_text']
        else:
            self.reflection_messages.append(_message('user', self.worker_history[-1], True))
            full = self._safe_call(respond, 'reflection', self.reflection_messages)
            reflection, reflection_thoughts = split_reflection(full)
            self.reflections.append(reflection)
            generator_text = ''
        if reflection:
            generator_text += c['reflection_planner_prefix'] + reflection + '\n'
        generator_text += c['text_buffer_suffix']
        self.planner_messages.append(_message('user', generator_text, True))
        transient_messages = self.planner_messages.copy()
        plan = ''
        for _ in range(c['retry_limit']):
            plan = self._safe_call(respond, 'planner', transient_messages)
            checked = validate_plan_response(plan, c['screenshot_width'], c['screenshot_height'], png)
            if not checked['errors']:
                break
            transient_messages.append(_message('assistant', plan))
            feedback = c['feedback_template'].replace('FORMATTING_FEEDBACK', '- ' + '\n- '.join(checked['errors']))
            transient_messages.append(_message('user', feedback))
            self.sleep(c['retry_sleep_seconds'])
        self.worker_history.append(plan)
        self.planner_messages.append(_message('assistant', plan))
        checked = validate_plan_response(plan, c['screenshot_width'], c['screenshot_height'], png)
        executable = checked['executable'] if checked['executable'] is not None else 'WAIT'
        self.turn_count += 1
        if len(self.planner_messages) > c['planner_message_limit']:
            del self.planner_messages[1:3]
        if len(self.reflection_messages) > c['reflection_message_limit']:
            del self.reflection_messages[1]
        self._strip_images()
        from chronotope.controller import parse_agents3
        result = {'plan': plan, 'plan_code': checked['code'], 'exec_code': executable,
                  'reflection': reflection, 'reflection_thoughts': reflection_thoughts,
                  'parsed': parse_agents3([executable])}
        if self.transport_errors:
            raise RuntimeError('Inference service error: ' + self.transport_errors[-1])
        return result


def build_cpu_fixtures(contract):
    """Fixed-response fixtures verified against Worker by the accompanying tests."""
    def code(text):
        return '```python\n' + text + '\n```'
    cases = {
        'reflection_and_format_retries': [
            [code('agent.wait(time=1.0)')],
            ['<thoughts>Check the screen.</thoughts><answer>Continue the task.</answer>',
             'No code here.', code('agent.click(x=-1,y=20)'), code('agent.click(x=320,y=240)')],
            ['Proceed.', code('agent.done()')],
        ],
        'retry_exhaustion': [
            ['no code', 'bad again', 'third malformed response'],
            ['Continue.', code('agent.wait()')],
        ],
        'history_twelve_turns': [[code('agent.wait()')]] + [
            [f'Continue step {i}.', code('agent.wait(time=0.5)')] for i in range(1, 12)],
        'historical_thinking_tags': [
            ['<think>Inspect the image.</think>\n\n' + code('agent.wait()')],
            ['<thoughts>Check.</thoughts><answer>Continue.</answer>', code('agent.wait(time=0.5)')],
            ['Proceed.', code('agent.done()')],
        ],
        'last_code_block_and_terminal': [
            [code('agent.wait()') + '\n' + code('agent.done()')],
            ['Task is complete.', code('agent.fail()')],
        ],
    }
    results = []
    for name, turns in cases.items():
        sleeps = []; state = PolicyReference(contract, sleep=sleeps.append); expected_turns = []
        for responses in turns:
            queue = list(responses)
            def respond(role, messages):
                return queue.pop(0)
            expected_turns.append(state.run_turn(respond))
            if queue:
                raise AssertionError('Unused fixture responses')
        results.append({'name': name, 'responses_by_turn': turns, 'expected_requests': state.requests,
            'expected_turns': expected_turns, 'sleep_seconds': sleeps,
            'final_planner_messages': state.planner_messages,
            'final_reflection_messages': state.reflection_messages})
    parser_inputs = [
        'agent.click(x=100,y=20)', 'agent.click(2.5,3.5)', 'agent.click(-0.49,-0.49)',
        'agent.click(639.5,479.5)', 'agent.click(x=1e2,y=2e1)', 'agent.click(0x64,0o24)',
        'agent.click(0b1100100,2_0)', 'agent.click((100),(20))', '(agent.click(x=100,y=20))',
        'agent.click(x=100,y=20) # a comment', 'agent.click(x=1,x=2,y=3)',
        'agent.click(True,20)', 'agent.click(x=+100,y=20)', 'agent.click(x=100,y=-(-20))',
        'agent.click((100,20))', 'agent.wait(-0.0)', 'agent.wait(time=-1)',
        'agent.wait(time=1+2)', 'agent.done()', 'agent.fail()', 'agent.click(1,2);agent.wait()',
        'agent.click(\n x=100,\n y=20\n)',
        'agent.click\n(100,20)', 'agent.\nclick(100,20)',
        'agent.click(x=-(+1),y=20)', 'agent.click(x=1e-9999,y=20)',
        'agent.click(x=1e9999,y=20)', 'agent.click(x=-0.5,y=20)',
        'agent.click(x=+(-1),y=20)', 'agent.click(x=1_0_0.,y=20)',
    ]
    parser_cases = [{'response': code(source), 'expected': validate_plan_response(
        code(source), contract['screenshot_width'], contract['screenshot_height'])} for source in parser_inputs]
    return {'policy_sha256': contract['policy_sha256'], 'experimental_trials': 0,
            'model_inference_calls': 0, 'cases': results, 'parser_cases': parser_cases}


def export_policy(path, **kwargs):
    contract = build_policy_contract(**kwargs)
    Path(path).write_text(json.dumps(contract, indent=2, ensure_ascii=False) + '\n')
    return contract


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--width', type=int, default=640); ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    ap.add_argument('--fixtures-out', type=Path)
    args = ap.parse_args()
    import sys
    candidate = Path(__file__).resolve().parent / 'src'
    if not candidate.exists(): candidate = Path(__file__).resolve().parents[1] / 'src'
    sys.path.insert(0, str(candidate))
    c = export_policy(args.out, width=args.width, height=args.height, model_id=args.model)
    if args.fixtures_out:
        args.fixtures_out.write_text(json.dumps(build_cpu_fixtures(c), indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'path': str(args.out), 'policy_sha256': c['policy_sha256'], 'experimental_trials': 0}))
