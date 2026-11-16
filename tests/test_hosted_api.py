"""CPU-only contract fixtures. No provider calls, GPU work, or evaluation trials."""
import contextlib
import base64
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'studies'), str(ROOT / 'src'), str(ROOT / 'chronotope/python')]
import httpx
from hosted_api import MODELS, HostedTransport, HostedAgentS3Adapter, normalize_response, provider_request


def response(model, text, *, truncated=False):
    common = {'id': 'fixture-only', 'model': model, 'usage': {'input_tokens': 100, 'output_tokens': 10}}
    if model == 'gpt-6.1-sol':
        return {**common, 'status': 'incomplete' if truncated else 'completed', 'output': [
            {'type': 'reasoning'}, {'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]}
    return {**common, 'stop_reason': 'max_tokens' if truncated else 'end_turn',
            'content': [{'type': 'thinking', 'thinking': 'private fixture reasoning'}, {'type': 'text', 'text': text}]}


class HostedContracts(unittest.TestCase):
    def test_wire_conversion_preserves_order_and_history(self):
        messages = [{'role': 'system', 'content': 'original system'},
                    {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
                                                 {'type': 'text', 'text': 'original user'}]},
                    {'role': 'assistant', 'content': 'original answer'},
                    {'role': 'user', 'content': 'next turn'}]
        before = copy.deepcopy(messages)
        for model in MODELS:
            wire = provider_request(model, messages)
            self.assertEqual(wire['model'], model)
            self.assertFalse(set(wire) & {'temperature', 'top_p', 'tools', 'chat_template_kwargs', 'previous_response_id'})
            turns = wire.get('input', wire.get('messages'))
            self.assertEqual(turns[-1]['content'][0]['text'], 'next turn')
            self.assertEqual(turns[-2]['content'][0]['text'], 'original answer')
            self.assertEqual(turns[-3]['content'][1]['text'], 'original user')
        self.assertEqual(messages, before)

    def test_refusals_truncation_and_identity(self):
        for model in MODELS:
            for text in ('', 'I cannot do this', '{"action":"wait"}'):
                result = normalize_response(model, response(model, text, truncated=True))
                self.assertEqual(result['choices'][0]['message']['content'], text)
                self.assertEqual(result['choices'][0]['finish_reason'], 'length')
                self.assertEqual(result['usage']['completion_tokens'], 10)
            wrong = response(model, 'answer'); wrong['model'] = 'substitute'
            with self.assertRaises(ValueError): normalize_response(model, wrong)

    def test_transport_uses_official_https_and_keeps_credentials_out_of_result(self):
        for model, cfg in MODELS.items():
            seen = []
            def handler(request):
                seen.append(request)
                self.assertEqual(str(request.url), cfg['endpoint'])
                return httpx.Response(200, json=response(model, 'visible answer'))
            transport = HostedTransport(model, httpx.MockTransport(handler))
            req = httpx.Request('POST', 'http://unused/chat/completions', json={'model': model, 'messages': [
                {'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'}]})
            with patch.dict(os.environ, {cfg['key_env']: 'fixture-secret'}):
                result = transport.handle_request(req).json()
            self.assertEqual(len(seen), 1)
            self.assertIn('provider_request', result)
            self.assertNotIn('fixture-secret', json.dumps(result))
            transport.close()

    def test_missing_credentials_and_http_failures_are_not_waits(self):
        model = 'gpt-6.1-sol'
        req = httpx.Request('POST', 'http://unused', json={'model': model, 'messages': [{'role': 'user', 'content': 'test'}]})
        transport = HostedTransport(model, httpx.MockTransport(lambda _: httpx.Response(429, headers={'retry-after': '1'}, json={'error': {'message': 'rate limited'}})))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError): transport.handle_request(req)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'fixture-secret'}):
            result = transport.handle_request(req)
        self.assertEqual(result.status_code, 429)
        self.assertEqual(result.headers['retry-after'], '1')
        self.assertNotIn('choices', result.json())

    def test_upstream_worker_retains_grounder_reflection_and_history(self):
        from PIL import Image
        from agents3_full import image_count
        reference = json.loads((ROOT / 'runs/notification_qwen08_edgellm0111_20261031/experiment/configs/full_agent_components.json').read_text())
        for model, cfg in MODELS.items():
            config = copy.deepcopy(reference); config['generation'] = {'max_tokens': 4096}
            for role, component in config['components'].items():
                if role != 'grounder':
                    component.update(model=model, runtime=cfg['runtime'], base_url=cfg['endpoint'])
            seen = []
            def factory(role):
                def handler(request):
                    wire = json.loads(request.content); seen.append((role, wire))
                    if role == 'grounder':
                        return httpx.Response(200, json={'id': 'ground-fixture', 'object': 'chat.completion', 'created': 0,
                            'model': reference['components']['grounder']['model'],
                            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '(322, 238)'}, 'finish_reason': 'stop'}],
                            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}})
                    answer = 'Trajectory reflection.' if role == 'reflection' else '(Grounded Action)\n```python\nagent.click("The blue button.")\n```'
                    return httpx.Response(200, json=response(model, answer))
                return httpx.MockTransport(handler)
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {cfg['key_env']: 'fixture-secret'}):
                adapter = HostedAgentS3Adapter(config, Path(tmp) / 'calls.jsonl', _transport_factory=factory)
                buf = io.BytesIO(); Image.new('RGB', (640, 480)).save(buf, format='PNG')
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        for _ in range(3):
                            result = adapter.invoke(buf.getvalue(), 'Click the blue notification button.', '')
                            self.assertEqual(result.parsed, {'action': 'click', 'x': 320, 'y': 240})
                    self.assertIn('reflection', [role for role, _ in seen])
                    self.assertIn('grounder', [role for role, _ in seen])
                    self.assertGreater(image_count(adapter.agent.executor.generator_agent.messages), 1)
                    self.assertFalse(adapter.errors)
                    self.assertNotIn('fixture-secret', (Path(tmp) / 'calls.jsonl').read_text())
                finally: adapter.close()

    def test_manifest_scope_and_controller_contract(self):
        from run_hosted_notification import prepare, validate, read, write
        with tempfile.TemporaryDirectory() as tmp:
            for model in MODELS:
                out = Path(tmp) / model
                prepare(out, model, ROOT / 'runs/hosted_api_notification_20261102/build')
                self.assertEqual(validate(out)['spec_checks'], 3000)
                plan = read(out / 'plan.json')
                self.assertEqual({j['task'] for j in plan['jobs']}, {'notification'})
                self.assertEqual({j['window_s'] for j in plan['jobs']}, {1, 2, 4, 8, 16})
                self.assertEqual(plan['qualification_trials'], 0)
                plan['jobs'][0]['task'] = 'jump'; write(out / 'plan.json', plan)
                with self.assertRaises((AssertionError, ValueError)): validate(out)

    def test_hosted_schema_does_not_relax_local_endpoint_checks(self):
        from run_hosted_notification import REFERENCE
        from core5_ledger_plan import read, load_cases, make_trial_spec
        from chronotope.spec import SpecError
        plan = read(REFERENCE / 'plan.json'); cases = load_cases(REFERENCE, plan)
        case = cases[plan['jobs'][0]['manifest']][0]
        config = read(REFERENCE / 'configs/notification_agents3_full.json')
        config.update(base_url='https://api.openai.com/v1/responses', model_id='gpt-6.1-sol')
        with self.assertRaises(SpecError): make_trial_spec(case, config, 'live')
        config['server'] = 'agents3_hosted_api'
        make_trial_spec(case, config, 'live')
        config['base_url'] = 'https://api.openai.com.attacker.invalid/v1/responses'
        with self.assertRaises(SpecError): make_trial_spec(case, config, 'live')

    def test_screenshot_audit_rejects_changed_pixels(self):
        from types import SimpleNamespace
        from chronotope.png import encode_png
        from run_hosted_notification import verify_planner_input, REFERENCE, read
        raw = bytes([0, 0, 0]) * (320 * 240)
        png = encode_png(raw, 320, 240, 2)
        compact = read(REFERENCE / 'configs/notification_compact_prompt.json')
        request = provider_request('gpt-6.1-sol', [
            {'role': 'system', 'content': compact['planner_system']},
            {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}},
                {'type': 'text', 'text': compact['initial_user_text']}]}])
        scenario = SimpleNamespace(render=lambda _: raw)
        verify_planner_input(request, 'legacy_cpp_detector', 'gpt-6.1-sol', {'source_time_ns': 0}, scenario)
        scenario.render = lambda _: bytes([255]) * len(raw)
        with self.assertRaises(ValueError):
            verify_planner_input(request, 'legacy_cpp_detector', 'gpt-6.1-sol', {'source_time_ns': 0}, scenario)


if __name__ == '__main__':
    import faulthandler
    faulthandler.dump_traceback_later(30)
    unittest.main()
