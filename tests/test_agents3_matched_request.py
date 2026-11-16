"""Offline request-contract checks; no inference or experimental trials."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'studies'), str(ROOT/'src')]
from agents3_matched_request import (
    MatchedRequestAgentS3Adapter, MODEL, SYSTEM_PROMPT, USER_TEXT, REQUEST_CONTRACT,
    contract_fingerprint, parse_native_action, native_canvas_coordinates,
)
from chronotope.controller import SYSTEM_PROMPTS


def png():
    image = Image.new('RGB', (640, 480), (17, 29, 41))
    buf = io.BytesIO(); image.save(buf, format='PNG')
    return buf.getvalue(), image.tobytes()


class MatchedRequestTests(unittest.TestCase):
    def test_prompt_matches_native_source_and_controller(self):
        self.assertEqual(SYSTEM_PROMPT, SYSTEM_PROMPTS[4])
        native = (ROOT/'chronotope/src/planner_vlm.cpp').read_text()
        # Exact C++ string literal, not a loosely equivalent prompt.
        self.assertIn(json.dumps(SYSTEM_PROMPT), native)

    def test_one_stateless_request_per_decision_with_full_generation_contract(self):
        requests = []
        def fake_open(req, timeout):
            body = json.loads(req.data); requests.append(body)
            self.assertEqual(timeout, 300)
            return io.BytesIO(json.dumps({'model': MODEL, 'usage': {'prompt_tokens': 449, 'completion_tokens': 25},
                'choices': [{'message': {'content': '```json\n[{"action":"click","coordinate":[350,200]'}, 'finish_reason': 'stop'}]}).encode())
        encoded, pixels = png()
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen', fake_open):
            log = Path(tmp)/'calls.jsonl'
            ad = MatchedRequestAgentS3Adapter('http://127.0.0.1:1/v1', log)
            one = ad.invoke(encoded, SYSTEM_PROMPT, USER_TEXT)
            two = ad.invoke(encoded, SYSTEM_PROMPT, USER_TEXT)
            self.assertEqual(one.parsed, {'action': 'click', 'x': 350, 'y': 200})
            self.assertEqual(two.parsed, one.parsed)
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(len(ad.agent.messages), 2)
            self.assertEqual(type(ad.agent).__name__, 'LMMAgent')
            self.assertEqual(type(ad.agent.engine).__name__, 'LMMEnginevLLM')
            body = requests[0]
            self.assertEqual([m['role'] for m in body['messages']], ['system', 'user'])
            self.assertEqual([x['type'] for x in body['messages'][1]['content']], ['image_url', 'text'])
            for k in ['model', 'max_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'stop', 'mm_processor_kwargs']:
                self.assertEqual(body[k], REQUEST_CONTRACT[k])
            self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
            self.assertNotIn('response_format', body)
            records = [json.loads(l) for l in log.read_text().splitlines()]
            self.assertEqual(records[0]['source_rgb_sha256'], hashlib.sha256(pixels).hexdigest())
            self.assertEqual(records[0]['source_image_wh'], [640, 480])
            self.assertEqual(records[0]['request_contract_sha256'], contract_fingerprint())
            self.assertNotIn('data:image', log.read_text())
            self.assertEqual(one.detail['reflection_calls'], 0)
            self.assertEqual(one.detail['history_turns'], 0)

    def test_altered_prompt_or_budget_fails_before_transport(self):
        encoded, _ = png()
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen') as request:
            ad = MatchedRequestAgentS3Adapter('http://127.0.0.1:1/v1', Path(tmp)/'calls.jsonl')
            with self.assertRaisesRegex(ValueError, 'prompt'):
                ad.invoke(encoded, SYSTEM_PROMPT + ' Extra instruction.', USER_TEXT)
            with self.assertRaisesRegex(ValueError, 'prompt'):
                ad.invoke(encoded, SYSTEM_PROMPT, 'Click now.')
            with self.assertRaisesRegex(ValueError, 'max_tokens'):
                MatchedRequestAgentS3Adapter('http://127.0.0.1:1/v1', Path(tmp)/'calls.jsonl', max_tokens=4096)
            request.assert_not_called()

    def test_malformed_response_is_no_action_without_retry(self):
        calls = []
        def fake_open(req, timeout):
            calls.append(req)
            return io.BytesIO(json.dumps({'model': MODEL, 'usage': {'prompt_tokens': 449, 'completion_tokens': 10},
                'choices': [{'message': {'content': 'I am done.'}, 'finish_reason': 'stop'}]}).encode())
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen', fake_open):
            ad = MatchedRequestAgentS3Adapter('http://127.0.0.1:1/v1', Path(tmp)/'calls.jsonl')
            result = ad.invoke(png()[0], SYSTEM_PROMPT, USER_TEXT)
            self.assertEqual(result.parsed, {'action': 'wait'})
            self.assertEqual(len(calls), 1)
            self.assertEqual(ad.errors, [])

    def test_wrong_response_model_is_infrastructure_error(self):
        def fake_open(req, timeout):
            return io.BytesIO(json.dumps({'model': 'different-model', 'usage': {'prompt_tokens': 1}, 'choices': []}).encode())
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen', fake_open):
            ad = MatchedRequestAgentS3Adapter('http://127.0.0.1:1/v1', Path(tmp)/'calls.jsonl')
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                ad.invoke(png()[0], SYSTEM_PROMPT, USER_TEXT)
            self.assertEqual(len(ad.calls), 1)
            self.assertEqual(len(ad.errors), 1)

    def test_native_parser_accepts_trimmed_and_array_coordinate_forms(self):
        cases = [
            ('{"action":"click","x":350,"y":200', {'action':'click','x':350,'y':200}),
            ('```json\n[{"action":"click","coordinate":[350.3,200.7]}', {'action':'click','x':350,'y':201}),
            ('{"action":"click","coordinate":[-1.8,-1.1]}', {'action':'click','x':-1,'y':0}),
            ('{"action":"click","x":12.9,"y":-2.1}', {'action':'click','x':12,'y':-2}),
            ('{"action":"wait"}', {'action':'wait'}),
            ('{"action":"click","coordinate":[4]}', {'action':'wait'}),
            ('agent.click(10,20)', {'action':'wait'}),
        ]
        for text, want in cases:
            with self.subTest(text=text): self.assertEqual(parse_native_action(text), want)

    def test_native_negative_coordinate_division_truncates(self):
        self.assertEqual(native_canvas_coordinates(-1, -1, 320, 240), {'kind':'click','x':0,'y':0})
        self.assertEqual(native_canvas_coordinates(350, 200, 320, 240), {'kind':'click','x':112,'y':48})


if __name__ == '__main__':
    unittest.main()
