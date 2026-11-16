"""Offline parity against installed Agent S3 Worker; no model/benchmark calls."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'studies'),str(ROOT/'src')]
from agents3_policy_contract import (
    build_policy_contract, PolicyReference, normalize_messages, generation_settings,
    parse_code, validate_plan_response, policy_fingerprint, split_reflection,
)
from agents3_stripped import StrippedAgentS3Adapter


def png():
    out=io.BytesIO();Image.new('RGB',(640,480),(17,29,41)).save(out,format='PNG');return out.getvalue()


def code(text):return '```python\n'+text+'\n```'


class PolicyContractTests(unittest.TestCase):
    def run_pair(self, turns):
        contract=build_policy_contract();reference_sleeps=[];actual_sleeps=[]
        reference=PolicyReference(contract,sleep=reference_sleeps.append)
        queue=[];actual_requests=[];transport_errors=[]
        def fake_open(req,timeout):
            body=json.loads(req.data)
            role='reflection' if body['messages'][0]['content'][0]['text']==contract['reflection_system'] else 'planner'
            actual_requests.append({'role':role,'messages':normalize_messages(body['messages']),
                'generation_settings':{k:v for k,v in body.items() if k not in ('model','messages')}})
            response=queue.pop(0)
            if isinstance(response,Exception):
                transport_errors.append(response);raise response
            return io.BytesIO(json.dumps({'model':contract['model'],'usage':{'prompt_tokens':100,'completion_tokens':10},
                'choices':[{'message':{'content':response},'finish_reason':'stop'}]}).encode())
        with tempfile.TemporaryDirectory() as tmp,patch('urllib.request.urlopen',fake_open),patch('gui_agents.s3.utils.common_utils.time.sleep',actual_sleeps.append):
            adapter=StrippedAgentS3Adapter('http://127.0.0.1:1/v1',Path(tmp)/'calls.jsonl',model_id=contract['model'])
            for responses in turns:
                queue[:]=responses
                ref_queue=list(responses)
                def respond(role,messages):
                    value=ref_queue.pop(0)
                    if isinstance(value,Exception):raise value
                    return value
                expected=reference.run_turn(respond,png())
                observed=adapter.invoke(png(),contract['base_instruction'],'ignored by original adapter')
                self.assertEqual(observed.parsed,expected['parsed'])
                self.assertEqual(observed.detail['plan_code'],expected['plan_code'])
                self.assertEqual(adapter.agent.executor.worker_history,reference.worker_history)
                self.assertEqual(normalize_messages(adapter.agent.executor.generator_agent.messages),reference.planner_messages)
                self.assertEqual(normalize_messages(adapter.agent.executor.reflection_agent.messages),reference.reflection_messages)
                self.assertEqual(actual_requests,reference.requests)
                self.assertEqual(actual_sleeps,reference_sleeps)
                self.assertEqual(queue,[]);self.assertEqual(ref_queue,[])
                self.assertEqual(adapter.policy_contract['policy_sha256'],contract['policy_sha256'])
            return reference,actual_requests,actual_sleeps

    def test_first_turn_then_reflection_and_transient_format_retries(self):
        reference,requests,sleeps=self.run_pair([
            [code('agent.wait(time=1.0)')],
            ['<thoughts>Check the screen.</thoughts><answer>Continue the task.</answer>',
             'No code here.',code('agent.click(x=-1,y=20)'),code('agent.click(x=320,y=240)')],
            ['Proceed.',code('agent.done()')],
        ])
        self.assertEqual([r['role'] for r in requests],['planner','reflection','planner','planner','planner','reflection','planner'])
        self.assertEqual(sleeps,[1.0,1.0])
        self.assertEqual(len(reference.worker_history),3)
        self.assertNotIn('No code here.',str(reference.planner_messages))

    def test_historical_thinking_tag_is_kept_in_policy_history(self):
        reference,_,_=self.run_pair([
            ['<think>Inspect the image.</think>\n\n'+code('agent.wait()')],
            ['<thoughts>Check.</thoughts><answer>Continue.</answer>',code('agent.wait(time=0.5)')],
            ['Proceed.',code('agent.done()')],
        ])
        self.assertIn('<think>Inspect the image.</think>',str(reference.planner_messages))

    def test_retry_exhaustion_waits_after_final_attempt_and_keeps_final_plan_only(self):
        reference,requests,sleeps=self.run_pair([
            ['no code','bad again','third malformed response'],
            ['Continue.',code('agent.wait()')],
        ])
        self.assertEqual(sleeps,[1.0,1.0,1.0])
        self.assertEqual(reference.worker_history[0],'third malformed response')
        self.assertNotIn('bad again',str(reference.planner_messages))

    def test_history_truncation_matches_actual_worker_after_twelve_turns(self):
        turns=[[code('agent.wait()')]]+[[f'Continue step {i}.',code('agent.wait(time=0.5)')] for i in range(1,12)]
        reference,requests,sleeps=self.run_pair(turns)
        self.assertEqual(len(reference.planner_messages),17)
        self.assertEqual(len(reference.reflection_messages),9)
        self.assertEqual(len(requests),23)
        for req in requests:
            self.assertEqual(sum(p['type']=='image' for m in req['messages'] for p in m['content']),1)

    def test_terminal_and_last_fenced_block_policy(self):
        multiple=code('agent.wait()')+'\n'+code('agent.done()')
        result=validate_plan_response(multiple,640,480)
        self.assertEqual(result['errors'],[])
        self.assertEqual(result['executable'],'DONE')
        self.assertEqual(parse_code(multiple),'agent.done()\n')
        reference,_,_=self.run_pair([[multiple],['Task is complete.',code('agent.fail()')]])
        self.assertEqual(reference.turn_count,2)  # DONE does not stop the gym adapter.

    def test_exact_settings_and_contract_fingerprint(self):
        c=build_policy_contract()
        self.assertEqual(c['generation_settings'],generation_settings())
        self.assertEqual(c['max_tokens'],4096)
        self.assertEqual(c['repetition_penalty'],1.05)
        self.assertEqual(c['stop'],[])
        self.assertNotIn('stop',c['generation_settings'])
        self.assertEqual(policy_fingerprint(c),c['policy_sha256'])
        modified=copy.deepcopy(c);modified['planner_system']+=' '
        with self.assertRaisesRegex(ValueError,'mismatch'):PolicyReference(modified)

    def test_code_validation_and_reflection_match_upstream_functions(self):
        from gui_agents.s3.utils.common_utils import parse_code_from_string,split_thinking_response
        from gui_agents.s3.utils.formatters import SINGLE_ACTION_FORMATTER,CODE_VALID_FORMATTER
        from agents3_stripped import DirectPixelACI,safe_action_code
        import gui_agents.s3.utils.formatters as module
        fixture=[code('agent.click(x=2.5,y=3.5)'),code('agent.click(640,20)'),code('agent.click(True,20)'),
            code('agent.wait(time=-1)'),code('agent.done()'),code('agent.click(1,2);agent.wait()'),
            code('agent.click(x=1+2,y=3)'),code('agent.click(1,2)')+'\n'+code('agent.fail()'),'',
            '```python\nagent.click(\n x=123,\n y=45\n)\n```']
        with patch.object(module,'create_pyautogui_code',safe_action_code):
            for response in fixture:
                expected=[]
                for ok,error in [SINGLE_ACTION_FORMATTER(response),CODE_VALID_FORMATTER(DirectPixelACI(640,480),{'screenshot':png()},response)]:
                    if not ok:expected.append(error)
                self.assertEqual(parse_code(response),parse_code_from_string(response))
                self.assertEqual(validate_plan_response(response,640,480,png())['errors'],expected)
        for text in ['Continue.','<thoughts>x</thoughts><answer>y</answer>','a<answer>b</answer>c<answer>d</answer>']:
            self.assertEqual(split_reflection(text),split_thinking_response(text))


if __name__=='__main__':unittest.main()
