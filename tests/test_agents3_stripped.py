"""Unit checks for action isolation, text history, and native Agent S3 wiring.
No model requests or benchmark trials are performed.
"""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'studies'),str(ROOT/'src')]
from agents3_stripped import DirectPixelACI,safe_action_code,strip_history_images,StrippedAgentS3Adapter,MODEL
from PIL import Image


def png():
    f=io.BytesIO();Image.new('RGB',(320,240),'white').save(f,format='PNG');return f.getvalue()


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.aci=DirectPixelACI(320,240);self.obs={'screenshot':png()}

    def test_literal_pixels_preserved(self):
        self.assertEqual(safe_action_code(self.aci,'agent.click(x=123, y=45)',self.obs),'pyautogui.click(123, 45)')

    def test_disallowed_python_never_executes(self):
        for code in ['__import__("os").system("false")','agent.click(*(1,2))','agent.click(1+2,4)',
                     'agent.generate_coords("button")','agent.click(1,2);agent.click(3,4)','agent.click(**{"x":1,"y":2})']:
            with self.subTest(code=code),self.assertRaises((ValueError,SyntaxError)):
                safe_action_code(self.aci,code,self.obs)

    def test_invalid_geometry_is_not_clipped(self):
        for xy in [(-1,10),(320,10),(10,240),(float('nan'),20),(True,2)]:
            with self.subTest(xy=xy),self.assertRaises(ValueError):self.aci.click(*xy)

    def test_text_history_preserved_without_past_images(self):
        from types import SimpleNamespace
        content=[{'type':'text','text':'previous action'},{'type':'image_url','image_url':{'url':'data:x'}}]
        agent=SimpleNamespace(messages=[{'role':'user','content':content}])
        worker=SimpleNamespace(generator_agent=agent,reflection_agent=agent,screenshot_inputs=[b'previous'])
        strip_history_images(worker)
        self.assertEqual(agent.messages[0]['content'],[{'type':'text','text':'previous action'}])
        self.assertEqual(worker.screenshot_inputs,[])

    def test_native_worker_reflection_and_single_current_image(self):
        class Response(io.BytesIO):
            pass
        requests=[]
        def fake_open(req,timeout):
            body=json.loads(req.data);requests.append(body)
            # First planner, second-turn reflection, then second planner.
            text='Continue the task.' if len(requests)==2 else '```python\nagent.click(123, 45)\n```'
            return Response(json.dumps({'model':MODEL,'choices':[{'message':{'content':text},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':20,'completion_tokens':10}}).encode())
        with tempfile.TemporaryDirectory() as tmp,patch('urllib.request.urlopen',fake_open):
            adapter=StrippedAgentS3Adapter('http://127.0.0.1:1/v1',Path(tmp)/'calls.jsonl')
            first=adapter.invoke(self.obs['screenshot'],'Click the blue notification button.','')
            second=adapter.invoke(self.obs['screenshot'],'Click the blue notification button.','')
            self.assertEqual(first.parsed,{'action':'click','x':123,'y':45})
            self.assertEqual(second.detail['reflection_calls'],1)
            self.assertEqual(second.detail['grounder_calls'],0)
            self.assertEqual(second.detail['history_turns'],2)
            self.assertEqual([c['role'] for c in adapter.calls],['planner','reflection','planner'])
            for body in requests:
                self.assertEqual(sum(p.get('type')=='image_url' for m in body['messages'] for p in m['content']),1)
                self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
            self.assertIn('agent.click(123, 45)',str(requests[-1]['messages']))
            self.assertNotIn('data:image',str([c['request'] for c in adapter.calls]))
            self.assertEqual(adapter.agent.executor.screenshot_inputs,[])

    def test_explicit_qwen08_model_reaches_request_and_trace(self):
        model='Qwen/Qwen3.5-0.8B';requests=[]
        def fake_open(req,timeout):
            body=json.loads(req.data);requests.append(body)
            return io.BytesIO(json.dumps({'model':model,'choices':[{'message':{'content':'```python\nagent.click(123, 45)\n```'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':20,'completion_tokens':10}}).encode())
        with tempfile.TemporaryDirectory() as tmp,patch('urllib.request.urlopen',fake_open):
            adapter=StrippedAgentS3Adapter('http://127.0.0.1:1/v1',Path(tmp)/'calls.jsonl',model_id=model)
            result=adapter.invoke(self.obs['screenshot'],'Click the blue notification button.','')
            self.assertEqual(result.parsed,{'action':'click','x':123,'y':45})
            self.assertTrue(requests)
            self.assertTrue(all(body['model']==model for body in requests))
            self.assertEqual(adapter.spec_fields()['model_id'],model)
            prompt=requests[0]['messages'][0]['content'][0]['text']
            self.assertIn('agent.click(x=123, y=45)',prompt)
            self.assertIn('def click(self, x:',prompt)
            self.assertNotIn('agent.call_code_agent',prompt)
            self.assertNotIn('agent.hotkey()',prompt)
            self.assertNotIn('agent.click("The menu button',prompt)


    def test_wrong_model_fails_closed(self):
        def fake_open(req,timeout):
            return io.BytesIO(json.dumps({'model':'wrong-checkpoint','usage':{'prompt_tokens':1},'choices':[]}).encode())
        with tempfile.TemporaryDirectory() as tmp,patch('urllib.request.urlopen',fake_open):
            ad=StrippedAgentS3Adapter('http://127.0.0.1:1/v1',Path(tmp)/'calls.jsonl')
            with self.assertRaisesRegex(RuntimeError,'identity'):
                ad._generate('planner',[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:x'}}]}])
            self.assertEqual(len(ad.errors),1)

if __name__=='__main__':unittest.main()
