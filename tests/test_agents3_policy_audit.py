import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies')]
from agents3_policy_contract import PolicyReference,build_policy_contract,generation_settings
from audit_agents3_policy import audit_trial_requests,model_input,token_tools
from chronotope.runner import PayloadStore
from chronotope.png import encode_png

class Trace:
    def __init__(self,events):self.events=events
    def all(self,key):return self.events.get(key,[])

class PolicyAuditTests(unittest.TestCase):
    def test_native_and_host_actual_request_contract_and_rejections(self):
        policy=build_policy_contract();ref=PolicyReference(policy)
        ref.run_turn(lambda role,messages:'```python\nagent.click(x=240,y=240)\n```')
        messages=ref.requests[0]['messages'];expected=model_input(messages)
        self.assertEqual(expected['input_tokens'],998)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'calls').mkdir();(out/'records').mkdir()
            payload=bytes([12,34,56])*(320*240);payload_hash=PayloadStore(out/'payloads').put(payload)
            ev={'model_reported':policy['model'],'observation_id':1,'input_tokens':998,'output_tokens':15,'planner_calls':1,'reflection_calls':0,'image_x':240,'image_y':240,'outcome':'click'}
            trace=Trace({'observation':[{'observation_id':1,'payload_hash':payload_hash}],'decision_invocation':[ev]})
            cfg={'model':policy['model'],'policy_sha256':policy['policy_sha256'],'request_contract':policy};scenario=SimpleNamespace(canvas_w_px=320,canvas_h_px=240)
            native={'turn':0,'request_kind':'planner','ok':True,'policy_sha256':policy['policy_sha256'],'messages':messages,
                    'generation_settings':generation_settings(),'input_tokens':998,'output_tokens':15,'response':'```python\nagent.click(x=240,y=240)\n```'}
            template,tokenizer=token_tools()
            native['input_token_ids']=tokenizer.encode(template.render(messages=messages,add_generation_prompt=True,enable_thinking=False).replace('<|image_pad|>','<|image_pad|>'*300),add_special_tokens=False).ids
            path=out/'calls/legacy_cpp_w1_000_live.jsonl';path.write_text(json.dumps(native)+'\n')
            evidence=audit_trial_requests(out,'legacy_cpp_w1_000_live','legacy_cpp',trace,scenario,cfg)
            (out/'records/legacy_cpp_w1_000_live.json').write_text(json.dumps({'request_audit':evidence}))
            body=copy.deepcopy(messages);png=encode_png(payload,320,240,2);url='data:image;base64,'+base64.b64encode(png).decode()
            for message in body:
                for p in message['content']:
                    if p['type']=='image':p.update(type='image_url',image_url={'sha256':hashlib.sha256(url.encode()).hexdigest()})
            host={'role':'planner','model':policy['model'],'policy_contract_sha256':policy['policy_sha256'],
                  'request':{'messages':body,**generation_settings()},'response':{'usage':{'prompt_tokens':998,'completion_tokens':15},'choices':[{'message':{'content':native['response']}}]}}
            hp=out/'calls/agents3_stripped_w1_000_live.jsonl';hp.write_text(json.dumps(host)+'\n')
            host_evidence=audit_trial_requests(out,'agents3_stripped_w1_000_live','agents3_stripped',trace,scenario,cfg)
            self.assertEqual(evidence[0]['messages_sha256'],host_evidence[0]['messages_sha256'])
            host['response']['usage']['prompt_tokens']=999;hp.write_text(json.dumps(host)+'\n')
            with self.assertRaisesRegex(AssertionError,'Input token mismatch'):audit_trial_requests(out,'agents3_stripped_w1_000_live','agents3_stripped',trace,scenario,cfg)
            native['generation_settings']['repetition_penalty']=1;path.write_text(json.dumps(native)+'\n')
            with self.assertRaisesRegex(AssertionError,'Generation settings'):audit_trial_requests(out,'legacy_cpp_w1_000_live','legacy_cpp',trace,scenario,cfg)

if __name__=='__main__':unittest.main()
