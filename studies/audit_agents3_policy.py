"""Audit actual model inputs after scored trials; never screen task competence."""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
_CHRONOTOPE_REPO_ROOT = _CHRONOTOPE_ENV.get('CHRONOTOPE_REPO_ROOT', str(next(
    (base for base in _CHRONOTOPE_PATH(__file__).resolve().parents if (base / 'pyproject.toml').is_file()),
    _CHRONOTOPE_PATH(_CHRONOTOPE_HOME) / 'chronotope')))
import base64
import functools
import hashlib
import io
import json
from pathlib import Path
import sys


def sha(value):
    return hashlib.sha256(value).hexdigest()


@functools.lru_cache(maxsize=4)
def token_tools(checkpoint_path=None):
    deps=Path(__file__).resolve().parent/'audit_deps'
    if deps.exists():sys.path.insert(0,str(deps))
    else:sys.path.append((_CHRONOTOPE_HOME + '/qwen-serve/.venv/lib/python3.12/site-packages'))
    import jinja2
    from tokenizers import Tokenizer
    checkpoint=Path(checkpoint_path or (_CHRONOTOPE_REPO_ROOT + '/runs/model_cache/qwen35_0p8b_legacy_source'))
    env=jinja2.Environment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
    def fail(message):raise ValueError(message)
    env.globals['raise_exception']=fail
    template=env.from_string((checkpoint/'chat_template.jinja').read_text())
    tokenizer=Tokenizer.from_file(str(checkpoint/'tokenizer.json'))
    return template,tokenizer


def model_input(messages,checkpoint_path=None,image_tokens=300):
    template,tokenizer=token_tools(checkpoint_path)
    prompt=template.render(messages=messages,add_generation_prompt=True,enable_thinking=False)
    # Both paths use640x480, patch16/merge2/temporal2 =>300visualtokens.
    expanded=prompt.replace('<|image_pad|>','<|image_pad|>'*image_tokens)
    ids=tokenizer.encode(expanded,add_special_tokens=False).ids
    return {'text_sha256':sha(prompt.encode()),'input_token_ids_sha256':sha(json.dumps(ids,separators=(',',':')).encode()),'input_tokens':len(ids)}


def audit_trial_requests(out,ident,arm,trace,scenario,execution):
    from agents3_policy_contract import normalize_messages,generation_settings,PolicyReference
    from chronotope.runner import PayloadStore
    from chronotope.png import encode_png
    from PIL import Image
    path=out/'calls'/f'{ident}.jsonl'
    events=trace.all('decision_invocation')
    if not path.exists():
        assert not events,'Missing model audit log'
        return []
    calls=[json.loads(line) for line in path.read_text().splitlines()]
    observations={r['observation_id']:r for r in trace.all('observation')}
    store=PayloadStore(out/'payloads');evidence=[];cursor=0
    reference=PolicyReference(execution['request_contract'])
    for turn,event in enumerate(events):
        assert event['model_reported']==execution['model']
        obs=observations[event['observation_id']]
        png=encode_png(store.get(obs['payload_hash']),scenario.canvas_w_px,scenario.canvas_h_px,2)
        with Image.open(io.BytesIO(png)) as image:
            assert image.size==(640,480)
            rgb_sha=sha(image.convert('RGB').tobytes())
        source_url='data:image;base64,'+base64.b64encode(png).decode()
        if arm=='legacy_cpp':
            own=[]
            while cursor<len(calls) and calls[cursor]['turn']==turn:own.append(calls[cursor]);cursor+=1
        else:
            count=event['planner_calls']+event['reflection_calls'];own=calls[cursor:cursor+count];cursor+=count
        assert own,'Decision has no audited model calls'
        replay_index=0
        def replay_model(role,messages):
            nonlocal replay_index
            assert replay_index<len(own),'Native policy requested fewer calls than Agent S3 reference'
            c=own[replay_index];replay_index+=1
            observed=c['messages'] if arm=='legacy_cpp' else normalize_messages(c['request']['messages'])
            assert c.get('request_kind',c.get('role'))==role,'Policy call order differs'
            assert observed==messages,'Actual prompt/history differs from Agent S3 Worker reference'
            return c['response'] if arm=='legacy_cpp' else c['response']['choices'][0]['message']['content']
        expected_action=reference.run_turn(replay_model,png)['parsed']
        assert replay_index==len(own),'Native policy requested more calls than Agent S3 reference'
        if arm=='legacy_cpp':
            if expected_action and expected_action.get('action')=='click':
                assert (event['image_x'],event['image_y'])==(expected_action['x'],expected_action['y']),'Native action parser differs'
                assert event['outcome']=='click'
            else:assert event['outcome']=='wait','Native emitted an unexpected action'

        assert own[0].get('request_kind',own[0].get('role'))==('planner' if turn==0 else 'reflection')
        ins=outs=0
        for call in own:
            if arm=='legacy_cpp':
                assert call['ok'],call.get('error')
                assert call['policy_sha256']==execution['policy_sha256']
                messages=call['messages'];settings=call['generation_settings']
                count_in,count_out=call['input_tokens'],call['output_tokens']
            else:
                assert call['model']==execution['model'] and call['policy_contract_sha256']==execution['policy_sha256']
                body=call['request'];messages=normalize_messages(body['messages'])
                settings={key:body[key] for key in generation_settings()}
                for message in body['messages']:
                    for part in message['content']:
                        if part['type']=='image_url':assert part['image_url']['sha256']==sha(source_url.encode())
                usage=call['response']['usage'];count_in,count_out=usage['prompt_tokens'],usage['completion_tokens']
            assert settings==generation_settings(),'Generation settings differ'
            assert sum(p['type']=='image' for m in messages for p in m['content'])==1
            expected=model_input(messages,execution.get('checkpoint_path'),execution.get('image_tokens',300))
            assert count_in==expected['input_tokens'],f"Input token mismatch: {count_in} vs {expected}"
            if arm=='legacy_cpp':
                actual_ids=call['input_token_ids']
                assert len(actual_ids)==count_in
                assert sha(json.dumps(actual_ids,separators=(',',':')).encode())==expected['input_token_ids_sha256'],'Actual native tokenizer IDs differ from HF'
            assert 0<=count_out<=4096,'Output budget violated'
            ins+=count_in;outs+=count_out
            evidence.append({'turn':turn,'request_kind':call.get('request_kind',call.get('role')),
                'source_payload_hash':obs['payload_hash'],'source_rgb_sha256':rgb_sha,
                'messages_sha256':sha(json.dumps(messages,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()),
                **expected,'output_tokens':count_out,'policy_sha256':execution['policy_sha256']})
        assert (ins,outs)==(event['input_tokens'],event['output_tokens']),'Per-call usage does not sum to trial trace'
    assert cursor==len(calls),'Unaffiliated model calls in audit'
    if evidence:
        paired=ident.replace(arm,'agents3_stripped' if arm=='legacy_cpp' else 'legacy_cpp',1)
        other=out/'records'/f'{paired}.json'
        if other.exists():
            counterpart=json.loads(other.read_text()).get('request_audit',[])
            if counterpart:
                for key in ['source_payload_hash','source_rgb_sha256','messages_sha256','input_token_ids_sha256','input_tokens','policy_sha256']:
                    assert evidence[0][key]==counterpart[0][key],f'First paired request differs: {key}'
    return evidence
