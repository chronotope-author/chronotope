"""Native Agent S3 Worker/reflection with CPU text history and direct pixel actions.

No grounder, detector, tracker, updater, code agent, or historical image input.
Native Worker response structure, reflection, format retries, and text trajectory
are retained; examples/guidelines match the stripped pixel-only action API. Only the current screenshot enters each request. Generated action code
is parsed as a single allowlisted call with literal arguments, never evaluated.
"""
import ast
import copy
import hashlib
import io
import json
import math
import time
import urllib.request
from pathlib import Path

MODEL = 'RadixArk/Qwen3.8-27B-NVFP4'
REVISION = '319f741cce68d7914884900c138a1fbb70a42f30'


def agent_action(func):
    func.is_agent_action = True
    return func


class DirectPixelACI:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.env = None
        self.notes = []

    def assign_screenshot(self, obs):
        # Geometry only; no screenshot inspection, OCR, or target detection.
        from PIL import Image
        with Image.open(io.BytesIO(obs['screenshot'])) as image:
            if image.size != (self.width, self.height):
                raise ValueError('Screenshot geometry changed during the trial')

    def set_task_instruction(self, instruction):
        self.instruction = instruction

    @agent_action
    def click(self, x: int, y: int):
        """Click the target center at absolute pixel coordinates x,y in the current screenshot. Infer the coordinates yourself from the image; no grounding model is available. The origin is the top-left corner."""
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x,y)):
            raise ValueError('Click coordinates must be finite numbers')
        x,y=round(x),round(y)
        if not (0<=x<self.width and 0<=y<self.height):
            raise ValueError('Click is outside the screenshot')
        return f'pyautogui.click({x}, {y})'

    @agent_action
    def wait(self, time: float = 1.0):
        """Wait for another observation when no task target is visible. This proposes no click."""
        if type(time) not in (int,float) or not math.isfinite(time) or time<0:
            raise ValueError('Invalid wait time')
        return 'WAIT'

    @agent_action
    def done(self):
        """Declare the task complete after it has been executed."""
        return 'DONE'

    @agent_action
    def fail(self):
        """Declare that the task cannot be completed."""
        return 'FAIL'


def safe_action_code(agent, code, obs):
    tree=ast.parse(code.strip(),mode='eval').body
    if not isinstance(tree,ast.Call) or not isinstance(tree.func,ast.Attribute):
        raise ValueError('Expected one agent action')
    if not isinstance(tree.func.value,ast.Name) or tree.func.value.id!='agent':
        raise ValueError('Only agent actions are supported')
    if tree.func.attr not in {'click','wait','done','fail'}:
        raise ValueError('Unsupported action')
    if any(k.arg is None for k in tree.keywords):
        raise ValueError('Keyword expansion is unsupported')
    args=[ast.literal_eval(a) for a in tree.args]
    kwargs={k.arg:ast.literal_eval(k.value) for k in tree.keywords}
    agent.assign_screenshot(obs)
    return getattr(agent,tree.func.attr)(*args,**kwargs)


def strip_history_images(worker):
    for agent in (worker.generator_agent,worker.reflection_agent):
        for message in agent.messages:
            if isinstance(message.get('content'),list):
                message['content']=[part for part in message['content'] if part.get('type')=='text']
    worker.screenshot_inputs.clear()


def direct_pixel_prompt(prompt):
    """Keep native Worker response/history instructions, adapt removed capabilities."""
    start=prompt.index('# GUIDELINES')
    end=prompt.index('### END OF GUIDELINES',start)+len('### END OF GUIDELINES')
    prompt=prompt[:start]+('# GUIDELINES\nUse only the listed click, wait, done, and fail actions. '
        'Infer click coordinates from the current screenshot yourself. '
        'No grounding model, code agent, or hotkey action is available.\n')+prompt[end:]
    example='agent.click("The menu button at the top right of the window", 1, "left")'
    if prompt.count(example)!=1:
        raise ValueError('Unexpected Agent S3 click example; inspect upstream prompt')
    prompt=prompt.replace(example,'agent.click(x=123, y=45)')
    prompt='\n'.join(line for line in prompt.splitlines() if not any(
        fragment in line for fragment in ['agent.hotkey()',"My computer's password",'hotkey on MacOS','Prefer hotkeys']))
    return prompt


def request_audit(body):
    audit=copy.deepcopy(body)
    for message in audit['messages']:
        if isinstance(message.get('content'),list):
            for part in message['content']:
                if part.get('type')=='image_url':
                    url=part['image_url']['url']
                    part['image_url']={'sha256':hashlib.sha256(url.encode()).hexdigest(),'encoded_bytes':len(url)}
    return audit


class StrippedAgentS3Adapter:
    def __init__(self, base_url, log, *, model_id=MODEL, max_tokens=4096):
        self.base_url=base_url.rstrip('/')
        self.model_id=model_id
        self.max_tokens=max_tokens
        self.log=Path(log)
        self.agent=None
        self.calls=[]
        self.errors=[]
        self.policy_contract=None

    def spec_fields(self):
        return {'server':'agents3_service','model_id':self.model_id,
                'implementation':'native AgentS3; DirectPixelACI; text-only past history',
                'reflection':True,'grounder':None,'updater':None,'thinking':False}

    def _generate(self, role, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        from agents3_policy_contract import generation_settings
        body={'model':self.model_id,'messages':messages,**generation_settings(self.max_tokens)}
        images=sum(part.get('type')=='image_url' for message in messages
                   for part in message.get('content',[]) if isinstance(part,dict))
        if images!=1:
            raise RuntimeError(f'Expected exactly the current screenshot, got {images}')
        encoded=json.dumps(body).encode()
        record={'role':role,'model':self.model_id,'start_ns':time.monotonic_ns(),
                'request_sha256':hashlib.sha256(encoded).hexdigest(),'request':request_audit(body),
                'policy_contract_sha256':self.policy_contract['policy_sha256'] if self.policy_contract else None}
        try:
            request=urllib.request.Request(self.base_url+'/chat/completions',data=encoded,
                                           headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=300) as response:
                result=json.load(response)
            record['response']=result
            if result.get('model')!=self.model_id or not result.get('usage'):
                raise RuntimeError('Model identity or usage missing from response')
            choice=result['choices'][0]
            answer=choice['message'].get('content')
            if not isinstance(answer,str):
                raise RuntimeError('Model returned no text')
            record['http_status']=200
            return answer
        except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'
            self.errors.append(record['error'])
            raise
        finally:
            record['end_ns']=time.monotonic_ns()
            self.calls.append(record)
            with self.log.open('a') as handle:
                handle.write(json.dumps(record,allow_nan=False)+'\n')

    def invoke(self,png,system,user_text,structured=False,schema=None,parser=None):
        from PIL import Image
        from chronotope.controller import RawInvocation,parse_agents3
        if self.agent is None:
            from gui_agents.s3.agents.agent_s import AgentS3
            import gui_agents.s3.agents.worker as worker_module
            import gui_agents.s3.utils.common_utils as common_module
            import gui_agents.s3.utils.formatters as formatters_module
            for module in (worker_module,common_module,formatters_module):
                module.create_pyautogui_code=safe_action_code
            with Image.open(io.BytesIO(png)) as image:
                self.width,self.height=image.size
            self.agent=AgentS3({'engine_type':'vllm','model':self.model_id,
                'base_url':self.base_url,'api_key':'dummy','temperature':0.0},
                DirectPixelACI(self.width,self.height),platform='linux',
                max_trajectory_length=8,enable_reflection=True)
            generator=self.agent.executor.generator_agent
            generator.add_system_prompt(direct_pixel_prompt(generator.system_prompt))
            for role,agent in [('planner',self.agent.executor.generator_agent),
                               ('reflection',self.agent.executor.reflection_agent)]:
                agent.engine.generate=lambda messages,_role=role,**kw:self._generate(_role,messages,**kw)
        worker=self.agent.executor
        strip_history_images(worker)
        start=time.monotonic_ns(); before=len(self.calls)
        if self.policy_contract is None:
            from agents3_policy_contract import build_policy_contract
            self.policy_contract=build_policy_contract(self.width,self.height,base_instruction=system,
                model_id=self.model_id,max_tokens=self.max_tokens)
        instruction=system+f' The screenshot is {self.width} pixels wide and {self.height} pixels high. Use absolute screenshot pixel coordinates.'
        info,actions=self.agent.predict(instruction,{'screenshot':png})
        strip_history_images(worker)
        if self.errors:
            raise RuntimeError('Inference service error: '+self.errors[-1])
        own=self.calls[before:]
        if not own:
            raise RuntimeError('Agent S3 made no measured model calls')
        return RawInvocation(parsed=parse_agents3(actions),invocation_id=f'agents3-stripped-{start}',detail={
            'model_reported':self.model_id,'finish_reason':'stop',
            'policy_contract_sha256':self.policy_contract['policy_sha256'],
            'input_tokens':sum(c['response']['usage']['prompt_tokens'] for c in own),
            'output_tokens':sum(c['response']['usage']['completion_tokens'] for c in own),
            'planner_calls':sum(c['role']=='planner' for c in own),
            'reflection_calls':sum(c['role']=='reflection' for c in own),
            'grounder_calls':0,'updater_calls':0,'history_representation':'CPU RAM text',
            'history_turns':worker.turn_count,'retained_history_images':0,
            'model_call_request_sha256':[c['request_sha256'] for c in own],
            'agent_elapsed_ns':time.monotonic_ns()-start,
            'plan_code':info.get('plan_code'),'raw_text':'\n'.join(actions),
            'screen_wh':[self.width,self.height]})
