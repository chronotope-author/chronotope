import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from chronotope.controller import ControlUpdate, Decision, Observation
from chronotope.hashing import sha256_hex
from chronotope.latest_frame import ActionAbstention, CudaNCC, LatestFrameController
from chronotope.tracker import cut_template, track


class Base:
    def __init__(self, action): self.action, self.calls = action, 0
    def admit(self, obs): return ControlUpdate('action', obs.payload_hash)
    def decide(self):
        self.calls += 1
        return Decision(self.action, 0)
    def spec_hash(self): return 'base'


def obs(a, i):
    b = a.tobytes()
    return Observation(i,i,i,i,sha256_hex(b),i,b)


def images():
    a = np.zeros((64,96,3), np.uint8)
    patch = np.random.default_rng(4).integers(20,240,(13,13,3),dtype=np.uint8)
    a[12:25,12:25] = patch
    b = np.zeros_like(a); b[32:45,52:65] = patch
    return a,b


def make(a,b,action=None,**kw):
    base = Base(action or {'kind':'click','x':18,'y':18})
    c = LatestFrameController(base,width=96,height=64,radius=6,**kw)
    c.admit(obs(a,1)); c.decide(); c.admit(obs(b,2))
    return c,base


def test_translation_preserves_operation_and_no_extra_call():
    a,b=images();c,base=make(a,b)
    correction=c.correct_at_commit(100)
    assert correction.action == {'kind':'click','x':58,'y':38}
    assert base.calls == 1 and correction.detail['additional_model_calls']==0


@pytest.mark.parametrize('failure',['missing','duplicate','recolored','flat'])
def test_safe_no_action(failure):
    a,b=images()
    if failure=='missing': b[:]=0
    if failure=='duplicate': b[12:25,12:25]=a[12:25,12:25]
    if failure=='recolored': b[32:45,52:65]=np.clip(b[32:45,52:65].astype(int)+10,0,255)
    if failure=='flat': a[:]=40;b[:]=41
    c,base=make(a,b)
    assert isinstance(c.correct_at_commit(100),ActionAbstention)
    assert base.calls == 1


def test_unsupported_operation_is_not_switched():
    a,b=images();c,_=make(a,b,{'kind':'press_key','key':'ENTER'})
    assert c.correct_at_commit(100).reason == 'unsupported_operation'


def test_unchanged_frame_preserves_model_action():
    a,_=images();c,base=make(a,a,{'kind':'click','x':90,'y':60})
    correction=c.correct_at_commit(100)
    assert correction.action==base.action
    assert correction.detail['unchanged_frame']


def test_harness_abstention_is_scored_no_action(tmp_path):
    from conftest import base_spec_dict
    from chronotope.spec import TrialSpec
    from chronotope.runner import PayloadStore,run_trial
    from chronotope.trace import Trace
    from chronotope.evaluate import evaluate_trace
    from chronotope.controller import ScriptedOracleController
    raw=base_spec_dict();raw['timestamp_source']='simulated'
    spec=TrialSpec.from_dict(raw);m=spec.manifest()
    class Suppressed(ScriptedOracleController):
        def correct_at_commit(self,commit_time_ns):
            return ActionAbstention('fixture_target_missing',1,{'track_ns':0})
    c=Suppressed(0,0,m.canvas_w_px,m.canvas_h_px,target_color=m.target_color())
    path=run_trial(spec,tmp_path/'fixture.jsonl',PayloadStore(tmp_path/'payloads'),controller=c)
    trace=Trace.load(path);result=evaluate_trace(trace)
    assert result.valid_trial and not result.contract_violations
    assert not result.valid_at_receipt and not result.action_committed
    assert trace.first('trial_end')['reason']=='safe_no_action'
    assert not trace.all('receipt') and len(trace.all('action_suppressed'))==1


def test_code_preserves_model_error_without_state_change():
    from chronotope.decoder import Decoded
    a,b=images();c,_=make(a,b,{'kind':'type','value':'9999'})
    with patch('chronotope.decoder.decode',return_value=Decoded('0123',(1,2,3,4),0,'ok')):
        assert c.correct_at_commit(100).action['value']=='9999'
    with patch('chronotope.decoder.decode',side_effect=[Decoded('0123',(1,2,3,4),0,'ok'),Decoded('0007',(1,2,3,4),0,'ok')]):
        assert c.correct_at_commit(100).action == {'kind':'type','value':'0007'}


def test_cuda_reference_parity():
    library=os.environ.get('LATEST_FRAME_TEST_LIBRARY')
    if not library: pytest.skip('Set LATEST_FRAME_TEST_LIBRARY for CUDA kernel contract check')
    a,b=images();matcher=CudaNCC(library,96,64,6)
    try:
        for target in (a,b,np.zeros_like(a)):
            ref=track(cut_template(a.tobytes(),96,64,18,18,6),target.tobytes(),96,64)
            got=matcher.match(a.tobytes(),target.tobytes(),18,18)
            assert (got.x,got.y,got.plateau)==(ref.x,ref.y,ref.plateau)
            assert abs(got.score-ref.score)<1e-5
            assert abs(got.second-ref.second)<1e-5
    finally: matcher.close()
