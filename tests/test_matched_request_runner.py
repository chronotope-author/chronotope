"""Checks of matched model-request evidence; no inference or experimental trials."""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'studies'),str(Path(__file__).resolve().parents[1]/'src')]
from run_qwen08_matched_requests import audit_requests,MODEL
from agents3_matched_request import contract_fingerprint


def fixture(tmp_path):
    raw=b'\x80'*(320*240*3);ph=hashlib.sha256(raw).hexdigest()
    (tmp_path/'payloads').mkdir();(tmp_path/'payloads'/f'{ph}.bin').write_bytes(raw)
    rgb=b'\x80'*(640*480*3)
    event={'decision_id':1,'observation_id':1,'model_reported':MODEL,'input_tokens':449,'output_tokens':25,
        'request_contract_sha256':contract_fingerprint(),'source_rgb_sha256':hashlib.sha256(rgb).hexdigest(),
        'source_image_wh':[640,480],'planner_calls':1,'reflection_calls':0,'history_turns':0}
    rows={'observation':[{'observation_id':1,'payload_hash':ph}],'decision_invocation':[event]}
    trace=SimpleNamespace(all=lambda key:rows.get(key,[]));scenario=SimpleNamespace(canvas_w_px=320,canvas_h_px=240)
    cfg={'expected_input_tokens':449,'request_contract_sha256':contract_fingerprint()}
    return event,trace,scenario,cfg


def test_valid_identical_request_evidence(tmp_path):
    event,trace,scenario,cfg=fixture(tmp_path)
    native=audit_requests(tmp_path,trace,scenario,cfg,'legacy_cpp')
    host=audit_requests(tmp_path,trace,scenario,cfg,'agents3_stripped')
    assert native==host and native[0]['input_tokens']==449


@pytest.mark.parametrize('field,value', [('input_tokens',450),('output_tokens',65),('history_turns',1),('reflection_calls',1),('source_rgb_sha256','wrong')])
def test_mismatched_request_evidence_fails(tmp_path,field,value):
    event,trace,scenario,cfg=fixture(tmp_path);event[field]=value
    with pytest.raises(AssertionError):audit_requests(tmp_path,trace,scenario,cfg,'agents3_stripped')
