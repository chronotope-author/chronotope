import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies'))
from prepare_deadline_envelope import observed_threshold,assert_model_match,with_window,WINDOWS,prepare
from chronotope.distributional.manifest import GenConfig,generate,build_manifest


def test_nonmonotonic_threshold_requires_wider_windows_to_pass():
    points=[{'window_s':w,'n':100,'successes':k} for w,k in [(1,91),(2,85),(4,94),(8,97)]]
    assert observed_threshold(points)=={'window_s':4,'lower_tested_window_s':2,'status':'observed'}


def test_incomplete_and_censored_thresholds_are_explicit():
    with pytest.raises(ValueError):observed_threshold([{'window_s':1,'n':99,'successes':99}])
    assert observed_threshold([{'window_s':16,'n':100,'successes':89}])['status']=='not_reached'
    assert observed_threshold([{'window_s':.125,'n':100,'successes':90}])['status']=='below_tested_range'


def test_matching_checkpoint_precision_with_different_runtimes():
    legacy={'checkpoint_sha256':'same-checkpoint','precision':'mixed NVFP4/FP8/BF16','runtime':'TensorRT-Edge-LLM'}
    agents3={**legacy,'runtime':'vLLM'}
    assert_model_match(legacy,agents3)
    agents3['precision']='BF16'
    with pytest.raises(ValueError,match='precision'):assert_model_match(legacy,agents3)
    with pytest.raises(ValueError,match='checkpoint'):assert_model_match({}, {})


def test_deadline_changes_preserve_geometry_and_seeds():
    row=generate(GenConfig(task_family='closing_window',n=1,master_seed=26102831,empirical_windows_s=(16.,))).rows[0]
    for w in WINDOWS:
        changed=with_window(row,w)
        assert changed.environment_seed==row.environment_seed
        assert changed.stimulus_seed==row.stimulus_seed
        assert changed.target_start_position==row.target_start_position
        assert build_manifest(changed).response_window_ns==round(w*1e9)
    with pytest.raises(ValueError):with_window(row,.3)


def test_generated_plan_retains_every_evaluation_cell_without_screening(tmp_path):
    import json
    from collections import Counter
    out=tmp_path/'deadline'
    prepare(out, "radixark_27b")
    plan=json.loads((out/'plan.json').read_text())
    jobs=plan['jobs']
    assert len(jobs)==4800
    assert {j['phase'] for j in jobs}=={'evaluation'}
    assert {j['task'] for j in jobs}=={'notification'}
    counts=Counter((j['model'],j['arm'],j['window_s'],j['mode']) for j in jobs)
    assert len(counts)==48 and set(counts.values())=={100}
    assert {j['index'] for j in jobs}==set(range(100))
    assert len(list((out/'manifests').glob('*.jsonl')))==8
    assert plan['models']['radixark_27b']['precision']=='NVFP4 MLP/lm_head; FP8 attention; BF16 vision'


def test_future_w8_jobs_have_only_evaluation_conditions():
    from agents3_legacy_w8 import evaluation_jobs as agents3_jobs
    from legacy_d1_w8 import evaluation_jobs as legacy_jobs
    for jobs,expected in [(agents3_jobs(),600),(legacy_jobs(),1800)]:
        assert len(jobs)==expected
        assert {j['phase'] for j in jobs}=={'evaluation'}
        assert {j['mode'] for j in jobs}=={'live','replay','paused'}
        assert {j['family'] for j in jobs}=={'closing_window','moving_target'}
        assert len({j['id'] for j in jobs})==expected


def test_radixark_parent_cannot_create_qwen08_comparison(tmp_path):
    import json
    from agents3_legacy_w8 import prepare as prepare_baseline
    parent=tmp_path/'old';parent.mkdir()
    (parent/'manifest.json').write_text(json.dumps({'model':'RadixArk/Qwen3.8-27B-NVFP4','precision':'NVFP4'}))
    (parent/'status.json').write_text(json.dumps({'state':'complete'}))
    out=tmp_path/'comparison'
    with pytest.raises(ValueError,match='Same-model comparison required'):
        prepare_baseline(out,parent,tmp_path/'model',tmp_path/'service')
    assert not out.exists()


def test_execution_runner_rejects_model_mismatch_and_extra_phase(tmp_path):
    import copy,json
    from run_same_model_deadline import validate_plan
    out=tmp_path/'execution';prepare(out, "radixark_27b")
    plan=json.loads((out/'plan.json').read_text());validate_plan(plan)
    mismatch=copy.deepcopy(plan);mismatch['models']['radixark_27b']['id']='Qwen/Qwen3.5-0.8B'
    with pytest.raises(AssertionError):validate_plan(mismatch)
    screening=copy.deepcopy(plan);screening['jobs'][0]['phase']='qualification'
    with pytest.raises(AssertionError):validate_plan(screening)
    extra_task=copy.deepcopy(plan);extra_task['jobs'][0]['task']='moving_balloon_select'
    with pytest.raises(AssertionError):validate_plan(extra_task)


def test_record_audit_accepts_real_header_and_keeps_competence_failure(tmp_path):
    import hashlib,json
    from watch_same_model_deadline import audit_record
    rows=[{'record':'trial_header','spec':{'controller':{'model_id':'RadixArk/Qwen3.8-27B-NVFP4'}}},
          {'record':'decision_invocation','planner':'tensorrt_edgellm','outcome':'wait'}]
    trace=tmp_path/'trace.jsonl';trace.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    record={'job':{'phase':'evaluation','model':'radixark_27b','mode':'live'},'trace':trace.name,
            'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
            'outcome':{'Y':0,'invalid_trial':False,'contract_violations':[]}}
    audit_record(tmp_path,record)
    rows[1]['outcome']='unparseable'
    trace.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    record['trace_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='TensorRT request failed'):audit_record(tmp_path,record)


def test_qwen08_pair_has_both_fresh_arms_and_rejects_mismatch(tmp_path):
    import copy,json
    from run_qwen08_deadline import validate_plan
    out=tmp_path/'qwen08';prepare(out,'qwen35_0p8b')
    plan=json.loads((out/'plan.json').read_text());validate_plan(plan)
    assert plan['jobs'][0]['window_s']==16
    assert plan['models']['qwen35_0p8b']['precision']=='FP16'
    assert {j['arm'] for j in plan['jobs']}=={'legacy_cpp','agents3_stripped'}
    bad=copy.deepcopy(plan);bad['models']['qwen35_0p8b']['precision']='BF16'
    with pytest.raises(AssertionError):validate_plan(bad)
    bad=copy.deepcopy(plan);bad['jobs'][0]['phase']='qualification'
    with pytest.raises(AssertionError):validate_plan(bad)


def test_qwen08_audit_checks_actual_model_identity(tmp_path):
    import hashlib,json
    from watch_same_model_deadline import audit_record
    rows=[{'record':'trial_header','spec':{'controller':{'model_id':'Qwen/Qwen3.5-0.8B'}}}]
    trace=tmp_path/'trace.jsonl';trace.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    record={'job':{'phase':'evaluation','model':'qwen35_0p8b','mode':'live'},'trace':trace.name,
        'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'outcome':{'Y':0,'invalid_trial':False,'contract_violations':[]}}
    audit_record(tmp_path,record)
    rows[0]['spec']['controller']['model_id']='RadixArk/Qwen3.8-27B-NVFP4'
    trace.write_text('\n'.join(json.dumps(x) for x in rows)+'\n');record['trace_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
    with pytest.raises(AssertionError,match='Model mismatch'):audit_record(tmp_path,record)
