"""Run the same Agent S3 Worker policy and RadixArk Qwen3.8-27B mixed precision in native C++ and vLLM.
Evaluation only; no competence filtering, extra smoke episodes, or implicit resume.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from collections import defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import urllib.request
from legacy_d1_w8 import digest, now, write, setup

ROOT = Path(__file__).resolve().parents[1]
MODEL='RadixArk/Qwen3.8-27B-NVFP4'
REVISION='319f741cce68d7914884900c138a1fbb70a42f30'
PRECISION='NVFP4 MLP/lm_head; FP8 attention; BF16 vision'
CHECKPOINT=ROOT/'runs/model_cache/radixark_qwen38_nvfp4_319f741c'
PREDECESSOR=ROOT/'runs/agents3_policy_qwen08_20261028/experiment'
ARMS=('legacy_cpp','agents3_stripped')
WINDOWS=(.125,.25,.5,1.,2.,4.,8.,16.)


def validate_plan(plan):
    assert set(plan['models'])=={'radixark_27b'}, 'One identical model in both arms'
    model=plan['models']['radixark_27b']
    assert (model['id'],model['revision'],model['precision'])==(MODEL,REVISION,PRECISION)
    assert plan['planned_trials']==4800 and plan['qualification_trials']==0
    jobs=plan['jobs'];assert len(jobs)==4800
    keys=set();groups=defaultdict(set)
    for j in jobs:
        assert j['phase']=='evaluation' and j['task']=='notification' and j['family']=='closing_window'
        assert j['model']=='radixark_27b' and j['arm'] in ARMS and j['window_s'] in WINDOWS
        assert j['mode'] in ('live','replay','paused') and 0<=j['index']<100
        k=(j['arm'],j['window_s'],j['index'],j['mode']);assert k not in keys;keys.add(k)
        groups[j['arm'],j['window_s'],j['mode']].add(j['index'])
    assert len(groups)==48 and all(v==set(range(100)) for v in groups.values())


def job_id(j):
    return f"{j['arm']}_w{round(j['window_s']*1e9)}_{j['index']:03d}_{j['mode']}"


def run(out, resume=False):
    require_predecessor_complete(out)
    setup(out/'code');sys.path.insert(0,str(out/'code'))
    from agents3_stripped import StrippedAgentS3Adapter
    from audit_agents3_policy import audit_trial_requests
    from chronotope.distributional.manifest import ManifestSet,build_manifest
    from chronotope.distributional.episode import ControllerSpec,trial_spec_for,make_controller,verify_trace_evidence,score_trace
    from chronotope.distributional.metrics import percentiles
    from chronotope.controller import CallBudget
    from chronotope.runner import run_trial,PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    plan=json.loads((out/'plan.json').read_text());validate_plan(plan)
    cfg=json.loads((out/'execution_manifest.json').read_text())
    os.environ['CHRONOTOPE_AGENT_S3_POLICY']=str(out/'configs/agents3_policy.json')
    for rel,sha in cfg['hashes'].items():assert digest(out/rel)==sha, rel
    for path,sha in {**cfg['checkpoint_hashes'],**cfg['engine_hashes']}.items():assert digest(path)==sha,path
    assert digest(cfg['plugin']['path'])==cfg['plugin']['sha256']
    with urllib.request.urlopen(cfg['model_service']+'/models',timeout=15) as response:models=json.load(response)
    assert [m['id'] for m in models['data']]==[MODEL]
    lock=(out.parent.parent/'legacy_d1_gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifests={name:ManifestSet.load(out/'manifests'/name) for name in plan['manifest_hashes']}
    configs={arm:json.loads((out/'configs'/f'{arm}.json').read_text()) for arm in ARMS}
    previous=json.loads((out/'status.json').read_text())
    existing=list((out/'records').glob('*.json'))
    if existing and not resume:raise RuntimeError('Use explicit --resume after validating partial records')
    records=[json.loads(p.read_text()) for p in sorted(existing)]
    planned={job_id(j):j for j in plan['jobs']}
    completed=set()
    for record in records:
        ident=job_id(record['job'])
        assert ident in planned and record['job']==planned[ident] and ident not in completed
        assert digest(out/record['trace'])==record['trace_sha256']
        assert not record['outcome']['invalid_trial'] and not record['outcome']['contract_violations']
        completed.add(ident)
    started=previous.get('started_at',now()) if resume else now();current=None
    def publish(state,error=None):
        groups=defaultdict(list)
        for r in records:groups[r['job']['arm'],r['job']['window_s'],r['job']['mode']].append(r)
        rows=[]
        for (arm,w,mode),rs in sorted(groups.items()):
            outcomes=[r['outcome'] for r in rs]
            row={'Task':'Disappearing Notification','Controller':cfg['arm_labels'][arm],
                'Model':MODEL,'Revision':REVISION,'Precision':PRECISION,
                'Runtime':'TensorRT-Edge-LLM' if arm=='legacy_cpp' else 'vLLM',
                'KV cache':'FP8 E4M3; static scales 1.0',
                'Window (s)':w,'Condition':mode,'n':len(rs),'Successes':sum(o['Y'] for o in outcomes),
                'SR':sum(o['Y'] for o in outcomes)/len(rs),'No receipt':sum(not o['R'] for o in outcomes),
                'Invalid':sum(o['invalid_trial'] for o in outcomes),'Replay pairs valid':sum(r['pair_valid'] is True for r in rs)}
            for label,key in [('E-to-O','L_EO'),('O-to-B','L_OB'),('B-to-D','L_BD'),('D-to-A','L_DA'),('E-to-A','L_EA')]:
                for pc,val in percentiles([o[key] for o in outcomes]).items():
                    if pc in ('p50','p95'):row[f'{label} {pc} (ms)']=None if val is None else val/1e6
            od=[o['D']-o['O'] for o in outcomes if o['D'] is not None and o['O'] is not None]
            row['O-to-D p50 (ms)']=None if not od else percentiles(od)['p50']/1e6
            rows.append(row)
        if rows:
            with (out/'summary.csv.tmp').open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
            (out/'summary.csv.tmp').replace(out/'summary.csv')
        write(out/'status.json',{'state':state,'started_at':started,'updated_at':now(),'completed_trials':len(records),
            'planned_trials':4800,'pid':os.getpid() if state=='running' else None,'current_job':current,
            'qualification_trials':0,'model':MODEL,'error':error,'automatic_resume':False})
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    publish('running')
    try:
        for current in plan['jobs']:
            ident=job_id(current)
            if ident in completed:continue
            publish('running');arm=current['arm'];mode=current['mode']
            m=manifests[current['manifest']];row=m.rows[current['index']];scenario=build_manifest(row)
            ctrl=ControllerSpec(arm,configs[arm])
            folder=out/'trials'/f"w{round(current['window_s']*1e9)}"/arm;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{row.episode_id}_{mode}.jsonl';assert not path.exists(),path
            live=Trace.load(folder/f'{row.episode_id}_live.jsonl') if mode=='replay' else None
            spec=trial_spec_for(row,ctrl,mode,live.file_hash() if live else None)
            adapter=StrippedAgentS3Adapter(cfg['model_service'],out/'calls'/f'{ident}.jsonl',model_id=MODEL,max_tokens=configs[arm]['max_tokens']) if arm=='agents3_stripped' else None
            os.environ['CHRONOTOPE_AGENT_S3_AUDIT']=str(out/'calls'/f'{ident}.jsonl')
            controller=make_controller(spec,scenario,ctrl,budget=CallBudget(100),vlm_adapter=adapter)
            t0=time.monotonic()
            try:run_trial(spec,path,PayloadStore(out/'payloads'),live_trace=live,controller=controller,manifest=scenario)
            finally:
                close=getattr(controller,'close',None)
                if close:close()
            trace=Trace.load(path)
            native_errors=[e for e in trace.all('decision_invocation') if e.get('planner')=='tensorrt_edgellm' and e.get('outcome')=='unparseable']
            if trace.all('decision_failed') or native_errors or (adapter and adapter.errors):raise RuntimeError(f'Controller/runtime error: {ident}')
            request_audit=audit_trial_requests(out,ident,arm,trace,scenario,cfg)
            verify_trace_evidence(row,scenario,trace,m.frozen_hash)
            assert trace.header()['spec']['controller']==configs[arm]
            outcome=score_trace(row,trace,m.frozen_hash,controller=arm,repeat_index=0)
            pair=validate_pair(trace,live) if live else None
            rec={'job':current,'request_audit':request_audit,'outcome':outcome.to_dict(),'trace':str(path.relative_to(out)),
                 'trace_sha256':trace.file_hash(),'pair_valid':pair.valid if pair else None,'pair_reason':pair.reason if pair else None,
                 'wall_seconds':time.monotonic()-t0,'completed_at':now()}
            write(out/'records'/f'{ident}.json',rec);records.append(rec)
            if outcome.invalid_trial or outcome.contract_violations or (pair and not pair.valid):raise RuntimeError(f'Invalid trace/pair: {ident}')
            print(f'{len(records)}/4800 {ident} success={outcome.Y}',flush=True);publish('running')
        current=None;publish('complete')
    except BaseException as exc:
        publish('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',f'{type(exc).__name__}: {exc}');raise


def read(path):return json.loads(Path(path).read_text())


def same_cohort(plan,parent):
    assert plan['manifest_hashes']==parent['manifest_hashes']
    assert [{k:v for k,v in j.items() if k!='model'} for j in plan['jobs']]==[
        {k:v for k,v in j.items() if k!='model'} for j in parent['jobs']]


def validate_specs(out,plan,code):
    setup(code)
    from chronotope.distributional.manifest import ManifestSet,build_manifest,verify_row_against_manifest
    from chronotope.distributional.episode import ControllerSpec,trial_spec_for
    manifests={n:ManifestSet.load(out/'manifests'/n) for n in plan['manifest_hashes']}
    configs={arm:read(out/'configs'/f'{arm}.json') for arm in ARMS}
    for key in ['model_id','prompt_version','max_tokens','image_scale','temperature','enable_thinking','invoke_on']:
        assert configs[ARMS[0]][key]==configs[ARMS[1]][key],key
    assert configs['legacy_cpp']['model_id']==MODEL
    assert configs['legacy_cpp']['planner']=='trt' and configs['legacy_cpp']['server']=='cpp_socket'
    for name,m in manifests.items():
        assert m.frozen_hash==plan['manifest_hashes'][name] and len(m.rows)==100
        for row in m.rows:verify_row_against_manifest(row,build_manifest(row))
    prior=set()
    for job in plan['jobs']:
        row=manifests[job['manifest']].rows[job['index']]
        assert row.task_family=='closing_window' and row.deadline_s==job['window_s'] and row.episode_id==job['episode_id']
        key=(job['arm'],job['window_s'],job['index'],job['mode'])
        if job['mode']=='replay':assert key[:3]+('live',) in prior
        prior.add(key)
        trial_spec_for(row,ControllerSpec(job['arm'],configs[job['arm']]),job['mode'],'0'*64 if job['mode']=='replay' else None)
    return len(plan['jobs'])


def prepare(out):
    assert not out.exists(), 'Existing successor is never overwritten'
    parent=read(PREDECESSOR/'plan.json');plan=json.loads(json.dumps(parent))
    plan.update(created_at=now(),state='prepared_awaiting_predecessor_audit',qualification_trials=0,
        predecessor=str(PREDECESSOR),predecessor_plan_sha256=digest(PREDECESSOR/'plan.json'),
        comparison='Fresh legacy C++/TensorRT-Edge-LLM and stripped Agent S3/vLLM; same Radix checkpoint and Worker policy')
    plan['models']={'radixark_27b':{'id':MODEL,'revision':REVISION,'precision':PRECISION,
        'checkpoint_path':str(CHECKPOINT),'kv_cache':'FP8 E4M3; static scales 1.0',
        'legacy_tensorrt_engine_status':'Deferred 16k build; freeze requires provenance and input-alignment proofs'}}
    for job in plan['jobs']:job['model']='radixark_27b'
    validate_plan(plan);same_cohort(plan,parent)
    out.mkdir(parents=True);shutil.copytree(PREDECESSOR/'manifests',out/'manifests');(out/'configs').mkdir()
    for arm in ARMS:
        cfg=read(PREDECESSOR/'configs'/f'{arm}.json');cfg['model_id']=MODEL
        cfg['base_url']=str(out.parent/'engines') if arm=='legacy_cpp' else 'http://127.0.0.1:8012/v1'
        write(out/'configs'/f'{arm}.json',cfg)
    from agents3_policy_contract import policy_fingerprint
    policy=read(PREDECESSOR/'configs/agents3_policy.json');policy['model']=MODEL
    policy['native_chat_template']={'strip_historical_thinking':False,
        'assistant_prefix':'<|im_start|>assistant\n<think>\n\n</think>\n\n'}
    policy['policy_sha256']=policy_fingerprint(policy)
    inputs=out.parent/'policy_inputs';inputs.mkdir(exist_ok=True)
    write(inputs/'agents3_policy.json',policy);write(out/'configs/agents3_policy.json',policy)
    fixtures=read(PREDECESSOR/'code/agents3_notification_policy_fixtures.json')
    fixtures['policy_sha256']=policy['policy_sha256'];write(inputs/'policy_fixtures.json',fixtures)
    write(out/'plan.json',plan);checks=validate_specs(out,plan,PREDECESSOR/'code')
    manifests={p.name:digest(p) for p in (out/'manifests').iterdir() if p.is_file()}
    assert manifests=={p.name:digest(p) for p in (PREDECESSOR/'manifests').iterdir() if p.is_file()}
    write(out/'preparation_report.json',{'state':'passed','created_at':now(),'spec_checks':checks,
        'experimental_trials':0,'model_inference_calls':0,'qualification_trials':0,'same_ordered_cohort':True,
        'predecessor':str(PREDECESSOR),'predecessor_plan_sha256':plan['predecessor_plan_sha256'],
        'manifest_file_sha256':manifests,'model':MODEL,'revision':REVISION,'precision':PRECISION,'policy_sha256':policy['policy_sha256']})
    write(out/'status.json',{'state':'prepared_awaiting_predecessor_audit','completed_trials':0,
        'planned_trials':4800,'qualification_trials':0,'pid':None,'model':MODEL,'automatic_resume':False})
    print(json.dumps({'state':'prepared','spec_checks':checks,'planned_trials':4800,'policy_sha256':policy['policy_sha256']}),flush=True)


def require_predecessor_complete(out):
    status=read(PREDECESSOR/'status.json')
    assert status['state']=='complete' and status['completed_trials']==4800 and status['planned_trials']==4800
    assert status.get('qualification_trials')==0 and not status.get('error')
    audit=read(out.parent/'predecessor_audit.json')
    assert audit['state']=='finished' and audit['records']==4800
    assert audit['infrastructure_errors']==0 and audit['task_failures_retained'] is True
    return audit


def verify_hashes(mapping):
    assert mapping
    for path,sha in mapping.items():assert digest(path)==sha,path


def validate_service(launch):
    assert launch['model']==MODEL and launch['revision']==REVISION
    assert launch['dtype']=='bfloat16' and launch['quantization']=='modelopt_mixed'
    assert launch['kv_cache_dtype']=='fp8_e4m3' and launch['calculate_kv_scales'] is False
    assert launch['max_model_len']==16384 and launch['max_num_seqs']==1
    assert launch['enable_prefix_caching'] is False


def freeze(out,engine_root):
    require_predecessor_complete(out)
    assert not (out/'execution_manifest.json').exists(), 'Already frozen'
    plan=read(out/'plan.json');validate_plan(plan);same_cohort(plan,read(PREDECESSOR/'plan.json'))
    assert digest(PREDECESSOR/'plan.json')==plan['predecessor_plan_sha256']
    root=out.parent;build=read(root/'engine_build_result.json');build_launch=read(root/'engine_build_launch.json')
    assert (build['state'],build['model'],build['revision'])==('built',MODEL,REVISION)
    for key in ['max_input_len','max_kv_cache_capacity']:assert build['builder_config'][key]==16384
    assert build['model_input_alignment']['aligned'] is True
    verify_hashes(build['hashes']);verify_hashes(build_launch['inputs']);verify_hashes(build['native_binary_hashes'])
    source_identity=read(ROOT/'runs/same_model_radixark_20261028/experiment/execution_manifest.json')
    assert (source_identity['model'],source_identity['revision'],source_identity['precision'])==(MODEL,REVISION,PRECISION)
    assert build['source_checkpoint_hashes']==source_identity['checkpoint_hashes']
    assert build_launch['source_checkpoint_hashes']==source_identity['checkpoint_hashes']
    assert set(build['native_binary_hashes'])=={str(root/'native_build'/n) for n in ('chronotope','libchronotope.so')}
    verify_hashes(build['source_checkpoint_hashes'])
    if build.get('native_source_hashes'):verify_hashes(build['native_source_hashes'])
    assert build['kv_cache']['dtype']=='FP8 E4M3' and build['kv_cache']['scales']=='static1.0'
    launch=read(root/'services/launch.json');validate_service(launch)
    report=read(root/'policy_verification.json');assert report['state']=='passed'
    assert report['normalization_precision']['bf16_first_roundtrip_equal'] is True
    assert report['normalization_precision']['full_patch_bf16_equal'] is True
    verify_hashes(report['source_file_sha256'])
    gpu=read(root/'runtime/gpu_input_verification.json')
    assert gpu['passed'] and gpu['inference_calls']==0 and gpu['evaluation_trials']==0
    verify_hashes(gpu['source_hashes'])
    assert digest(root/'runtime/gpu_input_verification.json')==build['model_input_alignment']['gpu_proof_sha256']
    runtime_build=read(root/'runtime/build_report.json')
    assert runtime_build['dedicated_model']==MODEL
    assert digest(root/'runtime/build/cpp/libedgellmCore.a')==runtime_build['patched_core_sha256']
    assert digest(root/'runtime/cpp/kernels/preprocessKernels/imageUtilKernels.cu')==runtime_build['normalization_source_sha256']
    from agents3_policy_contract import build_policy_contract,policy_fingerprint
    setup(PREDECESSOR/'code');policy=build_policy_contract(model_id=MODEL)
    assert policy==read(out/'configs/agents3_policy.json') and report['policy_sha256']==policy['policy_sha256']
    code=out/'code'
    shutil.copytree(PREDECESSOR/'code',code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.rmtree(code/'chronotope/src');shutil.copytree(root/'build_source/chronotope/src',code/'chronotope/src')
    shutil.copy2(root/'build_source/chronotope/CMakeLists.txt',code/'chronotope/CMakeLists.txt')
    for name in ('chronotope','libchronotope.so'):shutil.copy2(root/'native_build'/name,code/'chronotope/build'/name)
    for name in ('run_radix_agents3_policy.py','agents3_policy_contract.py','audit_agents3_policy.py','verify_agents3_policy.py','verify_radix_agents3_policy.py'):
        shutil.copy2(ROOT/'studies'/name,code/name)
    for source,dest in [('agents3_policy.json','agents3_notification_policy_contract.json'),('policy_fixtures.json','agents3_notification_policy_fixtures.json')]:
        shutil.copy2(root/'policy_inputs'/source,code/dest)
    shutil.rmtree(code/'cpu_policy_verification');shutil.copytree(root/'cpu_policy_verification',code/'cpu_policy_verification')
    for sub in ('records','calls'):(out/sub).mkdir()
    shutil.copy2(ROOT/'AGENTS.md',out/'AGENTS.md')
    for name in ['engine_build_result.json','engine_build_launch.json','policy_verification.json','predecessor_audit.json']:
        shutil.copy2(root/name,out/name)
    shutil.copy2(root/'services/launch.json',out/'service_launch.json')
    shutil.copy2(root/'runtime/gpu_input_verification.json',out/'gpu_input_verification.json')
    shutil.copy2(root/'runtime/build_report.json',out/'runtime_build_report.json')
    shutil.copy2(root/'runtime/cpp/kernels/preprocessKernels/imageUtilKernels.cu',code/'native_agents3_runtime/radix_imageUtilKernels.cu')
    checks=validate_specs(out,plan,code)
    artifacts={str(p.relative_to(out)):digest(p) for sub in ('code','configs','manifests') for p in (out/sub).rglob('*') if p.is_file()}
    for name in ['plan.json','preparation_report.json','AGENTS.md','engine_build_result.json','engine_build_launch.json',
                 'policy_verification.json','service_launch.json','gpu_input_verification.json','runtime_build_report.json','predecessor_audit.json']:
        artifacts[name]=digest(out/name)
    engines={str(p.resolve()):digest(p) for p in engine_root.rglob('*') if p.is_file()}
    assert engines and all(str(Path(p).resolve()).startswith(str(engine_root.resolve())+'/') for p in build['hashes'])
    execution={'created_at':now(),'model':MODEL,'revision':REVISION,'precision':PRECISION,'arms':ARMS,
        'planned_trials':4800,'checkpoint_path':str(CHECKPOINT),'image_tokens':300,
        'checkpoint_hashes':build['source_checkpoint_hashes'],'engine_hashes':engines,'hashes':artifacts,
        'plugin':{'path':(_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so'),
                  'sha256':digest((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so'))},
        'model_service':read(out/'configs/agents3_stripped.json')['base_url'],'spec_checks':checks,'qualification_trials':0,
        'request_contract':policy,'policy_sha256':policy_fingerprint(policy),'kv_cache':build['kv_cache'],
        'arm_labels':{'legacy_cpp':'Legacy C++ (Agent S3 policy)','agents3_stripped':'Agent S3 (stripped)'},
        'runtime_patch':runtime_build,'runtime_numerics':build['runtime_numerics'],'model_input_alignment':build['model_input_alignment'],
        'interpretation':['Fresh paired trials; identical checkpoint/component quantization and Agent S3 Worker policy',
            'Same screenshot, model-specific chat template, input token IDs, policy/history rules and generation settings',
            'BF16 normalized image values aligned through dedicated native BF16-first preprocessing',
            'Runtime kernels and generated model outputs may differ; subsequent generated text histories may differ',
            'Legacy C++ uses CPU text policy/history; not full GPU-history Chronotope',
            'Exact predecessor100Notificationcases and eight windows; no competence gates or qualification trials'],
        'cpu_work':['C++ Worker policy/history, runtime token loop, action parsing; Python gym/socket referee',
                    'Agent S3 Python Worker, CPU text history/reflection and vLLM HTTP service']}
    write(out/'execution_manifest.json',execution)
    write(out/'status.json',{'state':'frozen_ready','completed_trials':0,'planned_trials':4800,'pid':None,'qualification_trials':0})
    print(json.dumps({'state':'frozen_ready','spec_checks':checks,'same_model':MODEL,'policy_sha256':policy['policy_sha256']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    group=ap.add_mutually_exclusive_group(required=True)
    for action in ('prepare','freeze','run'):group.add_argument('--'+action,action='store_true')
    ap.add_argument('--engines',type=Path);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.resume and not args.run:ap.error('--resume requires --run')
    if args.prepare:prepare(args.out.resolve())
    elif args.freeze:
        if args.engines is None:ap.error('--freeze requires --engines')
        freeze(args.out.resolve(),args.engines.resolve())
    else:run(args.out.resolve(),resume=args.resume)
