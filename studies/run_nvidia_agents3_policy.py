"""Run the same Agent S3 Worker policy and NVIDIA Qwen3.8-27B mixed precision in native C++ and vLLM.
Evaluation only; no competence filtering, extra smoke episodes, or implicit resume.
"""
from __future__ import annotations
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
from legacy_d1_w8 import digest, now, write, setup as legacy_setup

ROOT = Path(__file__).resolve().parents[1]
MODEL='nvidia/Qwen3.8-27B-NVFP4'
REVISION='482ca0f3832238542f8f5295dde86b5f22711d80'
PRECISION='NVFP4 MLP/lm_head; FP8 attention; BF16 vision'
CHECKPOINT=ROOT/'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f'
PREDECESSOR=ROOT/'runs/agents3_policy_qwen08_20261028/experiment'
RUNTIME_PRECISION='NVFP4 MLP/lm_head; FP8 attention projections; FP16 vision and unquantized LLM tensors'
PLUGIN=ROOT/'runs/runtime_reversal_stock_20261029/build/libNvInfer_edgellm_plugin.so'
ARMS=('legacy_cpp','agents3_stripped')
WINDOWS=(.125,.25,.5,1.,2.,4.,8.,16.)


def setup(code):
    legacy_setup(code)
    sys.path.insert(0,str(code))
    os.environ['EDGELLM_PLUGIN_PATH']=str(PLUGIN)


def validate_plan(plan):
    assert set(plan['models'])=={'nvidia_27b'}, 'One identical model in both arms'
    model=plan['models']['nvidia_27b']
    assert (model['id'],model['revision'],model['precision'])==(MODEL,REVISION,PRECISION)
    assert plan['planned_trials']==4800 and plan['qualification_trials']==0
    jobs=plan['jobs'];assert len(jobs)==4800
    keys=set();groups=defaultdict(set)
    for j in jobs:
        assert j['phase']=='evaluation' and j['task']=='notification' and j['family']=='closing_window'
        assert j['model']=='nvidia_27b' and j['arm'] in ARMS and j['window_s'] in WINDOWS
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
                'Model':MODEL,'Revision':REVISION,'Source precision':PRECISION,'Precision':RUNTIME_PRECISION,
                'Runtime':'TensorRT-Edge-LLM' if arm=='legacy_cpp' else 'vLLM',
                'KV cache':'FP16',
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
    assert not out.exists(), 'Existing experiment is never overwritten'
    parent=read(PREDECESSOR/'plan.json');plan=json.loads(json.dumps(parent))
    plan.update(created_at=now(),state='prepared',qualification_trials=0,
        predecessor=str(PREDECESSOR),predecessor_plan_sha256=digest(PREDECESSOR/'plan.json'),
        comparison='Fresh legacy C++/stock TensorRT-Edge-LLM and stripped Agent S3/vLLM; same NVIDIA checkpoint and Worker policy')
    plan['models']={'nvidia_27b':{'id':MODEL,'revision':REVISION,'precision':PRECISION,
        'runtime_precision':RUNTIME_PRECISION,'checkpoint_path':str(CHECKPOINT),'kv_cache':'FP16',
        'legacy_tensorrt_engine_status':'Fresh NVIDIA export/build required before freeze'}}
    plan['supersedes']='runs/agents3_policy_radixark27b_20261028/experiment; no further Radix trials authorized'
    plan['settings_changes_from_predecessor']={'repetition_penalty':{'old':1.05,'new':1.0,'reason':'Pristine NVIDIA runtime exposes no repetition penalty field; equal1.0 in both arms.'}}
    for job in plan['jobs']:job['model']='nvidia_27b'
    validate_plan(plan);same_cohort(plan,parent)
    out.mkdir(parents=True);shutil.copytree(PREDECESSOR/'manifests',out/'manifests');(out/'configs').mkdir()
    code=out/'code';shutil.copytree(PREDECESSOR/'code',code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # These are isolated application-policy adapters; upstream backend sources are untouched.
    policy_source=(ROOT/'studies/agents3_policy_contract.py').read_text().replace('1.05','1.0')
    policy_source=policy_source.replace("if model_id == 'RadixArk/Qwen3.8-27B-NVFP4':", "if model_id == 'nvidia/Qwen3.8-27B-NVFP4':")
    policy_source=policy_source.replace("contract['policy_sha256'] = policy_fingerprint(contract)", "contract['native_request_mode']='checkpoint_formatted_public_api'\n    contract['policy_sha256'] = policy_fingerprint(contract)")
    (code/'agents3_policy_contract.py').write_text(policy_source)
    shutil.copy2(ROOT/'studies/audit_nvidia_agents3_policy.py',code/'audit_agents3_policy.py')
    watcher=(ROOT/'studies/watch_same_model_deadline.py').read_text().replace("MODELS={'radixark_27b':MODEL,'qwen35_0p8b':'Qwen/Qwen3.5-0.8B'}", "MODELS={'nvidia_27b':'nvidia/Qwen3.8-27B-NVFP4'}").replace("MODEL='RadixArk/Qwen3.8-27B-NVFP4'", "MODEL='nvidia/Qwen3.8-27B-NVFP4'")
    (code/'watch_same_model_deadline.py').write_text(watcher)
    for arm in ARMS:
        cfg=read(PREDECESSOR/'configs'/f'{arm}.json');cfg['model_id']=MODEL
        cfg['base_url']=str(out.parent/'engines') if arm=='legacy_cpp' else 'http://127.0.0.1:8012/v1'
        write(out/'configs'/f'{arm}.json',cfg)
    setup(code)
    from agents3_policy_contract import build_policy_contract,build_cpu_fixtures
    policy=build_policy_contract(model_id=MODEL)
    assert policy['repetition_penalty']==1.0 and policy['generation_settings']['repetition_penalty']==1.0
    inputs=out.parent/'policy_inputs';inputs.mkdir(exist_ok=True)
    write(inputs/'agents3_policy.json',policy);write(out/'configs/agents3_policy.json',policy)
    write(inputs/'policy_fixtures.json',build_cpu_fixtures(policy))
    write(out/'plan.json',plan);checks=validate_specs(out,plan,code)
    manifests={p.name:digest(p) for p in (out/'manifests').iterdir() if p.is_file()}
    assert manifests=={p.name:digest(p) for p in (PREDECESSOR/'manifests').iterdir() if p.is_file()}
    write(out/'preparation_report.json',{'state':'passed','created_at':now(),'spec_checks':checks,
        'experimental_trials':0,'model_inference_calls':0,'qualification_trials':0,'same_ordered_cohort':True,
        'predecessor':str(PREDECESSOR),'predecessor_plan_sha256':plan['predecessor_plan_sha256'],
        'manifest_file_sha256':manifests,'model':MODEL,'revision':REVISION,'precision':PRECISION,
        'runtime_precision':RUNTIME_PRECISION,'policy_sha256':policy['policy_sha256']})
    audit=read(PREDECESSOR/'audit_status.json');assert audit['state']=='finished' and audit['records']==4800 and audit['infrastructure_errors']==0
    write(out.parent/'predecessor_audit.json',audit)
    write(out/'status.json',{'state':'prepared','completed_trials':0,'planned_trials':4800,'qualification_trials':0,'pid':None,'model':MODEL,'automatic_resume':False})
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
    assert launch['dtype']=='half' and launch['quantization']=='modelopt_mixed'
    assert launch['kv_cache_dtype']=='auto' and launch['effective_kv_cache_dtype']=='float16'
    assert launch['max_model_len']==16384 and launch['max_num_seqs']==1
    assert launch['enable_prefix_caching'] is False


def freeze(out,engine_root):
    require_predecessor_complete(out)
    assert not (out/'execution_manifest.json').exists(), 'Already frozen'
    plan=read(out/'plan.json');validate_plan(plan);same_cohort(plan,read(PREDECESSOR/'plan.json'))
    assert digest(PREDECESSOR/'plan.json')==plan['predecessor_plan_sha256']
    root=out.parent;build=read(root/'engine_build_result.json')
    assert (build['state'],build['model'],build['revision'])==('built',MODEL,REVISION)
    for key in ['max_input_len','max_kv_cache_capacity']:assert build['builder_config'][key]==16384
    assert build['builder_config']['max_batch_size']==1
    assert build['upstream_commit']=='bb291453bc10d9bd1142070c605e090d4e483ee2' and build['source_patches']==[]
    for key in ['hashes','native_binary_hashes','native_source_hashes','source_checkpoint_hashes']:verify_hashes(build[key])
    assert set(build['native_binary_hashes'])=={str(root/'native_build'/n) for n in ('chronotope','libchronotope.so')}
    assert build['kv_cache']['dtype']=='FP16'
    launch=read(root/'services/launch.json');validate_service(launch)
    report=read(root/'policy_verification.json');assert report['state']=='passed'
    assert report['native_hf_chat_text_equal'] and report['native_hf_expanded_input_ids_equal']
    assert report['patch_fp16_equal'] is True
    verify_hashes(report['source_file_sha256'])
    code=out/'code';setup(code)
    from agents3_policy_contract import build_policy_contract,policy_fingerprint
    policy=build_policy_contract(model_id=MODEL)
    assert policy==read(out/'configs/agents3_policy.json') and report['policy_sha256']==policy['policy_sha256']
    shutil.rmtree(code/'chronotope/src');shutil.copytree(root/'build_source/chronotope/src',code/'chronotope/src')
    shutil.copy2(root/'build_source/chronotope/CMakeLists.txt',code/'chronotope/CMakeLists.txt')
    for name in ('chronotope','libchronotope.so'):shutil.copy2(root/'native_build'/name,code/'chronotope/build'/name)
    shutil.copy2(Path(__file__).resolve(),code/'run_nvidia_agents3_policy.py')
    shutil.copy2(ROOT/'studies/verify_nvidia_agents3_policy.py',code/'verify_nvidia_agents3_policy.py')
    for source,dest in [('agents3_policy.json','agents3_notification_policy_contract.json'),('policy_fixtures.json','agents3_notification_policy_fixtures.json')]:
        shutil.copy2(root/'policy_inputs'/source,code/dest)
    shutil.rmtree(code/'cpu_policy_verification');shutil.copytree(root/'cpu_policy_verification',code/'cpu_policy_verification')
    for sub in ('records','calls'):(out/sub).mkdir(exist_ok=True)
    shutil.copy2(ROOT/'AGENTS.md',out/'AGENTS.md')
    for name in ['engine_build_result.json','policy_verification.json','predecessor_audit.json']:shutil.copy2(root/name,out/name)
    shutil.copy2(root/'services/launch.json',out/'service_launch.json')
    checks=validate_specs(out,plan,code)
    artifacts={str(p.relative_to(out)):digest(p) for sub in ('code','configs','manifests') for p in (out/sub).rglob('*') if p.is_file()}
    for name in ['plan.json','preparation_report.json','AGENTS.md','engine_build_result.json','policy_verification.json','service_launch.json','predecessor_audit.json']:artifacts[name]=digest(out/name)
    engines={str(p.resolve()):digest(p) for p in engine_root.rglob('*') if p.is_file()}
    assert engines and engines==build['hashes']
    assert digest(engine_root/'llm/tokenizer.json')==digest(CHECKPOINT/'tokenizer.json')
    execution={'created_at':now(),'model':MODEL,'revision':REVISION,'precision':PRECISION,'runtime_precision':RUNTIME_PRECISION,'arms':ARMS,
        'planned_trials':4800,'checkpoint_path':str(CHECKPOINT),'image_tokens':300,
        'checkpoint_hashes':build['source_checkpoint_hashes'],'engine_hashes':engines,'hashes':artifacts,
        'plugin':{'path':str(PLUGIN),'sha256':digest(PLUGIN)},
        'model_service':read(out/'configs/agents3_stripped.json')['base_url'],'spec_checks':checks,'qualification_trials':0,
        'request_contract':policy,'policy_sha256':policy_fingerprint(policy),'kv_cache':build['kv_cache'],
        'arm_labels':{'legacy_cpp':'Legacy C++ (Agent S3 policy; stock TensorRT-Edge-LLM)','agents3_stripped':'Agent S3 (stripped)'},
        'upstream_commit':build['upstream_commit'],'source_patches':[],
        'runtime_numerics':build['runtime_numerics'],
        'model_input_alignment':{'scope':'Same checkpoint/componentquantization, FP16 visual/dense runtime tensors, matchedprompt/history/settings;CPU inputcontractverified',
            'native_direct_token_ids_available':False,'native_audit':'Exact publicruntime formattedtext/count plusCPUtokenization; nohiddenstatework enabled'},
        'interpretation':['Fresh paired trials, same NVIDIA checkpoint and Agent S3 Worker policy; no historical Radix rows',
            'Both use repetition penalty1.0 with same4096budget, greedy sampling and originalhistory/reflection/retryrules',
            'Both use FP16 vision and unquantized LLM tensors;sourceBF16tensors converted andquantizedNVFP4/FP8projectionsretained',
            'Native image preprocessing stays unmodified; backend arithmetic and subsequentgeneratedhistoriesmaydiffer',
            'Legacy C++ CPUtextpolicy/history; notfullGPUhistoryChronotope',
            'Exact100Notificationcases,eightwindows,LiveReplayPaused;noqualificationorcompetencegates'],
        'cpu_work':['C++ policy/history,hostimage staging,runtime tokenloop,actionparsing;Python gym/socket referee',
                    'AgentS3PythonWorker,CPUtexthistory/reflection,vLLMHTTPservice']}
    write(out/'execution_manifest.json',execution)
    write(out/'status.json',{'state':'frozen_ready','completed_trials':0,'planned_trials':4800,'pid':None,'qualification_trials':0,'model':MODEL})
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
