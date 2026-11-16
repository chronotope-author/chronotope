"""Run the same Qwen3.5-0.8B FP16 checkpoint in legacy C++ and stripped Agent S3.
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

ROOT=Path(__file__).resolve().parents[1]
MODEL='Qwen/Qwen3.5-0.8B'
REVISION='local-sha256:04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
PRECISION='FP16'
ARMS=('legacy_cpp','agents3_stripped')
WINDOWS=(.125,.25,.5,1.,2.,4.,8.,16.)


def validate_plan(plan):
    assert set(plan['models'])=={'qwen35_0p8b'}, 'One identical model in both arms'
    model=plan['models']['qwen35_0p8b']
    assert (model['id'],model['revision'],model['precision'])==(MODEL,REVISION,PRECISION)
    jobs=plan['jobs'];assert len(jobs)==4800
    keys=set();groups=defaultdict(set)
    for j in jobs:
        assert j['phase']=='evaluation' and j['task']=='notification' and j['family']=='closing_window'
        assert j['model']=='qwen35_0p8b' and j['arm'] in ARMS and j['window_s'] in WINDOWS
        assert j['mode'] in ('live','replay','paused') and 0<=j['index']<100
        k=(j['arm'],j['window_s'],j['index'],j['mode']);assert k not in keys;keys.add(k)
        groups[j['arm'],j['window_s'],j['mode']].add(j['index'])
    assert len(groups)==48 and all(v==set(range(100)) for v in groups.values())


def job_id(j):
    return f"{j['arm']}_w{round(j['window_s']*1e9)}_{j['index']:03d}_{j['mode']}"


def freeze(out,engine_root):
    plan=json.loads((out/'plan.json').read_text());validate_plan(plan)
    parent=ROOT/'runs/legacy_d1_cpp_w8_20261028'
    service=ROOT/'runs/agents3_qwen35_0p8b_fp16_w8_20261028_services'
    proof=json.loads((service/'model_identity.json').read_text())
    launch=json.loads((service/'launch.json').read_text())
    assert (proof['model_id'],proof['revision'],proof['precision'])==(MODEL,REVISION,PRECISION)
    assert proof['quantization'] is None and proof['verification']['embedding']['exact_fp16_match']
    assert all(proof['verification'][k]['direct_named_weights_matched']>=15 for k in ['llm','visual'])
    assert digest(parent/'manifest.json')==proof['parent_manifest_sha256']
    for path,sha in proof['legacy_external_hashes'].items():assert digest(path)==sha,path
    assert str((engine_root/'llm/llm.engine').resolve()) in proof['legacy_external_hashes']
    assert str((engine_root/'visual/visual.engine').resolve()) in proof['legacy_external_hashes']
    assert launch['model']==MODEL and launch['dtype']=='float16' and launch['quantization'] is None
    assert not (out/'execution_manifest.json').exists(), 'Already frozen'
    code=out/'code'
    for sub in ('src','specs','chronotope/python','chronotope/src'):
        source=ROOT/sub if sub=='specs' else parent/'code'/sub
        shutil.copytree(source,code/sub,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (code/'chronotope/build').mkdir(parents=True)
    for n in ('chronotope','libchronotope.so'):
        shutil.copy2(parent/'code/chronotope/build'/n,code/'chronotope/build'/n)
    for n in ('legacy_d1_w8.py','agents3_stripped.py','run_qwen08_deadline.py'):
        shutil.copy2(ROOT/'studies'/n,code/n)
    import gui_agents
    shutil.copytree(Path(gui_agents.__file__).parent,code/'gui_agents',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',out/'AGENTS.md')
    for sub in ('configs','records','calls'):(out/sub).mkdir()
    native=json.loads((ROOT/'runs/legacy_d1_cpp_w8_20261028/configs/closing_window_keep.json').read_text())
    native.update(base_url=str(engine_root),model_id=MODEL)
    baseline=json.loads((ROOT/'runs/agents3_qwen35_0p8b_fp16_w8_20261028/configs/agents3_stripped.json').read_text())
    assert native['model_id']==baseline['model_id']==MODEL
    assert native['planner']=='trt' and native['server']=='cpp_socket' and native['at_commit_policy']=='frozen'
    assert baseline['server']=='agents3_service'
    for arm,cfg in zip(ARMS,(native,baseline)):write(out/'configs'/f'{arm}.json',cfg)
    setup(code)
    from chronotope.distributional.manifest import ManifestSet,build_manifest,verify_row_against_manifest
    from chronotope.distributional.episode import ControllerSpec,trial_spec_for
    ms={n:ManifestSet.load(out/'manifests'/n) for n in plan['manifest_hashes']}
    configs=dict(zip(ARMS,(native,baseline)))
    for name,m in ms.items():
        assert m.frozen_hash==plan['manifest_hashes'][name]
        for row in m.rows:verify_row_against_manifest(row,build_manifest(row))
    for j in plan['jobs']:
        row=ms[j['manifest']].rows[j['index']]
        assert row.task_family=='closing_window' and row.deadline_s==j['window_s']
        trial_spec_for(row,ControllerSpec(j['arm'],configs[j['arm']]),j['mode'],'0'*64 if j['mode']=='replay' else None)
    previous=json.loads((ROOT/'runs/agents3_qwen35_0p8b_fp16_w8_20261028/manifest.json').read_text())
    assert previous['model']==MODEL and previous['revision']==REVISION
    model_hashes=previous['model_hashes']
    for p,sha in model_hashes.items():assert digest(p)==sha, f'Checkpoint changed: {p}'
    engines={str(p.resolve()):digest(p) for p in engine_root.rglob('*') if p.is_file()}
    for path,sha in proof['legacy_external_hashes'].items():
        if path in engines:assert engines[path]==sha
    shutil.copy2(service/'model_identity.json',out/'model_identity.json')
    shutil.copy2(service/'launch.json',out/'service_launch.json')
    artifact_hashes={str(p.relative_to(out)):digest(p) for sub in ('code','configs','manifests') for p in (out/sub).rglob('*') if p.is_file()}
    artifact_hashes['plan.json']=digest(out/'plan.json')
    artifact_hashes.update({n:digest(out/n) for n in ['model_identity.json','service_launch.json']})
    execution={'created_at':now(),'model':MODEL,'revision':REVISION,'precision':PRECISION,'arms':ARMS,
        'planned_trials':4800,'checkpoint_hashes':model_hashes,'engine_hashes':engines,'hashes':artifact_hashes,
        'plugin':{'path':(_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so'),
                  'sha256':digest((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so'))},
        'model_service':baseline['base_url'],'spec_checks':4800,'qualification_trials':0,
        'runtime_numerics':'Same BF16 source cast to FP16 in both runtimes; legacy ONNX has small FP32 constants; kernels and accumulation differ',
        'interpretation':['Fresh trials in both arms; no historical result rows reused',
            'Same source checkpoint cast to FP16; runtimes and controller prompting/history differ',
            'Legacy C++ has GPU event records but no model-consumed history embeddings',
            'Both policies have no detector/updater; all failures stay in the evaluation table',
            'Inference/control timing must be separated before an overhead-only attribution',
            'One Notification sweep; no implicit additional task variants'],
        'cpu_work':['Legacy runtime token loop and CPU action parser; Python gym and socket client',
                    'Agent S3 Python orchestration, text history/reflection and vLLM HTTP service']}
    write(out/'execution_manifest.json',execution)
    write(out/'status.json',{'state':'frozen_ready','completed_trials':0,'planned_trials':4800,'pid':None,'qualification_trials':0})
    print(json.dumps({'state':'frozen_ready','spec_checks':4800,'same_model':MODEL}),flush=True)


def run(out, resume=False):
    setup(out/'code');sys.path.insert(0,str(out/'code'))
    from agents3_stripped import StrippedAgentS3Adapter
    from chronotope.distributional.manifest import ManifestSet,build_manifest
    from chronotope.distributional.episode import ControllerSpec,trial_spec_for,make_controller,verify_trace_evidence,score_trace
    from chronotope.distributional.metrics import percentiles
    from chronotope.controller import CallBudget
    from chronotope.runner import run_trial,PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    plan=json.loads((out/'plan.json').read_text());validate_plan(plan)
    cfg=json.loads((out/'execution_manifest.json').read_text())
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
            row={'Task':'Disappearing Notification','Controller':'Legacy C++' if arm=='legacy_cpp' else 'Agent S3 (stripped)',
                'Model':MODEL,'Revision':REVISION,'Precision':PRECISION,
                'Runtime':'TensorRT-Edge-LLM' if arm=='legacy_cpp' else 'vLLM',
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
            controller=make_controller(spec,scenario,ctrl,budget=CallBudget(100),vlm_adapter=adapter)
            t0=time.monotonic()
            try:run_trial(spec,path,PayloadStore(out/'payloads'),live_trace=live,controller=controller,manifest=scenario)
            finally:
                close=getattr(controller,'close',None)
                if close:close()
            trace=Trace.load(path)
            native_errors=[e for e in trace.all('decision_invocation') if e.get('planner')=='tensorrt_edgellm' and e.get('outcome')=='unparseable']
            if trace.all('decision_failed') or native_errors or (adapter and adapter.errors):raise RuntimeError(f'Controller/runtime error: {ident}')
            verify_trace_evidence(row,scenario,trace,m.frozen_hash)
            assert trace.header()['spec']['controller']==configs[arm]
            outcome=score_trace(row,trace,m.frozen_hash,controller=arm,repeat_index=0)
            pair=validate_pair(trace,live) if live else None
            rec={'job':current,'outcome':outcome.to_dict(),'trace':str(path.relative_to(out)),
                 'trace_sha256':trace.file_hash(),'pair_valid':pair.valid if pair else None,'pair_reason':pair.reason if pair else None,
                 'wall_seconds':time.monotonic()-t0,'completed_at':now()}
            write(out/'records'/f'{ident}.json',rec);records.append(rec)
            if outcome.invalid_trial or outcome.contract_violations or (pair and not pair.valid):raise RuntimeError(f'Invalid trace/pair: {ident}')
            print(f'{len(records)}/4800 {ident} success={outcome.Y}',flush=True);publish('running')
        current=None;publish('complete')
    except BaseException as exc:
        publish('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--engines',type=Path);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.freeze==args.run:ap.error('Choose exactly one of --freeze or --run')
    if args.freeze:
        if args.engines is None:ap.error('--freeze requires --engines')
        freeze(args.out.resolve(),args.engines.resolve())
    else:run(args.out.resolve(),resume=args.resume)
