"""Stripped Agent S3 vs completed D1 C++: identical frozen window-8 task rows.

600 trials: two permitted task families x 100 Live, Replay, Paused evaluation.
No separate qualification or competence filtering. Parent results must use the
identical checkpoint/revision and component precision; mismatched models fail before preparation.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
MODEL='Qwen/Qwen3.5-0.8B'
TASKS={'closing_window':'Disappearing Notification','moving_target':'Moving Balloon'}
PRECISION='FP16'
LEGACY_PRECISION='FP16 weights in exported ONNX; no quantization operators; small FP32 constants'


def now(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def evaluation_jobs():
    return [dict(id=f'evaluation_{family}_{index:03d}_{mode}',phase='evaluation',
                 family=family,index=index,mode=mode)
            for index in range(100) for family in TASKS for mode in ['live','replay','paused']]


def setup(code):
    sys.path[:0]=[str(code),str(code/'src')]

def prepare(out,parent,model_dir,service_dir):
    if out.exists(): raise FileExistsError(out)
    old=json.loads((parent/'manifest.json').read_text())
    status=json.loads((parent/'status.json').read_text())
    if old.get('model') not in (MODEL,'Qwen3.5-0.8B') or old.get('precision') != LEGACY_PRECISION:
        raise ValueError('Same-model comparison required: parent must use Qwen3.5-0.8B FP16')
    proof=json.loads((service_dir/'model_identity.json').read_text())
    service=json.loads((service_dir/'launch.json').read_text())
    assert proof['model_id']==MODEL and proof['precision']==PRECISION and proof['quantization'] is None
    assert proof['parent_manifest_sha256']==digest(parent/'manifest.json')
    assert proof['legacy_external_hashes']==old['external_hashes']
    for path,sha in old['external_hashes'].items(): assert digest(path)==sha,path
    assert Path(proof['serving_checkpoint']).resolve()==model_dir.resolve()
    weights=list(model_dir.glob('model*.safetensors'))
    assert len(weights)==1 and digest(weights[0])==proof['source_weights_sha256']
    assert proof['verification']['embedding']['exact_fp16_match']
    assert all(proof['verification'][x]['direct_named_weights_matched']>=15 for x in ['llm','visual'])
    assert service['model']==MODEL and service['dtype']=='float16' and service['quantization'] is None
    assert Path(service['model_dir']).resolve()==model_dir.resolve()
    assert service['command'][service['command'].index('--dtype')+1]=='half'
    assert status['state']=='complete' and status['completed_trials']==old['planned_trials']
    assert set(old['task_names'])==set(TASKS) and old['window_seconds']==8
    out.mkdir();code=out/'code'
    shutil.copytree(parent/'code/src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['agents3_stripped.py','agents3_legacy_w8.py']:
        shutil.copy2(ROOT/'studies'/name,code/name)
    import gui_agents
    gui=Path(gui_agents.__file__).parent
    shutil.copytree(gui,code/'gui_agents',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for sub in ['configs','manifests','records','calls']: (out/sub).mkdir()
    shutil.copy2(ROOT/'AGENTS.md',out/'AGENTS.md')
    shutil.copy2(service_dir/'launch.json',out/'service_launch.json')
    shutil.copy2(service_dir/'model_identity.json',out/'model_identity.json')
    cfg={'kind':'vlm_direct','server':'agents3_service','base_url':service['base_url'],
         'model_id':MODEL,'prompt_version':17,'image_scale':2,'max_tokens':4096,
         'temperature':0.0,'enable_thinking':False,'reasoning_effort':None,'invoke_on':'frame_change'}
    write(out/'configs/agents3_stripped.json',cfg)
    setup(code)
    from chronotope.distributional.manifest import ManifestSet,build_manifest,verify_row_against_manifest
    from chronotope.distributional.episode import ControllerSpec,trial_spec_for
    checks=0
    for family in TASKS:
        for phase,n in [('evaluation',100)]:
            src=parent/'manifests'/f'{family}_{phase}.jsonl'
            shutil.copy2(src,out/'manifests'/src.name)
            shutil.copy2(Path(str(src)+'.sha256'),out/'manifests'/(src.name+'.sha256'))
            assert digest(src)==old['hashes'][f'manifests/{src.name}']
            ms=ManifestSet.load(out/'manifests'/src.name)
            assert len(ms.rows)==n
            for row in ms.rows:
                assert row.task_family==family and row.window_ns==8_000_000_000
                verify_row_against_manifest(row,build_manifest(row))
                for mode in ['live','replay','paused']:
                    trial_spec_for(row,ControllerSpec('agents3_stripped',cfg),mode,'0'*64 if mode=='replay' else None)
                    checks+=1
    jobs=evaluation_jobs()
    assert len(jobs)==600 and all(j['family'] in TASKS for j in jobs)
    hashes={str(p.relative_to(out)):digest(p) for sub in ['code','configs','manifests'] for p in (out/sub).rglob('*') if p.is_file()}
    model_files={str(p.resolve()):digest(p) for p in model_dir.iterdir() if p.is_file() and not p.name.startswith('.')}
    plan={'created_at':now(),'parent_run':str(parent),'parent_manifest_sha256':digest(parent/'manifest.json'),
          'model':MODEL,'revision':proof['revision'],'precision':PRECISION,'model_dir':str(model_dir),
          'runtime':'vLLM '+json.loads((service_dir/'launch.json').read_text())['vllm_version'],'agent_s3_version':importlib.metadata.version('gui-agents'),
          'model_service':service['base_url'],'image_scale':2,'window_seconds':8,
          'task_names':TASKS,'jobs':jobs,'planned_trials':len(jobs),'hashes':hashes,'model_hashes':model_files,
          'agent_adaptations':['Native AgentS3 Worker response structure and reflection retained; click example/guidelines adapted to pixel-only API','DirectPixelACI: planner supplies screenshot pixel coordinates',
              'No grounder/updater/OCR/detector/tracker/code agent','CPU text history; historical images removed; current screenshot only',
              'Allowlisted literal action parser replaces eval','Thinking disabled; temperature 0; max output 4096; native format retries'],
          'evaluation_policy':'No separate qualification; report competence failures in all evaluation cells',
          'comparison_limitations':['Same local Qwen3.5-0.8B checkpoint cast to FP16 in both arms; source fingerprint and legacy export correspondence verified',
              'Legacy engine contains small FP32 constants; inference kernels and accumulation can differ by runtime',
              'Different controller prompts and inference runtimes; system comparison, not a single-factor speedup',
              'One shared baseline per task; do not count it as independent replicates for each legacy arm',
              'Notification and established Moving Balloon drift only; no full five-task comparison',
              'Record each arm hardware/service residency separately; do not infer it from model identity']}
    plan['hashes']['model_identity.json']=digest(out/'model_identity.json')
    plan['hashes']['service_launch.json']=digest(out/'service_launch.json')
    write(out/'manifest.json',plan)
    write(out/'validation.json',{'manifest_sha256':digest(out/'manifest.json'),'job_count':600,'spec_checks':checks,
          'task_families':list(TASKS),'parent_manifests_identical':True,'extra_experimental_trials':0})
    write(out/'status.json',{'state':'prepared','planned_trials':600,'completed_trials':0,'pid':None})
    print(json.dumps({'prepared':str(out),'jobs':600,'spec_checks':checks}),flush=True)


def run(out):
    setup(out/'code')
    from agents3_stripped import StrippedAgentS3Adapter
    from chronotope.distributional.manifest import ManifestSet,build_manifest
    from chronotope.distributional.episode import ControllerSpec,trial_spec_for,make_controller,verify_trace_evidence,score_trace
    from chronotope.distributional.metrics import percentiles
    from chronotope.controller import CallBudget
    from chronotope.runner import run_trial,PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    plan=json.loads((out/'manifest.json').read_text())
    valid=json.loads((out/'validation.json').read_text())
    assert digest(out/'manifest.json')==valid['manifest_sha256']
    assert plan['jobs']==evaluation_jobs(), 'Only the complete evaluation-only job list is permitted'
    for rel,sha in plan['hashes'].items(): assert digest(out/rel)==sha,rel
    for path,sha in plan['model_hashes'].items(): assert digest(path)==sha,path
    lock=(out.parent/'legacy_d1_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with urllib.request.urlopen(plan['model_service']+'/models',timeout=15) as r: health=json.load(r)
    assert [x['id'] for x in health['data']]==[plan['model']],health
    manifests={(f,p):ManifestSet.load(out/'manifests'/f'{f}_{p}.jsonl') for f in TASKS for p in ['evaluation']}
    cfg=json.loads((out/'configs/agents3_stripped.json').read_text())
    ctrl=ControllerSpec('agents3_stripped',cfg)
    assert not list((out/'records').glob('*.json')),'No implicit resume; preserve partial runs'
    records=[];started=now();current=None
    def publish(state,error=None):
        groups=defaultdict(list)
        for rec in records:
            j=rec['job'];groups[j['phase'],j['family'],j['mode']].append(rec)
        rows=[]
        for (phase,family,mode),rs in sorted(groups.items()):
            os_=[r['outcome'] for r in rs]
            row={'Phase':phase,'Task':TASKS[family],'Controller':'Agent S3','Planner':plan['model'],
                 'Precision':plan['precision'],'Runtime':plan['runtime'],'History storage':'CPU RAM text','Grounder/Updater':'None',
                 'History representation':'Text','Condition':mode,'Window (s)':8,'n':len(rs),
                 'Successes':sum(o['Y'] for o in os_),'SR':sum(o['Y'] for o in os_)/len(rs),
                 'No receipt':sum(not o['R'] for o in os_),'Invalid':sum(o['invalid_trial'] for o in os_),
                 'Replay pairs valid':sum(r['pair_valid'] is True for r in rs)}
            for label,key in [('E-to-O','L_EO'),('O-to-B','L_OB'),('B-to-D','L_BD'),('D-to-A','L_DA'),('E-to-A','L_EA')]:
                for pc,val in percentiles([o[key] for o in os_]).items():
                    if pc in ('p50','p95'):row[f'{label} {pc} (ms)']=None if val is None else val/1e6
            od=[o['D']-o['O'] for o in os_ if o['D'] is not None and o['O'] is not None]
            row['O-to-D p50 (ms)']=None if not od else percentiles(od)['p50']/1e6
            rows.append(row)
        if rows:
            with (out/'summary.csv.tmp').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (out/'summary.csv.tmp').replace(out/'summary.csv')
        write(out/'status.json',{'state':state,'started_at':started,'updated_at':now(),'pid':os.getpid() if state=='running' else None,
              'planned_trials':600,'completed_trials':len(records),'current_job':current,'error':error,'automatic_resume':False})
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    publish('running')
    try:
        for current in plan['jobs']:
            publish('running')
            family,phase,mode=current['family'],current['phase'],current['mode']
            manifest=manifests[family,phase];row=manifest.rows[current['index']]
            assert row.window_ns==8_000_000_000 and row.task_family==family
            folder=out/'trials'/phase/family/'agents3_stripped';folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{row.episode_id}_{mode}.jsonl'
            assert not path.exists(),path
            live=Trace.load(folder/f'{row.episode_id}_live.jsonl') if mode=='replay' else None
            spec=trial_spec_for(row,ctrl,mode,live.file_hash() if live else None)
            scenario=build_manifest(row)
            adapter=StrippedAgentS3Adapter(plan['model_service'],out/'calls'/f"{current['id']}.jsonl",
                                          model_id=plan['model'],max_tokens=cfg['max_tokens'])
            controller=make_controller(spec,scenario,ctrl,budget=CallBudget(100),vlm_adapter=adapter)
            t0=time.monotonic()
            run_trial(spec,path,PayloadStore(out/'payloads'),live_trace=live,controller=controller,manifest=scenario)
            trace=Trace.load(path)
            if adapter.errors or trace.all('decision_failed'):
                raise RuntimeError(f"Failed model/controller call in {current['id']}; preserve trace and call log")
            if not adapter.calls: raise RuntimeError('No model calls observed')
            verify_trace_evidence(row,scenario,trace,manifest.frozen_hash)
            assert trace.header()['spec']['controller']==cfg
            outcome=score_trace(row,trace,manifest.frozen_hash,controller=ctrl.name,repeat_index=0)
            pair=validate_pair(trace,live) if live else None
            record={'job':current,'outcome':outcome.to_dict(),'trace':str(path.relative_to(out)),
                    'trace_sha256':trace.file_hash(),'pair_valid':pair.valid if pair else None,
                    'pair_reason':pair.reason if pair else None,'model_calls':len(adapter.calls),
                    'wall_seconds':time.monotonic()-t0,'completed_at':now()}
            write(out/'records'/f"{current['id']}.json",record);records.append(record)
            if outcome.invalid_trial or outcome.contract_violations or (pair and not pair.valid):
                raise RuntimeError(f'Invalid trial/pair: {record}')
            print(f"{len(records)}/600 {current['id']} success={outcome.Y} wall={record['wall_seconds']:.2f}s",flush=True)
            publish('running')
        current=None;publish('complete')
    except BaseException as exc:
        publish('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',f'{type(exc).__name__}: {exc}')
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true')
    ap.add_argument('--parent',type=Path,default=ROOT/'runs/legacy_d1_cpp_w8_20261028')
    ap.add_argument('--model-dir',type=Path,default=ROOT/'runs/model_cache/qwen35_0p8b_legacy_source')
    ap.add_argument('--service-dir',type=Path,default=ROOT/'runs/agents3_qwen35_0p8b_fp16_w8_20261028_services')
    a=ap.parse_args()
    if a.prepare==a.run:ap.error('Choose --prepare or --run')
    if a.prepare:prepare(a.out.resolve(),a.parent.resolve(),a.model_dir.resolve(),a.service_dir.resolve())
    else:run(a.out.resolve())
