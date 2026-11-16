"""Durable, completion-gated RadixArk follow-on to the matched Qwen0.8B sweep.

Preparing or waiting performs no model calls and touches no running GPU service.
The handoff requires the complete predecessor evidence, a released GPU lock,
and an idle, positively identified old vLLM process. No qualification episodes.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODEL='RadixArk/Qwen3.8-27B-NVFP4'
REVISION='319f741cce68d7914884900c138a1fbb70a42f30'
PREVIOUS_MODEL='Qwen/Qwen3.5-0.8B'
WINDOWS={.125,.25,.5,1.,2.,4.,8.,16.}
ARMS={'legacy_cpp','agents3_stripped'}
MODES={'live','replay','paused'}
TOTAL=4800


def now():return datetime.now(timezone.utc).isoformat()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def read(path):return json.loads(Path(path).read_text())

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def validate_jobs(plan,model_key):
    jobs=plan['jobs'];assert len(jobs)==TOTAL
    keys=set();finished=set()
    for j in jobs:
        assert j['phase']=='evaluation' and j['task']=='notification' and j['family']=='closing_window'
        assert j['model']==model_key and j['arm'] in ARMS and j['mode'] in MODES
        assert j['window_s'] in WINDOWS and type(j['index']) is int and 0<=j['index']<100
        key=(j['arm'],j['window_s'],j['index'],j['mode']);assert key not in keys;keys.add(key)
        if j['mode']=='replay':assert key[:3]+('live',) in finished
        finished.add(key)
    assert len(keys)==TOTAL
    return jobs


def predecessor_gate(status,audit):
    """Small/pure polling gate; failure or stopped never means completion."""
    if status.get('state')!='complete':
        return False,'waiting_for_predecessor_'+str(status.get('state','missing'))
    if status.get('completed_trials')!=TOTAL or status.get('planned_trials')!=TOTAL:
        return False,'waiting_for_complete_trial_count'
    if status.get('qualification_trials')!=0 or status.get('error'):
        return False,'predecessor_metadata_error'
    if not audit or audit.get('state')!='finished' or audit.get('records')!=TOTAL:
        return False,'waiting_for_final_predecessor_audit'
    if audit.get('infrastructure_errors')!=0 or audit.get('task_failures_retained') is not True:
        return False,'predecessor_audit_error'
    return True,'predecessor_complete_and_audited'


def command_line(pid):
    try:return Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')[:-1]
    except FileNotFoundError:return []

def process_matches(pid,expected):
    actual=[x.decode(errors='replace') for x in command_line(pid)]
    # Python entrypoint wrappers add an interpreter before the pinned command.
    return actual==expected or (len(actual)==len(expected)+1 and
        Path(actual[0]).name.startswith('python') and actual[1:]==expected)


def process_starttime(pid):
    try:
        raw=Path(f'/proc/{int(pid)}/stat').read_text()
        return int(raw[raw.rfind(')')+2:].split()[19])
    except FileNotFoundError:return None

def process_alive(pid):
    try:
        raw=Path(f'/proc/{int(pid)}/stat').read_text();return raw[raw.rfind(')')+2:].split()[0]!='Z'
    except FileNotFoundError:return False

def predecessor_processes(predecessor):
    marker=str(predecessor).encode();found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        cmd=command_line(entry.name)
        if marker in cmd and any(b'run_qwen08_agents3_policy.py' in x for x in cmd) and process_alive(entry.name):found.append(int(entry.name))
    return found


def full_predecessor_audit(predecessor):
    plan=read(predecessor/'plan.json');jobs=validate_jobs(plan,'qwen35_0p8b')
    planned={job_id(j):j for j in jobs};files=list((predecessor/'records').glob('*.json'))
    assert len(files)==TOTAL,'Predecessor record count differs'
    seen=set()
    for path in files:
        record=read(path);j=record['job'];ident=job_id(j)
        assert ident not in seen and j==planned[ident];seen.add(ident)
        outcome=record['outcome'];assert not outcome['invalid_trial'] and not outcome['contract_violations']
        assert digest(predecessor/record['trace'])==record['trace_sha256']
        if j['mode']=='replay':assert record['pair_valid'] is True
        # Every completed decision's exact policy/model-input audit was retained.
        assert isinstance(record['request_audit'],list)
    assert len(seen)==TOTAL
    return {'state':'passed','records':TOTAL,'qualification_trials':0,'created_at':now(),
            'plan_sha256':digest(predecessor/'plan.json')}


def job_id(j):return f"{j['arm']}_w{round(j['window_s']*1e9)}_{j['index']:03d}_{j['mode']}"

def get_json(url,timeout=10):
    with urllib.request.urlopen(url,timeout=timeout) as response:return json.load(response)

def endpoint_present(url):
    try:get_json(url+'/models');return True
    except (urllib.error.URLError,TimeoutError,ConnectionError):return False

def idle_service(base_url):
    with urllib.request.urlopen(base_url.removesuffix('/v1')+'/metrics',timeout=10) as response:text=response.read().decode()
    values={name:[] for name in ['num_requests_running','num_requests_waiting']}
    for line in text.splitlines():
        for name in values:
            if line.startswith('vllm:'+name+'{') or line.startswith('vllm:'+name+' '):values[name].append(float(line.rsplit(None,1)[1]))
    if not all(values.values()):raise RuntimeError('Cannot establish old service idle metrics')
    return all(v==0 for xs in values.values() for v in xs)


def stop_previous_service(launch,log_path,identity=None):
    assert launch['model']==PREVIOUS_MODEL and launch['base_url']=='http://127.0.0.1:8013/v1'
    command=launch['command'];assert '--served-model-name' in command and command[command.index('--served-model-name')+1]==PREVIOUS_MODEL
    assert command[command.index('--port')+1]=='8013'
    pid=launch['pid']
    if not process_alive(pid):
        if endpoint_present(launch['base_url']):raise RuntimeError('Old service endpoint alive under an unverified process; refusing to stop it')
        write(log_path,{'state':'already_stopped','pid':pid,'at':now()});return
    if not process_matches(pid,command):raise RuntimeError('Old service PID/command mismatch; refusing to stop it')
    if identity and (identity['pid']!=pid or identity['starttime_ticks']!=process_starttime(pid)):
        raise RuntimeError('Old service process start time changed; refusing to stop a reused PID')
    if os.getpgid(pid)!=pid:raise RuntimeError('Old service lacks its own process group; refusing broad termination')
    if not idle_service(launch['base_url']):raise RuntimeError('Old service has active or waiting requests')
    os.killpg(pid,signal.SIGTERM)
    deadline=time.monotonic()+90
    while process_alive(pid) or endpoint_present(launch['base_url']):
        if time.monotonic()>deadline:raise RuntimeError('Old service did not stop after SIGTERM; no SIGKILL issued')
        time.sleep(1)
    write(log_path,{'state':'stopped_verified_process_group','pid':pid,'command':command,'at':now()})


def service_command(root):
    return [(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/vllm'),'serve',
        str(ROOT/'runs/model_cache/radixark_qwen38_nvfp4_319f741c'),'--served-model-name',MODEL,
        '--dtype','bfloat16','--host','127.0.0.1','--port','8012','--max-model-len','16384',
        '--max-num-seqs','1','--gpu-memory-utilization','0.60','--kv-cache-memory-bytes','2147483648',
        '--kv-cache-dtype','fp8_e4m3','--no-calculate-kv-scales','--limit-mm-per-prompt','{"image":1}',
        '--generation-config','vllm','--reasoning-parser','qwen3','--no-enable-prefix-caching']


def base_environment():
    return dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'studies'),
        PATH=(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin:/usr/local/cuda/bin:')+os.environ.get('PATH',''),
        LD_LIBRARY_PATH=(_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:')+os.environ.get('LD_LIBRARY_PATH',''),
        EDGELLM_PLUGIN_PATH=(_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def service_packages():
    code="import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in ['vllm','transformers','torch','numpy','tokenizers']}))"
    result=subprocess.run([(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/python'),'-B','-c',code],
        check=True,capture_output=True,text=True,env=dict(base_environment(),CUDA_VISIBLE_DEVICES=''))
    return json.loads(result.stdout)


def service_runtime_configuration(log):
    lines=[line for line in log.splitlines() if 'Initializing a V1 LLM engine' in line]
    if not lines:raise RuntimeError('Service readiness has no observed engine configuration')
    line=lines[-1]
    for value in ['dtype=torch.bfloat16','quantization=modelopt_mixed','kv_cache_dtype=fp8_e4m3',
                  'max_seq_len=16384','served_model_name='+MODEL,'enable_prefix_caching=False']:
        if value not in line:raise RuntimeError('Unexpected observed RadixArk runtime configuration: missing '+value)
    return line


def start_service(root,command):
    base_url='http://127.0.0.1:8012/v1'
    if endpoint_present(base_url):raise RuntimeError('Port8012 already has a model service; refusing replacement/reuse')
    folder=root/'services';folder.mkdir(exist_ok=True)
    with (folder/'vllm.log').open('ab') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=base_environment())
    launch={'pid':process.pid,'command':command,'model':MODEL,'revision':REVISION,'model_dir':command[2],
        'dtype':'bfloat16','precision':'NVFP4 MLP/lm_head; FP8 attention; BF16 vision','quantization':'modelopt_mixed',
        'kv_cache_dtype':'fp8_e4m3','calculate_kv_scales':False,'kv_scales':'static1.0; no checkpoint KV scales',
        'max_model_len':16384,'max_num_seqs':1,'enable_prefix_caching':False,
        'base_url':base_url,'vllm_version':service_packages()['vllm'],'started_at':now(),'qualification_trials':0,'benchmark_trials':0}
    write(folder/'launch.json',launch)
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError(f'RadixArk vLLM startup exited {process.returncode}; see services/vllm.log')
        try:
            models=get_json(base_url+'/models',timeout=3)
            if [m['id'] for m in models.get('data',[])]==[MODEL]:
                observed=service_runtime_configuration((folder/'vllm.log').read_text(errors='replace'))
                launch.update(runtime_configuration_verified=True,process_starttime_ticks=process_starttime(process.pid))
                write(folder/'launch.json',launch)
                write(folder/'ready.json',{'state':'ready','pid':process.pid,'models':models,'at':now(),
                    'runtime_configuration':observed,'inference_calls':0});return launch
        except (urllib.error.URLError,TimeoutError,ConnectionError):pass
        time.sleep(2)
    raise TimeoutError('RadixArk model service did not become ready; no inference probe was sent')


def prepare(root,predecessor):
    root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'queue_manifest.json';assert not manifest_path.exists(),'Queue already prepared'
    out=root/'experiment';plan=read(out/'plan.json');validate_jobs(plan,'radixark_27b')
    old=read(predecessor/'plan.json');validate_jobs(old,'qwen35_0p8b')
    assert plan['manifest_hashes']==old['manifest_hashes'],'The same case/window manifests must be retained'
    assert [{k:v for k,v in j.items() if k!='model'} for j in plan['jobs']]==[{k:v for k,v in j.items() if k!='model'} for j in old['jobs']]
    assert plan['models']['radixark_27b']['id']==MODEL and plan['models']['radixark_27b']['revision']==REVISION
    python=(_CHRONOTOPE_HOME + '/agents3-venv/bin/python')
    commands={
      'build':[sys.executable,'-B',str(ROOT/'studies/prepare_radix_agents3_engines.py'),'--out',str(root),'--predecessor',str(predecessor),'--build'],
      'verify':[(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/python'),'-B',str(ROOT/'studies/verify_radix_agents3_policy.py'),'--root',str(root)],
      'freeze':[python,'-B',str(ROOT/'studies/run_radix_agents3_policy.py'),'--out',str(out),'--engines',str(root/'engines'),'--freeze'],
      'run':[python,'-B',str(out/'code/run_radix_agents3_policy.py'),'--out',str(out),'--run'],
      'audit':[str(ROOT/'.venv/bin/python'),'-B',str(out/'code/watch_same_model_deadline.py'),'--out',str(out)],
      'service':service_command(root)}
    sources=[Path(__file__),ROOT/'studies/prepare_radix_agents3_engines.py',ROOT/'studies/run_radix_agents3_policy.py',ROOT/'studies/verify_radix_agents3_policy.py']
    # Freeze all local modules used by preparation/freezing; active 0.8B code is already separately frozen.
    for name in ['agents3_policy_contract.py','agents3_stripped.py','audit_agents3_policy.py','audit_radix_agents3_policy.py','legacy_d1_w8.py','watch_same_model_deadline.py','prepare_deadline_envelope.py','verify_agents3_policy.py','run_qwen08_agents3_policy.py']:
        path=ROOT/'studies'/name
        if path.exists():sources.append(path)
    # Engine preparation snapshots native/runtime sources. Pin the exact snapshot
    # consumed by the deferred build as well as every prepared policy input.
    for name in ['engine_preparation.json','prepared_chat_template.json']:
        path=root/name;assert path.is_file(),f'Engine preparation missing: {path}';sources.append(path)
    for name in ['build_source','policy_inputs']:
        folder=root/name;assert folder.is_dir(),f'Prepared source directory missing: {folder}'
        sources.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    engine_preparation=read(root/'engine_preparation.json')
    for path,sha in engine_preparation['input_hashes'].items():
        assert digest(path)==sha,f'Prepared engine input changed: {path}'
        sources.append(Path(path))
    old_service=ROOT/'runs/agents3_qwen35_0p8b_fp16_w8_20261028_services/launch.json'
    old_launch=read(old_service);old_pid=old_launch['pid']
    if process_alive(old_pid):
        assert process_matches(old_pid,old_launch['command']),'Old service identity changed before queue preparation'
        previous_service_identity={'pid':old_pid,'starttime_ticks':process_starttime(old_pid)}
    else:previous_service_identity=None
    manifest={'schema':'radix-after-qwen08-queue-1','created_at':now(),'root':str(root),'predecessor':str(predecessor),
        'model':MODEL,'revision':REVISION,'planned_trials':TOTAL,'qualification_trials':0,
        'task':'Disappearing Notification','conditions':sorted(MODES),'windows_s':sorted(WINDOWS),
        'commands':commands,'source_hashes':{str(p):digest(p) for p in sources},
        'service_packages':service_packages(),
        'next_plan_sha256':digest(out/'plan.json'),'predecessor_plan_sha256':digest(predecessor/'plan.json'),
        'old_service_launch':str(old_service),'old_service_launch_sha256':digest(old_service),
        'old_service_identity':previous_service_identity,
        'gpu_lock':str(ROOT/'runs/legacy_d1_gpu.lock'),
        'on_predecessor_stopped_or_failed':'Keep waiting; never automatically resume it or start successor',
        'on_handoff_failure':'Stop queue, preserve logs/artifacts; no automatic pipeline restart'}
    write(manifest_path,manifest)
    write(root/'queue_status.json',{'state':'prepared_waiting','created_at':now(),'predecessor':str(predecessor),'pid':None,'qualification_trials':0})
    return manifest


def validate_pins(root,manifest):
    for path,sha in manifest['source_hashes'].items():assert digest(path)==sha,f'Queue source changed: {path}'
    assert digest(root/'experiment/plan.json')==manifest['next_plan_sha256']
    assert digest(Path(manifest['predecessor'])/'plan.json')==manifest['predecessor_plan_sha256']
    assert digest(manifest['old_service_launch'])==manifest['old_service_launch_sha256']
    assert service_packages()==manifest['service_packages'],'Serving environment changed after queue preparation'
    validate_jobs(read(root/'experiment/plan.json'),'radixark_27b')


def run_command(command,log_path):
    with log_path.open('ab') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=base_environment())
    if result.returncode:raise RuntimeError(f'Preparation command exited {result.returncode}; see {log_path}')


def run(root,poll_seconds=15):
    manifest=read(root/'queue_manifest.json');assert manifest['root']==str(root)
    lock=(root/'queue.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'handoff_started.json').exists():raise RuntimeError('Handoff was already attempted; inspect existing processes/artifacts before explicitly preparing another queue. Automatic restart is disabled.')
    except BaseException:
        lock.close();raise
    predecessor=Path(manifest['predecessor']);gpu=None
    def publish(state,**detail):
        write(root/'queue_status.json',{'state':state,'updated_at':now(),'pid':os.getpid(),'predecessor':str(predecessor),
            'successor':str(root/'experiment'),'model':MODEL,'revision':REVISION,'qualification_trials':0,**detail})
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        validate_pins(root,manifest)
        while True:
            status=read(predecessor/'status.json');ap=predecessor/'audit_status.json';audit=read(ap) if ap.exists() else None
            ready,reason=predecessor_gate(status,audit)
            publish('waiting',reason=reason,predecessor_completed=status.get('completed_trials'),predecessor_planned=status.get('planned_trials'))
            if ready:
                if predecessor_processes(predecessor):publish('waiting',reason='predecessor_process_still_exiting')
                else:
                    gpu=Path(manifest['gpu_lock']).open('a+')
                    try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:gpu.close();gpu=None;publish('waiting',reason='gpu_lock_busy')
                    else:break
            time.sleep(poll_seconds)
        publish('validating_predecessor_evidence')
        proof=full_predecessor_audit(predecessor);write(root/'predecessor_completion_verification.json',proof)
        final_audit=read(predecessor/'audit_status.json')
        assert predecessor_gate(read(predecessor/'status.json'),final_audit)[0]
        write(root/'predecessor_audit.json',final_audit)
        validate_pins(root,manifest)
        write(root/'handoff_started.json',{'at':now(),'pid':os.getpid(),'queue_manifest_sha256':digest(root/'queue_manifest.json')})
        publish('stopping_completed_predecessor_service')
        stop_previous_service(read(manifest['old_service_launch']),root/'old_service_stop.json',manifest.get('old_service_identity'))
        publish('building_radix_engine')
        run_command(manifest['commands']['build'],root/'queued_engine_build.log')
        publish('verifying_radix_policy')
        run_command(manifest['commands']['verify'],root/'queued_policy_verification.log')
        publish('starting_radix_service')
        start_service(root,manifest['commands']['service'])
        publish('freezing_radix_experiment')
        run_command(manifest['commands']['freeze'],root/'queued_freeze.log')
        out=root/'experiment';execution=read(out/'execution_manifest.json')
        assert execution['model']==MODEL and execution['revision']==REVISION and execution['planned_trials']==TOTAL and execution['qualification_trials']==0
        validate_jobs(read(out/'plan.json'),'radixark_27b')
        assert read(out/'status.json')['state']=='frozen_ready'
        fcntl.flock(gpu,fcntl.LOCK_UN);gpu.close();gpu=None
        env=dict(base_environment(),PYTHONPATH=str(out/'code/src')+':'+str(out/'code'))
        with (out/'runner.log').open('ab') as log:
            runner=subprocess.Popen(manifest['commands']['run'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        write(out/'launch.json',{'pid':runner.pid,'command':manifest['commands']['run'],'started_at':now(),'queue_manifest_sha256':digest(root/'queue_manifest.json')})
        publish('starting_radix_evaluation',runner_pid=runner.pid)
        deadline=time.monotonic()+900
        while read(out/'status.json')['state'] not in ('running','failed','stopped','complete'):
            if runner.poll() is not None:raise RuntimeError('RadixArk runner exited before entering running state')
            if time.monotonic()>deadline:raise TimeoutError('RadixArk runner startup exceeded900seconds')
            time.sleep(1)
        if read(out/'status.json')['state']!='running':raise RuntimeError('RadixArk runner did not enter running state')
        with (out/'auditor.log').open('ab') as log:
            auditor=subprocess.Popen(manifest['commands']['audit'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        write(out/'audit_launch.json',{'pid':auditor.pid,'command':manifest['commands']['audit'],'started_at':now()})
        while True:
            status=read(out/'status.json')
            publish('successor_running' if status['state']=='running' else 'successor_'+status['state'],runner_pid=runner.pid,
                    completed_trials=status.get('completed_trials'),planned_trials=TOTAL)
            if status['state']!='running':break
            if not process_alive(runner.pid):raise RuntimeError('Successor process disappeared without terminal status')
            if auditor.poll() is not None:raise RuntimeError('Successor audit process exited while evaluation is running')
            time.sleep(poll_seconds)
    except BaseException as error:
        publish('queue_stopped' if isinstance(error,KeyboardInterrupt) else 'queue_failed',error=f'{type(error).__name__}: {error}',automatic_resume=False)
        raise
    finally:
        if gpu is not None:fcntl.flock(gpu,fcntl.LOCK_UN);gpu.close()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--predecessor',type=Path,default=ROOT/'runs/agents3_policy_qwen08_20261028/experiment')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=15)
    args=parser.parse_args()
    if args.prepare==args.run:parser.error('Choose exactly one of --prepare or --run')
    if not 1<=args.poll_seconds<=60:parser.error('--poll-seconds must be1..60')
    if args.prepare:print(json.dumps({'state':'prepared','planned_trials':prepare(args.out.resolve(),args.predecessor.resolve())['planned_trials'],'inference_calls':0}))
    else:run(args.out.resolve(),args.poll_seconds)
