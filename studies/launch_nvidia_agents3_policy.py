"""Explicit NVIDIA service/evaluation launch; no screening requests or Radix resume.

The root agent calls each stage after reviewing freshly built/frozen artifacts.
The service readiness check uses GET /models and /metrics, never inference.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from legacy_d1_w8 import now,write,digest
from run_nvidia_agents3_policy import MODEL,REVISION,CHECKPOINT,ROOT,PLUGIN,validate_plan,validate_service


def read(path):return json.loads(Path(path).read_text())

def identity(pid):
    try:
        path=Path('/proc')/str(pid);stat=path.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        return {'pid':pid,'process_starttime_ticks':int(stat[19]),'command':[x.decode() for x in path.joinpath('cmdline').read_bytes().split(b'\0') if x]}
    except FileNotFoundError:return None


def stable_identity(process,expected):
    end=time.monotonic()+10
    while time.monotonic()<end:
        found=identity(process.pid)
        if found and len(found['command'])>=len(expected) and found['command'][-len(expected):]==expected:return found
        if process.poll() is not None:raise RuntimeError('Child exited before exec identity was available')
        time.sleep(.05)
    raise TimeoutError('Child exec identity unavailable')


def environment(out):
    env=dict(os.environ)
    for name in ['VLLM_NSYS_PROFILING','VLLM_PROFILER_CONFIG','VLLM_TORCH_PROFILER_DIR','CUDA_LAUNCH_BLOCKING']:
        env.pop(name,None)
    env.update(PYTHONPATH=':'.join([str(out/'code'),str(out/'code/src'),str(out/'code/chronotope/python')]),
        PATH=(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin:/usr/local/cuda/bin:')+os.environ.get('PATH',''),
        EDGELLM_PLUGIN_PATH=str(PLUGIN),
        LD_LIBRARY_PATH=':'.join([str(PLUGIN.parent),(_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'),'/usr/local/cuda/targets/sbsa-linux/lib']),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    return env


def service_command():
    return [(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/vllm'),'serve',str(CHECKPOINT),'--served-model-name',MODEL,
        '--dtype','half','--host','127.0.0.1','--port','8012','--max-model-len','16384','--max-num-seqs','1',
        '--gpu-memory-utilization','0.60','--kv-cache-memory-bytes','2147483648','--kv-cache-dtype','auto',
        '--limit-mm-per-prompt','{"image":1}','--generation-config','vllm','--reasoning-parser','qwen3','--no-enable-prefix-caching',
        '--enable-per-request-metrics']


def model_ready():
    with urllib.request.urlopen('http://127.0.0.1:8012/v1/models',timeout=5) as response:
        return [m['id'] for m in json.load(response)['data']]==[MODEL]


def start_service(root):
    out=root/'experiment';validate_plan(read(out/'plan.json'))
    build=read(root/'engine_build_result.json')
    assert build['state']=='built' and (build['model'],build['revision'])==(MODEL,REVISION)
    assert build['source_patches']==[] and build['kv_cache']['dtype']=='FP16'
    from runtime_reversal.batch_control import service_idle
    directory=root/'services';directory.mkdir(exist_ok=True)
    assert not (directory/'launch.json').exists(),'Do not replace/restart a recorded service implicitly'
    # Bind preflight, no existing service/process is stopped by this helper.
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',8012))
    command=service_command();env=environment(out)
    lock=(ROOT/'runs/legacy_d1_gpu.lock').open('a+')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (directory/'vllm.log').open('wb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ident=stable_identity(process,command)
        launch={**ident,'launch_command':command,'state':'starting','started_at':now(),
            'model':MODEL,'revision':REVISION,'model_dir':str(CHECKPOINT),'base_url':'http://127.0.0.1:8012/v1',
            'dtype':'half','quantization':'modelopt_mixed','kv_cache_dtype':'auto','effective_kv_cache_dtype':'float16',
            'source_precision':'NVFP4 MLP/lm_head;FP8 attention;BF16 vision/unquantizedsource',
            'runtime_precision':'NVFP4/FP8projections retained;FP16vision/unquantizedLLM;FP16KV',
            'max_model_len':16384,'max_num_seqs':1,'enable_prefix_caching':False,'qualification_trials':0,
            'inference_readiness_requests':0,'helper_sha256':digest(Path(__file__)),
            'environment_overrides':{key:env[key] for key in ['PATH','PYTHONPATH','LD_LIBRARY_PATH','EDGELLM_PLUGIN_PATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']}}
        write(directory/'launch.json',launch)
        try:
            deadline=time.monotonic()+1200
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError(f'NVIDIA service exited{process.returncode};see services/vllm.log')
                try:
                    if model_ready() and service_idle(launch['base_url']):break
                except (OSError,ValueError):pass
                time.sleep(2)
            else:raise TimeoutError('NVIDIA service did not become ready in1200seconds')
            log=(directory/'vllm.log').read_text(errors='replace')
            lines=[line for line in log.splitlines() if 'Initializing a V1 LLM engine' in line]
            assert len(lines)==1, 'Cannot independently confirm observed service configuration'
            line=lines[0]
            for value in ['dtype=torch.float16','quantization=modelopt_mixed','kv_cache_dtype=auto','max_seq_len=16384','served_model_name='+MODEL,'enable_prefix_caching=False']:
                assert value in line, f'Observed configuration missing{value}'
            launch.update(state='ready',ready_at=now(),observed_engine_configuration=line,
                observed_quantized_linear_backend='FlashInferCutlassNvFp4LinearKernel' if 'FlashInferCutlassNvFp4LinearKernel' in log else None)
            assert launch['observed_quantized_linear_backend'], 'NVFP4 runtime backend not confirmed'
            validate_service(launch);write(directory/'launch.json',launch)
        except BaseException as error:
            launch.update(state='failed',error=f'{type(error).__name__}: {error}',updated_at=now());write(directory/'launch.json',launch)
            current=identity(process.pid)
            if current and current['process_starttime_ticks']==ident['process_starttime_ticks']:
                os.killpg(process.pid,signal.SIGTERM)
            raise
    print(json.dumps({'state':'service_ready','pid':process.pid,'model':MODEL,'qualification_trials':0}),flush=True)


def launch_evaluation(root):
    out=root/'experiment';validate_plan(read(out/'plan.json'))
    execution=read(out/'execution_manifest.json');service=read(root/'services/launch.json');validate_service(service)
    assert read(out/'status.json')['state']=='frozen_ready'
    assert (execution['model'],execution['revision'])==(MODEL,REVISION)
    assert execution['qualification_trials']==0 and execution['planned_trials']==4800
    assert not list((out/'records').glob('*.json')),'Fresh NVIDIA run cannot include old rows'
    current=identity(service['pid']);assert current and current['process_starttime_ticks']==service['process_starttime_ticks']
    assert model_ready()
    command=[(_CHRONOTOPE_HOME + '/agents3-venv/bin/python'),'-B',str(out/'code/run_nvidia_agents3_policy.py'),'--out',str(out),'--run']
    env=environment(out)
    with (out/'runner.log').open('ab') as log:
        runner=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ident=stable_identity(runner,command);write(out/'launch.json',{**ident,'launch_command':command,'started_at':now(),'automatic_resume':False})
    deadline=time.monotonic()+1200
    while read(out/'status.json')['state']=='frozen_ready':
        if runner.poll() is not None:raise RuntimeError('Runner exited before entering running state;see runner.log')
        if time.monotonic()>deadline:raise TimeoutError('Runner did not enter running state in1200seconds')
        time.sleep(1)
    assert read(out/'status.json')['state']=='running'
    audit=[str(ROOT/'.venv/bin/python'),'-B',str(out/'code/watch_same_model_deadline.py'),'--out',str(out)]
    with (out/'auditor.log').open('ab') as log:
        auditor=subprocess.Popen(audit,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'audit_launch.json',{**stable_identity(auditor,audit),'launch_command':audit,'started_at':now()})
    print(json.dumps({'state':'running','runner_pid':runner.pid,'auditor_pid':auditor.pid,'out':str(out),'planned_trials':4800,'qualification_trials':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--start-service',action='store_true');group.add_argument('--launch',action='store_true')
    args=parser.parse_args();(start_service if args.start_service else launch_evaluation)(args.root.resolve())
