#!/usr/bin/env python3
"""Build NVIDIA paired-policy artifacts with the verified pristine Edge-LLM SDK.

This helper never executes model requests or application episodes. Export is a
separate root-owned stock-export phase; --build requires it to be complete.
Only the legacy C++ application is adapted to supported public APIs. No upstream
source, plugin, quantization recipe, or extracted chat template is modified.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/agents3_policy_nvidia27b_20261029'
STOCK=ROOT/'runs/runtime_reversal_stock_20261029'
SOURCE=STOCK/'source/TensorRT-Edge-LLM'
BUILD=STOCK/'build'
MODEL='nvidia/Qwen3.8-27B-NVFP4'
REVISION='482ca0f3832238542f8f5295dde86b5f22711d80'
CHECKPOINT=ROOT/'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f'
COMMIT='bb291453bc10d9bd1142070c605e090d4e483ee2'
PYTHON=Path((_CHRONOTOPE_HOME + '/edgellm-venv/bin/python'))
NUMERICS=[
 'Source checkpoint NVFP4 MLP/lm_head; FP8 attention projections; BF16 vision/unquantized tensors.',
 'Unmodified stock export converts BF16 floating LLM/vision weights to FP16; source quantized FP4/FP8 projections retained.',
 'Native and vLLM use FP16 floating weights/vision and FP16 KV cache; backend kernels/accumulation need not be bit-identical.',
 'Stock native image normalization and image patchification are unchanged; public ImageData receives an application-side CPU copy of the pinned GPU frame.',
 'Legacy application keeps Agent S3 CPU text history/reflection, builds raw checkpoint-equivalent ChatML, and performs a public-tokenizer input-budget check. This CPU work is included in latency.',
 'Both arms use repetition_penalty1.0 because stock Edge-LLM has no repetition-penalty request field; original4096-token generation budget is preserved.',
]

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def hashes(path):return {str(p):sha(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}
def verify(values):
 for path,expected in values.items():assert Path(path).is_file() and sha(path)==expected,path

def verify_stock():
 source=read(STOCK/'upstream_source_manifest.json');assert source['commit']==COMMIT and not source['local_source_patches']
 verify({str(SOURCE/path):meta['sha256'] for path,meta in source['files'].items()})
 compiled=read(STOCK/'native_build_result.json');assert compiled['state']=='complete';verify(compiled['artifact_hashes'])
 return source,compiled

def prepare():
 target=OUT/'engine_preparation.json'
 if target.exists():
  prepared=read(target);verify(prepared['native_source_hashes']);verify(prepared['input_hashes']);return prepared
 source,compiled=verify_stock();download=read(OUT/'checkpoint_download.json')
 assert (download['state'],download['model'],download['revision'])==('downloaded',MODEL,REVISION)
 assert Path(download['checkpoint'])==CHECKPOINT
 quant=read(CHECKPOINT/'hf_quant_config.json')
 assert quant['quantization']['kv_cache_quant_algo'] is None, 'Do not inherit Radix FP8 KV cache'
 app=OUT/'build_source/chronotope';assert (app/'src/planner_trt.cpp').is_file()
 prepared={'state':'prepared','created_at':now(),'model':MODEL,'revision':REVISION,
  'upstream_commit':COMMIT,'source_patches':[], 'native_source':str(app),
  'native_source_hashes':hashes(app),'source_checkpoint_hashes':download['hashes'],
  'input_hashes':{str(p):sha(p) for p in [Path(__file__).resolve(),OUT/'checkpoint_download.json',STOCK/'upstream_source_manifest.json',STOCK/'native_build_result.json']},
  'upstream_build_artifact_hashes':compiled['artifact_hashes'],'runtime_numerics':NUMERICS,
  'inference_calls':0,'evaluation_trials':0,'qualification_trials':0,
  'application_changes':['Public raw-format input for checkpoint template.','Explicit host screenshot copy for stock ImageData.','Public-tokenizer generation-capacity guard.','Formatted-text/count audit without hidden-state inference.','Penalty1.0 policy contract.'],
  'kv_cache':{'dtype':'FP16','source_quantization':None,'override':None}}
 write(target,prepared);return prepared

@contextmanager
def exclusive(lock_fd):
 if lock_fd is not None:
  assert Path(f'/proc/self/fd/{lock_fd}').resolve()==ROOT/'runs/legacy_d1_gpu.lock'
  fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
 else:
  with (ROOT/'runs/legacy_d1_gpu.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def environment():
 env=dict(os.environ)
 env.update(EDGELLM_PLUGIN_PATH=str(BUILD/'libNvInfer_edgellm_plugin.so'),
  LD_LIBRARY_PATH=':'.join([str(BUILD),(_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'),'/usr/local/cuda/targets/sbsa-linux/lib']),
  PYTHONDONTWRITEBYTECODE='1')
 return env

def run(command,log):
 with Path(log).open('wb') as stream:
  subprocess.run(list(map(str,command)),cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)

def audit_export():
 import onnx
 from collections import Counter
 import math,struct
 exported=OUT/'exports'
 result={'state':'passed','model':MODEL,'revision':REVISION,'upstream_commit':COMMIT,'source_patches':[],
  'runtime_precision':'NVFP4 MLP/lm_head; FP8 attention; FP16 floating LLM/vision; FP16 KV cache',
  'source_precision':'NVFP4 MLP/lm_head; FP8 attention; BF16 floating LLM/vision; unquantized KV cache',
  'inference_calls':0,'evaluation_trials':0,'qualification_trials':0}
 for kind in ['llm','visual']:
  model=onnx.load(exported/kind/'model.onnx',load_external_data=False)
  elems=Counter();count=Counter()
  for tensor in model.graph.initializer:
   name=onnx.TensorProto.DataType.Name(tensor.data_type);elems[name]+=math.prod(tensor.dims);count[name]+=1
  inputs={v.name:onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type) for v in model.graph.input}
  result[kind]={'initializer_elements_by_dtype':dict(elems),'initializer_count_by_dtype':dict(count),
   'node_count':len(model.graph.node),'inputs':inputs,'onnx_sha256':sha(exported/kind/'model.onnx')}
 assert result['llm']['initializer_count_by_dtype'].get('FLOAT4E2M1')==193
 assert result['llm']['initializer_elements_by_dtype'].get('FLOAT8E4M3FN',0)>0
 assert result['visual']['initializer_elements_by_dtype'].get('BFLOAT16',0)==0
 assert result['visual']['initializer_elements_by_dtype'].get('FLOAT16')==460603376
 kv={key:value for key,value in result['llm']['inputs'].items() if key.startswith('past_key_values_')}
 assert len(kv)==16 and set(kv.values())=={'FLOAT16'},kv
 with (exported/'llm/embedding.safetensors').open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
 assert {value['dtype'] for key,value in header.items() if key!='__metadata__'}=={'F16'}
 result['kv_inputs']=kv;result['embedding_dtype']='F16'
 write(OUT/'onnx_precision_audit.json',result)
 return result

def build(lock_fd):
 status=read(ROOT/'runs/agents3_policy_radixark27b_20261028/experiment/status.json')
 assert status['state'] in ['superseded','paused','stopped','complete'], 'Radix batch still active'
 with exclusive(lock_fd):
  prepared=prepare();verify(prepared['input_hashes']);verify(prepared['native_source_hashes']);verify(prepared['source_checkpoint_hashes']);verify_stock()
  assert (OUT/'export_result.json').exists(),'Root-owned stock export must finish first'
  export=read(OUT/'export_result.json');assert export['state'] in ['exported','complete']
  assert (export['model'],export['revision'])==(MODEL,REVISION)
  verify({str(OUT/'exports'/path):expected for path,expected in export['output_hashes'].items()})
  assert export['checkpoint_hashes']==prepared['source_checkpoint_hashes']
  result_path=OUT/'engine_build_result.json';assert not result_path.exists(),'Do not implicitly overwrite/retry a build'
  result={'state':'building','started_at':now(),'model':MODEL,'revision':REVISION,'upstream_commit':COMMIT,
   'source_patches':[],'runtime_numerics':NUMERICS,'source_checkpoint_hashes':prepared['source_checkpoint_hashes'],
   'inference_calls':0,'evaluation_trials':0,'qualification_trials':0}
  write(result_path,result)
  try:
   run([PYTHON,'-B',Path(__file__).resolve(),'--audit-export'],OUT/'onnx_precision_audit.log')
   # Freeze the exact complete export before the official engine builders read it.
   export_hashes=hashes(OUT/'exports');result['source_export_hashes']=export_hashes;write(result_path,result)
   native=OUT/'native_build';app=OUT/'build_source/chronotope'
   commands=[['cmake','-S',app,'-B',native,'-DCHRONOTOPE_EDGELLM_BUILD='+str(BUILD),
    '-DCHRONOTOPE_EDGELLM_SOURCE='+str(SOURCE),('-DCHRONOTOPE_TRT_PREFIX=' + _CHRONOTOPE_HOME + '/trt-prefix'),'-DCMAKE_BUILD_TYPE=Release'],
    ['cmake','--build',native,'--target','chronotope_main','chronotope','-j6'],
    [BUILD/'examples/llm/llm_build','--onnxDir',OUT/'exports/llm','--engineDir',OUT/'engines/llm',
     '--maxInputLen','16384','--maxKVCacheCapacity','16384','--maxBatchSize','1'],
    [BUILD/'examples/multimodal/visual_build','--onnxDir',OUT/'exports/visual','--engineDir',OUT/'engines',
     '--minImageTokens','64','--maxImageTokens','1024','--maxImageTokensPerImage','1024']]
   write(OUT/'native_engine_build_commands.json',{'commands':[[str(x) for x in cmd] for cmd in commands], 'source_patches':[]})
   for cmd,name in zip(commands,['native_configure.log','native_build.log','llm_build.log','visual_build.log']):run(cmd,OUT/name)
   verify(export_hashes);verify(prepared['native_source_hashes']);verify_stock()
   cfg=read(OUT/'engines/llm/config.json');bc=cfg['builder_config']
   assert (bc['max_input_len'],bc['max_kv_cache_capacity'],bc['max_batch_size'])==(16384,16384,1)
   assert cfg.get('kv_cache_dtype','fp16') in ['fp16','float16','FP16'],cfg.get('kv_cache_dtype')
   assert sorted(cfg['eos_token_id'])==[248044,248046]
   result.update(state='built',completed_at=now(),builder_config=bc,kv_cache={'dtype':'FP16','override':None},
    hashes=hashes(OUT/'engines'),native_binary_hashes={str(p):sha(p) for p in [native/'chronotope',native/'libchronotope.so']},
    native_source_hashes=prepared['native_source_hashes'],plugin={'path':str(BUILD/'libNvInfer_edgellm_plugin.so'),'sha256':sha(BUILD/'libNvInfer_edgellm_plugin.so')},
    model_input_alignment={'scope':'Exact checkpoint-rendered text verified by CPU fixtures and per-trial runtime text/count; FP16 image preprocessing stock path.',
     'direct_runtime_input_ids_available':False,'model_numerical_equality_claimed':False})
  except BaseException as error:
   result.update(state='failed',error=f'{type(error).__name__}: {error}',completed_at=now());raise
  finally:write(result_path,result)
  return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
 group.add_argument('--prepare',action='store_true');group.add_argument('--build',action='store_true');group.add_argument('--audit-export',action='store_true')
 parser.add_argument('--lock-fd',type=int);args=parser.parse_args()
 result=prepare() if args.prepare else audit_export() if args.audit_export else build(args.lock_fd)
 print(json.dumps({'state':result['state'],'model':MODEL,'inference_calls':0,'qualification_trials':0}))
