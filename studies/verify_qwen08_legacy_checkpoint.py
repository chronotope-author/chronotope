"""Verify the local 0.8B source against legacy exported weights; no inference."""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
from pathlib import Path
import hashlib,json
import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path((_CHRONOTOPE_HOME + '/tensorrt-edgellm-workspace/ckpt/Qwen3.5-0.8B'))
COPY=ROOT/'runs/model_cache/qwen35_0p8b_legacy_source'
LEGACY=Path((_CHRONOTOPE_HOME + '/tensorrt-edgellm-workspace/Qwen3.5-0.8B'))
PARENT=ROOT/'runs/legacy_d1_cpp_w8_20261028'
OUT=ROOT/'runs/agents3_qwen35_0p8b_fp16_w8_20261028_services/model_identity.json'

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

source=next(SOURCE.glob('*.safetensors'));copied=COPY/source.name
sha=digest(source);assert digest(copied)==sha
parent=json.loads((PARENT/'manifest.json').read_text())
for p,h in parent['external_hashes'].items():assert digest(p)==h, f'Legacy artifact changed: {p}'
report={'model_id':'Qwen/Qwen3.5-0.8B','legacy_raw_model_id':parent['model'],
 'precision':'FP16','source_checkpoint':str(SOURCE),'serving_checkpoint':str(COPY),
 'source_weights_sha256':sha,'revision':'local-sha256:'+sha,'parent_manifest_sha256':digest(PARENT/'manifest.json'),
 'legacy_external_hashes':parent['external_hashes'],'dtype_conversion':'Source BF16 weights cast to FP16 in both TensorRT export and vLLM --dtype half',
 'quantization':None,'verification':{},'experimental_trials':0}
with safe_open(str(source),framework='pt',device='cpu') as hf:
 keys=set(hf.keys())
 for part in ['llm','visual']:
  path=LEGACY/'onnx'/part/'model.onnx';m=onnx.load(path)
  assert not any(n.op_type in ['QuantizeLinear','DequantizeLinear'] for n in m.graph.node)
  matches=[];unmapped=[]
  for t in m.graph.initializer:
   if t.data_type!=onnx.TensorProto.FLOAT16 or not t.name.endswith('.weight'):continue
   if part=='llm':key=t.name.replace('_model.model.','model.language_model.',1)
   else:key='model.visual.'+t.name
   if key not in keys:unmapped.append(t.name);continue
   want=hf.get_tensor(key).to(torch.float16).numpy();actual=numpy_helper.to_array(t)
   assert want.shape==actual.shape and np.array_equal(want,actual), f'Weight mismatch: {part}/{t.name}'
   matches.append({'onnx':t.name,'source':key,'elements':int(actual.size)})
  assert len(matches)>=15,(part,len(matches))
  report['verification'][part]={'direct_named_weights_matched':len(matches),'matched_elements':sum(x['elements'] for x in matches),'matches':matches,'unmapped_named_weights':unmapped,'onnx_sha256':digest(path)}
  del m
 with safe_open(str(LEGACY/'engines/llm/embedding.safetensors'),framework='pt',device='cpu') as emb:
  actual=emb.get_tensor('embedding');want=hf.get_tensor('model.language_model.embed_tokens.weight').to(torch.float16)
  assert actual.dtype==torch.float16 and torch.equal(actual,want),'Embedding differs from legacy engine'
  report['verification']['embedding']={'exact_fp16_match':True,'elements':actual.numel(),'sha256':digest(LEGACY/'engines/llm/embedding.safetensors')}
report['verification_scope']='Exact copied source fingerprint, unchanged legacy engine hashes, direct named LLM/visual weights and full embedding comparison; folded anonymous ONNX constants are not individually reverse-mapped.'
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'model':report['model_id'],'precision':'FP16','source_weights_sha256':sha,'llm_named_matched':report['verification']['llm']['direct_named_weights_matched'],'visual_named_matched':report['verification']['visual']['direct_named_weights_matched'],'embedding_exact':True,'output':str(OUT)}))
