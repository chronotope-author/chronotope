"""CPU-only verification of the queued Radix Agent S3 policy and image inputs."""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]


def verify(root):
    import torch
    import verify_agents3_policy as common
    checkpoint=ROOT/'runs/model_cache/radixark_qwen38_nvfp4_319f741c'
    engines=root/'engines'
    if not (engines/'llm/processed_chat_template.json').exists():
        # Preview only tokenizer/template metadata; no engine/model load.
        engines=root/'policy_inputs/engine_preview'
        (engines/'llm').mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'prepared_chat_template.json',engines/'llm/processed_chat_template.json')
        shutil.copy2(ROOT/'runs/same_model_radixark_20261028/engine_build/engines/llm/tokenizer.json',engines/'llm/tokenizer.json')
    native=root/'build_source/chronotope/src'
    if not native.exists():native=ROOT/'chronotope/src'
    args=SimpleNamespace(contract=root/'policy_inputs/agents3_policy.json',fixtures=root/'policy_inputs/policy_fixtures.json',
        native_dir=native,engine_root=engines,out=root/'policy_verification.json',
        agents3_python=Path((_CHRONOTOPE_HOME + '/agents3-venv/bin/python')),
        json_include=Path((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/3rdParty/nlohmannJson/include')),
        checkpoint=checkpoint,reference_run=ROOT/'runs/agents3_policy_qwen08_20261028/experiment')
    report=common.verify(args)
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    x=((torch.arange(256,dtype=torch.float32)/255)-.5)/.5
    host=x.to(torch.bfloat16)
    legacy=x.to(torch.float16).to(torch.bfloat16)
    aligned=x.to(torch.bfloat16).to(torch.float16).to(torch.bfloat16)
    assert torch.equal(host,aligned), 'BF16-first storage changes normalized pixel values'
    # Compare model-consumed patches using the actual host processor/version.
    events=[json.loads(line) for line in (args.reference_run/'trials/w16000000000/legacy_cpp/closing_window-8400000_live.jsonl').read_text().splitlines()]
    begin=next(e for e in events if e['record']=='decision_begin')
    obs=next(e for e in events if e['record']=='observation' and e['observation_id']==begin['observation_id'])
    raw=np.frombuffer((args.reference_run/'payloads'/(obs['payload_hash']+'.bin')).read_bytes(),dtype=np.uint8).reshape(240,320,3)
    rgb=raw.repeat(2,axis=0).repeat(2,axis=1)
    processor=AutoProcessor.from_pretrained(str(checkpoint),local_files_only=True)
    kwargs=json.loads(args.contract.read_text())['generation_settings']['mm_processor_kwargs']
    pixels=processor.image_processor(images=[Image.fromarray(rgb)],return_tensors='pt',**kwargs).pixel_values
    normalized=(rgb.astype(np.float32)/np.float32(255)-np.float32(.5))/np.float32(.5)
    normalized=torch.from_numpy(normalized).to(torch.bfloat16).to(torch.float16).numpy()
    patches=np.stack([normalized,normalized]).transpose(0,3,1,2)
    patches=patches.reshape(1,2,3,15,2,16,20,2,16).transpose(0,3,6,4,7,2,1,5,8).reshape(1200,1536)
    native_patches=torch.from_numpy(patches).to(torch.bfloat16)
    host_patches=pixels.to(torch.bfloat16)
    assert torch.equal(native_patches,host_patches), 'Model-consumed BF16 image patches differ'
    report['model']='RadixArk/Qwen3.8-27B-NVFP4'
    report['revision']='319f741cce68d7914884900c138a1fbb70a42f30'
    report['native_template_mode']='Radix default preserve_thinking; generation enable_thinking=false'
    report['normalization_precision']={'rgb_levels_checked':256,
        'legacy_fp16_first_mismatches':int((host!=legacy).sum()),
        'legacy_max_absolute_error':float((host.float()-legacy.float()).abs().max()),
        'bf16_first_roundtrip_equal':True,'full_patch_bf16_equal':True,
        'full_patch_shape':list(host_patches.shape),
        'consumed_patch_sha256_bf16':hashlib.sha256(host_patches.view(torch.uint16).numpy().tobytes()).hexdigest(),
        'native_input':'FP32 normalization -> BF16 rounding -> FP16 storage -> BF16 visual graph',
        'host_input':'FP32 normalization -> BF16 visual dtype',
        'deferred_gpu_proof_required':str(root/'runtime/gpu_input_verification.json')}
    report['limits'].append('CPU patch agreement requires dedicated native BF16-first kernel; freeze additionally requires actual GPU preprocessing proof. Positional interpolation, inference kernels and accumulation may differ.')
    report['source_file_sha256'][str(Path(__file__).resolve())]=common.digest(Path(__file__).resolve())
    report['source_file_sha256'][str(ROOT/'studies/verify_agents3_policy.py')]=common.digest(ROOT/'studies/verify_agents3_policy.py')
    (root/'policy_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    try:
        report=verify(args.root.resolve())
    except Exception as error:
        args.root.mkdir(parents=True,exist_ok=True)
        (args.root/'policy_verification.json').write_text(json.dumps({'state':'failed','model_inference_calls':0,
            'experimental_trials':0,'error':f'{type(error).__name__}: {error}',
            'created_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
        raise
    print(json.dumps({k:report[k] for k in ['state','policy_sha256','first_request_input_tokens','normalization_precision']}))
