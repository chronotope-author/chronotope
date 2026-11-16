#!/usr/bin/env python3
"""Verify actual dedicated native image kernels against HF inputs; zero inference."""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse, hashlib, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); runtime=a.runtime.resolve(); source=Path(__file__).with_suffix('.cu')
    archive=runtime/'build/cpp/libedgellmCore.a'
    trt=(_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'); cuda='/usr/local/cuda/targets/sbsa-linux/lib'
    with tempfile.TemporaryDirectory(prefix='radix-image-kernel-') as temp:
        temp=Path(temp); exe=temp/'verify'; actual_file=temp/'actual.bin'
        cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-rdc=true',
            '--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]','-Wno-deprecated-declarations',
            '-I'+str(runtime/'cpp'),'-I'+str(runtime/'3rdParty/nlohmannJson/include'),
            ('-I' + _CHRONOTOPE_HOME + '/trt-prefix/usr/include/aarch64-linux-gnu'),str(source),str(archive),
            str(runtime/'cpp/kernels/cuteDSLArtifact/aarch64/sm_121/libcutedsl_aarch64.a'),
            '-L'+trt,'-L'+cuda,'-L'+cuda+'/stubs','-lnvonnxparser','-lnvinfer','-lcuda','-ldl','-lrt','-lpthread',
            '-Xlinker=--unresolved-symbols=ignore-in-shared-libs','-o',str(exe)]
        subprocess.run(cmd,check=True)
        env=dict(os.environ);env['LD_LIBRARY_PATH']=trt+':'+cuda+':'+env.get('LD_LIBRARY_PATH','')
        subprocess.run([str(exe),str(actual_file)],env=env,check=True)
        actual=np.fromfile(actual_file,dtype=np.float16).reshape(1200,1536)
        rgb=(np.arange(480*640*3,dtype=np.uint32)%256).astype(np.uint8).reshape(480,640,3)
        processor=AutoProcessor.from_pretrained(str(a.checkpoint),local_files_only=True)
        pixels=processor.image_processor(images=[rgb],return_tensors='pt')['pixel_values']
        expected=pixels.to(torch.bfloat16).float().numpy()
        observed=torch.from_numpy(actual).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(observed,expected)
        np.testing.assert_array_equal(actual.astype(np.float32),expected)
        x=torch.arange(256,dtype=torch.float32)
        normalized=(x/255.-.5)/.5
        direct=normalized.to(torch.bfloat16)
        fixed=direct.to(torch.float16).to(torch.bfloat16)
        assert torch.equal(direct,fixed)
        old=normalized.to(torch.float16).to(torch.bfloat16)
        mismatch=torch.where(old!=direct)[0].tolist()
        report={'passed':True,'inference_calls':0,'evaluation_trials':0,
            'test_type':'actual native CUDA normalization and Qwen patch-layout unit check against installed HF processor',
            'shape':[1200,1536],'all_256_values_verified':True,'patch_elements_bit_exact':int(actual.size),
            'comparison_dtype':'BF16 values before first visual convolution',
            'original_double_round_mismatch_values':mismatch,'normalization':'FP32 -> BF16 -> FP16 I/O -> BF16 graph input',
            'model_numerical_equality_claimed':False,'compile_command':cmd,
            'source_hashes':{str(q.resolve()):sha(q) for q in [source,Path(__file__),archive,
                runtime/'cpp/kernels/preprocessKernels/imageUtilKernels.cu',a.checkpoint/'preprocessor_config.json']}}
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'report':str(a.out),'inference_calls':0,'evaluation_trials':0}))
if __name__=='__main__': main()
