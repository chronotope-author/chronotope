#!/usr/bin/env python3
"""Freeze and build the exact Radix paired-policy native artifacts. No inference/trials.

--prepare performs CPU-only source/template preparation. --build is called by the
successor queue while it holds the GPU lock, after the predecessor audit passed.
This process deliberately does not acquire that lock a second time.
"""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
_CHRONOTOPE_REPO_ROOT = _CHRONOTOPE_ENV.get('CHRONOTOPE_REPO_ROOT', str(next(
    (base for base in _CHRONOTOPE_PATH(__file__).resolve().parents if (base / 'pyproject.toml').is_file()),
    _CHRONOTOPE_PATH(_CHRONOTOPE_HOME) / 'chronotope')))
import argparse, datetime, hashlib, json, os, re, shlex, shutil, subprocess
from pathlib import Path

REPO=Path((_CHRONOTOPE_REPO_ROOT + ''))
MODEL='RadixArk/Qwen3.8-27B-NVFP4'
REVISION='319f741cce68d7914884900c138a1fbb70a42f30'
CHECKPOINT=REPO/'runs/model_cache/radixark_qwen38_nvfp4_319f741c'
OLD=REPO/'runs/same_model_radixark_20261028/engine_build'
OLD_MANIFEST=OLD.parent/'experiment/execution_manifest.json'
RUNTIME=REPO/'runs/agents3_policy_qwen08_20261028/runtime'
UPSTREAM=Path((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM'))
PYTHON=Path((_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/python'))
PRECISION='NVFP4 MLP/lm_head; FP8 attention; BF16 vision source weights'
NUMERICS=['Preserved BF16 visual weights/internal attention; FP16 visual I/O and dense LLM sidecars.',
 'Pixel normalization explicitly rounds FP32 to BF16 before lossless FP16 I/O storage, matching vLLM BF16 vision input.',
 'Native position interpolation uses exported BF16 products/sums and FP16 interpolation-weight I/O; vLLM uses its fused Triton implementation. Internal backend numerical equality is not claimed.',
 'Native convolution state FP16, recurrent state FP32; native and vLLM backend arithmetic need not be bit-identical.']

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def treehash(root): return {str(p):sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}
def verify(hashes):
    for p,expected in hashes.items():
        if not Path(p).is_file() or sha(p)!=expected: raise RuntimeError('Pinned input changed: '+p)
def require_predecessor(path):
    status=read(path/'status.json');audit=read(path/'audit_status.json')
    if not(status.get('state')=='complete' and status.get('completed_trials')==4800 and status.get('planned_trials')==4800
        and audit.get('state')=='finished' and audit.get('records')==4800 and audit.get('infrastructure_errors')==0):
        raise RuntimeError('Predecessor must finish all4800 trials and evidence audit without infrastructure errors')
    return {str(p):sha(p) for p in [path/'status.json',path/'audit_status.json']}
def run(cmd,log,env=None,cwd=None):
    print('RUN '+shlex.join(map(str,cmd)),flush=True)
    with Path(log).open('wb') as f:
        result=subprocess.run(list(map(str,cmd)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd)
    if result.returncode:raise RuntimeError(f'Command exited {result.returncode}: see {log}')
def patch_image(path):
    text=path.read_text()
    anchor='#include <cuda_fp16.h>'
    if text.count(anchor)!=1: raise RuntimeError('Normalizer include anchor mismatch')
    text=text.replace(anchor,anchor+'\n#include <cuda_bf16.h>')
    old='    normalizedImage[tid] = __float2half(normalized);'
    new='    // Dedicated Radix BF16 vision: avoid FP16 double rounding before the graph BF16 cast.\n    normalizedImage[tid] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(normalized)));'
    if text.count(old)!=1:raise RuntimeError('Normalizer assignment anchor mismatch')
    arithmetic='    float normalized = (val / 255.0f - mean[channel]) / std[channel];'
    if text.count(arithmetic)!=1:raise RuntimeError('Normalizer arithmetic anchor mismatch')
    text=text.replace(arithmetic, '    float normalized = __fdiv_rn(__fsub_rn(__fdiv_rn(static_cast<float>(val), 255.0f), mean[channel]), std[channel]);')
    path.write_text(text.replace(old,new))
def prepare(out,predecessor):
    target=out/'engine_preparation.json'
    if target.exists():
        manifest=read(target);verify(manifest['input_hashes']);print('Preparation already pinned: '+str(target));return
    out.mkdir(parents=True,exist_ok=True)
    frozen=out/'build_source'
    if frozen.exists():raise RuntimeError('Partial source freeze exists; inspect it before retrying')
    (frozen/'chronotope').mkdir(parents=True)
    shutil.copy2(REPO/'chronotope/CMakeLists.txt',frozen/'chronotope/CMakeLists.txt')
    for name in ['src','tests']:shutil.copytree(REPO/'chronotope'/name,frozen/'chronotope'/name)
    shutil.copytree(RUNTIME/'cpp',frozen/'runtime/cpp',symlinks=True)
    patch_image(frozen/'runtime/cpp/kernels/preprocessKernels/imageUtilKernels.cu')
    helpers=frozen/'helpers';helpers.mkdir()
    for name in ['verify_radix_image.py','verify_radix_image.cu']:
        shutil.copy2(REPO/'studies/native_agents3_runtime'/name,helpers/name)
    shutil.copy2(Path(__file__),helpers/'prepare_radix_agents3_engines.py')
    original=read(REPO/'runs/agents3_policy_qwen08_20261028/engines/llm/processed_chat_template.json')
    original['model_path']=str(CHECKPOINT)
    original['roles']['assistant']['prefix']='<|im_start|>assistant\n<think>\n\n</think>\n\n'
    original['trim_content']=False
    write(out/'prepared_chat_template.json',original)
    inputs=treehash(frozen)
    for p in [out/'prepared_chat_template.json',OLD_MANIFEST,OLD/'onnx_precision_audit.json',
        OLD/'export_preserved_launch.json',OLD/'bf16_export_patch.json',RUNTIME/'build_report.json',
        RUNTIME/'build/cpp/libedgellmCore.a',UPSTREAM/'build/examples/llm/llm_build',
        UPSTREAM/'build/libNvInfer_edgellm_plugin.so',
        UPSTREAM/'build/cpp/CMakeFiles/edgellmCore.dir/flags.make',
        UPSTREAM/'build/cpp/CMakeFiles/edgellmCore.dir/includes_CUDA.rsp',Path(__file__).resolve()]:inputs[str(p)]=sha(p)
    manifest={'state':'prepared','prepared_at':now(),'model':MODEL,'revision':REVISION,'precision':PRECISION,
        'predecessor':str(predecessor),'input_hashes':inputs,'source_checkpoint_hashes':read(OLD_MANIFEST)['checkpoint_hashes'],
        'source_export':str(OLD/'onnx_preserved'),'runtime_source':str(RUNTIME),'runtime_numerics':NUMERICS,
        'inference_calls':0,'evaluation_trials':0,'qualification_trials':0,
        'build_requires':'successful predecessor completion/audit; caller holds exclusive GPU lock',
        'model_input_alignment':{'implementation':'FP32 -> BF16 -> FP16 I/O -> BF16 first visual Conv','aligned':False,'technical_gpu_proof_pending':True}}
    write(target,manifest);print(json.dumps({'state':'prepared','manifest':str(target)}))
def build_runtime(out,env):
    runtime=out/'runtime';frozen=out/'build_source'
    report=read(RUNTIME/'build_report.json')
    if sha(RUNTIME/'build/cpp/libedgellmCore.a')!=report['patched_core_sha256']:raise RuntimeError('Inherited S3 runtime archive hash mismatch')
    shutil.copytree(frozen/'runtime/cpp',runtime/'cpp',symlinks=True)
    for name in ['examples','3rdParty']:(runtime/name).symlink_to(UPSTREAM/name,target_is_directory=True)
    (runtime/'build/cpp').mkdir(parents=True)
    (runtime/'build/generated').symlink_to(UPSTREAM/'build/generated',target_is_directory=True)
    for p in (UPSTREAM/'build').glob('libNvInfer_edgellm_plugin.so*'):(runtime/'build'/p.name).symlink_to(p.resolve())
    objroot=UPSTREAM/'build/cpp/CMakeFiles/edgellmCore.dir'
    flags=(objroot/'flags.make').read_text()
    defines=shlex.split(re.search(r'^CUDA_DEFINES = (.*)$',flags,re.M).group(1))
    cuda_flags=shlex.split(re.search(r'^CUDA_FLAGS = (.*)$',flags,re.M).group(1))
    # CMake's flags contain redundant bare -fPIC, which nvcc does not accept.
    # Existing -Xcompiler=-fPIC forwards the same host option correctly.
    cuda_flags=[flag for flag in cuda_flags if flag != '-fPIC']
    # Preserve all upstream flags; only normalization uses explicit CUDA RN intrinsics.
    includes=shlex.split((objroot/'includes_CUDA.rsp').read_text().replace(str(UPSTREAM/'cpp'),str(runtime/'cpp')))
    obj=runtime/'build/imageUtilKernels.cu.o';source=runtime/'cpp/kernels/preprocessKernels/imageUtilKernels.cu'
    cmd=['/usr/local/cuda/bin/nvcc',*defines,*includes,*cuda_flags,'-dc',str(source),'-o',str(obj)]
    run(cmd,runtime/'compile_normalization.log',env)
    archive=runtime/'build/cpp/libedgellmCore.a';shutil.copy2(RUNTIME/'build/cpp/libedgellmCore.a',archive)
    members=subprocess.check_output(['ar','t',str(archive)],text=True).splitlines()
    if members.count(obj.name)!=1:raise RuntimeError('Image kernel archive member missing/ambiguous')
    subprocess.run(['ar','r',str(archive),str(obj)],check=True);subprocess.run(['ranlib',str(archive)],check=True)
    result={**report,'runtime':str(runtime),'parent_runtime':str(RUNTIME),'parent_core_sha256':report['patched_core_sha256'],
        'patched_core_sha256':sha(archive),'dedicated_model':MODEL,'normalization_compile_command':cmd,
        'normalization_source_sha256':sha(source),'inference_calls':0,'evaluation_trials':0}
    write(runtime/'build_report.json',result)
    run([PYTHON,frozen/'helpers/verify_radix_image.py','--runtime',runtime,'--checkpoint',CHECKPOINT,
        '--out',runtime/'gpu_input_verification.json'],runtime/'gpu_input_verification.log',env)
    proof=read(runtime/'gpu_input_verification.json')
    if not proof.get('passed') or proof.get('inference_calls')!=0 or proof.get('evaluation_trials')!=0:raise RuntimeError('GPU input verification failed')
    native=out/'native_build'
    run(['cmake','-S',frozen/'chronotope','-B',native,'-DCHRONOTOPE_EDGELLM_BUILD='+str(runtime/'build'),
        ('-DCHRONOTOPE_TRT_PREFIX=' + _CHRONOTOPE_HOME + '/trt-prefix'),'-DCMAKE_BUILD_TYPE=Release'],out/'native_configure.log',env)
    run(['cmake','--build',native,'--target','chronotope_main','chronotope','-j6'],out/'native_build.log',env)
    return proof

def build(out,predecessor):
    preparation=read(out/'engine_preparation.json');verify(preparation['input_hashes'])
    predecessor_proof=require_predecessor(predecessor)
    result_path=out/'engine_build_result.json'
    if result_path.exists():
        previous=read(result_path)
        if previous.get('state')=='built':verify(previous['hashes']);verify(previous['native_binary_hashes']);print('Already built and verified');return
        raise RuntimeError('Existing partial/failed build; inspect and resolve it before retrying')
    if (out/'runtime').exists() or (out/'engines').exists():raise RuntimeError('Untracked build output exists')
    result={'state':'building','started_at':now(),'model':MODEL,'revision':REVISION,'precision':PRECISION,
        'inference_calls':0,'evaluation_trials':0,'qualification_trials':0,'predecessor_proof':predecessor_proof}
    write(result_path,result)
    try:
        verify(preparation['source_checkpoint_hashes'])
        audit=read(OLD/'onnx_precision_audit.json')
        for kind in ['llm','visual']:
            if sha(OLD/'onnx_preserved'/kind/'model.onnx')!=audit[kind]['onnx_sha256']:raise RuntimeError('Export ONNX provenance mismatch: '+kind)
        if audit['visual']['initializer_elements_by_dtype'].get('BFLOAT16',0)!=460603376:raise RuntimeError('BF16 vision source was changed')
        if audit['llm']['initializer_elements_by_dtype'].get('FLOAT4E2M1',0)!=18384158720 or audit['llm']['initializer_elements_by_dtype'].get('FLOAT8E4M3FN',0)!=8363212800:raise RuntimeError('Mixed source quantization was changed')
        oldmanifest=read(OLD_MANIFEST)
        if oldmanifest['model']!=MODEL or oldmanifest['revision']!=REVISION:raise RuntimeError('Historical source identity mismatch')
        verify({p:h for p,h in oldmanifest['engine_hashes'].items() if '/visual/' in p})
        env=dict(os.environ)
        env['EDGELLM_PLUGIN_PATH']=str(UPSTREAM/'build/libNvInfer_edgellm_plugin.so')
        env['LD_LIBRARY_PATH']=(_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib:')+env.get('LD_LIBRARY_PATH','')
        command=[UPSTREAM/'build/examples/llm/llm_build','--onnxDir',OLD/'onnx_preserved/llm','--engineDir',out/'engines/llm',
            '--maxInputLen','16384','--maxKVCacheCapacity','16384','--maxBatchSize','1']
        write(out/'engine_build_launch.json',{'command':list(map(str,command)),'inputs':treehash(OLD/'onnx_preserved'),
            'model':MODEL,'revision':REVISION,'source_checkpoint_hashes':preparation['source_checkpoint_hashes'],
            'source_preparation_sha256':sha(out/'engine_preparation.json'),
            'external_data_provenance':'Current full external-data SHA pinned before build. Historical export command, source checkpoint SHA and ONNX graph SHA are recorded; no historical external-data SHA was recorded.',
            'inference_calls':0,'evaluation_trials':0})
        proof=build_runtime(out,env)
        shutil.copytree(OLD/'engines/visual',out/'engines/visual')
        run(command,out/'llm_build.log',env)
        llm=out/'engines/llm';shutil.copy2(out/'prepared_chat_template.json',llm/'processed_chat_template.json')
        config=read(llm/'config.json');bc=config['builder_config']
        if bc['max_input_len']!=16384 or bc['max_kv_cache_capacity']!=16384 or bc['max_batch_size']!=1:raise RuntimeError('Incorrect built engine context/batch limits')
        if config.get('kv_cache_dtype')!='fp8' or sorted(config['eos_token_id'])!=[248044,248046]:raise RuntimeError('Incorrect KV cache or EOS settings')
        result.update(state='built',completed_at=now(),builder_config=bc,kv_cache={'dtype':'FP8 E4M3','scales':'static1.0','calculate_kv_scales':False},
            runtime_numerics=NUMERICS,source_export=str(OLD/'onnx_preserved'),source_checkpoint_hashes=preparation['source_checkpoint_hashes'],
            hashes=treehash(out/'engines'),native_binary_hashes={str(p):sha(p) for p in [out/'native_build/chronotope',out/'native_build/libchronotope.so']},
            native_source_hashes=treehash(out/'build_source/chronotope'),
            model_input_alignment={'aligned':True,'scope':'BF16 image values entering first visual Conv; policy/template separately verified',
                'gpu_proof':str(out/'runtime/gpu_input_verification.json'),'gpu_proof_sha256':sha(out/'runtime/gpu_input_verification.json'),
                'model_numerical_equality_claimed':False})
        write(result_path,result);print(json.dumps({'state':'built','result':str(result_path)}))
    except BaseException as error:
        result.update(state='failed',updated_at=now(),error=f'{type(error).__name__}: {error}')
        write(result_path,result);raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--predecessor',type=Path,default=REPO/'runs/agents3_policy_qwen08_20261028/experiment')
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--prepare',action='store_true');modes.add_argument('--build',action='store_true')
    a=parser.parse_args();out=a.out.resolve();predecessor=a.predecessor.resolve()
    if not out.is_relative_to(REPO/'runs') or out==predecessor or out==OLD:raise RuntimeError('Output must be a fresh successor root under runs/')
    if a.prepare:prepare(out,predecessor)
    else:build(out,predecessor)
if __name__=='__main__':main()
