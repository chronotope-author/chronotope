"""Deferred official Gemma assets/services; never executes evaluation probes.

Every mutating phase requires the completed Qwen studies and the study GPU lock.
No checkpoint substitution, exporter patch, model warmup, or automatic retry.
"""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/legacy_detector_full_agents3_gemma4_26b_20261029'
OLD = ROOT / 'runs/legacy_detector_full_agents3_20261029'
REMAINING = ROOT / 'runs/legacy_detector_full_agents3_core5_qwen_20261029/experiment'
STOCK = ROOT / 'runs/runtime_reversal_stock_20261029'
SOURCE = STOCK / 'source/TensorRT-Edge-LLM'
BUILD = STOCK / 'build'
MODEL = 'nvidia/Gemma-4-26B-A4B-NVFP4'
REVISION = 'a19cfe00be84568a6867111c9a68c9c44fdcffe6'
CHECKPOINT = ROOT / 'runs/model_cache/nvidia_gemma4_26b_a4b_nvfp4_a19cfe0'
COMMIT = 'bb291453bc10d9bd1142070c605e090d4e483ee2'
PYTHON = (_CHRONOTOPE_HOME + '/edgellm-venv/bin/python')
GROUNDER = Path((_CHRONOTOPE_HOME + '/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e'))
sys.path.insert(0, str(ROOT / 'studies'))
from legacy_d1_w8 import now, write, digest

def read(p): return json.loads(Path(p).read_text())
def hashes(root):
    return {str(p): digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and '.cache' not in p.parts}
def verify(mapping):
    for p, expected in mapping.items():
        assert digest(p) == expected, p

def complete_predecessors():
    from resume_notification_schedule import completion_gate_for
    for path, count in [(OLD / 'experiment', 4800), (REMAINING, 10200)]:
        s = read(path / 'status.json')
        if path == OLD / 'experiment':
            ready, reason = completion_gate_for(path, s)
            assert ready, f'{path}: {reason}'
        else:
            assert s['state'] == 'complete' and s['completed_trials'] == s['planned_trials'] == count, str(path)
            assert s.get('qualification_trials') == 0 and not s.get('error'), str(path)
        pid = s.get('pid')
        if pid and Path(f'/proc/{pid}/cmdline').exists():
            assert str(path).encode() not in Path(f'/proc/{pid}/cmdline').read_bytes(), 'Predecessor still active'

@contextmanager
def exclusive(fd):
    complete_predecessors()
    if fd is not None:
        assert Path(f'/proc/self/fd/{fd}').resolve() == ROOT / 'runs/legacy_d1_gpu.lock'
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    else:
        with (ROOT / 'runs/legacy_d1_gpu.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

def verify_stock():
    m = read(STOCK / 'upstream_source_manifest.json')
    assert m['commit'] == COMMIT and m['local_source_patches'] == []
    verify({str(SOURCE / p): info['sha256'] for p, info in m['files'].items()})
    b = read(STOCK / 'native_build_result.json')
    assert b['state'] == 'complete'
    verify(b['artifact_hashes'])

def environment():
    env = dict(os.environ)
    for k in ['CUDA_LAUNCH_BLOCKING', 'VLLM_NSYS_PROFILING', 'VLLM_PROFILER_CONFIG', 'VLLM_TORCH_PROFILER_DIR']:
        env.pop(k, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', EDGELLM_NVFP4_MOE_TARGET='sm12x', EDGELLM_PLUGIN_PATH=str(BUILD / 'libNvInfer_edgellm_plugin.so'),
               LD_LIBRARY_PATH=':'.join([str(BUILD), (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'), '/usr/local/cuda/targets/sbsa-linux/lib']),
               PATH=(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin:/usr/local/cuda/bin:') + env.get('PATH', ''))
    return env

def command(args, log, env=None):
    with Path(log).open('xb') as stream:
        subprocess.run([str(x) for x in args], cwd=ROOT, env=env or environment(), stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)

def download():
    from huggingface_hub import snapshot_download, HfApi
    assert not (OUT / 'checkpoint_download.json').exists()
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=CHECKPOINT, max_workers=2)
    files = hashes(CHECKPOINT)
    for f in info.siblings:
        assert (CHECKPOINT / f.rfilename).is_file(), f.rfilename
        lfs = f.lfs
        if lfs:
            expected = lfs.sha256 if hasattr(lfs, 'sha256') else lfs['sha256']
            assert files[str(CHECKPOINT / f.rfilename)] == expected, f.rfilename
    c = read(CHECKPOINT / 'config.json'); q = read(CHECKPOINT / 'hf_quant_config.json')
    assert c['model_type'] == 'gemma4' and c['text_config']['enable_moe_block']
    assert q['quantization']['quant_algo'] == 'NVFP4' and q['quantization']['kv_cache_quant_algo'] == 'FP8'
    write(OUT / 'checkpoint_download.json', {'state': 'downloaded', 'completed_at': now(),
        'model': MODEL, 'revision': REVISION, 'checkpoint': str(CHECKPOINT), 'hashes': files,
        'source_precision': 'NVFP4 routed expert weights/activations; BF16 floating components; FP8 KV calibration',
        'inference_calls': 0, 'qualification_trials': 0})

def export():
    verify_stock(); d = read(OUT / 'checkpoint_download.json'); verify(d['hashes'])
    assert (d['model'], d['revision'], d['state']) == (MODEL, REVISION, 'downloaded')
    assert not (OUT / 'exports').exists() and not (OUT / 'export_result.json').exists()
    cmd = [PYTHON, '-B', '-m', 'tensorrt_edgellm.scripts.export', str(CHECKPOINT), str(OUT / 'exports')]
    env = environment(); env.update(PYTHONPATH=str(SOURCE), LLM_SDK_DIR=str(SOURCE), CUDA_VISIBLE_DEVICES='',
                                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = {'state': 'exporting', 'started_at': now(), 'model': MODEL, 'revision': REVISION,
              'command': cmd, 'upstream_commit': COMMIT, 'source_patches': [], 'checkpoint_hashes': d['hashes'],
              'inference_calls': 0, 'qualification_trials': 0}
    write(OUT / 'export_result.json', result)
    try:
        command(cmd, OUT / 'export.log', env)
        verify_stock(); verify(d['hashes'])
        result.update(state='exported', output_hashes=hashes(OUT / 'exports'))
    except BaseException as e:
        result.update(state='failed', error=f'{type(e).__name__}: {e}'); raise
    finally:
        result['completed_at'] = now(); write(OUT / 'export_result.json', result)

def planned_kv_cache():
    value = read(OUT / 'experiment/configs/model.json')['kv_cache']
    assert value in ('FP16', 'FP8 E4M3'), f'Unreviewed KV precision: {value}'
    return value


def audit_export():
    import onnx
    from collections import Counter
    import math
    result = {'state': 'passed', 'created_at': now(), 'model': MODEL, 'revision': REVISION,
              'source_patches': [], 'inference_calls': 0, 'qualification_trials': 0}
    for name in ['llm', 'visual']:
        model = onnx.load(OUT / f'exports/{name}/model.onnx', load_external_data=False)
        types = Counter(); ops = Counter(n.op_type for n in model.graph.node)
        for t in model.graph.initializer: types[onnx.TensorProto.DataType.Name(t.data_type)] += math.prod(t.dims)
        inputs = {v.name: onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type) for v in model.graph.input}
        result[name] = {'initializer_elements_by_dtype': dict(types), 'inputs': inputs, 'ops': dict(ops)}
        assert not types.get('BFLOAT16', 0), 'Stock floating runtime precision changed; do not mislabel the pair'
    assert result['visual']['initializer_elements_by_dtype'].get('FLOAT16', 0) > 0
    assert any('nvfp4' in k.lower() for k in result['llm']['ops']) or result['llm']['initializer_elements_by_dtype'].get('FLOAT4E2M1', 0), 'Quantized experts missing'
    cfg = read(OUT / 'exports/llm/config.json')
    result['export_config'] = cfg
    accepted = ('fp16', 'float16') if planned_kv_cache() == 'FP16' else ('fp8', 'float8', 'fp8_e4m3')
    assert str(cfg.get('kv_cache_dtype', '')).lower() in accepted, 'Stock export KV precision differs from the paired experiment plan'
    write(OUT / 'onnx_precision_audit.json', result)

def build():
    verify_stock(); e = read(OUT / 'export_result.json'); assert e['state'] == 'exported'; verify(e['output_hashes'])
    assert not (OUT / 'engine_build_result.json').exists()
    audit_export()
    cmds = [
        [BUILD / 'examples/llm/llm_build', '--onnxDir', OUT / 'exports/llm', '--engineDir', OUT / 'engines/llm',
         '--maxInputLen', '16384', '--maxKVCacheCapacity', '16384', '--maxBatchSize', '1'],
        [BUILD / 'examples/multimodal/visual_build', '--onnxDir', OUT / 'exports/visual', '--engineDir', OUT / 'engines',
         '--minImageTokens', '64', '--maxImageTokens', '280', '--maxImageTokensPerImage', '280']]
    result = {'state': 'building', 'started_at': now(), 'model': MODEL, 'revision': REVISION,
              'upstream_commit': COMMIT, 'source_patches': [], 'commands': [[str(x) for x in c] for c in cmds],
              'source_checkpoint_hashes': read(OUT / 'checkpoint_download.json')['hashes'],
              'source_export_hashes': e['output_hashes'], 'kv_cache': {'dtype': planned_kv_cache(), 'override': None},
              'runtime_precision': 'NVFP4 routed expert weights with FP16 activations (W4A16); FP16 floating LLM and vision tensors; ' + planned_kv_cache() + ' KV',
              'inference_calls': 0, 'evaluation_trials': 0, 'qualification_trials': 0}
    write(OUT / 'engine_build_result.json', result)
    try:
        for cmd, log in zip(cmds, ['llm_build.log', 'visual_build.log']): command(cmd, OUT / log)
        verify_stock(); verify(e['output_hashes'])
        result.update(state='built', hashes=hashes(OUT / 'engines'), builder_config=read(OUT / 'engines/llm/config.json')['builder_config'])
    except BaseException as exc:
        result.update(state='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        result['completed_at'] = now(); write(OUT / 'engine_build_result.json', result)

def stop_old_services():
    import nvidia_timing_rollout as ctl
    from enable_nvidia_request_metrics import descendants
    report = {'state': 'stopping_completed_qwen_services', 'started_at': now(), 'services': []}
    assert not (OUT / 'services_handoff.json').exists()
    write(OUT / 'services_handoff.json', report)
    for role in ['planner', 'grounder']:
        launch = read(OLD / f'services/{role}/launch.json')
        proc = ctl.process(launch['pid'])
        if proc is None:
            report['services'].append({'role': role, 'already_exited': True}); continue
        assert proc['starttime_ticks'] == launch['process_starttime_ticks'] and proc['command'] == launch['command']
        assert ctl.service_idle(launch['base_url'])
        children = descendants(proc['pid'])
        assert ctl.still_same(proc)
        os.kill(proc['pid'], signal.SIGTERM)
        end = time.monotonic() + 120
        while any(ctl.still_same(x) for x in [proc, *children]):
            if time.monotonic() > end: raise TimeoutError(f'{role} did not exit; no broad process signal sent')
            time.sleep(1)
        report['services'].append({'role': role, 'process': proc, 'descendants': children})
        write(OUT / 'services_handoff.json', report)
    report.update(state='complete', completed_at=now()); write(OUT / 'services_handoff.json', report)

def services(launcher=None):
    assert read(OUT / 'services_handoff.json')['state'] == 'complete'
    assert read(OUT / 'engine_build_result.json')['state'] == 'built'
    assert read(OUT / 'native_build_result.json')['state'] == 'complete'
    spec = importlib.util.spec_from_file_location('original_service_launcher', launcher or OLD / 'launch_services.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.RUN = OUT
    env = environment(); env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('PYTHONPATH', None)
    dependency_pin = OUT / 'serving_dependency_pin.json'
    pinned = read(dependency_pin) if dependency_pin.exists() else None
    if pinned:
        assert pinned['package'] == 'transformers' and pinned['version'] == '5.14.1'
        verify(pinned['hashes'])
        env['PYTHONPATH'] = pinned['path']
    common = ['--host', '127.0.0.1', '--max-num-seqs', '1', '--generation-config', 'vllm',
              '--no-enable-prefix-caching', '--enable-per-request-metrics']
    kv = planned_kv_cache()
    kv_arg = 'float16' if kv == 'FP16' else 'fp8_e4m3'
    cmd = [(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/vllm'), 'serve', str(CHECKPOINT), '--served-model-name', MODEL,
           '--dtype', 'half', '--port', '8012', '--max-model-len', '16384', '--gpu-memory-utilization', '0.60',
           '--kv-cache-memory-bytes', '2147483648', '--kv-cache-dtype', kv_arg,
           '--limit-mm-per-prompt', '{"image":9}', '--reasoning-parser', 'gemma4', '--moe-backend', 'marlin', *common]
    component = read(OUT / 'experiment/configs/full_agent_components.json')['components']['planner']
    planner = module.start('planner', cmd, env, {
        'model': MODEL, 'revision': REVISION, 'runtime': 'vLLM', 'dtype': 'half',
        'serving_dependency_pin': pinned,
        'precision': component['precision'],
        'moe_backend': 'marlin',
        'kv_cache_dtype': kv_arg, 'effective_kv_cache_dtype': kv, 'kv_cache': kv,
        'quantization': 'modelopt', 'max_model_len': 16384, 'max_num_seqs': 1,
        'max_images_per_prompt': 9, 'enable_prefix_caching': False, 'enable_per_request_metrics': True,
        'checkpoint_path': str(CHECKPOINT), 'source_precision': 'NVFP4 routed experts; BF16 floating components',
        'required_log_markers': ['dtype=torch.float16', 'quantization=modelopt', 'kv_cache_dtype=' + kv_arg, 'max_seq_len=16384', 'enable_prefix_caching=False']})
    # The compatibility pin applies only to the Gemma planner service.
    env.pop('PYTHONPATH', None)
    gcmd = [(_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/vllm'), 'serve', str(GROUNDER),
        '--served-model-name', 'ByteDance-Seed/UI-TARS-1.5-7B', '--dtype', 'bfloat16', '--port', '8013',
        '--max-model-len', '8192', '--gpu-memory-utilization', '0.20', '--kv-cache-memory-bytes', '1073741824',
        '--kv-cache-dtype', 'auto', '--limit-mm-per-prompt', '{"image":1}', *common]
    grounder = module.start('grounder', gcmd, env, {
        'model': 'ByteDance-Seed/UI-TARS-1.5-7B', 'revision': GROUNDER.name, 'runtime': 'vLLM',
        'precision': 'BF16', 'dtype': 'bfloat16', 'kv_cache_dtype': 'auto', 'effective_kv_cache_dtype': 'bfloat16',
        'checkpoint_path': str(GROUNDER), 'max_model_len': 8192, 'max_num_seqs': 1,
        'max_images_per_prompt': 1, 'enable_prefix_caching': False, 'enable_per_request_metrics': True,
        'required_log_markers': ['dtype=torch.bfloat16', 'quantization=None', 'max_seq_len=8192', 'enable_prefix_caching=False']})
    write(OUT / 'services/launch.json', {'state': 'ready', 'created_at': now(), 'qualification_trials': 0,
        'inference_readiness_requests': 0, 'components': {'planner': planner, 'reflection': planner,
        'text_grounder': planner, 'grounder': grounder}})

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['download', 'stop-services', 'export', 'build', 'services'], required=True)
    ap.add_argument('--lock-fd', type=int)
    args = ap.parse_args()
    with exclusive(args.lock_fd):
        {'download': download, 'stop-services': stop_old_services, 'export': export, 'build': build, 'services': services}[args.phase]()
    print(json.dumps({'state': 'complete', 'phase': args.phase, 'model': MODEL, 'inference_probes': 0}), flush=True)
