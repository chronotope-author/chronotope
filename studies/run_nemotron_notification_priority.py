"""Deferred exact-checkpoint Nemotron pair after audited Gemma Notification completion."""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
from pathlib import Path
from collections import Counter
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies'))
import gemma_ledger_assets as assets
import run_gemma_notification_priority as common
from core5_ledger_plan import validate_plan, load_cases
from legacy_d1_w8 import now, write, digest

OUT = ROOT / 'runs/notification_nemotron_bolt_20261030'
EXP = OUT / 'experiment'
PREDECESSOR = ROOT / 'runs/notification_gemma4_26b_20261030'
MODEL = 'nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16'
REVISION = '49843eed8db1f1c616c47af75cfd36e07aec0667'
CHECKPOINT = ROOT / 'runs/model_cache/nvidia_nemotron35_bolt_bf16_49843ee'
AGENT_PY = (_CHRONOTOPE_HOME + '/agents3-venv/bin/python')
TOKEN_PY = (_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/python')
ORIGINAL_SERVICE_SOURCE = ROOT / 'runs/legacy_detector_full_agents3_20261029/launch_services.py'

def read(path):
    return json.loads(Path(path).read_text())

def publish(state, **detail):
    write(OUT / 'priority_status.json', dict(state=state, updated_at=now(), pid=os.getpid(),
          model=MODEL, revision=REVISION, planned_trials=3000, qualification_trials=0,
          automatic_resume=False, predecessor=str(PREDECESSOR), **detail))

def validate_scope():
    plan = read(EXP / 'plan.json')
    assert validate_plan(plan) == 3000
    assert {j['task'] for j in plan['jobs']} == {'notification'}
    assert {j['window_s'] for j in plan['jobs']} == {16., 8., 4., 2., 1.}
    assert (plan['model'], plan['revision']) == (MODEL, REVISION)
    assert plan['qualification_trials'] == 0
    load_cases(EXP, plan)
    parent = read(PREDECESSOR / 'experiment/plan.json')
    assert plan['case_file_sha256'] == parent['case_file_sha256']

def predecessor_ready():
    status = read(PREDECESSOR / 'experiment/status.json')
    proof_path = PREDECESSOR / 'completion_audit.json'
    if status.get('state') != 'complete' or not proof_path.exists():
        return False
    proof = read(proof_path)
    assert status['completed_trials'] == status['planned_trials'] == 3000 and not status['error']
    assert proof['state'] == 'passed' and proof['completed_trials'] == proof['planned_trials'] == 3000
    assert proof['plan_sha256'] == digest(PREDECESSOR / 'experiment/plan.json')
    assert proof['execution_manifest_sha256'] == digest(PREDECESSOR / 'experiment/execution_manifest.json')
    assert proof['model_inference_calls'] == proof['experimental_trials'] == proof['qualification_trials'] == 0
    return read(PREDECESSOR / 'priority_status.json')['state'] == 'complete'

def download():
    from huggingface_hub import HfApi, snapshot_download
    assert not (OUT / 'checkpoint_download.json').exists()
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=CHECKPOINT, max_workers=2)
    files = assets.hashes(CHECKPOINT)
    for item in info.siblings:
        assert (CHECKPOINT / item.rfilename).is_file()
        if item.lfs:
            expected = item.lfs.sha256 if hasattr(item.lfs, 'sha256') else item.lfs['sha256']
            assert files[str(CHECKPOINT / item.rfilename)] == expected
    config = read(CHECKPOINT / 'config.json')
    assert config['model_type'] == 'gemma4' and config['architectures'] == ['Gemma4ForConditionalGeneration']
    assert config['dtype'] == 'bfloat16' and not config['text_config']['enable_moe_block']
    keys = read(CHECKPOINT / 'model.safetensors.index.json')['weight_map']
    assert any(key.startswith('model.vision_tower.') for key in keys), 'Missing vision tower weights'
    assert 'model.embed_vision.embedding_projection.weight' in keys, 'Missing Gemma4 vision projection weights'
    write(OUT / 'checkpoint_download.json', dict(state='downloaded', completed_at=now(), model=MODEL,
          revision=REVISION, checkpoint=str(CHECKPOINT), hashes=files, source_precision='BF16',
          runtime_conversion='Stock BF16-to-FP16; matching vLLM dtype half', inference_calls=0, qualification_trials=0))

def export():
    assets.verify_stock(); proof = read(OUT / 'checkpoint_download.json'); assets.verify(proof['hashes'])
    assert not (OUT / 'exports').exists() and not (OUT / 'export_result.json').exists()
    cmd = [assets.PYTHON, '-B', '-m', 'tensorrt_edgellm.scripts.export', str(CHECKPOINT),
           str(OUT / 'exports'), '--dtype', 'float16', '--skip-audio']
    env = assets.environment()
    env.update(PYTHONPATH=str(assets.SOURCE), LLM_SDK_DIR=str(assets.SOURCE), CUDA_VISIBLE_DEVICES='',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = dict(state='exporting', started_at=now(), model=MODEL, revision=REVISION, command=cmd,
                  upstream_commit=assets.COMMIT, source_patches=[], checkpoint_hashes=proof['hashes'],
                  inference_calls=0, qualification_trials=0)
    write(OUT / 'export_result.json', result)
    try:
        assets.command(cmd, OUT / 'export.log', env)
        assets.verify_stock(); assets.verify(proof['hashes'])
        result.update(state='exported', output_hashes=assets.hashes(OUT / 'exports'))
    except BaseException as exc:
        result.update(state='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        result['completed_at'] = now(); write(OUT / 'export_result.json', result)

def build():
    import onnx
    assets.verify_stock(); exported = read(OUT / 'export_result.json')
    assert exported['state'] == 'exported'; assets.verify(exported['output_hashes'])
    assert not (OUT / 'engine_build_result.json').exists()
    audit = dict(state='passed', model=MODEL, revision=REVISION, source_patches=[])
    for name in ['llm', 'visual']:
        graph = onnx.load(OUT / f'exports/{name}/model.onnx', load_external_data=False).graph
        types = Counter()
        for tensor in graph.initializer:
            types[onnx.TensorProto.DataType.Name(tensor.data_type)] += math.prod(tensor.dims)
        assert not types.get('BFLOAT16', 0)
        assert types.get('FLOAT16', 0) > 0
        assert not any('nvfp4' in node.op_type.lower() for node in graph.node)
        audit[name] = {'initializer_elements_by_dtype': dict(types)}
    config = read(OUT / 'exports/llm/config.json')
    assert config['kv_cache_dtype'] == 'fp16'; audit['export_config'] = config
    write(OUT / 'onnx_precision_audit.json', audit)
    commands = [
        [assets.BUILD / 'examples/llm/llm_build', '--onnxDir', OUT / 'exports/llm', '--engineDir', OUT / 'engines/llm',
         '--maxInputLen', '16384', '--maxKVCacheCapacity', '16384', '--maxBatchSize', '1'],
        [assets.BUILD / 'examples/multimodal/visual_build', '--onnxDir', OUT / 'exports/visual', '--engineDir', OUT / 'engines',
         '--minImageTokens', '64', '--maxImageTokens', '280', '--maxImageTokensPerImage', '280']]
    result = dict(state='building', started_at=now(), model=MODEL, revision=REVISION,
                  upstream_commit=assets.COMMIT, source_patches=[], commands=[[str(x) for x in cmd] for cmd in commands],
                  source_checkpoint_hashes=read(OUT / 'checkpoint_download.json')['hashes'],
                  source_export_hashes=exported['output_hashes'], kv_cache={'dtype': 'FP16', 'override': None},
                  runtime_precision='FP16 LLM and vision; FP16 KV; source checkpoint BF16',
                  inference_calls=0, evaluation_trials=0, qualification_trials=0)
    write(OUT / 'engine_build_result.json', result)
    try:
        for cmd, name in zip(commands, ['llm_build.log', 'visual_build.log']): assets.command(cmd, OUT / name)
        assets.verify_stock(); assets.verify(exported['output_hashes'])
        result.update(state='built', hashes=assets.hashes(OUT / 'engines'),
                      builder_config=read(OUT / 'engines/llm/config.json')['builder_config'])
    except BaseException as exc:
        result.update(state='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        result['completed_at'] = now(); write(OUT / 'engine_build_result.json', result)

def services():
    assert read(OUT / 'services_handoff.json')['state'] == 'complete'
    assert read(OUT / 'engine_build_result.json')['state'] == 'built'
    assert read(OUT / 'native_build_result.json')['state'] == 'complete'
    spec = importlib.util.spec_from_file_location('priority_service_launcher', ORIGINAL_SERVICE_SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.RUN = OUT
    env = assets.environment(); env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'); env.pop('PYTHONPATH', None)
    pinned = read(OUT / 'serving_dependency_pin.json')
    assert pinned['package'] == 'transformers' and pinned['version'] == '5.14.1'
    assets.verify(pinned['hashes'])
    env['PYTHONPATH'] = pinned['path']
    base = ['--host', '127.0.0.1', '--max-num-seqs', '1', '--generation-config', 'vllm',
            '--no-enable-prefix-caching', '--enable-per-request-metrics']
    vllm = (_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/vllm')
    cmd = [vllm, 'serve', str(CHECKPOINT), '--served-model-name', MODEL, '--dtype', 'half', '--port', '8012',
           '--max-model-len', '16384', '--gpu-memory-utilization', '0.60', '--kv-cache-memory-bytes', '2147483648',
           '--kv-cache-dtype', 'auto', '--limit-mm-per-prompt', '{"image":9,"video":0,"audio":0}',
           '--reasoning-parser', 'gemma4', *base]
    component = read(EXP / 'configs/full_agent_components.json')['components']['planner']
    planner = module.start('planner', cmd, env, dict(model=MODEL, revision=REVISION, runtime='vLLM',
        serving_dependency_pin=pinned,
        dtype='half', precision=component['precision'], kv_cache_dtype='auto', effective_kv_cache_dtype='float16',
        kv_cache='FP16', quantization=None, max_model_len=16384, max_num_seqs=1, max_images_per_prompt=9,
        enable_prefix_caching=False, enable_per_request_metrics=True, checkpoint_path=str(CHECKPOINT),
        source_precision='BF16', required_log_markers=['dtype=torch.float16', 'quantization=None',
        'max_seq_len=16384', 'enable_prefix_caching=False']))
    env.pop('PYTHONPATH', None)
    grounder_id = 'ByteDance-Seed/UI-TARS-1.5-7B'
    cmd = [vllm, 'serve', str(assets.GROUNDER), '--served-model-name', grounder_id, '--dtype', 'bfloat16',
           '--port', '8013', '--max-model-len', '8192', '--gpu-memory-utilization', '0.20',
           '--kv-cache-memory-bytes', '1073741824', '--kv-cache-dtype', 'auto', '--limit-mm-per-prompt', '{"image":1}', *base]
    grounder = module.start('grounder', cmd, env, dict(model=grounder_id, revision=assets.GROUNDER.name,
        runtime='vLLM', precision='BF16', dtype='bfloat16', kv_cache_dtype='auto', effective_kv_cache_dtype='bfloat16',
        checkpoint_path=str(assets.GROUNDER), max_model_len=8192, max_num_seqs=1, max_images_per_prompt=1,
        enable_prefix_caching=False, enable_per_request_metrics=True,
        required_log_markers=['dtype=torch.bfloat16', 'quantization=None', 'max_seq_len=8192', 'enable_prefix_caching=False']))
    write(OUT / 'services/launch.json', dict(state='ready', created_at=now(), qualification_trials=0,
          inference_readiness_requests=0, components=dict(planner=planner, reflection=planner, text_grounder=planner, grounder=grounder)))

def main(refresh_queued_setup=False):
    validate_scope()
    assets.OUT, assets.OLD, assets.MODEL, assets.REVISION, assets.CHECKPOINT = OUT, PREDECESSOR, MODEL, REVISION, CHECKPOINT
    common.OUT, common.EXP = OUT, EXP
    with (OUT / 'priority.lock').open('a+') as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start_record = OUT / 'priority_started.json'
        if refresh_queued_setup:
            # This repairs only a waiting setup queue, never a started evaluation.
            prior = read(OUT / 'priority_status.json')
            state = read(EXP / 'status.json')
            assert prior['state'] == 'queued_waiting_gemma'
            assert not Path(f"/proc/{prior['pid']}").exists()
            assert state['state'] == 'prepared' and state['completed_trials'] == 0
            assert not (EXP / 'execution_manifest.json').exists()
            assert not (OUT / 'checkpoint_download.json').exists()
            assert not list((EXP / 'records').glob('*.json'))
            start_record = OUT / 'queue_setup_repair/started.json'
        assert not start_record.exists(), 'No automatic restart'
        write(start_record, dict(at=now(), model=MODEL, revision=REVISION,
              plan_sha256=digest(EXP / 'plan.json'), source_sha256=digest(__file__), user_authorized_order=['Gemma26', 'Nemotron Bolt']))
        try:
            while not predecessor_ready():
                prior = read(PREDECESSOR / 'priority_status.json')
                publish('queued_waiting_gemma', predecessor_state=prior['state'], predecessor_phase=prior.get('phase'))
                time.sleep(15)
            validate_scope()
            with (ROOT / 'runs/legacy_d1_gpu.lock').open('a+') as gpu:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                assert predecessor_ready()
                for name, operation in [('download', download), ('stop_gemma_services', assets.stop_old_services),
                                        ('export', export), ('build_engines', build)]:
                    publish('preparing_model', phase=name); operation()
                for name, argv in [
                    ('verify_native', [TOKEN_PY, '-B', str(ROOT / 'studies/prepare_ledger_native.py'), '--root', str(OUT), '--verify-cpu', '--checkpoint', str(CHECKPOINT)]),
                    ('build_native', [TOKEN_PY, '-B', str(ROOT / 'studies/prepare_ledger_native.py'), '--root', str(OUT), '--build', '--lock-fd', str(gpu.fileno())]),
                    ('verify_agents3', [AGENT_PY, '-B', str(ROOT / 'studies/verify_full_agents3.py'), '--config', str(EXP / 'configs/full_agent_components.json'), '--out', str(OUT / 'full_agents3_cpu_verification.json')])]:
                    publish('preparing_model', phase=name); common.run_command(name, argv, gpu)
                publish('preparing_model', phase='start_services'); services()
                publish('preparing_model', phase='freeze')
                common.run_command('freeze', [AGENT_PY, '-B', str(ROOT / 'studies/run_core5_detector_full_agents3.py'), '--out', str(EXP), '--freeze', '--services', str(OUT / 'services/launch.json')], gpu)
            publish('running_trials')
            common.run_command('evaluation', [AGENT_PY, '-B', str(EXP / 'code/run_core5_detector_full_agents3.py'), '--out', str(EXP), '--run'])
            status = read(EXP / 'status.json')
            assert status['state'] == 'complete' and status['completed_trials'] == status['planned_trials'] == 3000
            publish('auditing_completed_study')
            log = common.run_command('completion_audit', [AGENT_PY, '-B', str(EXP / 'code/run_core5_detector_full_agents3.py'), '--out', str(EXP), '--audit-completed'])
            proof = json.loads([line for line in log.read_text().splitlines() if line.strip()][-1])
            assert proof['state'] == 'passed' and proof['completed_trials'] == 3000
            write(OUT / 'completion_audit.json', proof); publish('complete', completed_trials=3000)
        except BaseException as exc:
            publish('failed', error=f'{type(exc).__name__}: {exc}'); raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-queued-setup', action='store_true')
    args = parser.parse_args()
    main(args.refresh_queued_setup)
