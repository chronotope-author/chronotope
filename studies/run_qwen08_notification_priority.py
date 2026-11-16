"""Fresh Qwen0.8B Notification evaluation with the accepted full configurations.

Reuses the unchanged legacy application and stock Edge-LLM 0.11.1. Only model
assets and experiment configuration change. No model screening or successor jobs.
"""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/notification_qwen08_edgellm0111_20261031'
EXP = OUT / 'experiment'
PREVIOUS = ROOT / 'runs/notification_qwen27b_edgellm0111_20261030'
HISTORICAL = ROOT / 'runs/legacy_detector_full_agents3_20261029'
NEMO = ROOT / 'runs/notification_nemotron_bolt_edgellm0111_20261030'
BACKEND = ROOT / 'runs/edgellm_0p11p1_20261030'
CHECKPOINT = ROOT / 'runs/model_cache/qwen35_0p8b_legacy_source'
MODEL = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
WEIGHT_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
COMMIT = '358de11c7c5f30fdaac75f0726a3b2df7e842778'
AGENT_PY = (_CHRONOTOPE_HOME + '/agents3-venv/bin/python')
TOKEN_PY = (_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/python')
sys.path.insert(0, str(ROOT / 'studies'))
import gemma_ledger_assets as assets
from legacy_d1_w8 import digest, now, write
from core5_ledger_plan import validate_plan, load_cases


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    assets.OUT, assets.OLD, assets.STOCK = OUT, PREVIOUS, BACKEND
    assets.SOURCE, assets.BUILD = BACKEND / 'source/TensorRT-Edge-LLM', BACKEND / 'build'
    assets.MODEL, assets.REVISION, assets.CHECKPOINT, assets.COMMIT = MODEL, REVISION, CHECKPOINT, COMMIT


def guard():
    plan = read(EXP / 'plan.json')
    assert validate_plan(plan) == 3000 and plan['qualification_trials'] == 0
    assert {j['task'] for j in plan['jobs']} == {'notification'}
    assert {j['window_s'] for j in plan['jobs']} == {1., 2., 4., 8., 16.}
    assert {j['arm'] for j in plan['jobs']} == {'legacy_cpp_detector', 'agents3_full'}
    assert {j['mode'] for j in plan['jobs']} == {'live', 'replay', 'paused'}
    assert (plan['model'], plan['revision']) == (MODEL, REVISION)
    assert plan['case_file_sha256'] == read(PREVIOUS / 'experiment/plan.json')['case_file_sha256']
    load_cases(EXP, plan)
    for root, state, count in [(PREVIOUS, 'stopped', 297), (NEMO, 'paused', 308)]:
        old = read(root / 'experiment/status.json')
        assert old['state'] == state and old['pid'] is None and old['completed_trials'] == count


def publish(state, **details):
    write(OUT / 'priority_status.json', dict(state=state, updated_at=now(),
        pid=os.getpid() if state not in ('complete', 'failed', 'prepared') else None,
        model=MODEL, revision=REVISION, backend='TensorRT-Edge-LLM 0.11.1',
        planned_trials=3000, qualification_trials=0, automatic_resume=False, **details))


def verify_checkpoint():
    url = f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    assert metadata['sha'] == REVISION
    hashes, upstream = {}, {}
    for item in metadata['siblings']:
        path = CHECKPOINT / item['rfilename']
        assert path.is_file(), path
        actual = digest(path)
        if item.get('lfs'):
            assert actual == item['lfs']['sha256'], path
        else:
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert blob == item['blobId'], path
        hashes[str(path)] = actual
        upstream[item['rfilename']] = item
    assert digest(CHECKPOINT / 'model.safetensors-00001-of-00001.safetensors') == WEIGHT_SHA
    assert set(hashes) == set(assets.hashes(CHECKPOINT)), 'Unexpected unpinned checkpoint files'
    config = read(CHECKPOINT / 'config.json')
    assert config['model_type'] == 'qwen3_5' and 'quantization_config' not in config
    write(OUT / 'checkpoint_download.json', dict(state='verified_existing_cache', checked_at=now(),
        model=MODEL, revision=REVISION, checkpoint=str(CHECKPOINT), hashes=hashes,
        upstream_file_metadata=upstream, source_weight_sha256=WEIGHT_SHA,
        source_precision='BF16', runtime_conversion='FP16 LLM/vision in both arms; FP16 KV',
        quantization=None, model_inference_calls=0, qualification_trials=0))


def prepare():
    from run_core5_detector_full_agents3 import prepare as prepare_cohort, execution_jobs
    import prepare_ledger_native as native
    if OUT.exists():
        # A CPU-only preparation failure may be repaired before assets are frozen.
        assert not (OUT / 'authorization.json').exists()
        assert not (OUT / 'started.json').exists()
        assert not (OUT / 'native_source_preparation.json').exists()
        assert not (EXP / 'execution_manifest.json').exists()
        assert not list((EXP / 'records').glob('*.json'))
        assert read(EXP / 'status.json')['state'] == 'prepared'
    else:
        OUT.mkdir()
    write(OUT / 'backend.json', dict(tag='v0.11.1', commit=COMMIT,
        source=str(assets.SOURCE), build=str(assets.BUILD),
        plugin=str(assets.BUILD / 'libNvInfer_edgellm_plugin.so'),
        tokenizer_verification=str(OUT / 'tokenizer_verification.json')))
    verify_checkpoint()
    prepare_cohort(EXP, 'qwen35_0p8b', True, tasks=['notification'],
        notification_windows=(16., 8., 4., 2., 1.), engine_directory=OUT / 'engines')
    support = dict(configuration_reviewed=True, code_action='defer_unsupported',
        user_authorized=False, reason='Notification comparison; typing not required')
    (OUT / 'inputs').mkdir(exist_ok=True)
    write(OUT / 'inputs/action_support.json', support)
    ready, deferred = execution_jobs(read(EXP / 'plan.json'), support)
    assert len(ready) == 3000 and not deferred
    shutil.copy2(HISTORICAL / 'grounder_checkpoint_verification.json', OUT / 'grounder_checkpoint_verification.json')
    old_source = read(PREVIOUS / 'native_source_preparation.json')
    old_build = read(PREVIOUS / 'native_build_result.json')
    assert old_build['state'] == 'complete' and old_build['upstream_commit'] == COMMIT
    assets.verify(old_source['source_hashes'])
    assets.verify(old_build['source_hashes']); assets.verify(old_build['binary_hashes'])
    source = OUT / 'build_source/chronotope'
    shutil.copytree(PREVIOUS / 'build_source/chronotope', source)
    old_source.update(created_at=now(), source_hashes=native.source_hashes(source),
        copied_unchanged_from=str(PREVIOUS / 'build_source/chronotope'))
    write(OUT / 'native_source_preparation.json', old_source)
    (OUT / 'native_build').mkdir()
    for name in ['chronotope', 'libchronotope.so']:
        shutil.copy2(PREVIOUS / 'native_build' / name, OUT / 'native_build' / name)
    old_build.update(source_hashes=native.source_hashes(source),
        binary_hashes={str(OUT / 'native_build' / name): digest(OUT / 'native_build' / name)
                      for name in ['chronotope', 'libchronotope.so']},
        reused_application_build=str(PREVIOUS / 'native_build_result.json'),
        reused_application_build_sha256=digest(PREVIOUS / 'native_build_result.json'), completed_at=now())
    write(OUT / 'native_build_result.json', old_build)
    write(OUT / 'authorization.json', dict(at=now(), user_request='Can we just do the Qwen/Qwen3.5-0.8B for now?',
        previous_runs_held=True, fresh_evaluation_trials=3000, qualification_trials=0,
        tasks=['notification'], windows_s=[16, 8, 4, 2, 1], controller_policy_changes=False,
        successor_jobs=[], plan_sha256=digest(EXP / 'plan.json')))
    guard(); publish('prepared', phase='cpu_contract_verification_pending')


def prepare_prompts():
    import prepare_ledger_native as native
    native.verify_cpu(OUT, CHECKPOINT)
    prompt = read(OUT / 'inputs/task_prompts.json')['notification']
    original = read(HISTORICAL / 'experiment/configs/legacy_compact_prompt.json')
    for key in ['planner_system', 'initial_user_text', 'generation_settings', 'history', 'reflection']:
        assert prompt[key] == original[key], key
    contract = OUT / 'inputs/notification_tokenizer_contract.json'
    write(contract, prompt)
    assets.command([BACKEND / 'tokenizer_diagnostic', CHECKPOINT, contract,
        OUT / 'tokenizer_diagnostic.json'], OUT / 'tokenizer_diagnostic.log')
    observed = read(OUT / 'tokenizer_diagnostic.json')
    assert observed['stock_ids'] == observed['reference_ids']
    write(OUT / 'tokenizer_verification.json', dict(state='passed', checked_at=now(),
        upstream_commit=COMMIT, upstream_tag='v0.11.1', all_token_ids_match=True,
        stock_tokens=len(observed['stock_ids']), reference_tokens=len(observed['reference_ids']),
        model_inference_calls=0, experimental_trials=0, qualification_trials=0))
    write(OUT / 'historical_prompt_equivalence.json', dict(state='passed', checked_at=now(),
        system_and_user_text_identical=True, generation_settings_identical=True,
        application_binary_identical=True, image_tokens=prompt['image_tokens'],
        input_tokens=len(observed['stock_ids']), model_inference_calls=0))
    assets.command([AGENT_PY, '-B', ROOT / 'studies/verify_full_agents3.py', '--config',
        EXP / 'configs/full_agent_components.json', '--out', OUT / 'full_agents3_cpu_verification.json'],
        OUT / 'full_agents3_cpu_verification.log')
    guard(); publish('prepared', phase='ready_for_export_and_build')


def export_and_build():
    import onnx
    assets.verify_stock()
    cp = read(OUT / 'checkpoint_download.json'); assets.verify(cp['hashes'])
    cmd = [assets.PYTHON, '-B', '-m', 'tensorrt_edgellm.scripts.export', str(CHECKPOINT),
        str(OUT / 'exports'), '--dtype', 'float16']
    env = assets.environment()
    env.update(PYTHONPATH=str(assets.SOURCE), LLM_SDK_DIR=str(assets.SOURCE),
        CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = dict(state='exporting', started_at=now(), model=MODEL, revision=REVISION,
        command=cmd, upstream_commit=COMMIT, source_patches=[], checkpoint_hashes=cp['hashes'],
        inference_calls=0, qualification_trials=0)
    write(OUT / 'export_result.json', result)
    assets.command(cmd, OUT / 'export.log', env)
    result.update(state='exported', completed_at=now(), output_hashes=assets.hashes(OUT / 'exports'))
    write(OUT / 'export_result.json', result)
    precision = dict(state='passed', model=MODEL, revision=REVISION, source_patches=[])
    for part in ['llm', 'visual']:
        graph = onnx.load(OUT / f'exports/{part}/model.onnx', load_external_data=False).graph
        types = Counter()
        for tensor in graph.initializer:
            types[onnx.TensorProto.DataType.Name(tensor.data_type)] += math.prod(tensor.dims)
        assert types.get('FLOAT16', 0) > 0 and not types.get('BFLOAT16', 0)
        assert not any('FLOAT4' in name or 'FLOAT8' in name for name in types)
        assert not any(n.op_type in ('QuantizeLinear', 'DequantizeLinear') for n in graph.node)
        precision[part] = dict(initializer_elements_by_dtype=dict(types))
    assert read(OUT / 'exports/llm/config.json')['kv_cache_dtype'] == 'fp16'
    write(OUT / 'onnx_precision_audit.json', precision)
    commands = [
        [assets.BUILD / 'examples/llm/llm_build', '--onnxDir', OUT / 'exports/llm',
         '--engineDir', OUT / 'engines/llm', '--maxInputLen', '16384',
         '--maxKVCacheCapacity', '16384', '--maxBatchSize', '1'],
        [assets.BUILD / 'examples/multimodal/visual_build', '--onnxDir', OUT / 'exports/visual',
         '--engineDir', OUT / 'engines', '--minImageTokens', '64',
         '--maxImageTokens', '1024', '--maxImageTokensPerImage', '1024']]
    proof = dict(state='building', started_at=now(), model=MODEL, revision=REVISION,
        upstream_commit=COMMIT, source_patches=[], commands=[[str(x) for x in c] for c in commands],
        source_checkpoint_hashes=cp['hashes'], source_export_hashes=result['output_hashes'],
        kv_cache=dict(dtype='FP16', override=None),
        runtime_precision=read(EXP / 'configs/model.json')['runtime_precision'],
        inference_calls=0, evaluation_trials=0, qualification_trials=0)
    write(OUT / 'engine_build_result.json', proof); publish('preparing', phase='build_engines')
    for cmd, name in zip(commands, ['llm_build.log', 'visual_build.log']):
        assets.command(cmd, OUT / name)
    assets.verify_stock(); assets.verify(cp['hashes'])
    proof.update(state='built', completed_at=now(), hashes=assets.hashes(OUT / 'engines'),
        builder_config=read(OUT / 'engines/llm/config.json')['builder_config'])
    assert read(OUT / 'engines/llm/config.json')['kv_cache_dtype'] == 'fp16'
    write(OUT / 'engine_build_result.json', proof)


def services():
    spec = importlib.util.spec_from_file_location('historical_service_launcher', HISTORICAL / 'launch_services.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.RUN = OUT
    old = read(HISTORICAL / 'services/launch.json')['components']
    fields = ['model', 'revision', 'runtime', 'precision', 'dtype', 'kv_cache_dtype',
        'effective_kv_cache_dtype', 'quantization', 'max_model_len', 'max_num_seqs',
        'max_images_per_prompt', 'enable_prefix_caching', 'enable_per_request_metrics',
        'source_precision', 'checkpoint_path', 'config_sha256', 'required_log_markers']
    env = assets.environment(); env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'); env.pop('PYTHONPATH', None)
    components = {}
    for role in ['planner', 'grounder']:
        command = list(old[role]['command'])
        config = {key: old[role][key] for key in fields if key in old[role]}
        if role == 'planner':
            command[command.index('serve') + 1] = str(CHECKPOINT)
            command[command.index('--served-model-name') + 1] = MODEL
            model = read(EXP / 'configs/model.json')
            config.update(model=MODEL, revision=REVISION, quantization=None,
                precision=model['runtime_precision'], source_precision=model['source_precision'],
                checkpoint_path=str(CHECKPOINT), config_sha256=digest(CHECKPOINT / 'config.json'),
                required_log_markers=['dtype=torch.float16', 'quantization=None',
                    'max_seq_len=16384', 'enable_prefix_caching=False'])
        components[role] = module.start(role, command, env, config)
    components.update(reflection=components['planner'], text_grounder=components['planner'])
    write(OUT / 'services/launch.json', dict(state='ready', created_at=now(), components=components,
        qualification_trials=0, inference_readiness_requests=0))


def run():
    guard()
    with (OUT / 'priority.lock').open('a+') as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / 'started.json').exists(), 'No automatic restart'
        write(OUT / 'started.json', dict(at=now(), pid=os.getpid(), driver_sha256=digest(__file__),
            authorization=str(OUT / 'authorization.json'), plan_sha256=digest(EXP / 'plan.json')))
        try:
            with (ROOT / 'runs/legacy_d1_gpu.lock').open('a+') as gpu:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                guard(); publish('preparing', phase='stop_idle_qwen27b_services'); assets.stop_old_services()
                publish('preparing', phase='export'); export_and_build()
                publish('preparing', phase='start_model_services'); services()
                publish('preparing', phase='freeze')
                assets.command([AGENT_PY, '-B', ROOT / 'studies/run_core5_detector_full_agents3.py',
                    '--out', EXP, '--freeze', '--services', OUT / 'services/launch.json'], OUT / 'freeze.log')
            guard(); publish('running_trials', phase='evaluation')
            assets.command([AGENT_PY, '-B', EXP / 'code/run_core5_detector_full_agents3.py',
                '--out', EXP, '--run'], OUT / 'evaluation.log')
            publish('auditing_completed_study')
            assets.command([AGENT_PY, '-B', EXP / 'code/run_core5_detector_full_agents3.py',
                '--out', EXP, '--audit-completed'], OUT / 'completion_audit.log')
            proof = json.loads([line for line in (OUT / 'completion_audit.log').read_text().splitlines() if line.strip()][-1])
            assert proof['state'] == 'passed' and proof['completed_trials'] == 3000
            write(OUT / 'completion_audit.json', proof); publish('complete', completed_trials=3000)
        except BaseException as exc:
            publish('failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'prepare-prompts', 'run'])
    args = parser.parse_args(); configure()
    {'prepare': prepare, 'prepare-prompts': prepare_prompts, 'run': run}[args.action]()
