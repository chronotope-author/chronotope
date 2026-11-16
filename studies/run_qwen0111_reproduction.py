"""Fresh historical Qwen Notification reproduction using stock Edge-LLM 0.11.1."""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/notification_qwen27b_edgellm0111_20261030'
EXP = OUT / 'experiment'
OLD = ROOT / 'runs/legacy_detector_full_agents3_20261029'
NEMO = ROOT / 'runs/notification_nemotron_bolt_edgellm0111_20261030'
BACKEND = ROOT / 'runs/edgellm_0p11p1_20261030'
CHECKPOINT = ROOT / 'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f'
MODEL = 'nvidia/Qwen3.8-27B-NVFP4'
REVISION = '482ca0f3832238542f8f5295dde86b5f22711d80'
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
    assets.OUT, assets.OLD, assets.STOCK = OUT, NEMO, BACKEND
    assets.SOURCE, assets.BUILD = BACKEND / 'source/TensorRT-Edge-LLM', BACKEND / 'build'
    assets.MODEL, assets.REVISION, assets.CHECKPOINT = MODEL, REVISION, CHECKPOINT
    assets.COMMIT = COMMIT

def guard():
    plan = read(EXP / 'plan.json')
    assert validate_plan(plan) == 3000 and plan['qualification_trials'] == 0
    assert {j['task'] for j in plan['jobs']} == {'notification'}
    assert {j['window_s'] for j in plan['jobs']} == {1., 2., 4., 8., 16.}
    assert {j['arm'] for j in plan['jobs']} == {'legacy_cpp_detector', 'agents3_full'}
    assert {j['mode'] for j in plan['jobs']} == {'live', 'replay', 'paused'}
    assert (plan['model'], plan['revision']) == (MODEL, REVISION)
    load_cases(EXP, plan)
    held = read(NEMO / 'experiment/status.json')
    assert held['state'] == 'paused' and held['pid'] is None
    assert held['completed_trials'] == 308

def publish(state, **details):
    write(OUT / 'priority_status.json', dict(state=state, updated_at=now(),
          pid=os.getpid() if state not in ('complete', 'failed', 'prepared') else None,
          model=MODEL, revision=REVISION, backend='TensorRT-Edge-LLM 0.11.1',
          planned_trials=3000, qualification_trials=0, automatic_resume=False, **details))

def prepare():
    from run_core5_detector_full_agents3 import prepare as prepare_cohort, execution_jobs
    assert not OUT.exists(), 'Never overwrite an existing reproduction'
    OUT.mkdir()
    write(OUT / 'backend.json', dict(tag='v0.11.1', commit=COMMIT,
          source=str(assets.SOURCE), build=str(assets.BUILD),
          plugin=str(assets.BUILD / 'libNvInfer_edgellm_plugin.so'),
          tokenizer_verification=str(OUT / 'tokenizer_verification.json')))
    prepare_cohort(EXP, 'nvidia_27b', True, tasks=['notification'],
                   notification_windows=(16., 8., 4., 2., 1.), engine_directory=OUT / 'engines')
    (OUT / 'inputs').mkdir()
    support = dict(configuration_reviewed=True, code_action='defer_unsupported',
                   user_authorized=False, reason='Notification reproduction; no typing required')
    write(OUT / 'inputs/action_support.json', support)
    ready, deferred = execution_jobs(read(EXP / 'plan.json'), support)
    assert len(ready) == 3000 and not deferred
    checkpoint = read(ROOT / 'runs/agents3_policy_nvidia27b_20261029/checkpoint_download.json')
    assert checkpoint['model'] == MODEL and checkpoint['revision'] == REVISION
    write(OUT / 'checkpoint_download.json', checkpoint)
    shutil.copy2(OLD / 'grounder_checkpoint_verification.json', OUT / 'grounder_checkpoint_verification.json')
    source = OUT / 'build_source/chronotope'
    shutil.copytree(NEMO / 'build_source/chronotope', source)
    import prepare_ledger_native as native
    proof = read(NEMO / 'native_source_preparation.json')
    assets.verify(proof['source_hashes'])
    proof.update(created_at=now(), source_hashes=native.source_hashes(source),
                 copied_unchanged_from=str(NEMO / 'build_source/chronotope'))
    write(OUT / 'native_source_preparation.json', proof)
    build = read(NEMO / 'native_build_result.json')
    assert build['state'] == 'complete' and build['upstream_commit'] == COMMIT
    assets.verify(build['binary_hashes']); assets.verify(build['source_hashes'])
    (OUT / 'native_build').mkdir()
    for name in ['chronotope', 'libchronotope.so']:
        shutil.copy2(NEMO / 'native_build' / name, OUT / 'native_build' / name)
    build.update(source_hashes=native.source_hashes(source),
                 binary_hashes={str(OUT / 'native_build' / name): digest(OUT / 'native_build' / name)
                                for name in ['chronotope', 'libchronotope.so']},
                 reused_application_build=str(NEMO / 'native_build_result.json'), completed_at=now())
    write(OUT / 'native_build_result.json', build)
    write(OUT / 'authorization.json', dict(at=now(), user_request=
          'With this 0.11.1, see if we could reproduce the old results with nvidia qwen 3.8-27b nvfp4',
          reference=str(OLD / 'experiment'), nemotron_remains_paused=True,
          plan_sha256=digest(EXP / 'plan.json'), fresh_evaluation_trials=3000,
          qualifications=0, inherited_policy_changes=False))
    guard(); publish('prepared', phase='cpu_prompt_verification_pending')

def prepare_prompts():
    import prepare_ledger_native as native
    native.verify_cpu(OUT, CHECKPOINT)
    prompts = read(OUT / 'inputs/task_prompts.json')
    prompt = prompts['notification']
    reference = read(OLD / 'experiment/execution_manifest.json')['native_prompt_reference']
    assert prompt['formatted_text'] == reference['formatted_text'], 'Historical Qwen formatted prompt changed'
    old = read(OLD / 'experiment/configs/legacy_compact_prompt.json')
    for key in ['planner_system', 'initial_user_text', 'generation_settings', 'history', 'reflection']:
        assert prompt[key] == old[key], key
    path = OUT / 'inputs/notification_tokenizer_contract.json'; write(path, prompt)
    assets.command([BACKEND / 'tokenizer_diagnostic', CHECKPOINT, path,
                    OUT / 'tokenizer_diagnostic.json'], OUT / 'tokenizer_diagnostic.log')
    observed = read(OUT / 'tokenizer_diagnostic.json')
    assert observed['stock_ids'] == observed['reference_ids']
    write(OUT / 'tokenizer_verification.json', dict(state='passed', checked_at=now(),
          upstream_commit=COMMIT, upstream_tag='v0.11.1', all_token_ids_match=True,
          stock_tokens=len(observed['stock_ids']), reference_tokens=len(observed['reference_ids']),
          model_inference_calls=0, experimental_trials=0, qualification_trials=0))
    write(OUT / 'historical_prompt_equivalence.json', dict(state='passed', checked_at=now(),
          old_formatted_prompt_identical=True, system_and_user_text_identical=True,
          generation_settings_identical=True, image_tokens=prompt['image_tokens'],
          input_tokens=len(observed['stock_ids']), model_inference_calls=0))

def export_and_build():
    assets.verify_stock(); cp = read(OUT / 'checkpoint_download.json'); assets.verify(cp['hashes'])
    cmd = [assets.PYTHON, '-B', '-m', 'tensorrt_edgellm.scripts.export', str(CHECKPOINT),
           str(OUT / 'exports'), '--dtype', 'float16']
    env = assets.environment(); env.update(PYTHONPATH=str(assets.SOURCE),
          LLM_SDK_DIR=str(assets.SOURCE), CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = dict(state='exporting', started_at=now(), model=MODEL, revision=REVISION,
          command=cmd, upstream_commit=COMMIT, source_patches=[], checkpoint_hashes=cp['hashes'],
          inference_calls=0, qualification_trials=0)
    write(OUT / 'export_result.json', result)
    assets.command(cmd, OUT / 'export.log', env)
    result.update(state='exported', completed_at=now(), output_hashes=assets.hashes(OUT / 'exports'))
    write(OUT / 'export_result.json', result)
    assets.audit_export()
    precision = read(OUT / 'onnx_precision_audit.json')['llm']['initializer_elements_by_dtype']
    assert any('FLOAT8' in k and v > 0 for k, v in precision.items()), 'FP8 attention projections missing'
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
    for cmd, name in zip(commands, ['llm_build.log', 'visual_build.log']): assets.command(cmd, OUT / name)
    assets.verify_stock(); assets.verify(cp['hashes'])
    proof.update(state='built', completed_at=now(), hashes=assets.hashes(OUT / 'engines'),
                 builder_config=read(OUT / 'engines/llm/config.json')['builder_config'])
    assert read(OUT / 'engines/llm/config.json')['kv_cache_dtype'] == 'fp16'
    write(OUT / 'engine_build_result.json', proof)

def services():
    spec = importlib.util.spec_from_file_location('qwen_original_services', OLD / 'launch_services.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.RUN = OUT
    previous = read(OLD / 'services/launch.json')['components']
    env = assets.environment(); env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'); env.pop('PYTHONPATH', None)
    fields = ['model', 'revision', 'runtime', 'precision', 'dtype', 'kv_cache_dtype',
              'effective_kv_cache_dtype', 'quantization', 'max_model_len', 'max_num_seqs',
              'max_images_per_prompt', 'enable_prefix_caching', 'enable_per_request_metrics',
              'source_precision', 'checkpoint_path', 'config_sha256', 'required_log_markers']
    components = {}
    for role in ['planner', 'grounder']:
        old = previous[role]; cfg = {k: old[k] for k in fields if k in old}
        components[role] = module.start(role, old['command'], env, cfg)
        assert components[role]['command'] == old['command']
    components.update(reflection=components['planner'], text_grounder=components['planner'])
    write(OUT / 'services/launch.json', dict(state='ready', created_at=now(), components=components,
          qualification_trials=0, inference_readiness_requests=0, original_service_commands_identical=True))

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
                guard(); publish('preparing', phase='stop_idle_nemotron_services'); assets.stop_old_services()
                publish('preparing', phase='export'); export_and_build()
                publish('preparing', phase='start_original_qwen_services'); services()
                publish('preparing', phase='freeze')
                assets.command([AGENT_PY, '-B', ROOT / 'studies/run_core5_detector_full_agents3.py',
                    '--out', EXP, '--freeze', '--services', OUT / 'services/launch.json'], OUT / 'freeze.log')
            guard(); publish('running_trials', phase='evaluation')
            assets.command([AGENT_PY, '-B', EXP / 'code/run_core5_detector_full_agents3.py',
                            '--out', EXP, '--run'], OUT / 'evaluation.log')
            publish('auditing_completed_study')
            assets.command([AGENT_PY, '-B', EXP / 'code/run_core5_detector_full_agents3.py',
                            '--out', EXP, '--audit-completed'], OUT / 'completion_audit.log')
            proof = json.loads([x for x in (OUT / 'completion_audit.log').read_text().splitlines() if x.strip()][-1])
            assert proof['state'] == 'passed' and proof['completed_trials'] == 3000
            write(OUT / 'completion_audit.json', proof); publish('complete', completed_trials=3000)
        except BaseException as exc:
            publish('failed', error=f'{type(exc).__name__}: {exc}'); raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'prepare-prompts', 'run'])
    args = parser.parse_args(); configure()
    {'prepare': prepare, 'prepare-prompts': prepare_prompts, 'run': run}[args.action]()
