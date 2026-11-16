"""Fresh five-arm extension of the accepted Qwen0.8 comparison.

Preparation and contract checks perform no inference. No service management,
qualification, automatic resume, or successor jobs are provided.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
from collections import defaultdict
import copy
import csv
from dataclasses import replace
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

from run_latest_frame_qwen08 import (ROOT, MODEL, REVISION, TASKS, SETTINGS,
    read, write, digest, now, verify, metrics, empty_metrics, wilson)

REFERENCE = ROOT / 'runs/latest_frame_qwen08_20261108/experiment'
ARMS = ('legacy_cpp_detector', 'agents3_full', 'aoi', 'mobiledreamer', 'aapt')
LABELS = {'legacy_cpp_detector': 'Legacy C++ + detector', 'agents3_full': 'Agent S3 (Keep)',
    'aoi': 'AOI visual (buffered processing)',
    'mobiledreamer': 'MobileDreamer released one-step selection (adapted)',
    'aapt': 'AAPT (paper-derived; historical latency proxy)'}
TOTAL = len(TASKS) * len(SETTINGS) * len(ARMS) * 100
BUFFER_HOOK = '''            # Opt-in research observers retain inter-step frames before coalescing.
            # Existing controllers have no hook and retain their original behavior.
            buffer_observation = getattr(controller, "buffer_observation", None)
            if buffer_observation is not None and buffer_observation(obs):
                tw.append("observation_buffered", observation_id=obs.observation_id,
                          harness_time_ns=harness.now_ns(), observer=type(controller).__name__)
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def setup(out):
    code = out / 'code'
    sys.path[:0] = [str(code), str(code / 'src'), str(code / 'chronotope/python')]
    from chronotope.paper_baselines.factory import register
    from chronotope_core.glue import register as native_register
    register(); native_register()


def make_jobs(cells, cases):
    jobs = []
    for index in range(100):
        for k, cell in enumerate(cells):
            case = cases[cell['manifest']][index]
            offset = (index + k) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                jobs.append(dict(task=cell['task'], setting_s=cell['setting_s'],
                    cell_id=cell['cell_id'], task_deadline_s=cell['task_deadline_s'],
                    manifest=cell['manifest'], index=index, seed=case['seed'],
                    episode_id=case['episode_id'], arm=arm, mode='live', phase='evaluation',
                    id=f"{arm}_{cell['cell_id']}_{index:03d}_live"))
    return jobs


def latency_proxy(reference):
    samples = []; sources = {}
    for path in sorted((reference / 'calls').glob('agents3_full_*.jsonl')):
        used = False
        for line in path.read_text().splitlines():
            call = json.loads(line)
            if call.get('role') == 'planner' and call.get('http_status') == 200:
                elapsed = call['end_ns'] - call['start_ns']
                require(elapsed > 0, 'Invalid historical call duration')
                samples.append(elapsed / 1e6); used = True
        if used:
            sources[str(path)] = digest(path)
    require(bool(samples), 'No historical planner latency evidence')
    p95 = sorted(samples)[math.ceil(.95 * len(samples)) - 1]
    return dict(kind='historical Agent S3 planner latency proxy; NOT measured AAPT p95',
        source_files=sources, samples=len(samples), estimator='nearest-rank p95, rounded up to ms',
        historical_p95_ms=p95, planner_budget_ms=math.ceil(p95),
        qualification_trials=0, additional_inference_calls=0,
        limitation='Worker output length differs from a policy tree. This is an initial proxy, '
                    'not validation of the AAPT coverage rule. Actual tree latency is logged in evaluation.')


def with_response_window(case, seconds):
    """Change only the authorized deadline/horizon; retain stimulus and geometry."""
    from chronotope.distributional.manifest import (EpisodeRow, build_manifest,
        expected_event_hash, expected_observation_hash, proxy_scenario, verify_row_against_manifest)
    result = copy.deepcopy(case)
    scenario = result['base_spec']['scenario']
    horizon = scenario['event_time_ns'] + seconds * 10**9 + 500_000_000
    if result.get('distributional_row'):
        old = EpisodeRow.from_dict(result['distributional_row'])
        row = replace(old, deadline_s=float(seconds),
                      task_payload=dict(old.task_payload, episode_horizon_ns=horizon))
        manifest = build_manifest(row)
        row = replace(row, expected_event_hash=expected_event_hash(manifest),
                      expected_observation_hash=expected_observation_hash(manifest, None))
        verify_row_against_manifest(row, manifest)
        require(manifest.target_box_at(row.event_ns) == build_manifest(old).target_box_at(old.event_ns),
                'Deadline change moved the stimulus')
        result['distributional_row'] = row.to_dict()
        result['base_spec']['scenario'] = proxy_scenario(row)
    else:
        scenario.update(response_window_ns=seconds * 10**9, episode_horizon_ns=horizon)
    return result


def prepare(out, baseline_configs=None, response_window_s=None):
    require(response_window_s in (None, 64), 'Only the explicitly authorized 64-second amendment is supported')
    require(not out.exists(), 'Use a fresh output directory')
    reference = read(REFERENCE / 'execution_manifest.json')
    verify(reference['hashes'], REFERENCE)
    require(read(REFERENCE / 'status.json')['state'] == 'stopped', 'Expected preserved stopped reference')
    out.mkdir(parents=True)
    shutil.copytree(REFERENCE / 'code', out / 'code',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('run_qwen08_five_arm.py', 'paper_baselines.py'):
        shutil.copy2(ROOT / 'studies' / name, out / 'code' / name)
    shutil.copytree(ROOT / 'src/chronotope/paper_baselines', out / 'code/src/chronotope/paper_baselines',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Freeze the observation hook and the explicit, opt-in episode boundary.
    inherited = out / 'code/src/chronotope/runner.py'
    updated = (ROOT / 'src/chronotope/runner.py').read_text()
    without_hooks = updated.replace(BUFFER_HOOK, '').replace(
        'manifest: Manifest | None = None,\n              enforce_horizon: bool = False) -> Path:',
        'manifest: Manifest | None = None) -> Path:').replace(
        'if (multi or enforce_horizon) and harness.now_ns() >= end_h:',
        'if multi and harness.now_ns() >= end_h:')
    require(updated.count(BUFFER_HOOK) == 1 and without_hooks == inherited.read_text(),
            'Unexpected harness differences beyond the observation and horizon hooks')
    inherited.write_text(updated)
    for folder in ('configs', 'cases', 'records', 'trials', 'calls', 'payloads', 'upstream'):
        (out / folder).mkdir()
    setup(out)
    from chronotope.paper_baselines.factory import check_config
    from chronotope.paper_baselines.upstream import LOCK_PATH, locked_file
    lock = read(LOCK_PATH)
    for name in ('aoi', 'mobiledreamer'):
        source = ROOT / 'runs/paper_baseline_sources' / name
        for relative in lock[name]['files']:
            target = out / 'upstream' / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(locked_file(source, name, relative), target)
    proxy = latency_proxy(REFERENCE)
    write(out / 'configs/aapt_latency_proxy.json', proxy)
    for name in ('full_agent_components.json', 'service_launch_reference.json', 'task_instructions.json'):
        shutil.copy2(REFERENCE / 'configs' / name, out / 'configs' / name)
    instructions = read(out / 'configs/task_instructions.json')
    baselines = {}
    for arm in ARMS[2:]:
        source = (baseline_configs / f'{arm}.json' if baseline_configs else
                  ROOT / 'configs/paper_baselines' / f'{arm}.example.json')
        config = read(source)
        config['model']['base_url'] = 'http://127.0.0.1:8012/v1'
        config['model']['extra_body'].update(top_p=.8, repetition_penalty=1.0)
        if arm in ('aoi', 'mobiledreamer'):
            config['source_root'] = str(out / 'upstream' / arm)
        if arm == 'aapt':
            config['parameters'].update(planner_p95_ms=proxy['planner_budget_ms'],
                planner_latency_evidence=str(out / 'configs/aapt_latency_proxy.json'))
        if arm == 'mobiledreamer' and baseline_configs is None:
            # Planned runtime; the runtime check requires independent launch evidence.
            config['world_model']['identity'].update(
                source_precision='FP32 merged checkpoint (first-shard header verified; all-shard audit required)',
                precision='FP16 (planned stock vLLM conversion)', kv_cache_precision='FP16')
            root = ROOT / 'runs/model_cache/paper_baselines'
            config['parameters']['ocr'].update(
                layout_detection_model_dir=str(root / 'PP-DocLayoutV2'),
                vl_rec_model_dir=str(root / 'PaddleOCR-VL'))
        check_config(config)
        target = out / 'configs' / f'{arm}.json'
        write(target, config); baselines[arm] = target
    cells = copy.deepcopy(read(REFERENCE / 'plan.json')['cells'])
    if response_window_s is not None:
        cells = [c for c in cells if c['task'] != 'notification' or c['setting_s'] == 16]
    cases = {}
    for cell in cells:
        source = REFERENCE / cell['manifest']
        own = [json.loads(s) for s in source.read_text().splitlines()]
        if response_window_s is not None:
            own = [with_response_window(c, response_window_s) for c in own]
            cell['task_deadline_s'] = response_window_s
            if cell['task'] == 'notification':
                cell.update(setting_s=response_window_s, cell_id=f'notification_{response_window_s}s',
                            manifest=f'cases/notification_{response_window_s}s.jsonl')
                cohort = dict(schema='derived-notification-deadline-1', source_cases=str(source),
                    source_sha256=digest(source), response_window_s=response_window_s,
                    rows=[c['distributional_row'] for c in own])
                cohort_path = out / 'cases/notification_distributional_manifest.json'
                write(cohort_path, cohort)
                for case in own:
                    case['distributional_manifest_hash'] = digest(cohort_path)
        target = out / cell['manifest']
        target.write_text(''.join(json.dumps(c, sort_keys=True) + '\n' for c in own)) if response_window_s else shutil.copy2(source, target)
        cell.update(source_cases=str(source), source_sha256=digest(source), case_sha256=digest(target))
        cases[cell['manifest']] = own
    preserved = {}
    for task in TASKS:
        for name in (f'{task}_compact_prompt.json', *[f'{task}_{arm}.json' for arm in ARMS[:2]]):
            source = REFERENCE / 'configs' / name
            shutil.copy2(source, out / 'configs' / name)
            preserved[f'configs/{name}'] = digest(source)
        for arm, path in baselines.items():
            write(out / 'configs' / f'{task}_{arm}.json', dict(kind='paper_baseline', baseline=arm,
                task=task, instruction=instructions[task], config_path=str(path), config_sha256=digest(path)))
    total = len(cells) * len(ARMS) * 100
    plan = dict(schema='qwen08-five-arm-1', model=MODEL, revision=REVISION, cells=cells,
        jobs=make_jobs(cells, cases), planned_trials=total, qualification_trials=0,
        modes=['live'], settings_s=sorted({c['setting_s'] for c in cells}, reverse=True),
        response_window_s=response_window_s, enforce_horizon=response_window_s is not None,
        trials_per_cell=100, arms=list(ARMS),
        automatic_resume=False, successor_jobs=[],
        authorization='Add AOI, MobileDreamer and AAPT to the existing legacy C++ versus Agent S3 Qwen0.8 comparison.',
        scope_basis=str(REFERENCE / 'plan.json'),
        deferred={'code': 'Existing context-limit issue and unsupported native typing',
                  'drift': 'Outside the latest three-task comparison; full-five plan remains separate'},
        historical_results_reused=False, previous_runs_preserved=True)
    if response_window_s is not None:
        plan['authorization'] = "2026-11-09: let's do it. And use 64 second windows"
        plan['schedule_amendment'] = ('64-second response deadlines for all tasks; Notification has one window; '
            'Button/Trading retain their five state-change delays. Fresh cases derived from the same stimuli.')
    write(out / 'plan.json', plan)
    cfg = copy.deepcopy(reference)
    cfg.update(schema='qwen08-five-arm-execution-1', created_at=now(), planned_trials=total,
        runnable_trials=None, readiness='pending runtime checks', arm_labels=LABELS,
        source_reference=str(REFERENCE), reference_manifest_sha256=digest(REFERENCE / 'execution_manifest.json'),
        preserved_controller_hashes=preserved, qualification_trials=0,
        interpretation='Complete configurations sharing a planner checkpoint; legacy C++ is not full Chronotope.',
        integration_limitations=[
            'Legacy detector/compact prompt and original Agent S3 Worker/UI-TARS remain unchanged.',
            'Different prompts, token budgets, histories and auxiliary models; not an isolated controller-overhead test.',
            'New baselines use raw screenshot pixel coordinates/image scale 1; accepted arms retain their original scaling.',
            'AOI buffers published observations; CLIP processing is synchronous and measured.',
            'MobileDreamer reuses released one-step helpers, with a separate learned world model and OCR.',
            'AAPT uses the same Qwen model for tree planning and routing, with a historical latency proxy.',
            'AAPT preparation consumes the ordinary trial clock; no extra pre-event preparation is granted.',
            'Scheduled 20ms harness receipts are not physical application acknowledgements.',
            'No claim that this three-task Live extension completes the full core-five, three-condition study.'])
    if response_window_s is not None:
        cfg['integration_limitations'].extend([
            'AOI uses bounded structured narration/actions adapted to the original parser; no coordinate clamping.',
            'MobileDreamer uses discriminated bounded pixel-action arguments and accepts harmless wait metadata.',
            'AAPT tree text is bounded; structured decoding suppresses whitespace; invalid-output retry includes feedback.',
            'All five arms forbid starting another decision after the episode horizon; in-flight calls may finish late.',
            'These repairs and the 64-second deadline define a new version; historical failures remain separate.'])
    cfg['hashes'] = {str(p.relative_to(out)): digest(p)
        for folder in ('code', 'configs', 'cases', 'upstream')
        for p in sorted((out / folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    cfg['hashes']['plan.json'] = digest(out / 'plan.json')
    write(out / 'execution_manifest.json', cfg)
    write(out / 'status.json', dict(state='prepared', planned_trials=total, completed_trials=0,
        attempted_trials=0, infrastructure_errors=0, pid=None, automatic_resume=False, updated_at=now()))
    validate(out); summarize(out, [])


def validate(out):
    setup(out)
    from core5_ledger_plan import make_trial_spec
    from paper_baselines import case_manifest
    from chronotope.paper_baselines.factory import check_config, check_scope
    from agents3_full import source_fingerprints
    cfg = read(out / 'execution_manifest.json'); plan = read(out / 'plan.json')
    verify(cfg['hashes'], out); verify(cfg['preserved_controller_hashes'], out)
    require((plan['schema'], plan['model'], plan['revision']) == ('qwen08-five-arm-1', MODEL, REVISION), 'Plan identity')
    require(plan['qualification_trials'] == 0 and plan['successor_jobs'] == []
            and plan['automatic_resume'] is False, 'No qualification/resume/queue allowed')
    amended = plan.get('response_window_s') == 64
    expected_cells = {(t, s) for t in TASKS for s in ((64,) if amended and t == 'notification' else SETTINGS)}
    total = len(expected_cells) * len(ARMS) * 100
    require(plan.get('response_window_s') in (None, 64), 'Unauthorized deadline amendment')
    require(plan['planned_trials'] == total and plan['arms'] == list(ARMS)
            and plan['modes'] == ['live'] and plan['settings_s'] == sorted({s for _, s in expected_cells}, reverse=True), 'Plan scope')
    require(bool(plan.get('enforce_horizon')) == amended, 'Horizon policy mismatch')
    cases = {}; groups = defaultdict(set)
    for cell in plan['cells']:
        require((cell['task'], cell['setting_s']) in expected_cells, 'Out-of-scope cell')
        path = out / cell['manifest']
        require(path.resolve().is_relative_to((out / 'cases').resolve()), 'Case path escapes frozen directory')
        require(digest(path) == cell.get('case_sha256', cell['source_sha256']), 'Cases changed')
        require(digest(Path(cell['source_cases'])) == cell['source_sha256'], 'Source cases changed')
        own = [json.loads(s) for s in path.read_text().splitlines()]
        require(len(own) == 100, 'Expected exactly 100 frozen cases')
        cases[cell['manifest']] = own
        for case in own:
            case_manifest(case, cell['task'])
    require({(c['task'], c['setting_s']) for c in plan['cells']} == expected_cells
            and len(cases) == len(plan['cells']) == len(expected_cells), 'Duplicate or missing cells')
    require(plan['jobs'] == make_jobs(plan['cells'], cases) and len(plan['jobs']) == total, 'Job plan changed')
    for arm in ARMS[2:]:
        config = read(out / 'configs' / f'{arm}.json'); check_config(config)
        identity = config['model']['identity']
        require((identity['model'], identity['revision'], identity['precision'], identity['kv_cache_precision'])
                == (MODEL, REVISION, 'FP16 planner and vision', 'FP16'), 'Planner identity/precision mismatch')
        require(config['model']['base_url'] == 'http://127.0.0.1:8012/v1', 'Use the same planner service')
    for job in plan['jobs']:
        case = cases[job['manifest']][job['index']]
        config = read(out / 'configs' / f"{job['task']}_{job['arm']}.json")
        spec = make_trial_spec(case, config, job['mode'])
        require(spec.raw['scenario']['response_window_ns'] == job['task_deadline_s'] * 10**9, 'Deadline mismatch')
        if amended:
            require(job['task_deadline_s'] == 64, 'Every amended response window must be 64 seconds')
        if job['task'] != 'notification':
            field = 'change_delay_ns' if job['task'] == 'button' else 'reversal_delay_ns'
            require(spec.raw['scenario'][field] == job['setting_s'] * 10**9, 'State-change timing changed')
        if job['arm'] in ARMS[2:]:
            check_scope(spec.raw)
        require((case['index'], case['seed'], case['episode_id']) ==
                (job['index'], job['seed'], job['episode_id']), 'Case identity mismatch')
        groups[job['task'], job['setting_s'], job['arm']].add(job['seed'])
    require(len(groups) == len(expected_cells) * len(ARMS) and all(len(seeds) == 100 for seeds in groups.values()), 'Unbalanced seed sets')
    for task, setting in expected_cells:
        require(all(groups[task, setting, arm] == groups[task, setting, ARMS[0]] for arm in ARMS), 'Unmatched cases')
    original = cfg['full_agent_source_verification']['source_file_sha256']
    for path, sha in source_fingerprints().items():
        suffix = path[path.index('/gui_agents/'):] if '/gui_agents/' in path else '/agents3_full.py'
        require([v for p, v in original.items() if p.endswith(suffix)] == [sha], 'Agent S3 source changed')
    write(out / 'validation.json', dict(state='passed', at=now(), planned_trials=total,
        task_families=list(TASKS), modes=['live'], matched_seed_sets=True,
        preserved_controller_configs=True, qualification_trials=0, model_inference_calls=0,
        plan_sha256=digest(out / 'plan.json')))
    return cases


def check_runtime(out, *, with_cases=False):
    """Read-only setup checks, never task episodes or model-generation probes.

    Auxiliary evidence is produced by installation/launch, outside the frozen plan.
    It must name every asset and the actual world-model process. No model alias is
    accepted as proof of checkpoint or precision.
    """
    cases = validate(out)
    from run_claim1_agents3_remaining3 import check_services, same_process
    blockers = []; cfg = read(out / 'execution_manifest.json')
    def check(label, operation):
        try:
            operation()
        except Exception as exc:
            blockers.append(f'{label}: {type(exc).__name__}: {exc}')
    check('Accepted planner/grounder services', lambda: check_services(out))
    check('Native engines', lambda: verify(cfg['engine_hashes']))
    check('Native plugin', lambda: require(digest(cfg['plugin']['path']) == cfg['plugin']['sha256'], 'Changed plugin'))
    for name in ('torch', 'clip', 'paddleocr', 'paddle', 'PIL', 'numpy'):
        check('Python dependency ' + name,
              lambda name=name: require(importlib.util.find_spec(name) is not None, 'Not installed in ' + sys.executable))
    mobile = read(out / 'configs/mobiledreamer.json')
    for key in ('layout_detection_model_dir', 'vl_rec_model_dir'):
        check('MobileDreamer ' + key,
              lambda key=key: require(Path(mobile['parameters']['ocr'][key]).is_dir(), 'Missing model asset directory'))
    evidence_path = out / 'auxiliary_runtime.json'
    def auxiliary():
        evidence = read(evidence_path)
        require(evidence['schema'] == 'paper-baseline-runtime-1', 'Unsupported runtime evidence')
        require(evidence['python'] == str(Path(sys.executable).absolute()), 'Different Python environment')
        require(evidence['plan_sha256'] == digest(out / 'plan.json'), 'Runtime evidence for another plan')
        for name in ('clip', 'ocr_layout', 'ocr_vl', 'world_model'):
            asset = evidence['assets'][name]
            require(asset['file_hashes'] and asset['revision'] and asset['source_precision']
                    and asset['effective_precision'], 'Incomplete auxiliary provenance: ' + name)
            verify(asset['file_hashes'])
            root = Path(asset['root']).resolve()
            require(all(Path(p).resolve().is_relative_to(root) for p in asset['file_hashes']),
                    'Asset files outside declared root: ' + name)
        for asset_name, key in (('ocr_layout', 'layout_detection_model_dir'), ('ocr_vl', 'vl_rec_model_dir')):
            asset = evidence['assets'][asset_name]
            root = Path(mobile['parameters']['ocr'][key]).resolve()
            require(Path(asset['root']).resolve() == root, 'OCR evidence does not describe the configured assets')
            required_files = {str(p) for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts}
            require(required_files <= set(asset['file_hashes']), 'Incomplete OCR asset inventory')
        import clip
        require(callable(getattr(clip, 'load', None)) and callable(getattr(clip, 'tokenize', None)),
                'Expected OpenAI CLIP package')
        clip_path = str(Path.home() / '.cache/clip/ViT-B-16.pt')
        require(clip_path in evidence['assets']['clip']['file_hashes'], 'Missing default CLIP ViT-B/16 asset')
        world_asset = evidence['assets']['world_model']
        world_root = Path(world_asset['root']).resolve()
        index = read(world_root / 'model.safetensors.index.json')
        required = set(index['weight_map'].values()) | {'model.safetensors.index.json', 'config.json',
                                                        'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'}
        require({str(world_root / name) for name in required} <= set(world_asset['file_hashes']),
                'Incomplete merged world-model checkpoint inventory')
        world = evidence['world_service']
        require(same_process(world), 'World-model service changed or stopped')
        identity = mobile['world_model']['identity']
        require(world['model'] == identity['model'] and world['revision'] == identity['revision']
                and world['dtype'] == 'half' and world['effective_kv_cache_dtype'] == 'float16',
                'World-model identity/precision mismatch')
        require(world['checkpoint_subfolder'] == 'merge_model', 'Wrong world-model artifact')
        require(world_asset['revision'] == identity['revision'], 'World-model asset revision mismatch')
        command = world['command']
        require(command[command.index('serve') + 1] == str(world_root), 'Different served checkpoint directory')
        require(command[command.index('--dtype') + 1] == 'half', 'World service must convert floating tensors to FP16')
        require(command[command.index('--kv-cache-dtype') + 1] == 'auto', 'Expected FP16 automatic KV dtype')
        require(command[command.index('--served-model-name') + 1] == identity['model'], 'Wrong world service alias')
        with urllib.request.urlopen(mobile['world_model']['base_url'] + '/models', timeout=5) as response:
            models = json.load(response)
        require(identity['model'] in {item['id'] for item in models['data']}, 'World model not served')
    check('Auxiliary model installation and launch provenance', auxiliary)
    result = dict(state='blocked_setup' if blockers else 'ready', checked_at=now(), blockers=blockers,
        python=sys.executable, model_inference_calls=0, qualification_trials=0,
        plan_sha256=digest(out / 'plan.json'),
        auxiliary_runtime_sha256=digest(evidence_path) if evidence_path.exists() else None)
    write(out / 'readiness.json', result)
    return (result, cases) if with_cases else result


def launch(out):
    """Explicit launch only; refuses duplicate markers and any incomplete setup."""
    require(read(out / 'status.json')['state'] == 'prepared', 'Fresh prepared study required')
    require(not (out.parent / 'launch.json').exists(), 'No automatic relaunch')
    require(check_runtime(out)['state'] == 'ready', 'Setup incomplete; see readiness.json')
    cfg = read(out / 'execution_manifest.json')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=':'.join([
        str(Path(cfg['plugin']['path']).parent), (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'),
        '/usr/local/cuda/targets/sbsa-linux/lib']))
    env.pop('PYTHONPATH', None)
    command = [sys.executable, '-B', str(out / 'code/run_qwen08_five_arm.py'), 'run', '--out', str(out)]
    with (out.parent / 'run.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(out.parent / 'launch.json', dict(pid=child.pid, command=command, at=now(),
        plan_sha256=digest(out / 'plan.json'), automatic_resume=False))


def summarize(out, records):
    groups = defaultdict(list)
    for record in records:
        job = record['job']; groups[job['task'], job['setting_s'], job['arm']].append(record)
    rows = []
    plan = read(out / 'plan.json') if (out / 'plan.json').exists() else None
    cells = plan['cells'] if plan else [dict(task=t, setting_s=s) for t in TASKS for s in SETTINGS]
    def percentile(values, q):
        if not values: return None
        values = sorted(values); pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    lines = ['# Qwen3.5-0.8B five-arm comparison', '',
        'Fresh Live evaluation: Notification, Button and Trading; 100 cases per cell. '
        'All planners use revision `' + REVISION + '`, FP16 planner/vision/KV. '
        'No historical successes enter these denominators.', '',
        '| Task | Setting (s) | ' + ' | '.join(LABELS.values()) + ' |',
        '|---|---:|' + '---:|' * len(ARMS)]
    for task in TASKS:
        for setting in [c['setting_s'] for c in cells if c['task'] == task]:
            display = []
            for arm in ARMS:
                own = groups[task, setting, arm]; scored = [r for r in own if r['status'] == 'scored']
                n = len(scored); k = sum(r['success'] for r in scored); lo, hi = wilson(k, n)
                latencies = [r.get('metrics', {}).get('e_to_receipt_ms') for r in scored]
                latencies = [v for v in latencies if v is not None]
                observed = [r.get('metrics', {}).get('observed_handler_e_to_a_ms') for r in scored]
                observed = [v for v in observed if v is not None]
                decisions = [v for r in scored for v in r.get('metrics', {}).get('decision_B_to_D_ms', [])]
                rows.append(dict(task=TASKS[task], setting_s=setting, mode='live', arm=arm,
                    controller=LABELS[arm], successes=k, scored_trials=n, planned_trials=100,
                    infrastructure_errors=len(own)-n, success_rate=k/n if n else None,
                    ci95_low=lo, ci95_high=hi, action_receipts=len(latencies), no_action_receipt=n-len(latencies),
                    event_to_receipt_p50_ms=percentile(latencies, .5), event_to_receipt_p95_ms=percentile(latencies, .95),
                    observed_handler_p50_ms=percentile(observed, .5), observed_handler_p95_ms=percentile(observed, .95),
                    decision_count=len(decisions), decision_mean_ms=sum(decisions)/len(decisions) if decisions else None))
                display.append(f'{k}/{n}' if n else '— (0 measured)')
            lines.append(f'| {TASKS[task]} | {setting} | ' + ' | '.join(display) + ' |')
    if plan and plan.get('response_window_s'):
        lines.extend(['', f"All response deadlines are {plan['response_window_s']} seconds. "
            'Setting is the response window for Notification and the state-change delay for Button/Trading.'])
    lines.extend(['', 'Event-to-action latency (milliseconds; benchmark receipt, including late/incorrect actions):', '',
        '| Task | Setting (s) | Controller | Receipts / scored | E→A median | E→A p95 |',
        '|---|---:|---|---:|---:|---:|'])
    for row in rows:
        def display(value): return f'{value:.1f}' if value is not None else '—'
        lines.append(f"| {row['task']} | {row['setting_s']} | {row['controller']} | "
                     f"{row['action_receipts']}/{row['scored_trials']} | "
                     f"{display(row['event_to_receipt_p50_ms'])} | {display(row['event_to_receipt_p95_ms'])} |")
    lines.extend(['', 'No action receipt means unmeasurable E→A, never zero. Failures remain in all trial denominators. '
        'Observed handler timestamps and decision-completion durations are separate columns in summary.csv.'])
    lines.extend(['', 'Counts show completed scored cases only; infrastructure errors are separate in summary.csv.',
        '', 'Legacy C++ retains its detector and compact prompt. Agent S3 retains its Worker, history, reflection '
        'and BF16 UI-TARS grounder. This is not full GPU-history Chronotope or an identical-request comparison.',
        '', 'AOI uses the released visual keyframe code with buffered processing. MobileDreamer uses the released '
        'one-step selector, a separate learned world model and OCR. AAPT is paper-derived and uses a historical '
        'Worker latency proxy, not a measured policy-tree p95. Their runtime readiness is reported in readiness.json.',
        '', 'Code remains deferred; Moving Balloon remains in the separate full-five plan. '
        'This extension follows the latest three-task Live scope; it does not complete the full-five study.'])
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n')
    with (out / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def baseline_calls(controller):
    calls = []
    for role, model in (('planner', getattr(controller, 'model', None)),
                        ('world_model', getattr(controller, 'world_model', None))):
        if model is not None:
            calls.extend(dict(call, role=role) for call in model.calls)
    return calls


def run(out):
    require(read(out / 'status.json')['state'] == 'prepared' and
            not any((out / 'records').iterdir()) and not any((out / 'trials').iterdir()),
            'Fresh prepared run required; no resume')
    locks = []; identities = set()
    for path in (ROOT / 'runs/legacy_d1_gpu.lock',
                 ROOT / 'runs/hosted_api_deepseek_nothinking_20261105/source/runs/legacy_d1_gpu.lock'):
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity not in identities:
            stream = path.open('a+'); fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(stream); identities.add(identity)
    readiness, cases = check_runtime(out, with_cases=True)
    require(readiness['state'] == 'ready', 'Setup incomplete; see readiness.json. No trials launched.')
    cfg = read(out / 'execution_manifest.json'); plan = read(out / 'plan.json')
    total = plan['planned_trials']
    from agents3_full import FullAgentS3Adapter
    from core5_ledger_plan import make_trial_spec
    from run_core5_detector_full_agents3 import make_host
    from audit_legacy_detector_full_agents3 import audit_trial_requests
    from chronotope.controller import CallBudget
    from chronotope.runner import run_trial, PayloadStore
    from chronotope.trace import Trace
    from chronotope.evaluate import evaluate_trace
    from chronotope_core.controller import build as build_native
    from chronotope.paper_baselines.factory import build as build_baseline
    from paper_baselines import case_manifest
    from chronotope.distributional.episode import verify_trace_evidence
    for key in ('CHRONOTOPE_AGENT_S3_POLICY', 'CHRONOTOPE_AGENT_S3_PROMPT', 'CHRONOTOPE_AGENT_S3_AUDIT'):
        os.environ.pop(key, None)
    os.environ.update(CHRONOTOPE_BIN=str(out / 'code/chronotope/build/chronotope'),
        CHRONOTOPE_LIB=str(out / 'code/chronotope/build/libchronotope.so'), EDGELLM_PLUGIN_PATH=cfg['plugin']['path'])
    records = []; current = None; started = now()
    def publish(state, error=None):
        scored = sum(r['status'] == 'scored' for r in records)
        write(out / 'status.json', dict(state=state, started_at=started, updated_at=now(),
            completed_trials=scored, attempted_trials=len(records), infrastructure_errors=len(records)-scored,
            planned_trials=total, current_job=current, pid=os.getpid() if state == 'running' else None,
            error=error, automatic_resume=False, qualification_trials=0))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for current in plan['jobs']:
            publish('running'); job = current; ident = job['id']; arm = job['arm']; task = job['task']
            case = cases[job['manifest']][job['index']]
            spec = make_trial_spec(case, read(out / 'configs' / f'{task}_{arm}.json'), 'live')
            row, overlay = case_manifest(case, task); scenario = overlay or spec.manifest()
            path = out / 'trials' / f'{ident}.jsonl'; log = out / 'calls' / f'{ident}.jsonl'
            require(not path.exists() and not log.exists(), 'Existing trial output')
            controller = adapter = trace = evaluation = error = None
            request_audit = []; start = time.monotonic()
            try:
                if arm == ARMS[0]:
                    os.environ['CHRONOTOPE_PLANNER_PROMPT'] = str(out / 'configs' / f'{task}_compact_prompt.json')
                    os.environ['CHRONOTOPE_PLANNER_AUDIT'] = str(log)
                    controller = build_native(spec, budget=CallBudget(100))
                elif arm == ARMS[1]:
                    adapter = FullAgentS3Adapter(cfg['agent_s3_components'], log)
                    controller = make_host(spec, task, adapter, {'code_action': 'not_needed'})
                else:
                    controller = build_baseline(spec)
                run_trial(spec, path, PayloadStore(out / 'payloads'), controller=controller, manifest=scenario,
                          enforce_horizon=plan.get('enforce_horizon', False))
                trace = Trace.load(path)
                if trace.all('decision_failed') or (adapter and adapter.errors):
                    raise RuntimeError('Model/runtime error; retained unscored')
                result = evaluate_trace(trace)
                require(result.valid_trial and not result.contract_violations, 'Trace contract failure')
                evaluation = result.to_dict()
                if arm in ARMS[:2]:
                    request_audit = audit_trial_requests(out, ident, arm, trace, scenario,
                        dict(cfg, native_prompt_contract=cfg['native_task_prompts'][task],
                             native_prompt_reference=cfg['native_task_prompt_references'][task]))
                if row:
                    verify_trace_evidence(row, scenario, trace, case['distributional_manifest_hash'])
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                (out / 'records' / f'{ident}.error.txt').write_text(traceback.format_exc())
            finally:
                if arm in ARMS[2:] and controller is not None:
                    log.write_text(''.join(json.dumps(c, allow_nan=False) + '\n' for c in baseline_calls(controller)))
                try:
                    close = getattr(controller, 'close', None)
                    if close: close()
                finally:
                    if adapter: adapter.close()
            if trace is None and path.exists(): trace = Trace.load(path)
            calls = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
            record = dict(job=job, status='infrastructure_error' if error else 'scored', error=error,
                success=None if error else int(bool(evaluation['valid_at_receipt'])), evaluation=evaluation,
                metrics=metrics(trace, calls, task) if trace else empty_metrics(), request_audit=request_audit,
                trace=str(path.relative_to(out)) if path.exists() else None,
                trace_sha256=digest(path) if path.exists() else None,
                calls_sha256=digest(log) if log.exists() else None,
                completed_at=now(), wall_seconds=time.monotonic()-start)
            if trace:
                begins = {e['decision_id']: e['begin_time_ns'] for e in trace.all('decision_begin')}
                record['metrics']['decision_B_to_D_ms'] = [
                    (e['end_time_ns'] - begins[e['decision_id']]) / 1e6
                    for e in trace.all('decision_invocation') if e['decision_id'] in begins]
            write(out / 'records' / f'{ident}.json', record); records.append(record)
            summarize(out, records); publish('running')
            print(f'{len(records)}/{total} {ident} {record["status"]} success={record["success"]}', flush=True)
            # A runtime outage is not evidence about competence. Stop with its trace;
            # invalid model actions remain scored failures and never gate later cases.
            if error: raise RuntimeError(error)
        current = None; audit(out); publish('complete')
    except BaseException as exc:
        summarize(out, records)
        publish('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', f'{type(exc).__name__}: {exc}')
        raise


def audit(out):
    validate(out)
    from chronotope.trace import Trace
    from chronotope.evaluate import evaluate_trace
    records = [read(p) for p in sorted((out / 'records').glob('*.json'))]
    jobs = {j['id']: j for j in read(out / 'plan.json')['jobs']}
    for record in records:
        require(record['job'] == jobs[record['job']['id']], 'Record job mismatch')
        if record['trace']:
            path = out / record['trace']; require(digest(path) == record['trace_sha256'], 'Trace hash mismatch')
            if record['status'] == 'scored':
                require(evaluate_trace(Trace.load(path)).to_dict() == record['evaluation'], 'Score mismatch')
        if record['calls_sha256']:
            require(digest(out / 'calls' / (record['job']['id'] + '.jsonl')) == record['calls_sha256'], 'Call log changed')
    summarize(out, records)
    result = dict(state='passed', at=now(), audited_records=len(records),
        complete_plan=len(records) == len(jobs) and all(r['status'] == 'scored' for r in records))
    write(out / 'evidence_audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'check-runtime', 'launch', 'run', 'audit', 'report'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-configs', type=Path,
                        help='Optional aoi.json, mobiledreamer.json, aapt.json for a fresh preparation')
    parser.add_argument('--response-window-s', type=int, choices=(64,),
                        help='Authorized 64-second deadline amendment; only used for fresh preparation')
    args = parser.parse_args(); out = args.out.resolve()
    if args.action == 'prepare': prepare(out, args.baseline_configs, args.response_window_s)
    elif args.action == 'report': summarize(out, [read(p) for p in sorted((out / 'records').glob('*.json'))])
    elif args.action == 'check-runtime':
        result = check_runtime(out); print(json.dumps(result, indent=2))
        if result['state'] != 'ready': raise SystemExit(2)
    else: globals()[args.action](out)


if __name__ == '__main__':
    main()
