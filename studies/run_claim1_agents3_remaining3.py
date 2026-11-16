"""Fresh Keep-action Agent S3 evaluation, queued after the active DeepSeek run.

Preparation and validation are CPU-only. The queue never resumes either study,
changes model services, or runs qualification episodes. All scoring uses receipt.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'runs/notification_qwen08_edgellm0111_20261031/experiment'
PREDECESSOR = ROOT / 'runs/hosted_api_deepseek_nothinking_20261105/deepseek-flash'
PYTHON = (_CHRONOTOPE_HOME + '/agents3-venv/bin/python')
MODEL = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
TASKS = {'code': 'Expiring Authenticator Code', 'button': 'Displaced Button',
         'trading': 'Reversing Trading Signal'}
SETTINGS = (16, 8, 4, 2, 1)
MODES = ('live', 'replay', 'paused')
TOTAL = 4500


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_hashes(hashes, base=Path('/')):
    for name, expected in hashes.items():
        if digest(base / name) != expected:
            raise ValueError('Frozen evidence changed: ' + name)


def setup(out):
    code = out / 'code'
    sys.path[:0] = [str(code), str(code / 'src'), str(code / 'chronotope/python')]
    for key in ('CHRONOTOPE_AGENT_S3_POLICY', 'CHRONOTOPE_AGENT_S3_PROMPT',
                'CHRONOTOPE_AGENT_S3_AUDIT', 'CHRONOTOPE_PLANNER_PROMPT'):
        os.environ.pop(key, None)


def process_identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': pid, 'process_starttime_ticks': int(stat[19]),
                'command': (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected):
    actual = process_identity(expected['pid'])
    return actual is not None and all(actual[key] == expected[key]
        for key in ('pid', 'process_starttime_ticks', 'command'))


def completion_gate(status, evidence, plan_hash, execution_hash):
    return (status.get('state') == 'complete' and
            status.get('completed_trials') == status.get('planned_trials') == 3000 and
            evidence.get('state') == 'passed' and evidence.get('completed_trials') == 3000 and
            evidence.get('plan_sha256') == plan_hash and
            evidence.get('execution_manifest_sha256') == execution_hash)


def prepare(out):
    if out.exists():
        raise FileExistsError('Never replace a prepared run: ' + str(out))
    out.mkdir(parents=True)
    shutil.copytree(REFERENCE / 'code', out / 'code',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, out / 'code' / Path(__file__).name)
    for name in ('cases', 'configs', 'records', 'trials', 'calls', 'payloads'):
        (out / name).mkdir()
    setup(out)
    from core5_ledger_plan import historical_cells
    from run_core5_detector_full_agents3 import task_instruction
    full = read(REFERENCE / 'configs/full_agent_components.json')
    assert (full['components']['planner']['model'], full['components']['planner']['revision']) == (MODEL, REVISION)
    write(out / 'configs/full_agent_components.json', full)
    source_controller = read(REFERENCE / 'configs/notification_agents3_full.json')
    instructions = {task: task_instruction(task) for task in TASKS}
    write(out / 'configs/task_instructions.json', instructions)
    cells, jobs, case_hashes = [], [], {}
    templates = {cell['task']: (cell, cases) for cell, cases in historical_cells(TASKS)
                 if cell['setting_value'] == 2}
    for task in TASKS:
        template_cell, template_cases = templates[task]
        controller = dict(source_controller, image_scale=template_cell['image_scale'])
        write(out / 'configs' / f'{task}_agents3_full.json', controller)
        field = 'reversal_delay_ns' if task == 'trading' else 'change_delay_ns'
        for setting in SETTINGS:
            cell_id = f'{task}_change{setting}s'
            manifest = f'cases/{cell_id}.jsonl'
            cases = copy.deepcopy(template_cases)
            for case in cases:
                case['base_spec']['scenario'][field] = setting * 1_000_000_000
            path = out / manifest
            path.write_text(''.join(json.dumps(c, sort_keys=True) + '\n' for c in cases))
            case_hashes[manifest] = digest(path)
            cells.append(dict(template_cell, cell_id=cell_id, setting_value=setting,
                              manifest=manifest, action_policy='Keep action',
                              task_deadline_s=32, cases=100))
            for case in cases:
                for mode in MODES:
                    jobs.append(dict(phase='evaluation', task=task, arm='agents3_full', mode=mode,
                        index=case['index'], seed=case['seed'], episode_id=case['episode_id'],
                        setting_s=setting, task_deadline_s=32, cell_id=cell_id, manifest=manifest,
                        id=f"agents3_full_{cell_id}_{case['index']:03d}_{mode}"))
    write(out / 'plan.json', dict(schema='claim1-agents3-remaining3-1', model=MODEL, revision=REVISION,
        cells=cells, jobs=jobs, case_file_sha256=case_hashes, planned_trials=TOTAL,
        qualification_trials=0, action_policy='Keep action', automatic_resume=False,
        successor_jobs=[], reference_notification_run=str(REFERENCE),
        parameter_definition='Seconds from task event E until code replacement, banner insertion, or signal reversal',
        deadline_definition='32 seconds from E; distinct from the varied state-change time',
        code_action_bridge='Historical focused-field atomic typing; AST parses model text; no OCR or update',
        authorization='User selected full Agent S3 Qwen3.5-0.8B Notification table and requested the other three tasks after the current DeepSeek API run'))
    verification = read(REFERENCE.parent / 'full_agents3_cpu_verification.json')
    write(out / 'configs/full_agent_source_verification.json', verification)
    services = read(REFERENCE.parent / 'services/launch.json')
    write(out / 'configs/service_launch_reference.json', services)
    checkpoint = read(REFERENCE.parent / 'checkpoint_download.json')
    grounder = read(REFERENCE.parent / 'grounder_checkpoint_verification.json')
    write(out / 'configs/planner_checkpoint_reference.json', checkpoint)
    write(out / 'configs/grounder_checkpoint_reference.json', grounder)
    predecessor_status = read(PREDECESSOR / 'status.json')
    identity = process_identity(predecessor_status['pid'])
    if not identity or str(PREDECESSOR) not in identity['command']:
        raise RuntimeError('The selected running predecessor has changed')
    predecessor_runner = PREDECESSOR.parent / 'source/studies/run_hosted_notification.py'
    write(out.parent / 'queue_manifest.json', dict(created_at=now(), predecessor=str(PREDECESSOR),
        predecessor_identity=identity, predecessor_planned_trials=3000,
        predecessor_plan_sha256=digest(PREDECESSOR / 'plan.json'),
        predecessor_execution_manifest_sha256=digest(PREDECESSOR / 'execution_manifest.json'),
        predecessor_runner=str(predecessor_runner), predecessor_runner_sha256=digest(predecessor_runner),
        out=str(out), planned_trials=TOTAL, required_successful_completion=True,
        required_evidence_audit=True, automatic_resume=False, qualification_trials=0,
        successor_jobs=[], source_notification_run=str(REFERENCE)))
    cfg = dict(schema='claim1-agents3-execution-1', created_at=now(), model=MODEL, revision=REVISION,
        runtime='vLLM', planner_precision='FP16 planner/vision and KV; BF16 source, no quantization',
        grounder_precision='BF16 weights/activations and KV', action_policy='Keep action',
        arm_labels={'agents3_full': 'Full Agent S3'}, agent_s3_components=full,
        full_agent_source_verification=verification, model_inference_calls_during_preparation=0,
        checkpoint_hashes=checkpoint['hashes'], grounder_checkpoint_hashes=grounder['file_hashes'],
        reference_execution_manifest_sha256=digest(REFERENCE / 'execution_manifest.json'),
        code_action_bridge='host_focused_field_historical_type_bridge',
        native_controller_present=False, source_snapshot=str(REFERENCE / 'code'))
    cfg['hashes'] = {str(p.relative_to(out)): digest(p) for folder in ('code', 'configs', 'cases')
                     for p in sorted((out / folder).rglob('*')) if p.is_file()}
    cfg['hashes']['plan.json'] = digest(out / 'plan.json')
    cfg['queue_manifest_sha256'] = digest(out.parent / 'queue_manifest.json')
    write(out / 'execution_manifest.json', cfg)
    validate(out)
    write(out / 'status.json', dict(state='prepared', completed_trials=0, planned_trials=TOTAL,
        qualification_trials=0, automatic_resume=False, pid=None, updated_at=now()))


def validate(out):
    setup(out)
    from core5_ledger_plan import make_trial_spec
    from run_core5_detector_full_agents3 import task_instruction
    plan, cfg = read(out / 'plan.json'), read(out / 'execution_manifest.json')
    verify_hashes(cfg['hashes'], out)
    assert digest(out.parent / 'queue_manifest.json') == cfg['queue_manifest_sha256']
    assert (plan['model'], plan['revision'], plan['planned_trials']) == (MODEL, REVISION, TOTAL)
    assert plan['qualification_trials'] == 0 and plan['action_policy'] == 'Keep action'
    assert plan['successor_jobs'] == [] and plan['automatic_resume'] is False
    assert len(plan['cells']) == 15 and len(plan['jobs']) == TOTAL
    assert len({job['id'] for job in plan['jobs']}) == TOTAL
    assert {(c['task'], c['setting_value']) for c in plan['cells']} == {(t, s) for t in TASKS for s in SETTINGS}
    expected_jobs = []; cases_by_file = {}
    for cell in plan['cells']:
        task, setting = cell['task'], cell['setting_value']
        cases = [json.loads(line) for line in (out / cell['manifest']).read_text().splitlines()]
        assert len(cases) == 100 and [c['seed'] for c in cases] == list(range(1000, 1100))
        assert digest(out / cell['manifest']) == plan['case_file_sha256'][cell['manifest']]
        controller = read(out / 'configs' / f'{task}_agents3_full.json')
        assert controller == dict(read(REFERENCE / 'configs/notification_agents3_full.json'),
                                  image_scale=1 if task == 'button' else 2)
        cases_by_file[cell['manifest']] = cases
        field = 'reversal_delay_ns' if task == 'trading' else 'change_delay_ns'
        for index, case in enumerate(cases):
            assert case['index'] == index and case['episode_id'] == f'{task}-{1000 + index}'
            scenario = case['base_spec']['scenario']
            assert scenario[field] == setting * 1_000_000_000
            assert scenario['response_window_ns'] == 32_000_000_000
            assert scenario['episode_horizon_ns'] > scenario['event_time_ns'] + scenario['response_window_ns']
            for mode in MODES:
                make_trial_spec(case, controller, mode, '0' * 64 if mode == 'replay' else None).manifest()
                expected_jobs.append(dict(phase='evaluation', task=task, arm='agents3_full', mode=mode,
                    index=index, seed=1000 + index, episode_id=case['episode_id'],
                    setting_s=setting, task_deadline_s=32, cell_id=cell['cell_id'], manifest=cell['manifest'],
                    id=f"agents3_full_{cell['cell_id']}_{index:03d}_{mode}"))
    assert expected_jobs == plan['jobs'], 'Unexpected, reordered, or out-of-scope evaluation jobs'
    assert read(out / 'configs/task_instructions.json') == {t: task_instruction(t) for t in TASKS}
    assert cfg['agent_s3_components'] == read(REFERENCE / 'configs/full_agent_components.json')
    # Verify the actual frozen upstream modules, not just configuration names.
    from agents3_full import source_fingerprints
    original = cfg['full_agent_source_verification']['source_file_sha256']
    for path, actual in source_fingerprints().items():
        suffix = path[path.index('/gui_agents/'):] if '/gui_agents/' in path else '/agents3_full.py'
        expected = [h for name, h in original.items() if name.endswith(suffix)]
        assert expected == [actual], 'Agent S3 source differs: ' + suffix
    return cases_by_file


def make_controller(spec, task, adapter):
    from run_core5_detector_full_agents3 import make_host, parse_historical_type_bridge, task_instruction
    from chronotope.controller import CallBudget, Decision, VLMDirectController
    from chronotope.png import encode_png
    if task != 'code':
        return make_host(spec, task, adapter, {'code_action': 'not_needed'})

    class TypedController(VLMDirectController):
        def decide(self):
            obs = self._newest
            self._baseline_hash = obs.payload_hash
            self.budget.take()
            result = self.adapter.invoke(encode_png(obs.payload, self.w, self.h, self.scale),
                                         self.system, 'Screenshot attached.')
            parsed = parse_historical_type_bridge(result.detail['raw_text']) or result.parsed or {}
            value = parsed.get('value')
            valid = parsed.get('action') == 'type' and isinstance(value, str) and len(value) == 4 and value.isdigit()
            if valid:
                self._done = True
            return Decision({'kind': 'type', 'value': value} if valid else None, None,
                invocation_id=result.invocation_id, invocation_detail={**result.detail,
                'observation_id': obs.observation_id, 'action_bridge': 'host_focused_field_historical_type_bridge',
                'outcome': 'type' if valid else 'wait' if parsed.get('action') == 'wait' else 'unparseable'})

    m = spec.manifest()
    controller = TypedController(adapter=adapter, budget=CallBudget(100), prompt_version=17,
        image_scale=spec.raw['controller']['image_scale'], invoke_on='frame_change',
        canvas_w_px=m.canvas_w_px, canvas_h_px=m.canvas_h_px, skin='notification', target_color=m.target_color())
    controller.system = task_instruction(task)
    return controller


def check_services(out):
    cfg = read(out / 'execution_manifest.json')
    verify_hashes(cfg['checkpoint_hashes']); verify_hashes(cfg['grounder_checkpoint_hashes'])
    services = read(out / 'configs/service_launch_reference.json')
    for role in ('planner', 'grounder'):
        service = services['components'][role]
        assert same_process(service), f'{role} service identity changed; no replacement authorized by this runner'
        expected_dtype = 'half' if role == 'planner' else 'bfloat16'
        assert service['command'][service['command'].index('--dtype') + 1] == expected_dtype
        assert service['effective_kv_cache_dtype'] == ('float16' if role == 'planner' else 'bfloat16')
        with urllib.request.urlopen(service['base_url'] + '/models', timeout=15) as response:
            assert [m['id'] for m in json.load(response)['data']] == [service['model']]
    write(out / 'service_verification.json', dict(state='passed', verified_at=now(),
        reused_service_launch=str(REFERENCE.parent / 'services/launch.json'),
        components={r: services['components'][r] for r in ('planner', 'grounder')},
        checkpoint_hashes_verified=True, model_inference_calls=0, qualification_trials=0))


def summarize(out, records):
    groups = defaultdict(list)
    for record in records:
        job = record['job']; groups[job['task'], job['setting_s'], job['mode']].append(record)
    rows = []
    for task in TASKS:
        for setting in SETTINGS:
            for mode in ('paused', 'replay', 'live'):
                own = groups[task, setting, mode]
                rows.append(dict(task=TASKS[task], setting_s=setting, task_deadline_s=32, mode=mode,
                    successes=sum(int(bool(r['evaluation']['valid_at_receipt'])) for r in own), trials=len(own),
                    planned_trials=100, coverage='complete' if len(own) == 100 else 'partial' if own else 'unmeasured',
                    action_policy='Keep action', controller='Full Agent S3', model=MODEL, revision=REVISION,
                    precision='FP16', kv_cache='FP16', runtime='vLLM',
                    grounder='ByteDance-Seed/UI-TARS-1.5-7B', grounder_precision='BF16',
                    seed_set=json.dumps(sorted(r['job']['seed'] for r in own)),
                    source_records=json.dumps([str(out / 'records' / (r['job']['id'] + '.json')) for r in own])))
    path = out / 'summary.csv.tmp'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    path.replace(out / 'summary.csv')


def audit(out):
    cases = validate(out)
    from chronotope.trace import Trace
    from chronotope.evaluate import evaluate_trace
    from chronotope.pair import validate_pair
    from audit_legacy_detector_full_agents3 import audit_trial_requests
    plan, cfg = read(out / 'plan.json'), read(out / 'execution_manifest.json')
    records = {p.stem: read(p) for p in (out / 'records').glob('*.json')}
    assert set(records) == {job['id'] for job in plan['jobs']}
    for job in plan['jobs']:
        record = records[job['id']]; assert record['job'] == job
        trace = Trace.load(out / record['trace'])
        assert trace.file_hash() == record['trace_sha256']
        evaluated = evaluate_trace(trace)
        assert evaluated.valid_trial and not evaluated.contract_violations
        assert evaluated.to_dict() == record['evaluation']
        from core5_ledger_plan import make_trial_spec
        controller = read(out / 'configs' / f"{job['task']}_agents3_full.json")
        case = cases[job['manifest']][job['index']]
        live = Trace.load(out / 'trials' / job['cell_id'] / f"{job['episode_id']}_live.jsonl") if job['mode'] == 'replay' else None
        expected_spec = make_trial_spec(case, controller, job['mode'], live.file_hash() if live else None)
        assert trace.header()['spec'] == expected_spec.raw
        assert evaluated.n_restarts == 0 and evaluated.binding == 'state_at_B'
        assert record['outcome']['Y'] == int(bool(evaluated.valid_at_receipt))
        scenario = expected_spec.manifest()
        assert audit_trial_requests(out, job['id'], 'agents3_full', trace, scenario, cfg) == record['request_audit']
        if job['mode'] == 'replay':
            assert record['pair_valid'] and validate_pair(trace, live).valid
    result = dict(state='passed', completed_trials=len(records), planned_trials=TOTAL,
        plan_sha256=digest(out / 'plan.json'), execution_manifest_sha256=digest(out / 'execution_manifest.json'),
        model_inference_calls=0, experimental_trials=0, qualification_trials=0)
    write(out / 'completion_audit.json', result)
    return result


def run(out):
    if read(out / 'status.json')['state'] != 'prepared' or any((out / 'records').iterdir()) or any((out / 'trials').iterdir()):
        raise RuntimeError('Only a fresh prepared run may start; automatic resume is disabled')
    handoff = read(out.parent / 'handoff.json')
    assert handoff['state'] == 'passed' and handoff['plan_sha256'] == digest(out / 'plan.json')
    assert handoff['execution_manifest_sha256'] == digest(out / 'execution_manifest.json')
    assert os.getppid() == handoff['queue_identity']['pid'] and same_process(handoff['queue_identity'])
    cases = validate(out); check_services(out)
    from agents3_full import FullAgentS3Adapter
    from core5_ledger_plan import make_trial_spec
    from chronotope.runner import run_trial, PayloadStore
    from chronotope.trace import Trace
    from chronotope.evaluate import evaluate_trace
    from chronotope.pair import validate_pair
    from audit_legacy_detector_full_agents3 import audit_trial_requests
    cfg, plan = read(out / 'execution_manifest.json'), read(out / 'plan.json')
    records = []; current = None

    def publish(state, error=None):
        summarize(out, records)
        write(out / 'status.json', dict(state=state, updated_at=now(), model=MODEL,
            completed_trials=len(records), planned_trials=TOTAL, qualification_trials=0,
            pid=os.getpid() if state in ('running', 'auditing') else None, current_job=current,
            error=error, automatic_resume=False))

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for current in plan['jobs']:
            publish('running')
            task, mode, ident = current['task'], current['mode'], current['id']
            folder = out / 'trials' / current['cell_id']; folder.mkdir(exist_ok=True)
            path = folder / f"{current['episode_id']}_{mode}.jsonl"
            assert not path.exists(), 'Never overwrite an interrupted trial'
            live = Trace.load(folder / f"{current['episode_id']}_live.jsonl") if mode == 'replay' else None
            controller_config = read(out / 'configs' / f'{task}_agents3_full.json')
            spec = make_trial_spec(cases[current['manifest']][current['index']], controller_config,
                                   mode, live.file_hash() if live else None)
            scenario = spec.manifest(); start = time.monotonic()
            adapter = FullAgentS3Adapter(cfg['agent_s3_components'], out / 'calls' / f'{ident}.jsonl')
            controller = make_controller(spec, task, adapter)
            try:
                run_trial(spec, path, PayloadStore(out / 'payloads'), live_trace=live,
                          controller=controller, manifest=scenario)
            finally:
                adapter.close()
            trace = Trace.load(path)
            if trace.all('decision_failed') or adapter.errors:
                raise RuntimeError('Infrastructure failure; retain trace unscored')
            evaluated = evaluate_trace(trace); pair = validate_pair(trace, live) if live else None
            if not evaluated.valid_trial or evaluated.contract_violations or (pair and not pair.valid):
                raise RuntimeError('Invalid trace or Replay pair; retain evidence unscored')
            requests = audit_trial_requests(out, ident, 'agents3_full', trace, scenario, cfg)
            record = dict(job=current, evaluation=evaluated.to_dict(),
                outcome={'Y': int(bool(evaluated.valid_at_receipt)), 'invalid_trial': False,
                         'contract_violations': [], 'R': int(bool(trace.all('receipt')))},
                request_audit=requests, trace=str(path.relative_to(out)), trace_sha256=trace.file_hash(),
                pair_valid=pair.valid if pair else None, completed_at=now(), wall_seconds=time.monotonic()-start)
            write(out / 'records' / f'{ident}.json', record); records.append(record)
            print(f"{len(records)}/{TOTAL} {ident} success={record['outcome']['Y']}", flush=True)
        current = None; publish('auditing'); audit(out); publish('complete')
    except BaseException as exc:
        publish('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', f'{type(exc).__name__}: {exc}')
        raise


def queue(out):
    root = out.parent; cfg = read(root / 'queue_manifest.json')
    lock = (root / 'queue.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'handoff.json').exists() or (root / 'successor_launch.json').exists():
        raise RuntimeError('A prior handoff cannot be resumed automatically')
    predecessor = Path(cfg['predecessor']); child = None; gpu_locks = []

    def publish(state, **extra):
        write(root / 'queue_status.json', dict(state=state, updated_at=now(), pid=os.getpid(),
            predecessor=str(predecessor), successor=str(out), planned_trials=TOTAL,
            automatic_resume=False, qualification_trials=0, **extra))

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        validate(out)
        while True:
            status = read(predecessor / 'status.json')
            state = status['state']
            if state in ('failed', 'stopped', 'paused', 'blocked', 'complete_scoped'):
                publish('blocked_predecessor', predecessor_state=state,
                        reason='Successful completion required; no automatic resume')
                return
            if state == 'complete' and not same_process(cfg['predecessor_identity']):
                break
            if state not in ('running', 'auditing', 'complete') or not same_process(cfg['predecessor_identity']):
                raise RuntimeError('Predecessor is no longer running and has not completed successfully')
            publish('waiting_for_predecessor', predecessor_state=state,
                    predecessor_completed_trials=status['completed_trials'], predecessor_planned_trials=3000)
            time.sleep(30)
        assert digest(predecessor / 'plan.json') == cfg['predecessor_plan_sha256']
        assert digest(predecessor / 'execution_manifest.json') == cfg['predecessor_execution_manifest_sha256']
        assert digest(cfg['predecessor_runner']) == cfg['predecessor_runner_sha256']
        # Lock both the canonical study lock and the hosted runner's frozen-root lock.
        locked_inodes = set()
        for path in (ROOT / 'runs/legacy_d1_gpu.lock', Path(cfg['predecessor_runner']).parents[1] / 'runs/legacy_d1_gpu.lock'):
            inode = (path.stat().st_dev, path.stat().st_ino)
            if inode in locked_inodes: continue
            handle = path.open('a+'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); gpu_locks.append(handle)
            locked_inodes.add(inode)
        publish('auditing_predecessor')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        with (root / 'predecessor_completion_audit.log').open('xb') as log:
            child = subprocess.Popen([PYTHON, '-B', cfg['predecessor_runner'], 'audit', '--out', str(predecessor)],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if child.wait() != 0: raise RuntimeError('Predecessor evidence audit failed')
        child = None
        evidence = read(predecessor / 'completion_audit.json')
        assert completion_gate(read(predecessor / 'status.json'), evidence,
            cfg['predecessor_plan_sha256'], cfg['predecessor_execution_manifest_sha256'])
        publish('checking_successor_configuration')
        validate(out); check_services(out)
        write(root / 'handoff.json', dict(state='passed', checked_at=now(), predecessor_audit=evidence,
            plan_sha256=digest(out / 'plan.json'), execution_manifest_sha256=digest(out / 'execution_manifest.json'),
            predecessor_process_exited=True, qualification_trials=0, queue_identity=process_identity(os.getpid())))
        command = [PYTHON, '-B', '-u', str(out / 'code' / Path(__file__).name), 'run', '--out', str(out)]
        with (root / 'evaluation.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(h.fileno() for h in gpu_locks))
            write(root / 'successor_launch.json', dict(pid=child.pid, command=command, started_at=now(),
                planned_trials=TOTAL, qualification_trials=0, successor_jobs=[]))
            publish('running_successor', successor_pid=child.pid)
            returncode = child.wait()
        child = None
        if returncode != 0 or read(out / 'status.json')['state'] != 'complete':
            raise RuntimeError(f'Successor failed or stopped (exit {returncode}); no automatic resume')
        publish('complete', completed_trials=TOTAL)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate(); child.wait()
        publish('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for handle in gpu_locks: handle.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'queue', 'run', 'audit'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); out = args.out.resolve()
    if args.action == 'prepare': prepare(out)
    elif args.action == 'validate':
        validate(out); print(json.dumps(dict(state='passed', spec_checks=TOTAL, model_inference_calls=0,
                                           experimental_trials=0, qualification_trials=0)))
    elif args.action == 'queue': queue(out)
    elif args.action == 'run': run(out)
    else: print(json.dumps(audit(out)))
