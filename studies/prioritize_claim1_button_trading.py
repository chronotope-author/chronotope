"""Run the untouched Button/Trading cases first after explicit user reordering.

The failed Code study and all of its evidence remain immutable. Only scheduling
scope changes; frozen agent, renderer, model, grounder, and scoring code are kept.
"""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from contextlib import ExitStack
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/claim1_agents3_qwen08_remaining3_20261105/experiment'
RUN = ROOT / 'runs/claim1_agents3_qwen08_button_trading_20261106'
OUT = RUN / 'experiment'
PYTHON = (_CHRONOTOPE_HOME + '/agents3-venv/bin/python')
RUNNER_NAME = 'run_claim1_agents3_button_trading.py'
TASKS = ('button', 'trading')


def load_study(path):
    spec = importlib.util.spec_from_file_location('claim1_priority_study', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare():
    if RUN.exists():
        raise FileExistsError('Never overwrite a prepared priority run')
    source = load_study(SOURCE / 'code/run_claim1_agents3_remaining3.py')
    source.validate(SOURCE)
    source_status = source.read(SOURCE / 'status.json')
    assert source_status['state'] == 'failed' and source_status['completed_trials'] == 192
    old_records = [source.read(p) for p in (SOURCE / 'records').glob('*.json')]
    assert len(old_records) == 192 and {r['job']['task'] for r in old_records} == {'code'}
    assert not source.process_identity(source.read(SOURCE.parent / 'successor_launch.json')['pid'])
    assert not source.process_identity(source.read(SOURCE.parent / 'dependency_skip_launch.json')['pid'])
    OUT.mkdir(parents=True)
    shutil.copytree(SOURCE / 'code', OUT / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(SOURCE / 'configs', OUT / 'configs')
    for name in ('cases', 'records', 'trials', 'calls', 'payloads'):
        (OUT / name).mkdir()
    original_plan = source.read(SOURCE / 'plan.json')
    plan = copy.deepcopy(original_plan)
    plan['cells'] = [c for c in original_plan['cells'] if c['task'] in TASKS]
    plan['jobs'] = [j for j in original_plan['jobs'] if j['task'] in TASKS]
    plan['case_file_sha256'] = {c['manifest']: original_plan['case_file_sha256'][c['manifest']]
                               for c in plan['cells']}
    for name in plan['case_file_sha256']:
        shutil.copy2(SOURCE / name, OUT / name)
    assert len(plan['jobs']) == 3000
    assert [j['task'] for j in plan['jobs']] == ['button'] * 1500 + ['trading'] * 1500
    plan.update(schema='claim1-agents3-button-trading-priority-1', planned_trials=3000,
        original_planned_trials=4500, source_plan=str(SOURCE / 'plan.json'),
        source_plan_sha256=source.digest(SOURCE / 'plan.json'),
        deferred_tasks=['code'], deferred_uncompleted_code_trials=1308,
        authorization='User instruction, November 6, 2026: run the others first',
        task_order=['button', 'trading'], automatic_resume=False, successor_jobs=[])
    source.write(OUT / 'plan.json', plan)
    instructions = source.read(OUT / 'configs/task_instructions.json')
    source.write(OUT / 'configs/task_instructions.json', {t: instructions[t] for t in TASKS})
    # Scope-only derivative of the audited runner. Its inference, controller,
    # action bridge, per-trial loop, request audit, and scorer remain byte-for-byte.
    original_runner = (SOURCE / 'code/run_claim1_agents3_remaining3.py').read_text()
    text = original_runner[:original_runner.index('\ndef queue(')]
    prepare_start, validate_start = text.index('\ndef prepare('), text.index('\ndef validate(')
    text = text[:prepare_start] + text[validate_start:]
    replacements = {
        "TASKS = {'code': 'Expiring Authenticator Code', 'button': 'Displaced Button',\n         'trading': 'Reversing Trading Signal'}":
        "TASKS = {'button': 'Displaced Button', 'trading': 'Reversing Trading Signal'}",
        'TOTAL = 4500': 'TOTAL = 3000',
        "assert len(plan['cells']) == 15": "assert len(plan['cells']) == 10",
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    text += '''\nif __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'run', 'audit'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); out = args.out.resolve()
    if args.action == 'validate':
        validate(out)
        print(json.dumps(dict(state='passed', spec_checks=3000, model_inference_calls=0,
                              experimental_trials=0, qualification_trials=0)))
    elif args.action == 'run': run(out)
    else: print(json.dumps(audit(out)))
'''
    (OUT / 'code' / RUNNER_NAME).write_text(text)
    shutil.copy2(__file__, RUN / 'priority_supervisor.py')
    auth = dict(created_at=source.now(), user_request='run the others first',
        task_order=['button', 'trading'], planned_trials=3000, qualification_trials=0,
        source_experiment=str(SOURCE), source_status=source_status,
        preserved_completed_code_trials=192, deferred_uncompleted_code_trials=1308,
        source_status_sha256=source.digest(SOURCE / 'status.json'),
        source_plan_sha256=source.digest(SOURCE / 'plan.json'),
        source_execution_manifest_sha256=source.digest(SOURCE / 'execution_manifest.json'),
        source_record_hashes={str(p): source.digest(p) for p in (SOURCE / 'records').glob('*.json')},
        source_interrupted_trace_sha256=source.digest(SOURCE / 'trials/code_change16s/code-1064_live.jsonl'),
        controller_policy_changes=False, service_changes=False, generation_changes=False,
        context_limit_changes=False, automatic_resume=False, successor_jobs=[],
        supervisor_sha256=source.digest(RUN / 'priority_supervisor.py'))
    source.write(RUN / 'authorization.json', auth)
    source.write(RUN / 'queue_manifest.json', dict(created_at=source.now(), out=str(OUT),
        task_order=['button', 'trading'], planned_trials=3000, qualification_trials=0,
        automatic_resume=False, successor_jobs=[], deferred_tasks=['code'],
        dependency='Explicit user priority supersedes failed Code predecessor',
        authorization_sha256=source.digest(RUN / 'authorization.json')))
    cfg = copy.deepcopy(source.read(SOURCE / 'execution_manifest.json'))
    cfg.update(created_at=source.now(), source_scope=str(SOURCE),
        source_scope_plan_sha256=source.digest(SOURCE / 'plan.json'),
        source_scope_execution_manifest_sha256=source.digest(SOURCE / 'execution_manifest.json'),
        active_tasks=['button', 'trading'], planned_trials=3000,
        scope_only_runner_changes=list(replacements.items()))
    cfg['hashes'] = {str(p.relative_to(OUT)): source.digest(p) for folder in ('code', 'configs', 'cases')
                     for p in sorted((OUT / folder).rglob('*')) if p.is_file()}
    cfg['hashes']['plan.json'] = source.digest(OUT / 'plan.json')
    cfg['queue_manifest_sha256'] = source.digest(RUN / 'queue_manifest.json')
    source.write(OUT / 'execution_manifest.json', cfg)
    source.write(OUT / 'status.json', dict(state='prepared', completed_trials=0, planned_trials=3000,
        qualification_trials=0, automatic_resume=False, pid=None, updated_at=source.now()))
    source.write(SOURCE.parent / 'schedule_amendment_button_trading_20261106.json',
                 dict(authorization=str(RUN / 'authorization.json'), active_experiment=str(OUT),
                      preserved_code_trials=192, deferred_code_trials=1308,
                      user_request='run the others first', original_evidence_unchanged=True))


def validate():
    study = load_study(OUT / 'code' / RUNNER_NAME)
    auth = study.read(RUN / 'authorization.json')
    assert auth['user_request'] == 'run the others first'
    assert study.digest(RUN / 'priority_supervisor.py') == auth['supervisor_sha256']
    assert study.digest(SOURCE / 'status.json') == auth['source_status_sha256']
    assert study.digest(SOURCE / 'plan.json') == auth['source_plan_sha256']
    assert study.digest(SOURCE / 'execution_manifest.json') == auth['source_execution_manifest_sha256']
    study.verify_hashes(auth['source_record_hashes'])
    assert study.digest(SOURCE / 'trials/code_change16s/code-1064_live.jsonl') == auth['source_interrupted_trace_sha256']
    plan, source_plan = study.read(OUT / 'plan.json'), study.read(SOURCE / 'plan.json')
    assert plan['jobs'] == [j for j in source_plan['jobs'] if j['task'] in TASKS]
    assert plan['cells'] == [c for c in source_plan['cells'] if c['task'] in TASKS]
    assert plan['qualification_trials'] == 0 and plan['successor_jobs'] == []
    for task in TASKS:
        assert study.digest(OUT / 'configs' / f'{task}_agents3_full.json') == study.digest(SOURCE / 'configs' / f'{task}_agents3_full.json')
    study.validate(OUT)
    return study


def run():
    study = validate(); child = None
    assert study.read(OUT / 'status.json')['state'] == 'prepared'
    assert not any((OUT / 'records').iterdir()) and not any((OUT / 'trials').iterdir())
    assert not (RUN / 'handoff.json').exists() and not (RUN / 'successor_launch.json').exists()

    def publish(state, **details):
        study.write(RUN / 'queue_status.json', dict(state=state, updated_at=study.now(),
            pid=os.getpid(), successor=str(OUT), planned_trials=3000, task_order=['button', 'trading'],
            deferred_tasks=['code'], automatic_resume=False, qualification_trials=0, **details))

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with ExitStack() as stack:
        locked = set(); locks = []
        paths = [RUN / 'queue.lock', SOURCE.parent / 'queue.lock', ROOT / 'runs/legacy_d1_gpu.lock',
            ROOT / 'runs/hosted_api_deepseek_nothinking_20261105/source/runs/legacy_d1_gpu.lock']
        for path in paths:
            handle = stack.enter_context(path.open('a+'))
            inode = (path.stat().st_dev, path.stat().st_ino)
            if inode in locked: continue
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locked.add(inode); locks.append(handle)
        try:
            publish('checking_configuration')
            study.check_services(OUT)
            study.write(RUN / 'handoff.json', dict(state='passed', checked_at=study.now(),
                authorization_sha256=study.digest(RUN / 'authorization.json'),
                plan_sha256=study.digest(OUT / 'plan.json'),
                execution_manifest_sha256=study.digest(OUT / 'execution_manifest.json'),
                queue_identity=study.process_identity(os.getpid()), qualification_trials=0,
                predecessor_dependency='Explicitly deferred Code; run Button then Trading'))
            command = [PYTHON, '-B', '-u', str(OUT / 'code' / RUNNER_NAME), 'run', '--out', str(OUT)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONPATH', None)
            with (RUN / 'evaluation.log').open('xb') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(h.fileno() for h in locks))
                study.write(RUN / 'successor_launch.json', dict(pid=child.pid, command=command,
                    started_at=study.now(), planned_trials=3000, qualification_trials=0, successor_jobs=[]))
                publish('running', evaluation_pid=child.pid)
                result = child.wait()
            child = None
            if result != 0 or study.read(OUT / 'status.json')['state'] != 'complete':
                raise RuntimeError(f'Evaluation failed or stopped (exit {result}); no automatic resume')
            publish('complete', completed_trials=3000)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate(); child.wait()
            publish('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed',
                    error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'run'))
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'validate':
        validate()
        print(json.dumps(dict(state='passed', spec_checks=3000, model_inference_calls=0,
                              experimental_trials=0, qualification_trials=0)))
    else: run()
