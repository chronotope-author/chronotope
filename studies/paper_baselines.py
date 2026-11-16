"""Prepare/check/run explicit core-five baseline plans; no qualifications or queues.

Preparation accepts existing case JSONL files and preserves their base_spec.
Nothing in this module manages GPU services or resumes an existing output folder.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from chronotope.hashing import hash_obj
from chronotope.paper_baselines.common import SetupError, TASK_NAMES
from chronotope.paper_baselines.factory import (check_config, check_scope, build,
                                                digest, register, LABELS)
from chronotope.paper_baselines.upstream import LOCK_PATH, locked_file


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def fetch_sources(destination):
    destination.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK_PATH.read_text())
    for name in ('aoi', 'mobiledreamer'):
        repo, target = lock[name], destination / name
        if not target.exists():
            subprocess.run(['git', 'clone', '--no-checkout', repo['url'], str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', repo['revision']], check=True)
        # Never reset a user's existing checkout.
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != repo['revision']:
            raise SetupError(f'Existing checkout differs: {target}')
        for relative in repo['files']:
            locked_file(target, name, relative)
        print(f'{name}: {actual} {target}')


def code_hashes():
    files = list((ROOT / 'src/chronotope').rglob('*.py')) + [LOCK_PATH, Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def case_manifest(case, task):
    """The selected Notification cases have a distributional renderer overlay."""
    value = case.get('distributional_row')
    if value is None:
        return None, None
    from chronotope.distributional.manifest import (
        EpisodeRow, build_manifest, proxy_scenario, verify_row_against_manifest)
    if task != 'notification' or value.get('task_family') != 'closing_window':
        raise SetupError('Only the existing Notification distributional overlay is supported')
    row = EpisodeRow.from_dict(value)
    if proxy_scenario(row) != case['base_spec']['scenario']:
        raise SetupError('Case proxy differs from the distributional row')
    if row.delivery != case['base_spec']['delivery'] or row.publication_delay_ns != case['base_spec']['observation']['publication_delay_ns']:
        raise SetupError('Distributional timing differs from the base case')
    manifest = build_manifest(row)
    verify_row_against_manifest(row, manifest)
    if not case.get('distributional_manifest_hash'):
        raise SetupError('Missing original distributional manifest provenance')
    return row, manifest


def prepare(case_file, config_file, task, instruction, out):
    register()
    from chronotope.spec import TrialSpec
    config = json.loads(config_file.read_text())
    check_config(config)
    cases = [json.loads(line) for line in case_file.read_text().splitlines() if line.strip()]
    if not cases:
        raise SetupError('Empty case manifest')
    # Validate every future evaluation cell before making a runnable plan.
    controller = dict(kind='paper_baseline', baseline=config['baseline'], task=task,
        instruction=instruction, config_path=str((out / 'config.json').resolve()),
        config_sha256=digest(config_file))
    specs = []
    for index, case in enumerate(cases):
        case_manifest(case, task)
        for mode in ('live', 'replay', 'paused'):
            raw = copy.deepcopy(case['base_spec'])
            raw.update(controller=controller, task_clock_policy=mode,
                       replay_source_live_trace_sha256='0' * 64 if mode == 'replay' else None)
            TrialSpec.from_dict(raw)
            specs.append((f'{index:04d}_{mode}', raw, f'{index:04d}_live' if mode == 'replay' else None, index))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_bytes(config_file.read_bytes())
    (out / 'cases.jsonl').write_bytes(case_file.read_bytes())
    (out / 'specs').mkdir()
    jobs = []
    for ident, raw, replay_of, index in specs:
        check_scope(raw)
        path = out / 'specs' / f'{ident}.json'
        write(path, raw)
        jobs.append(dict(id=ident, task=task, phase='evaluation', spec=str(path.relative_to(out)),
                         spec_sha256=digest(path), replay_of=replay_of, case_index=index))
    plan = dict(schema='paper-baselines-plan-1', qualification_trials=0, successor_jobs=[],
        automatic_resume=False, source_cases=str(case_file.resolve()), source_cases_sha256=digest(case_file),
        task_label=TASK_NAMES[task], baseline=config['baseline'], baseline_label=LABELS[config['baseline']], jobs=jobs,
        planned_trials=len(jobs), code_sha256=code_hashes())
    write(out / 'plan.json', plan)
    validate(out)
    print(f'Prepared {len(jobs)} evaluation cells; no model calls: {out}')


def validate(out):
    register()
    from chronotope.spec import TrialSpec
    plan = json.loads((out / 'plan.json').read_text())
    if (plan.get('schema') != 'paper-baselines-plan-1' or plan.get('qualification_trials') != 0
            or plan.get('successor_jobs') != [] or plan.get('automatic_resume') is not False):
        raise SetupError('Unsupported plan/qualification/queue configuration')
    if plan['code_sha256'] != code_hashes():
        raise SetupError('Baseline code changed since preparation; prepare a fresh plan')
    if not plan['jobs'] or len(plan['jobs']) != plan['planned_trials']:
        raise SetupError('Empty plan or wrong trial count')
    if digest(out / 'cases.jsonl') != plan['source_cases_sha256']:
        raise SetupError('Frozen cases changed')
    cases = [json.loads(line) for line in (out / 'cases.jsonl').read_text().splitlines() if line.strip()]
    verified_cases = set()
    seen = {}
    for job in plan['jobs']:
        if job['phase'] != 'evaluation' or job['id'] in seen or not job['id'].replace('_', '').isalnum():
            raise SetupError('Invalid or duplicate evaluation job')
        path = (out / job['spec']).resolve()
        if not path.is_relative_to(out.resolve()) or digest(path) != job['spec_sha256']:
            raise SetupError('Spec location/hash mismatch')
        spec = TrialSpec.from_dict(json.loads(path.read_text()))
        config = check_scope(spec.raw)
        check_config(config)
        if job['task'] != spec.raw['controller']['task'] or config['baseline'] != plan['baseline']:
            raise SetupError('Job task/baseline mismatch')
        index = job['case_index']
        if type(index) is not int or not 0 <= index < len(cases):
            raise SetupError('Invalid frozen case index')
        case = cases[index]
        if index not in verified_cases:
            case_manifest(case, job['task'])
            verified_cases.add(index)
        expected = copy.deepcopy(case['base_spec'])
        expected.update(controller=spec.raw['controller'], task_clock_policy=spec.policy,
                        replay_source_live_trace_sha256=spec.raw['replay_source_live_trace_sha256'])
        if expected != spec.raw:
            raise SetupError('Evaluation spec changed the frozen case')
        if spec.policy == 'replay':
            prior = seen.get(job['replay_of'])
            if prior is None or prior.policy != 'live' or prior.matching_key() != spec.matching_key():
                raise SetupError('Replay needs a prior matching Live case in this plan')
        elif job['replay_of'] is not None:
            raise SetupError('Only Replay may have a source')
        seen[job['id']] = spec
    return plan


def run(out):
    from chronotope.runner import PayloadStore, run_trial
    from chronotope.spec import TrialSpec
    from chronotope.trace import Trace
    from chronotope.evaluate import evaluate_trace
    plan = validate(out)
    cases = [json.loads(line) for line in (out / 'cases.jsonl').read_text().splitlines() if line.strip()]
    execution = out / 'execution'
    execution.mkdir(exist_ok=False)  # refuses automatic resume/duplicate execution
    store = PayloadStore(execution / 'payloads')
    records = []
    write(execution / 'status.json', {'state': 'running', 'completed': 0})
    try:
        for job in plan['jobs']:
            raw = json.loads((out / job['spec']).read_text())
            live = None
            if job['replay_of']:
                live = Trace.load(execution / (job['replay_of'] + '.jsonl'))
                raw['replay_source_live_trace_sha256'] = live.file_hash()
            spec = TrialSpec.from_dict(raw)
            case = cases[job['case_index']]
            row, manifest = case_manifest(case, job['task'])
            path = execution / (job['id'] + '.jsonl')
            controller = build(spec)
            try:
                run_trial(spec, path, store, live_trace=live, controller=controller, manifest=manifest)
            finally:
                # Preserve requests and infrastructure errors even if the decision
                # raised before an invocation record could be written.
                calls = {'planner': controller.model.calls}
                if hasattr(controller, 'world_model'):
                    calls['world_model'] = controller.world_model.calls
                write(execution / (job['id'] + '.calls.json'), calls)
            trace = Trace.load(path)
            if trace.all('decision_failed'):
                raise SetupError(f'Infrastructure error in {job["id"]}; trace retained unscored')
            result = evaluate_trace(trace)
            if not result.valid_trial or result.contract_violations:
                raise SetupError(f'Trace contract error in {job["id"]}: {result.contract_violations}')
            outcome = None
            if row is not None:
                from chronotope.distributional.episode import score_trace, verify_trace_evidence
                provenance = case['distributional_manifest_hash']
                verify_trace_evidence(row, manifest, trace, provenance)
                outcome = score_trace(row, trace, provenance, controller=plan['baseline']).to_dict()
            records.append(dict(id=job['id'], task=TASK_NAMES[job['task']], baseline=plan['baseline'],
                                baseline_label=LABELS[plan['baseline']],
                                mode=spec.policy, status='scored', evaluation=result.to_dict(),
                                distributional_outcome=outcome))
            write(execution / 'records.json', records)
            write(execution / 'status.json', dict(state='running', completed=len(records), planned=len(plan['jobs'])))
    except BaseException as exc:
        write(execution / 'status.json', dict(state='stopped', completed=len(records), planned=len(plan['jobs']),
            current_job=job['id'] if 'job' in locals() else None, error=f'{type(exc).__name__}: {exc}',
            automatic_resume=False))
        raise
    write(execution / 'status.json', dict(state='complete', completed=len(records), planned=len(plan['jobs'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('fetch-sources')
    fetch.add_argument('--out', type=Path, default=ROOT / 'runs/paper_baseline_sources')
    prep = sub.add_parser('prepare')
    prep.add_argument('--cases', type=Path, required=True)
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--task', choices=tuple(TASK_NAMES), required=True)
    prep.add_argument('--instruction-file', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    for name in ('check', 'run'):
        sub.add_parser(name).add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fetch-sources':
        fetch_sources(args.out)
    elif args.command == 'prepare':
        prepare(args.cases, args.config, args.task, args.instruction_file.read_text().strip(), args.out)
    elif args.command == 'check':
        plan = validate(args.out)
        print(f'Validated {plan["planned_trials"]} cases without inference')
    else:
        run(args.out)


if __name__ == '__main__':
    main()
