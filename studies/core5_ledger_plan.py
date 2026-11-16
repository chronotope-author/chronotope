"""CPU-only case planning for the authorized core-five experiment ledger.

No evaluation, model calls, qualification episodes, or service operations occur
here. Notification keeps the current frozen distributional cases. Other tasks
retain historical scenario definitions, varying only the requested parameter
and the predeclared evaluation seeds.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / 'runs/legacy_detector_full_agents3_20261029/experiment'
ARMS = ('legacy_cpp_detector', 'agents3_full')
MODES = ('live', 'replay', 'paused')
WINDOWS = (16., 8., 4., 2., 1., .5, .25, .125)
LABELS = {'notification': 'Disappearing Notification', 'drift': 'Moving Balloon',
          'code': 'Expiring Authenticator Code', 'button': 'Displaced Button',
          'trading': 'Reversing Trading Signal'}
SOURCES = {
    'drift': 'runs/c1_qwen38_nothink_100/live_v8_frozen_s1000.jsonl',
    'code': 'runs/c4_qwen38_nothink_100/live_ch2_frozen_s1000.jsonl',
    'button': 'runs/g1_reflow_qwen38_nothink_100/live_ch2_frozen_s1000.jsonl',
    'trading': 'runs/b1_qwen38_nothink_100/prl_r2000_s204_live.jsonl',
}
PARAMETERS = {
    'drift': ('velocity_x_px_per_s', 'speed', 'px/s', 'v'),
    'code': ('change_delay_ns', 'code change delay', 's', 'ch'),
    'button': ('change_delay_ns', 'banner insertion delay', 's', 'ch'),
    'trading': ('reversal_delay_ns', 'signal reversal delay', 's', 'r'),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def setup():
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'chronotope/python')]


def make_trial_spec(case, controller, mode, live_hash=None):
    """Construct a real validated spec; controller selection belongs to runner."""
    from chronotope.spec import TrialSpec
    if controller['kind'] == 'chronotope':
        from chronotope_core.glue import register
        register()
    assert mode in MODES
    if mode == 'replay':
        assert isinstance(live_hash, str) and len(live_hash) == 64
    else:
        assert live_hash is None
    raw = copy.deepcopy(case['base_spec'])
    raw.update(controller=copy.deepcopy(controller), task_clock_policy=mode,
               replay_source_live_trace_sha256=live_hash if mode == 'replay' else None)
    return TrialSpec.from_dict(raw)


def historical_cells(tasks=None):
    for task, source in SOURCES.items():
        if tasks is not None and task not in tasks:
            continue
        with (ROOT / source).open() as stream:
            header = json.loads(next(stream))
        template = copy.deepcopy(header['spec'])
        assert template['scenario']['response_window_ns'] == 16_000_000_000
        # The ledger includes changes/reversals at E+16s. Keep those events
        # inside the episode rather than aliasing them to a no-change control.
        if task in ('code', 'button', 'trading'):
            template['scenario']['response_window_ns'] = 32_000_000_000
            template['scenario']['episode_horizon_ns'] += 16_000_000_000
        field, name, unit, prefix = PARAMETERS[task]
        for value in ([2, 4, 8, 16, None] if task == 'trading' else [2, 4, 8, 16]):
            cell_id = f'{task}_{prefix}{value if value is not None else "none"}'
            cell = {'cell_id': cell_id, 'task': task, 'task_label': LABELS[task],
                    'family': template['scenario']['scenario_id'], 'setting_name': name,
                    'setting_value': value, 'setting_unit': unit,
                    'window_s': template['scenario']['response_window_ns'] / 1e9,
                    'image_scale': template['controller']['image_scale'],
                    'legacy_prompt_version': template['controller']['prompt_version'],
                    'source': source, 'source_sha256': digest(ROOT / source)}
            cases = []
            for index in range(100):
                raw = copy.deepcopy(template)
                raw.pop('controller')
                raw['task_clock_policy'] = 'live'
                raw['replay_source_live_trace_sha256'] = None
                raw['scenario']['scenario_seed'] = seed = 1000 + index
                raw['scenario'][field] = value if task == 'drift' else (0 if value is None else value * 1_000_000_000)
                case = {'index': index, 'episode_id': f'{task}-{seed}', 'seed': seed,
                        'base_spec': raw, 'distributional_row': None,
                        'distributional_manifest_hash': None}
                # Geometry/spec validation uses the original controller schema,
                # not a trial or an assertion of pending native capabilities.
                make_trial_spec(case, template['controller'], 'live')
                cases.append(case)
            yield cell, cases


def notification_cells(windows=WINDOWS):
    from chronotope.distributional.manifest import ManifestSet, build_manifest, verify_row_against_manifest
    from chronotope.distributional.episode import ControllerSpec, trial_spec_for
    plan = read(PREDECESSOR / 'plan.json')
    controller = ControllerSpec('source_only', read(PREDECESSOR / 'configs/agents3_full.json'))
    for window in windows:
        name = f'evaluation_notification_w{round(window * 1e9)}.jsonl'
        source = PREDECESSOR / 'manifests' / name
        manifest = ManifestSet.load(source)
        assert manifest.frozen_hash == plan['manifest_hashes'][name] and len(manifest.rows) == 100
        cell_id = f'notification_w{round(window * 1e9)}'
        cell = {'cell_id': cell_id, 'task': 'notification', 'task_label': LABELS['notification'],
                'family': 'closing_window', 'setting_name': 'response window', 'setting_value': window,
                'setting_unit': 's', 'window_s': window, 'image_scale': 2,
                'legacy_prompt_version': 4, 'source': str(source), 'source_sha256': digest(source),
                'distributional_manifest_hash': manifest.frozen_hash}
        cases = []
        for index, row in enumerate(manifest.rows):
            assert row.task_family == 'closing_window' and row.deadline_s == window
            verify_row_against_manifest(row, build_manifest(row))
            raw = copy.deepcopy(trial_spec_for(row, controller, 'live', None).raw)
            raw.pop('controller')
            cases.append({'index': index, 'episode_id': row.episode_id, 'seed': row.environment_seed,
                          'base_spec': raw, 'distributional_row': row.to_dict(),
                          'distributional_manifest_hash': manifest.frozen_hash})
        yield cell, cases


def job_id(job):
    return f"{job['arm']}_{job['cell_id']}_{job['index']:03d}_{job['mode']}"


def validate_plan(plan):
    assert plan['schema'] == 'core5-ledger-plan-1'
    assert plan['arms'] == list(ARMS) and plan['modes'] == list(MODES)
    assert plan['qualification_trials'] == 0 and plan['trials_per_cell'] == 100
    selected = plan.get('selected_tasks', list(LABELS) if plan['include_notification'] else list(SOURCES))
    assert selected and len(selected) == len(set(selected)) and set(selected) <= set(LABELS)
    assert plan['include_notification'] == ('notification' in selected)
    windows = plan.get('notification_windows_s', list(WINDOWS) if plan['include_notification'] else [])
    assert len(windows) == len(set(windows)) and set(windows) <= set(WINDOWS)
    assert bool(windows) == plan['include_notification']
    task_counts = {'notification': len(windows), 'drift': 4, 'code': 4, 'button': 4, 'trading': 5}
    expected_counts = {task: task_counts[task] for task in selected}
    cells = {c['cell_id']: c for c in plan['cells']}
    assert len(cells) == len(plan['cells']) == sum(expected_counts.values())
    assert {c['task'] for c in cells.values()} == set(selected)
    assert Counter(c['task'] for c in cells.values()) == Counter(expected_counts)
    assert {c['window_s'] for c in cells.values() if c['task'] == 'notification'} == set(windows)
    assert len(plan['jobs']) == plan['planned_trials'] == len(cells) * 600
    seen = set(); groups = defaultdict(set)
    for job in plan['jobs']:
        cell = cells[job['cell_id']]
        assert job['id'] == job_id(job) and job['id'] not in seen
        assert job['phase'] == 'evaluation' and job['model'] == plan['model_key']
        assert job['arm'] in ARMS and job['mode'] in MODES and type(job['index']) is int and 0 <= job['index'] < 100
        assert all(job[k] == cell[k] for k in ('task', 'family', 'window_s'))
        assert job['manifest'] == f"cases/{job['cell_id']}.jsonl"
        if job['mode'] == 'replay':
            assert job_id(dict(job, mode='live')) in seen
        seen.add(job['id']); groups[job['cell_id'], job['arm'], job['mode']].add(job['index'])
    assert len(groups) == len(cells) * 6 and all(indices == set(range(100)) for indices in groups.values())
    return len(plan['jobs'])


def load_cases(out, plan=None):
    out = Path(out); plan = plan or read(out / 'plan.json'); validate_plan(plan)
    result = {}
    for name, expected in plan['case_file_sha256'].items():
        path = out / name
        assert digest(path) == expected
        values = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(values) == 100 and [case['index'] for case in values] == list(range(100))
        result[name] = values
    for job in plan['jobs']:
        case = result[job['manifest']][job['index']]
        assert case['episode_id'] == job['episode_id'] and case['seed'] == job['seed']
    return result


def prepare_cohort(out, model_key, model_id, revision, include_notification, *, tasks=None, notification_windows=None):
    setup(); out = Path(out).resolve()
    assert not (out / 'plan.json').exists(), 'Never overwrite a prepared/frozen cohort'
    assert len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
    selected = list(tasks) if tasks is not None else list(LABELS if include_notification else SOURCES)
    assert selected and len(selected) == len(set(selected)) and set(selected) <= set(LABELS)
    include_notification = 'notification' in selected
    windows = list(notification_windows) if notification_windows is not None else list(WINDOWS) if include_notification else []
    assert len(windows) == len(set(windows)) and set(windows) <= set(WINDOWS)
    assert bool(windows) == include_notification
    sources = list(notification_cells(windows)) if include_notification else []
    sources.extend(historical_cells(selected))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cases').mkdir(exist_ok=True)
    assert not any((out / 'cases').iterdir()), 'Never overwrite existing case files'
    plan = {'schema': 'core5-ledger-plan-1', 'model_key': model_key, 'model': model_id, 'revision': revision,
            'arms': list(ARMS), 'modes': list(MODES), 'trials_per_cell': 100, 'include_notification': include_notification,
            'selected_tasks': selected, 'notification_windows_s': windows,
            'planned_trials': len(sources) * 600, 'qualification_trials': 0, 'cells': [], 'jobs': [],
            'case_file_sha256': {}, 'predecessor': str(PREDECESSOR),
            'notification_source_plan_sha256': digest(PREDECESSOR / 'plan.json'),
            'historical_seed_range': [1000, 1099], 'generator_sha256': digest(__file__),
            'interpretation': 'Setting values mean response windows only for Notification. Drift keeps a 16-second window; code, button and trading use a fixed 32-second window so changes at E+16s precede expiry.',
            'forbidden_variants': ['jump', 'select'], 'successor_jobs': [],
            'native_code_capability': 'Must be verified by execution preparation; case validation does not establish native typing support.'}
    for cell, cases in sources:
        name = f"cases/{cell['cell_id']}.jsonl"
        path = out / name
        path.write_text(''.join(json.dumps(case, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n' for case in cases))
        plan['cells'].append(cell); plan['case_file_sha256'][name] = digest(path)
        for case in cases:
            for arm in ARMS:
                for mode in MODES:
                    job = dict(phase='evaluation', model=model_key, arm=arm, mode=mode,
                               index=case['index'], seed=case['seed'], episode_id=case['episode_id'],
                               cell_id=cell['cell_id'], task=cell['task'], family=cell['family'],
                               window_s=cell['window_s'], manifest=name)
                    job['id'] = job_id(job); plan['jobs'].append(job)
    validate_plan(plan); write(out / 'plan.json', plan); load_cases(out, plan)
    write(out / 'cohort_verification.json', {'state': 'passed', 'planned_trials': plan['planned_trials'],
        'case_count': len(sources) * 100, 'task_cells': dict(Counter(c['task'] for c in plan['cells'])),
        'plan_sha256': digest(out / 'plan.json'), 'model_inference_calls': 0, 'experimental_trials': 0,
        'qualification_trials': 0, 'geometry_spec_checks': len(sources) * 100,
        'controller_capability_verified': False})
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-key', required=True); parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True); parser.add_argument('--include-notification', action='store_true')
    parser.add_argument('--tasks', nargs='+', choices=tuple(LABELS))
    parser.add_argument('--notification-windows', nargs='+', type=float, choices=WINDOWS)
    args = parser.parse_args()
    plan = prepare_cohort(args.out, args.model_key, args.model, args.revision, args.include_notification,
                          tasks=args.tasks, notification_windows=args.notification_windows)
    print(json.dumps({'state': 'prepared', 'planned_trials': plan['planned_trials'], 'experimental_trials': 0}))
