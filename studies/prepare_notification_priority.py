"""Prepare the authorized fresh Gemma Notification comparison without inference."""
import argparse
from pathlib import Path

from core5_ledger_plan import load_cases, validate_plan
from run_core5_detector_full_agents3 import MODEL_SETTINGS, ROOT, execution_jobs, prepare, read
from legacy_d1_w8 import digest, now, write

DEFAULT_ROOT = ROOT / 'runs/notification_gemma4_26b_20261030'
WINDOWS = (16., 8., 4., 2., 1.)


def prepare_priority(root):
    root = Path(root).resolve(); out = root / 'experiment'
    assert not (out / 'execution_manifest.json').exists(), 'Never change a frozen experiment'
    assert not list((out / 'records').glob('*.json')), 'Fresh results required'
    prepare(out, 'gemma4_26b_a4b', True, tasks=['notification'], notification_windows=WINDOWS,
            engine_directory=root / 'engines')
    plan = read(out / 'plan.json')
    assert validate_plan(plan) == 3000
    assert {job['task'] for job in plan['jobs']} == {'notification'}
    assert {job['window_s'] for job in plan['jobs']} == set(WINDOWS)
    assert len(load_cases(out, plan)) == 5
    support = {'configuration_reviewed': True, 'code_action': 'defer_unsupported', 'user_authorized': False,
               'reason': 'Notification-only comparison does not require typed-action transport; controllers remain unchanged.'}
    (root / 'inputs').mkdir(exist_ok=True)
    write(root / 'inputs/action_support.json', support)
    ready, deferred = execution_jobs(plan, support)
    assert len(ready) == 3000 and not deferred
    model = MODEL_SETTINGS['gemma4_26b_a4b']
    proof = {'state': 'prepared', 'created_at': now(), 'model': model['id'], 'revision': model['revision'],
             'experiment': str(out), 'planned_trials': 3000, 'cases_per_window': 100,
             'tasks': ['notification'], 'windows_s': list(WINDOWS), 'arms': plan['arms'], 'modes': plan['modes'],
             'model_inference_calls': 0, 'experimental_trials': 0, 'qualification_trials': 0,
             'plan_sha256': digest(out / 'plan.json'), 'case_file_sha256': plan['case_file_sha256'],
             'notification_source_plan_sha256': plan['notification_source_plan_sha256'],
             'engine_directory': str(root / 'engines'), 'native_typing_required': False,
             'launch_state': 'not_launched', 'model_services_changed': False}
    write(root / 'priority_cohort_preparation.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = prepare_priority(args.root)
    print(f"Prepared {result['planned_trials']} Notification trials; no inference or GPU work performed.")
