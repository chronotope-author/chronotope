"""CPU-only coverage for explicit scheduling; no controller/model is invoked."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'studies/resume_notification_schedule.py'
spec = importlib.util.spec_from_file_location('notification_schedule', MODULE)
scheduling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduling)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path):
    jobs = [{'phase': 'evaluation', 'task': 'notification', 'family': 'closing_window',
             'arm': arm, 'window_s': window, 'index': index, 'mode': mode}
            for window in (16., 8., 4., 2., 1., .5, .25, .125)
            for index in range(100)
            for arm in ('legacy_cpp_detector', 'agents3_full')
            for mode in ('live', 'replay', 'paused')]
    write(tmp_path / 'plan.json', {'planned_trials': 4800, 'qualification_trials': 0, 'jobs': jobs})
    write(tmp_path / 'execution_manifest.json', {'frozen': True})
    frozen = tmp_path / scheduling.FROZEN_RUNNER
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text('# frozen runner fixture; never executed\n')
    retained = [jobs[0]] + [job for job in jobs if job['window_s'] == .5][:2]
    for index, job in enumerate(retained):
        trace = tmp_path / f'traces/{index}.jsonl'
        trace.parent.mkdir(exist_ok=True)
        trace.write_text('{}\n')
        write(tmp_path / 'records' / (scheduling.job_id(job) + '.json'),
              {'job': job, 'trace': str(trace.relative_to(tmp_path)),
               'trace_sha256': scheduling.digest(trace),
               'outcome': {'invalid_trial': False, 'contract_violations': []}})
    write(tmp_path / 'status.json', {'state': 'stopped', 'completed_trials': 3,
                                   'planned_trials': 4800, 'pid': None})
    return tmp_path


def test_exact_subset_retains_completed_subsecond_and_immutable_inputs(experiment):
    originals = {name: scheduling.digest(experiment / name)
                 for name in ('plan.json', 'execution_manifest.json', scheduling.FROZEN_RUNNER)}
    schedule = scheduling.create_schedule(experiment, 'Save the <1s window runs for smaller models <=4B.')
    selected = scheduling.scheduled_jobs(experiment)
    assert len(selected) == scheduling.planned_count(experiment) == 3002
    assert schedule['deferred_trials'] == 1798
    assert schedule['deferred_windows_s'] == [.5, .25, .125]
    assert sum(job['window_s'] == .5 for job in selected) == 2
    assert {job['window_s'] for job in selected} == {16., 8., 4., 2., 1., .5}
    assert all(scheduling.digest(experiment / name) == digest for name, digest in originals.items())
    assert len(schedule['retained_completed_ids']) == 3
    assert scheduling.validate_schedule(experiment) == schedule


def test_explicit_half_second_only_subset_remains_supported(experiment):
    schedule = scheduling.create_schedule(experiment, 'Defer only 0.5s', deferred_windows_s=[.5])
    assert schedule['planned_trials'] == 4202
    assert schedule['deferred_trials'] == 598
    assert {job['window_s'] for job in scheduling.scheduled_jobs(experiment)} == scheduling.ORIGINAL_WINDOWS


@pytest.mark.parametrize('windows', [[], [1.], [.5, 2.], [.0625]])
def test_only_existing_subsecond_windows_can_be_deferred(experiment, windows):
    with pytest.raises(AssertionError):
        scheduling.create_schedule(experiment, 'Save subsecond runs for smaller models', deferred_windows_s=windows)
    assert not (experiment / scheduling.SCHEDULE_NAME).exists()


def test_no_schedule_keeps_original_count_and_jobs(experiment):
    assert scheduling.planned_count(experiment) == 4800
    assert len(scheduling.scheduled_jobs(experiment)) == 4800


@pytest.mark.parametrize('state', ['running', 'failed', 'complete'])
def test_create_requires_controlled_stop(experiment, state):
    write(experiment / 'status.json', {'state': state, 'completed_trials': 3})
    with pytest.raises(AssertionError):
        scheduling.create_schedule(experiment, 'Defer only 0.5s')
    assert not (experiment / scheduling.SCHEDULE_NAME).exists()


def test_create_requires_actual_authorization_and_never_overwrites(experiment):
    with pytest.raises(AssertionError):
        scheduling.create_schedule(experiment, '')
    scheduling.create_schedule(experiment, 'Defer only 0.5s')
    with pytest.raises(AssertionError):
        scheduling.create_schedule(experiment, 'Defer only 0.5s')


@pytest.mark.parametrize('field', ['original_plan_sha256', 'original_execution_sha256',
                                 'frozen_runner_sha256', 'helper_sha256'])
def test_every_provenance_pin_is_enforced(experiment, field):
    schedule = scheduling.create_schedule(experiment, 'Defer only 0.5s')
    schedule[field] = '0' * 64
    with pytest.raises(AssertionError):
        scheduling.validate_schedule(experiment, schedule)


def test_unknown_window_rejected_even_when_plan_hash_is_updated(experiment):
    schedule = scheduling.create_schedule(experiment, 'Defer only 0.5s')
    plan = scheduling.read(experiment / 'plan.json')
    plan['jobs'][-1]['window_s'] = 64.
    write(experiment / 'plan.json', plan)
    schedule['original_plan_sha256'] = scheduling.digest(experiment / 'plan.json')
    with pytest.raises(AssertionError):
        scheduling.validate_schedule(experiment, schedule)


def test_cannot_defer_another_window_or_drop_completed_records(experiment):
    original = scheduling.create_schedule(experiment, 'Defer only 0.5s')
    for key in ('scheduled_job_ids', 'retained_completed_ids'):
        schedule = dict(original, **{key: original[key][1:]})
        with pytest.raises(AssertionError):
            scheduling.validate_schedule(experiment, schedule)


def test_retained_result_modification_is_detected(experiment):
    schedule = scheduling.create_schedule(experiment, 'Defer only 0.5s')
    path = experiment / 'records' / (schedule['retained_completed_ids'][0] + '.json')
    record = scheduling.read(path)
    record['outcome']['invented_success'] = True
    write(path, record)
    with pytest.raises(AssertionError):
        scheduling.validate_schedule(experiment)


def test_completion_requires_new_state_exact_count_and_schedule_digest(experiment):
    schedule = scheduling.create_schedule(experiment, 'Defer only 0.5s')
    status = {'state': 'complete_scheduled', 'completed_trials': 3002, 'planned_trials': 3002,
              'qualification_trials': 0, 'error': None,
              'schedule_sha256': scheduling.digest(experiment / scheduling.SCHEDULE_NAME)}
    assert scheduling.completion_gate_for(experiment, status)[0]
    for replacement in ({'state': 'complete'}, {'completed_trials': 3001},
                        {'planned_trials': 4800}, {'schedule_sha256': 'bad'}):
        assert not scheduling.completion_gate_for(experiment, dict(status, **replacement))[0]
    assert schedule['planned_trials'] == 3002


def test_run_requires_explicit_resume_before_any_import_or_inference(experiment):
    with pytest.raises(AssertionError, match='explicit --resume'):
        scheduling.run(experiment)


def test_already_scored_scope_completes_without_loading_models_or_network(experiment, monkeypatch):
    plan = scheduling.read(experiment / 'plan.json')
    trace = experiment / 'traces/0.jsonl'
    trace_hash = scheduling.digest(trace)
    for job in plan['jobs']:
        if job['window_s'] >= 1.:
            path = experiment / 'records' / (scheduling.job_id(job) + '.json')
            if not path.exists():
                write(path, {'job': job, 'trace': str(trace.relative_to(experiment)),
                             'trace_sha256': trace_hash,
                             'outcome': {'invalid_trial': False, 'contract_violations': []}})
    status = scheduling.read(experiment / 'status.json')
    status['completed_trials'] = 3002
    write(experiment / 'status.json', status)
    scheduling.create_schedule(experiment, 'Save the <1s window runs for smaller models <=4B.')
    originals = {path: scheduling.digest(path) for path in (experiment / 'records').glob('*.json')}
    (experiment / 'summary.csv').write_text('Existing scored summary remains unchanged\n')
    def unexpected(*args, **kwargs):
        raise AssertionError('No model loading or readiness network request is allowed')
    monkeypatch.setattr(scheduling, '_load_frozen_runner', unexpected)
    monkeypatch.setattr(scheduling.urllib.request, 'urlopen', unexpected)
    scheduling.run(experiment, resume=True)
    status = scheduling.read(experiment / 'status.json')
    assert status['state'] == 'complete_scheduled'
    assert status['planned_trials'] == status['completed_trials'] == 3002
    assert status['model_inference_calls_this_continuation'] == 0
    assert status['completed_without_new_trials'] is True
    assert scheduling.completion_gate_for(experiment, status)[0]
    assert all(scheduling.digest(path) == expected for path, expected in originals.items())
    assert (experiment / 'summary.csv').read_text() == 'Existing scored summary remains unchanged\n'
