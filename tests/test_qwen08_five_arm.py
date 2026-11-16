"""CPU-only plan/integration tests; no evaluation or qualification episodes."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies'))
import run_qwen08_five_arm as comparison


def test_full_job_grid_is_paired_and_counterbalanced():
    cells = []; cases = {}
    for task in comparison.TASKS:
        for setting in comparison.SETTINGS:
            key = f'{task}_{setting}s'
            cells.append(dict(task=task, setting_s=setting, cell_id=key,
                manifest=key, task_deadline_s=setting if task == 'notification' else 32))
            cases[key] = [dict(index=i, seed=1000+i, episode_id=f'{key}_{i}') for i in range(100)]
    jobs = comparison.make_jobs(cells, cases)
    assert len(jobs) == 7500 and len({j['id'] for j in jobs}) == 7500
    assert {j['phase'] for j in jobs} == {'evaluation'}
    assert {j['mode'] for j in jobs} == {'live'}
    assert {j['task'] for j in jobs} == {'notification', 'button', 'trading'}
    assert {j['setting_s'] for j in jobs} == {1, 2, 4, 8, 16}
    positions = {arm: [0] * 5 for arm in comparison.ARMS}
    for offset in range(0, len(jobs), 5):
        block = jobs[offset:offset+5]
        assert {j['arm'] for j in block} == set(comparison.ARMS)
        assert len({(j['manifest'], j['index'], j['seed']) for j in block}) == 1
        for position, job in enumerate(block): positions[job['arm']][position] += 1
    assert all(counts == [300] * 5 for counts in positions.values())


def test_64_second_case_derivation_preserves_stimuli_and_change_delays():
    import copy
    sys.path.insert(0, str(ROOT / 'src'))
    cells = []; cases = {}
    for task in comparison.TASKS:
        for setting in ((64,) if task == 'notification' else comparison.SETTINGS):
            source_setting = 16 if task == 'notification' else setting
            source = comparison.REFERENCE / 'cases' / f'{task}_{source_setting}s.jsonl'
            old = json.loads(source.read_text().splitlines()[0])
            new = comparison.with_response_window(old, 64)
            before, after = old['base_spec']['scenario'], new['base_spec']['scenario']
            assert after['response_window_ns'] == 64 * 10**9
            assert after['episode_horizon_ns'] == after['event_time_ns'] + 64_500_000_000
            assert {k:v for k,v in before.items() if k not in ('response_window_ns','episode_horizon_ns')} == {
                k:v for k,v in after.items() if k not in ('response_window_ns','episode_horizon_ns')}
            assert (old['seed'],old['episode_id']) == (new['seed'],new['episode_id'])
            key = f'{task}_{setting}s'
            cells.append(dict(task=task,setting_s=setting,cell_id=key,manifest=key,task_deadline_s=64))
            cases[key] = [dict(index=i,seed=i,episode_id=str(i)) for i in range(100)]
    jobs = comparison.make_jobs(cells,cases)
    assert len(jobs) == 5500
    assert {j['task_deadline_s'] for j in jobs} == {64}


def test_live_latency_reporting_keeps_no_action_failures(tmp_path):
    comparison.write(tmp_path/'plan.json', {'cells':[{'task':'notification','setting_s':64}], 'response_window_s':64})
    comparison.summarize(tmp_path, [dict(job=dict(task='notification',setting_s=64,arm='aapt'),
        status='scored',success=0,metrics={'e_to_receipt_ms':None,'decision_B_to_D_ms':[2000.]})])
    import csv
    row = next(r for r in csv.DictReader((tmp_path/'summary.csv').open()) if r['arm']=='aapt')
    assert row['scored_trials']=='1' and row['no_action_receipt']=='1'
    assert row['event_to_receipt_p50_ms']=='' and row['decision_mean_ms']=='2000.0'


def test_latency_proxy_excludes_failures_and_is_explicitly_not_aapt(tmp_path):
    (tmp_path / 'calls').mkdir()
    calls = [dict(role='planner', http_status=200, start_ns=0, end_ns=i * 1_000_000)
             for i in range(1, 21)]
    calls += [dict(role='grounder', http_status=200, start_ns=0, end_ns=10**12),
              dict(role='planner', http_status=500, start_ns=0, end_ns=10**12)]
    (tmp_path / 'calls/agents3_full_fixture.jsonl').write_text('\n'.join(map(json.dumps, calls)))
    evidence = comparison.latency_proxy(tmp_path)
    assert evidence['samples'] == 20 and evidence['planner_budget_ms'] == 19
    assert 'NOT measured AAPT' in evidence['kind']
    assert evidence['additional_inference_calls'] == 0


def test_unmeasured_and_infrastructure_cases_never_become_model_failures(tmp_path):
    records = [dict(job=dict(task='notification', setting_s=16, arm='aoi'),
                    status='infrastructure_error', success=None)]
    comparison.summarize(tmp_path, records)
    import csv
    rows = list(csv.DictReader((tmp_path / 'summary.csv').open()))
    assert len(rows) == 75
    aoi = next(r for r in rows if r['arm'] == 'aoi' and r['setting_s'] == '16'
               and r['task'] == 'Disappearing Notification')
    assert aoi['scored_trials'] == '0' and aoi['success_rate'] == ''
    assert aoi['infrastructure_errors'] == '1'
    assert '— (0 measured)' in (tmp_path / 'comparison.md').read_text()


def test_run_refuses_stopped_or_nonempty_study_before_runtime_access(tmp_path, monkeypatch):
    (tmp_path / 'records').mkdir(); (tmp_path / 'trials').mkdir()
    comparison.write(tmp_path / 'status.json', {'state': 'stopped'})
    monkeypatch.setattr(comparison, 'check_runtime', lambda _: pytest.fail('Must not inspect/start services'))
    with pytest.raises(ValueError, match='no resume'): comparison.run(tmp_path)
    comparison.write(tmp_path / 'status.json', {'state': 'prepared'})
    (tmp_path / 'trials/interrupted.jsonl').write_text('retained trace')
    with pytest.raises(ValueError, match='no resume'): comparison.run(tmp_path)


def test_baseline_call_audit_keeps_auxiliary_usage_separate():
    from types import SimpleNamespace
    controller = SimpleNamespace(model=SimpleNamespace(calls=[{'wall_ns': 3}]),
                                 world_model=SimpleNamespace(calls=[{'wall_ns': 9}]))
    assert comparison.baseline_calls(controller) == [
        {'wall_ns': 3, 'role': 'planner'}, {'wall_ns': 9, 'role': 'world_model'}]
    assert 'role' not in controller.model.calls[0]


def test_mobile_schema_transport_preserves_tool_arguments_without_a_server_parser():
    from chronotope.paper_baselines.common import ChatModel, ModelIdentity
    from chronotope.paper_baselines.mobiledreamer import ACTION_TOOLS, ModelBridge, mobile_action
    bodies = []
    def transport(body):
        bodies.append(body)
        return {'choices': [{'message': {'content': '{"action":"click","coordinate":[17,23]}'}}]}
    model = ChatModel(ModelIdentity('qwen', 'pinned', 'BF16', 'vLLM', 'FP16', 'FP16'),
                      'http://unused', transport=transport)
    message, ok, _ = ModelBridge(model, {}).call_llm([{'role': 'user', 'content': 'fixture'}], tools=ACTION_TOOLS)
    assert ok and mobile_action(message, 100, 100) == {'kind': 'click', 'x': 17, 'y': 23}
    assert 'tools' not in bodies[0] and 'tool_choice' not in bodies[0]
    assert bodies[0]['response_format']['json_schema']['schema'] == ACTION_TOOLS[0]['function']['parameters']
    ModelBridge(model, {}).call_llm([{'role': 'user', 'content': 'fixture'}])
    assert 'tool_choice' not in bodies[1]  # world-model and selector text calls
    assert 'response_format' not in bodies[1]


def test_ocr_bridge_keeps_official_block_text_and_coordinates():
    from types import SimpleNamespace
    from chronotope.paper_baselines.ocr import elements_from_result
    from chronotope.paper_baselines.common import SetupError
    expected = [{'label': 'text', 'text': "Continue", 'bbox': [2, 3, 40, 20]}]
    assert elements_from_result({'parsing_res_list': [SimpleNamespace(
        label='text', content='Continue', bbox=[2, 3, 40, 20])]}) == expected
    assert elements_from_result({'parsing_res_list': [dict(
        block_label='text', block_content='Continue', block_bbox=[2, 3, 40, 20])]}) == expected
    assert elements_from_result({'parsing_res_list': []}) == []
    with pytest.raises(SetupError, match='schema'):
        elements_from_result({'unexpected': []})
