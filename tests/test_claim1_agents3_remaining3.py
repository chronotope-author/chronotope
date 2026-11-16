"""Offline contract checks; no model requests or benchmark episodes."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies'))
import run_claim1_agents3_remaining3 as study

OUT = ROOT / 'runs/claim1_agents3_qwen08_remaining3_20261105/experiment'
study.setup(OUT)


@pytest.mark.parametrize('state,count,audit_state,audit_count,hash_ok,allowed', [
    ('complete', 3000, 'passed', 3000, True, True),
    ('running', 3000, 'passed', 3000, True, False),
    ('complete_scoped', 3000, 'passed', 3000, True, False),
    ('stopped', 3000, 'passed', 3000, True, False),
    ('paused', 3000, 'passed', 3000, True, False),
    ('failed', 3000, 'passed', 3000, True, False),
    ('complete', 2999, 'passed', 3000, True, False),
    ('complete', 3000, 'failed', 3000, True, False),
    ('complete', 3000, 'passed', 2999, True, False),
    ('complete', 3000, 'passed', 3000, False, False),
])
def test_handoff_requires_complete_audited_exact_predecessor(state, count, audit_state, audit_count, hash_ok, allowed):
    status = dict(state=state, completed_trials=count, planned_trials=3000)
    audit = dict(state=audit_state, completed_trials=audit_count, plan_sha256='plan',
                 execution_manifest_sha256='execution' if hash_ok else 'different')
    assert study.completion_gate(status, audit, 'plan', 'execution') is allowed


def test_manifest_complete_and_action_support():
    cases = study.validate(OUT)
    from core5_ledger_plan import make_trial_spec
    for task in study.TASKS:
        config = study.read(OUT / 'configs' / f'{task}_agents3_full.json')
        case = cases[f'cases/{task}_change16s.jsonl'][0]
        spec = make_trial_spec(case, config, 'live')
        controller = study.make_controller(spec, task, object())
        # Keep action has neither a late-binding updater nor a restart hook.
        assert not hasattr(controller, 'correct_at_commit')
        assert controller.system == study.read(OUT / 'configs/task_instructions.json')[task]


@pytest.mark.parametrize('answer,expected', [
    ('agent.type(element_description="The focused code input below the yellow panel.", text="0123", overwrite=True)',
     {'kind': 'type', 'value': '0123'}),
    ('agent.type(text="123")', None),
    ('agent.wait(1.0)', None),
])
def test_full_upstream_worker_to_focused_field_bridge(tmp_path, answer, expected):
    import httpx
    from agents3_full import FullAgentS3Adapter
    from chronotope.controller import Observation
    from core5_ledger_plan import make_trial_spec
    config = study.read(OUT / 'configs/full_agent_components.json')
    request_roles = []

    def transport(role):
        def respond(request):
            request_roles.append(role)
            content = '(322, 238)' if role == 'grounder' else f'```python\n{answer}\n```'
            return httpx.Response(200, json={'id': 'offline-contract', 'object': 'chat.completion',
                'created': 0, 'model': config['components'][role]['model'],
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})
        return httpx.MockTransport(respond)

    case = json.loads((OUT / 'cases/code_change1s.jsonl').read_text().splitlines()[0])
    spec = make_trial_spec(case, study.read(OUT / 'configs/code_agents3_full.json'), 'live')
    adapter = FullAgentS3Adapter(config, tmp_path / 'calls.jsonl', _transport_factory=transport)
    controller = study.make_controller(spec, 'code', adapter)
    m = spec.manifest()
    # Synthetic pixels exercise parsing, not competence or a scored episode.
    observation = Observation(1, 1, 0, 0, 'fixture', 1, bytes(m.canvas_w_px*m.canvas_h_px*3))
    controller.admit(observation)
    try:
        result = controller.decide()
        assert result.action == expected
        assert not adapter.errors
        assert result.invocation_detail['updater_calls'] == 0
        assert result.invocation_detail['code_agent_calls'] == 0
        assert request_roles[0] == 'planner'
        if expected:
            assert 'grounder' in request_roles
    finally:
        adapter.close()


def test_no_receipt_failure_stays_in_denominator(tmp_path):
    record = {'job': {'task': 'code', 'setting_s': 1, 'mode': 'live', 'seed': 1000, 'id': 'fixture'},
              'evaluation': {'valid_at_receipt': None}}
    study.summarize(tmp_path, [record])
    import csv
    rows = list(csv.DictReader((tmp_path / 'summary.csv').open()))
    row = next(r for r in rows if r['task'] == study.TASKS['code'] and r['setting_s'] == '1' and r['mode'] == 'live')
    assert (row['successes'], row['trials'], row['coverage']) == ('0', '1', 'partial')
    assert all(r['coverage'] == 'unmeasured' for r in rows if r is not row)
