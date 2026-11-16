"""CPU contract fixtures only: no model services or qualification episodes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chronotope.controller import Observation, Restart
from chronotope.hashing import sha256_hex
from chronotope.paper_baselines.common import (ChatModel, InvalidOutput, ModelIdentity,
                                              SetupError, parse_command, strict_json)
from chronotope.paper_baselines.aapt import AAPTController, ChangeGate, validate_tree
from chronotope.paper_baselines.factory import check_scope
from chronotope.paper_baselines.upstream import locked_file

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'runs/paper_baseline_sources'


def identity():
    return ModelIdentity('fixture', 'fixture-revision', 'FP16', 'fixture', 'FP16', 'FP16')


def model(outputs):
    responses = iter(outputs)
    def transport(body):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        message = {'role': 'assistant', 'content': value} if isinstance(value, str) else value
        return {'id': 'fixture', 'choices': [{'message': message, 'finish_reason': 'stop'}],
                'usage': {'completion_tokens': 7}}
    return ChatModel(identity(), 'http://127.0.0.1:1/v1', transport=transport)


def obs(i, *, time_ns=0, color=0):
    payload = np.full((48, 64, 3), color, dtype=np.uint8).tobytes()
    return Observation(i, 1234567, 987654321, time_ns, sha256_hex(payload), i, payload)


def kwargs(m, config=None):
    return dict(task='notification', instruction='Click the notification when it appears.',
                width=64, height=48, config=config or {}, model=m)


def tree():
    return {'reason': 'Bounded visual choices', 'cover_ms': 3500, 'root_id': 'idle', 'nodes': [
        dict(id='idle', depth=0, guard='No notification', action='WAIT', confidence=1.0,
             deadline_ms=3500, risk='low', expected_state='Idle', children=['shown']),
        dict(id='shown', depth=1, guard='Notification visible at the prepared location',
             action='click(20, 20)', confidence=0.8, deadline_ms=500,
             risk='low', expected_state='Notification accepted', children=[])]}


def validate(value):
    return validate_tree(value, width=64, height=48, required_cover_ms=3500)


def test_action_parser_never_executes_or_clamps():
    assert parse_command('type("0012")', 64, 48) == {'kind': 'type', 'value': '0012'}
    assert parse_command('click(20, 21)', 64, 48) == {'kind': 'click', 'x': 20, 'y': 21}
    for command in ('click(True, 1)', 'click(64, 0)', 'click(1.1, 2)',
                    '__import__("os").system("echo bad")', 'click(1,2); click(2,3)'):
        with pytest.raises(InvalidOutput):
            parse_command(command, 64, 48)
    for value in ('{"a":1,"a":2}', '{"a": NaN}'):
        with pytest.raises(InvalidOutput):
            strict_json(value)


@pytest.mark.parametrize('damage', [
    lambda t: t.update(cover_ms=3499),
    lambda t: t['nodes'][1].update(action='click(64,0)'),
    lambda t: t['nodes'][1].update(confidence=float('nan')),
    lambda t: t['nodes'][1].update(depth=0),
    lambda t: t['nodes'][1].update(children=['idle']),
    lambda t: t['nodes'][0].update(children=['absent']),
    lambda t: t['nodes'][0].update(children=[]),
    lambda t: t['nodes'][1].update(id='idle'),
    lambda t: t['nodes'][1].update(deadline_ms=True),
    lambda t: t['nodes'][1].update(guard=''),
    lambda t: t['nodes'][0].update(action='click(1,1)'),
    lambda t: t['nodes'][1].update(extra='unrecognized'),
])
def test_aapt_rejects_invalid_trees(damage):
    value = tree()
    damage(value)
    with pytest.raises(InvalidOutput):
        validate(value)


def test_aapt_valid_tree_and_risk_limit():
    nodes, actions = validate(tree())
    assert actions['shown'] == {'kind': 'click', 'x': 20, 'y': 20}
    with pytest.raises(InvalidOutput):
        validate_tree(tree(), width=64, height=48, required_cover_ms=3500,
                      risk_horizons_ms={'low': 100, 'medium': 50, 'high': 0})


def test_change_gate_static_probe_and_small_change():
    gate = ChangeGate()
    gray = np.zeros((48, 64), dtype=np.uint8)
    assert gate.accepts(Image.fromarray(gray), 0)
    assert not gate.accepts(Image.fromarray(gray), 499_999_999)
    assert gate.accepts(Image.fromarray(gray), 500_000_000)
    gray[:4, :4] = 20
    assert gate.accepts(Image.fromarray(gray), 510_000_000)


def test_aapt_plan_then_observer_only_routes_prepared_action():
    m = model([json.dumps(tree()), '{"branch_id":"shown","confidence":0.1,"reason":"visible"}'])
    c = AAPTController(**kwargs(m, {'planner_p95_ms': 3000}))
    now = [0]
    c.bind_clock(lambda: now[0])
    c.admit(obs(1))
    assert c.decide().invocation_detail['outcome'] == 'tree_armed'
    now[0] = 100_000_000
    c.admit(obs(2, time_ns=now[0], color=255))
    action = c.decide()
    assert action.action == {'kind': 'click', 'x': 20, 'y': 20}
    assert m.calls[-1]['request']['max_tokens'] == 64
    assert 'click(20' not in json.dumps(m.calls[-1]['request']['messages'])
    assert '1234567' not in json.dumps(m.calls)
    assert '987654321' not in json.dumps(m.calls)
    assert c.correct_at_commit(600_000_000) is None
    assert isinstance(c.correct_at_commit(600_000_001), Restart)


@pytest.mark.parametrize('branch,confidence,outcome', [
    ('NONE', 0.8, 'no_match'), ('unknown', 0.8, 'ineligible_branch'),
    ('shown', 0.44, 'planner_confidence_below_floor')])
def test_aapt_abstains_without_inventing_actions(branch, confidence, outcome):
    value = tree()
    value['nodes'][1]['confidence'] = confidence
    m = model([json.dumps(value), json.dumps(dict(branch_id=branch, confidence=1, reason='fixture'))])
    c = AAPTController(**kwargs(m, {'planner_p95_ms': 3000}))
    c.bind_clock(lambda: 0)
    c.admit(obs(1)); c.decide()
    decision = c.decide()
    assert decision.action is None and decision.invocation_detail['outcome'] == outcome
    assert not c.ready_to_decide()


def test_aapt_invalid_planner_is_scored_no_action_with_one_retry():
    c = AAPTController(**kwargs(model(['invalid', 'invalid']), {'planner_p95_ms': 3000}))
    c.admit(obs(1))
    decision = c.decide()
    assert decision.action is None and decision.invocation_detail['outcome'] == 'invalid_tree'
    assert len(decision.invocation_detail['model_calls']) == 2
    assert not c.ready_to_decide()
    assert 'previous output was invalid' in c.model.calls[1]['request']['messages'][-1]['content']


def test_aapt_schema_bounds_repetitive_text_and_whitespace():
    from chronotope.paper_baselines.aapt import tree_schema
    schema = tree_schema(required_cover_ms=3500, max_cover_ms=64000)
    assert schema['properties']['reason']['maxLength'] == 96
    assert schema['properties']['cover_ms']['minimum'] == 3500
    m = model([json.dumps(tree())])
    c = AAPTController(**kwargs(m, {'planner_p95_ms': 3000}))
    c.admit(obs(1)); c.decide()
    assert m.calls[0]['request']['structured_outputs']['disable_any_whitespace'] is True
    assert m.calls[0]['request']['structured_outputs']['json'] == m.calls[0]['request']['response_format']['json_schema']['schema']


def test_transport_failure_is_not_a_competence_failure():
    m = model([OSError('fixture unavailable')])
    with pytest.raises(SetupError):
        m.complete([{'role': 'user', 'content': 'hello'}], max_tokens=64, temperature=0.0)
    assert len(m.errors) == len(m.calls) == 1


def test_aapt_observer_latency_is_charged_against_publication_age():
    now = [0]
    m = model([json.dumps(tree()), '{"branch_id":"shown","confidence":1,"reason":"visible"}'])
    c = AAPTController(**kwargs(m, {'planner_p95_ms': 3000}))
    c.bind_clock(lambda: now[0])
    c.admit(obs(1)); c.decide()
    original = m.transport
    def delayed(body):
        now[0] += 500_000_001
        return original(body)
    m.transport = delayed
    c.admit(obs(2, time_ns=0, color=255))
    result = c.decide()
    assert result.action is None and result.invocation_detail['outcome'] == 'late_observer'


def test_upstream_tampering_is_rejected(tmp_path):
    (tmp_path / 'aoi').mkdir()
    (tmp_path / 'aoi/keyframe_extractor.py').write_text('tampered')
    with pytest.raises(SetupError, match='hash mismatch'):
        locked_file(tmp_path, 'aoi', 'aoi/keyframe_extractor.py')


@pytest.fixture
def sources():
    if not SOURCES.exists():
        pytest.skip('Fetch pinned public source code for upstream integration tests')
    return SOURCES


def test_aoi_real_extractor_keeps_intermediate_frames_and_narration(sources, monkeypatch):
    from chronotope.paper_baselines.aoi import AOIController
    from chronotope.paper_baselines.upstream import load_module
    extractor = load_module(sources / 'aoi', 'aoi', 'aoi/keyframe_extractor.py')
    monkeypatch.setattr(extractor.KeyframeExtractor, '_load_clip', lambda self: None)
    m = model(['NARRATION: A notification appeared.\nACTION: wait()',
               'NARRATION: No visual change.\nACTION: click(20, 20)'])
    c = AOIController(source_root=sources / 'aoi', **kwargs(m))
    c.extractor._encode_clip = lambda image: np.ones(512, dtype=np.float32) / np.sqrt(512)
    samples = [obs(1), obs(2, time_ns=400_000_000, color=255), obs(3, time_ns=800_000_000)]
    for sample in samples:
        assert c.buffer_observation(sample)
    c.admit(samples[-1])
    result = c.decide()
    assert result.invocation_detail['sampled_observation_ids'] == [1, 2, 3]
    assert result.invocation_detail['keyframes'] == 2
    assert len(m.calls[0]['request']['messages'][0]['content']) == 5
    c.admit(obs(4, time_ns=900_000_000))
    assert c.decide().action['x'] == 20
    assert 'A notification appeared.' in json.dumps(m.calls[1]['request'])


def test_aoi_structured_transport_keeps_narration_and_actual_geometry(sources, monkeypatch):
    from chronotope.paper_baselines.aoi import AOIController
    from chronotope.paper_baselines.upstream import load_module
    extractor = load_module(sources / 'aoi', 'aoi', 'aoi/keyframe_extractor.py')
    monkeypatch.setattr(extractor.KeyframeExtractor, '_load_clip', lambda self: None)
    m = model([json.dumps({'narration': 'Visible button.', 'action': {'kind': 'click', 'x': 20, 'y': 21}}),
               json.dumps({'narration': 'Bad coordinate.', 'action': {'kind': 'click', 'x': 640, 'y': 360}})])
    c = AOIController(source_root=sources / 'aoi', **kwargs(m, {'structured_actions': True}))
    c.admit(obs(1)); assert c.decide().action == {'kind': 'click', 'x': 20, 'y': 21}
    request = m.calls[0]['request']; text = json.dumps(request['messages'])
    assert '64x48' in text and 'click(640, 360)' not in text
    branches = request['response_format']['json_schema']['schema']['properties']['action']['oneOf']
    assert branches[1]['properties']['x']['maximum'] == 63
    assert branches[1]['properties']['y']['maximum'] == 47
    c.admit(obs(2, color=255)); result = c.decide()
    assert result.action is None and 'invalid pixel coordinates' in result.invocation_detail['outcome']


def test_mobile_wait_metadata_is_not_misclassified_as_an_invalid_click():
    from chronotope.paper_baselines.mobiledreamer import mobile_action, action_tools
    value = mobile(10)
    value['tool_calls'][0]['function']['arguments'] = '{"action":"wait","coordinate":[0,0]}'
    assert mobile_action(value, 64, 48) is None
    schema = action_tools(64, 48)[0]['function']['parameters']
    assert schema['oneOf'][0]['required'] == ['action']
    coordinates = schema['oneOf'][1]['properties']['coordinate']['prefixItems']
    assert [v['maximum'] for v in coordinates] == [63, 47]


def mobile(x):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': 'fixture', 'type': 'function', 'function': {'name': 'computer_use',
        'arguments': json.dumps({'action': 'click', 'coordinate': [x, 20]})}}]}


class OCR:
    def image_to_text_description(self, image):
        return "label = text | text='Notification' | bbox=[10,10,40,30]"


def test_mobiledreamer_calls_released_candidate_world_selection(sources):
    from chronotope.paper_baselines.mobiledreamer import MobileDreamerController
    planner = model([mobile(10), mobile(20), mobile(30), '<selection>Action Number: 2</selection>'])
    world = model(['state one', 'state two', 'state three'])
    c = MobileDreamerController(source_root=sources / 'mobiledreamer', world_model=world,
                               ocr=OCR(), **kwargs(planner))
    c.admit(obs(1))
    result = c.decide()
    assert result.action == {'kind': 'click', 'x': 20, 'y': 20}
    assert len(planner.calls) == 4 and len(world.calls) == 3
    assert result.invocation_detail['rollout']['selected_action_idx'] == 1
    assert '[Current Screen (OCR+BBOX)]' in world.calls[0]['request']['messages'][0]['content']


def test_mobiledreamer_deduplicates_and_does_not_hide_failed_world_model(sources):
    from chronotope.paper_baselines.mobiledreamer import MobileDreamerController
    planner = model([mobile(10), mobile(10), mobile(10)])
    c = MobileDreamerController(source_root=sources / 'mobiledreamer',
        world_model=model([OSError('world model offline')]), ocr=OCR(), **kwargs(planner))
    c.admit(obs(1))
    with pytest.raises(SetupError):
        c.decide()


def test_mobiledreamer_ocr_error_is_not_a_model_failure(sources):
    import logging
    from chronotope.paper_baselines.mobiledreamer import MobileDreamerController
    class FailedOCR:
        def image_to_text_description(self, image):
            logging.error('OCR inference failed')
            return ''
    planner = model([])
    c = MobileDreamerController(source_root=sources / 'mobiledreamer', world_model=model([]),
                               ocr=FailedOCR(), **kwargs(planner))
    c.admit(obs(1))
    with pytest.raises(SetupError, match='OCR'):
        c.decide()
    assert not planner.calls


def study_module():
    spec = importlib.util.spec_from_file_location('baseline_study_fixture', ROOT / 'studies/paper_baselines.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_audits_all_jobs_and_replay_without_model_calls(tmp_path):
    from conftest import base_spec_dict
    from dataclasses import asdict
    config = {'schema': 'paper-baseline-config-1', 'baseline': 'aapt', 'planner_total_parameters': 800000000,
        'model': {'identity': asdict(identity()), 'base_url': 'http://127.0.0.1:1/v1'},
        'parameters': {'planner_p95_ms': 3000, 'planner_latency_evidence': 'fixture only'}}
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps(config))
    raw = base_spec_dict()
    raw['scenario']['skin'] = 'notification'
    raw['scenario']['response_window_ns'] = 1_000_000_000
    raw['timestamp_source'] = 'monotonic_ns'
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'base_spec': raw, 'distributional_row': None}) + '\n')
    study = study_module()
    out = tmp_path / 'prepared'
    study.prepare(cases, cfg, 'notification', 'Click the notification.', out)
    plan = study.validate(out)
    assert len(plan['jobs']) == 3 and plan['qualification_trials'] == 0
    assert not (out / 'execution').exists()
    raw = json.loads((out / plan['jobs'][0]['spec']).read_text())
    raw['scenario']['scenario_id'] = 'jump_target'
    with pytest.raises(SetupError, match='Task/scenario'):
        check_scope(raw)
    with pytest.raises(FileExistsError):
        study.prepare(cases, cfg, 'notification', 'Click the notification.', out)


def test_runner_publication_hook_retains_frames_while_model_is_busy(tmp_path):
    from conftest import base_spec_dict
    from chronotope.controller import ControlUpdate, Decision
    from chronotope.spec import TrialSpec
    from chronotope.runner import run_trial, PayloadStore
    from chronotope.trace import Trace
    raw = base_spec_dict()
    raw['scenario']['skin'] = 'notification'
    class BufferedFixture:
        def __init__(self):
            self.buffer, self.seen, self.calls = [], None, 0
        def spec_hash(self): return 'fixture'
        def buffer_observation(self, observation):
            self.buffer.append(observation.observation_id)
            return True
        def admit(self, observation):
            self.seen = observation
            return ControlUpdate('action', 'fixture')
        def ready_to_decide(self): return self.seen is not None and self.calls < 1
        def decide(self):
            self.calls += 1
            return Decision(None, 300_000_000)
    c = BufferedFixture()
    path = run_trial(TrialSpec.from_dict(raw), tmp_path / 'fixture.jsonl',
                     PayloadStore(tmp_path / 'payloads'), controller=c)
    trace = Trace.load(path)
    assert trace.all('observation_superseded')
    assert len(trace.all('observation_buffered')) == len(trace.all('observation')) == len(c.buffer)


def test_study_runs_all_three_conditions_despite_model_format_failures(tmp_path, monkeypatch):
    from conftest import base_spec_dict
    from dataclasses import asdict
    from chronotope.clock import SimulatedHarnessClock
    import chronotope.runner as runner
    class FixtureClock(SimulatedHarnessClock):
        source = 'monotonic_ns'
    monkeypatch.setattr(runner, 'MonotonicHarnessClock', FixtureClock)
    raw = base_spec_dict()
    raw['scenario'].update(skin='notification', response_window_ns=1_000_000_000,
                           frame_period_ns=333_333_333)
    raw['timestamp_source'] = 'monotonic_ns'
    config = {'schema': 'paper-baseline-config-1', 'baseline': 'aapt', 'planner_total_parameters': 800000000,
        'model': {'identity': asdict(identity()), 'base_url': 'http://127.0.0.1:1/v1'},
        'parameters': {'planner_p95_ms': 3000, 'planner_latency_evidence': 'fixture only'}}
    cfg, cases = tmp_path / 'config.json', tmp_path / 'cases.jsonl'
    cfg.write_text(json.dumps(config))
    cases.write_text(json.dumps({'base_spec': raw}) + '\n')
    study = study_module()
    out = tmp_path / 'prepared'
    study.prepare(cases, cfg, 'notification', 'Click the notification.', out)
    monkeypatch.setattr(study, 'build', lambda spec: AAPTController(
        **kwargs(model(['invalid'] * 100), {'planner_p95_ms': 3000})))
    study.run(out)
    records = json.loads((out / 'execution/records.json').read_text())
    assert len(records) == 3
    assert [r['mode'] for r in records] == ['live', 'replay', 'paused']
    assert all(r['status'] == 'scored' and not r['evaluation']['action_committed'] for r in records)
    assert json.loads((out / 'execution/status.json').read_text())['state'] == 'complete'
    assert len(list((out / 'execution').glob('*.calls.json'))) == 3
    with pytest.raises(FileExistsError):
        study.run(out)


def test_frozen_notification_overlay_is_not_silently_replaced():
    case_file = ROOT / 'runs/notification_qwen08_edgellm0111_20261031/experiment/cases/notification_w1000000000.jsonl'
    if not case_file.exists():
        pytest.skip('Historical case provenance not available')
    case = json.loads(case_file.read_text().splitlines()[0])
    row, manifest = study_module().case_manifest(case, 'notification')
    assert list(manifest.target_box_at(row.event_ns).center()) == row.target_start_position
    case['base_spec']['scenario']['scenario_seed'] += 1
    with pytest.raises(SetupError, match='proxy differs'):
        study_module().case_manifest(case, 'notification')
