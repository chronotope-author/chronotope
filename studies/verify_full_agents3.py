"""CPU-only contract fixtures for the historical full Agent S3 adapter.

All inference responses are canned through httpx.MockTransport. This script
does not contact model services, run benchmark episodes, or execute actions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

from agents3_full import FullAgentS3Adapter, image_count, source_fingerprints


def verify(config, out):
    import httpx
    from PIL import Image
    from gui_agents.s3.agents.agent_s import AgentS3
    from gui_agents.s3.agents.grounding import OSWorldACI
    from gui_agents.s3.agents.worker import Worker
    from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
    from gui_agents.s3.utils import common_utils

    before = source_fingerprints()
    methods = (AgentS3.predict, Worker.generate_next_action, Worker.flush_messages,
               Worker._generate_reflection, OSWorldACI.click, common_utils.create_pyautogui_code)
    png_buffer = io.BytesIO()
    Image.new('RGB', (640, 480), color=(30, 40, 50)).save(png_buffer, format='PNG')
    png = png_buffer.getvalue()
    instruction = 'CPU fixture: click the indicated point.'
    click = '(Grounded Action)\n```python\nagent.click("The indicated point in the screenshot.")\n```'
    all_requests = []
    log_files = []

    def factory_for(mode):
        counts = Counter()

        def factory(role):
            def handler(request):
                body = json.loads(request.content)
                counts[role] += 1
                all_requests.append({'fixture': mode, 'role': role, 'body': body})
                assert body['model'] == config['components'][role]['model']
                expected = config['grounder_generation' if role == 'grounder' else 'generation']
                for key in ('max_tokens', 'temperature', 'top_p', 'repetition_penalty'):
                    assert body[key] == expected[key], (mode, role, key, body)
                if 'enable_thinking' in expected:
                    assert body['chat_template_kwargs']['enable_thinking'] == expected['enable_thinking']
                if mode == 'http_retry' and role == 'planner' and counts[role] == 1:
                    return httpx.Response(500, json={'error': {'message': 'fixture service error'}})
                if role == 'grounder':
                    answer = '(322, 238)'
                elif role == 'reflection':
                    answer = 'The trajectory is proceeding as planned.'
                elif mode == 'format_retry' and counts[role] == 1:
                    answer = 'This fixture intentionally omits an action.'
                elif mode == 'unsupported_action':
                    answer = '```python\nagent.hotkey(["ctrl", "l"])\n```'
                else:
                    answer = click
                return httpx.Response(200, json={
                    'id': f'fixture-{role}-{counts[role]}', 'object': 'chat.completion',
                    'created': 0, 'model': body['model'],
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer},
                                 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 100 + counts[role], 'completion_tokens': 10,
                              'total_tokens': 110 + counts[role]},
                    'metrics': {'fixture_only': True, 'time_to_first_token': 0.001}})
            return httpx.MockTransport(handler)
        return factory

    with tempfile.TemporaryDirectory(prefix='chronotope-full-agent-cpu-') as tmp:
        root = Path(tmp)
        history = FullAgentS3Adapter(config, root / 'history.jsonl',
                                    _transport_factory=factory_for('history'))
        details = []
        # Ten canned decisions exercise upstream image eviction, not ten
        # experimental trials. No model, application or desktop is involved.
        with contextlib.redirect_stdout(io.StringIO()):
            for turn in range(10):
                result = history.invoke(png, instruction, 'ignored, as in historical service')
                assert result.parsed == {'action': 'click', 'x': 320, 'y': 240}
                detail = result.detail
                own = history.calls[-detail['model_calls']:]
                assert detail['input_tokens'] == sum(c['response']['usage']['prompt_tokens'] for c in own)
                assert detail['output_tokens'] == sum(c['response']['usage']['completion_tokens'] for c in own)
                assert detail['planner_calls'] == 1 and detail['grounder_calls'] == 2
                assert detail['reflection_calls'] == int(turn > 0)
                assert detail['text_grounder_calls'] == detail['code_agent_calls'] == 0
                assert detail['retained_history_images'] == min(turn + 1, 8)
                assert detail['retained_reflection_images'] == min(turn + 1, 8)
                details.append(detail)
        worker = history.agent.executor
        expected_prompt = PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
            OSWorldACI, skipped_actions=['call_code_agent']).replace('CURRENT_OS', 'linux')
        assert worker.generator_agent.system_prompt == expected_prompt.replace('TASK_DESCRIPTION', instruction)
        assert type(history.aci) is OSWorldACI and type(history.agent) is AgentS3 and type(worker) is Worker
        assert history.aci.env is None and worker.max_trajectory_length == 8 and worker.enable_reflection
        prompt_tools = [name for name in dir(OSWorldACI)
                        if getattr(getattr(OSWorldACI, name), 'is_agent_action', False)
                        and name != 'call_code_agent']
        assert all(f'def {name}' in worker.generator_agent.system_prompt for name in prompt_tools)
        assert 'def call_code_agent' not in worker.generator_agent.system_prompt
        history_requests = [r for r in all_requests if r['fixture'] == 'history']
        assert max(image_count(r['body']['messages']) for r in history_requests if r['role'] == 'planner') == 9
        assert max(image_count(r['body']['messages']) for r in history_requests if r['role'] == 'reflection') == 9
        assert not history.errors
        for record in history.calls:
            assert record['response']['metrics']['fixture_only']
            for message in record['request']['messages']:
                if isinstance(message.get('content'), list):
                    for part in message['content']:
                        if part.get('type') == 'image_url':
                            assert 'url' not in part['image_url'] and 'sha256' in part['image_url']
        history.close()
        log_files.append(root / 'history.jsonl')

        retry = FullAgentS3Adapter(config, root / 'format_retry.jsonl',
                                  _transport_factory=factory_for('format_retry'))
        with contextlib.redirect_stdout(io.StringIO()):
            retried = retry.invoke(png, instruction, '')
        assert retried.parsed == {'action': 'click', 'x': 320, 'y': 240}
        assert retried.detail['planner_calls'] == 2 and not retry.errors
        retry.close()
        log_files.append(root / 'format_retry.jsonl')

        unsupported = FullAgentS3Adapter(config, root / 'unsupported_action.jsonl',
                                        _transport_factory=factory_for('unsupported_action'))
        with contextlib.redirect_stdout(io.StringIO()):
            no_click = unsupported.invoke(png, instruction, '')
        assert no_click.parsed is None and not unsupported.errors
        assert 'pyautogui.hotkey' in no_click.detail['raw_text']
        unsupported.close()
        log_files.append(root / 'unsupported_action.jsonl')

        http_retry = FullAgentS3Adapter(config, root / 'http_retry.jsonl',
                                       _transport_factory=factory_for('http_retry'))
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                http_retry.invoke(png, instruction, '')
                raise AssertionError('Infrastructure error was not surfaced')
            except RuntimeError as exc:
                assert str(exc).startswith('Inference service error: ')
        assert http_retry.calls[0]['http_status'] == 500
        assert http_retry.calls[1]['http_status'] == 200
        assert http_retry.errors and len(http_retry.calls) == 4
        http_retry.close()
        log_files.append(root / 'http_retry.jsonl')

        out.parent.mkdir(parents=True, exist_ok=True)
        evidence_dir = out.parent / 'full_agents3_cpu_fixture_logs'
        evidence_dir.mkdir(exist_ok=True)
        for path in log_files:
            (evidence_dir / path.name).write_bytes(path.read_bytes())

    assert methods == (AgentS3.predict, Worker.generate_next_action, Worker.flush_messages,
                       Worker._generate_reflection, OSWorldACI.click, common_utils.create_pyautogui_code)
    assert source_fingerprints() == before
    before[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {'state': 'passed', 'model_inference_calls': 0, 'experimental_trials': 0,
              'qualification_trials': 0, 'network_requests': 0,
              'mock_http_attempts': len(all_requests), 'source_file_sha256': before,
              'component_config_sha256': history.component_config_sha256,
              'upstream_worker_template_sha256': history.upstream_worker_template_sha256,
              'generation': config['generation'], 'grounder_generation': config['grounder_generation'],
              'policy_guarantees': {'upstream_sources_unchanged': True, 'worker_methods_unchanged': True,
                  'upstream_prompt_exact': True, 'all_upstream_gui_actions_preserved': True,
                  'eight_turn_image_and_text_history': True, 'reflection_retained': True,
                  'upstream_format_retries_retained': True, 'upstream_sdk_retries_retained': True,
                  'all_http_attempts_logged': True, 'response_metrics_preserved': True,
                  'optional_code_execution': 'disabled upstream default',
                  'historical_action_parser_unchanged': True, 'unsupported_action_is_not_infrastructure_error': True,
                  'no_returned_desktop_action_executed': True},
              'max_images_observed_in_mock_requests': 9, 'gui_action_methods': prompt_tools,
              'fixture_logs': str(evidence_dir),
              'fixture_log_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in evidence_dir.glob('*.jsonl')}}
    out.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    # Direct script execution puts studies/, but not the src-layout package,
    # on sys.path. The Agent S3 venv need not have this repository installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = verify(json.loads(args.config.read_text()), args.out)
    print(json.dumps({'state': report['state'], 'model_inference_calls': 0,
                      'experimental_trials': 0, 'mock_http_attempts': report['mock_http_attempts'],
                      'out': str(args.out)}))
