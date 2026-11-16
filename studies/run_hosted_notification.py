"""Fresh hosted-planner Notification pairs. Preparation/validation make no API calls.

Run each model explicitly. No qualification, automatic resume, or successor jobs.
This is legacy C++ with remote inference, not GPU-history Chronotope.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'runs/notification_qwen08_edgellm0111_20261031/experiment'
ARMS = ('legacy_cpp_detector', 'agents3_full')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'chronotope/python')]
from core5_ledger_plan import digest, read, write, validate_plan, load_cases, make_trial_spec
from hosted_api import MODELS, GENERATION, HostedAgentS3Adapter


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def configure(out):
    # Existing controller schemas and evaluator, with the opt-in native transport.
    from chronotope_core.glue import register
    register()
    for name in ('CHRONOTOPE_AGENT_S3_POLICY', 'CHRONOTOPE_AGENT_S3_PROMPT', 'CHRONOTOPE_AGENT_S3_AUDIT', 'CHRONOTOPE_PLANNER_PROMPT'):
        os.environ.pop(name, None)
    build = read(out / 'config.json')['build']
    os.environ['CHRONOTOPE_BIN'] = str(Path(build) / 'chronotope')
    os.environ['CHRONOTOPE_LIB'] = str(Path(build) / 'libchronotope.so')


def prepare(out, model, build):
    if out.exists():
        raise FileExistsError('Never overwrite a prepared cohort: ' + str(out))
    out.mkdir(parents=True)
    old = read(REFERENCE / 'status.json')
    if old['state'] != 'complete' or old['completed_trials'] != 3000:
        raise ValueError('The accepted Qwen0.8B predecessor is not complete')
    predecessor_audit = read(REFERENCE.parent / 'completion_audit.json')
    if predecessor_audit['state'] != 'passed' or predecessor_audit['completed_trials'] != 3000:
        raise ValueError('The accepted predecessor evidence audit has not passed')
    plan = copy.deepcopy(read(REFERENCE / 'plan.json'))
    validate_plan(plan)
    plan.update(model_key=model, model=model, revision=None, source_precision='provider undisclosed',
                runtime=MODELS[model]['runtime'], kv_cache='provider undisclosed',
                predecessor=str(REFERENCE), predecessor_plan_sha256=digest(REFERENCE / 'plan.json'),
                matching_scope='Same hosted planner ID/settings; distinct legacy and full Agent S3 policies',
                label='Legacy C++ + detector + remote API versus Agent S3 + remote API + UI-TARS',
                automatic_resume=False, successor_jobs=[])
    for job in plan['jobs']:
        job['model'] = model
    shutil.copytree(REFERENCE / 'cases', out / 'cases')
    for folder in ('configs', 'records', 'calls', 'trials', 'payloads'):
        (out / folder).mkdir()
    native = read(REFERENCE / 'configs/notification_legacy_cpp_detector.json')
    native.update(planner='api', model_id=model, base_url=MODELS[model]['endpoint'],
                  max_tokens=4096, temperature=None, enable_thinking=True, reasoning_effort='low')
    host = read(REFERENCE / 'configs/notification_agents3_full.json')
    host.update(server='agents3_hosted_api', model_id=model, base_url=MODELS[model]['endpoint'], max_tokens=4096,
                temperature=0.0, enable_thinking=True, reasoning_effort='low')
    full = read(REFERENCE / 'configs/full_agent_components.json')
    full['generation'] = {'max_tokens': 4096}
    full['provider_generation'] = GENERATION
    for role, component in full['components'].items():
        if role == 'grounder':
            continue
        component.update(model=model, revision='provider-managed; exact API model ID recorded',
                         precision='provider undisclosed', source_precision='provider undisclosed',
                         kv_cache='provider undisclosed', runtime=MODELS[model]['runtime'],
                         base_url=MODELS[model]['endpoint'])
    write(out / 'configs/notification_legacy_cpp_detector.json', native)
    write(out / 'configs/notification_agents3_full.json', host)
    write(out / 'configs/full_agent_components.json', full)
    write(out / 'config.json', {'model': model, 'build': str(build.resolve()),
        'provider': MODELS[model], 'generation': GENERATION,
        'model_predecessor': str(out.parent / 'gpt-6.1-sol') if model == 'claude-fable-5-1' else None,
        'native_transport_retries': 0, 'agent_s3_retries': 'existing SDK and engine backoff plus Worker formatting retries',
        'transport_timeout_s': 600, 'image_input': 'existing 640x480 PNG; OpenAI detail=high',
        'schema_placeholders': 'Host controller.temperature=0.0 is required by the legacy schema; hosted transport omits temperature. Provider requests are authoritative.',
        'generation_changes': ['Both API arms use low effort and 4096 total output tokens including reasoning.',
            'Native local-model cap was 64 visible tokens; hosted cap includes hidden reasoning.',
            'Sampling and vLLM-only fields are omitted where provider APIs do not support them.',
            'Native Claude keeps the brace stop sequence; Responses has no stop-string parameter, so native OpenAI parses after full completion.',
            'No constrained JSON decoding or provider computer-use tools are added.'],
        'grounder_checkpoint_provenance': str(REFERENCE.parent / 'grounder_checkpoint_verification.json'),
        'grounder_service_provenance': str(REFERENCE / 'service_launch.json'),
        'api_internal_precision_verified': False, 'launch_authorization': 'User: Let\'s do it. Make sure to show me the diff',
        'planned_trials': 3000, 'qualification_trials': 0})
    write(out / 'plan.json', plan)
    configure(out)
    validate(out)
    write(out / 'status.json', {'state': 'prepared', 'completed_trials': 0, 'planned_trials': 3000,
                               'qualification_trials': 0, 'pid': None, 'automatic_resume': False})


def validate(out):
    plan = read(out / 'plan.json')
    if validate_plan(plan) != 3000 or plan['selected_tasks'] != ['notification']:
        raise ValueError('Expected exactly 3000 Notification trials')
    if set(plan['notification_windows_s']) != {1, 2, 4, 8, 16} or plan.get('successor_jobs'):
        raise ValueError('Unsupported window or chained job')
    if plan['model'] not in MODELS or plan['revision'] is not None:
        raise ValueError('Hosted model identity/precision cannot be a local checkpoint claim')
    cases = load_cases(out, plan)
    from chronotope_core.glue import register
    register()
    for job in plan['jobs']:
        controller = read(out / 'configs' / f'notification_{job["arm"]}.json')
        make_trial_spec(cases[job['manifest']][job['index']], controller, job['mode'],
                        '0' * 64 if job['mode'] == 'replay' else None)
    full = read(out / 'configs/full_agent_components.json')
    # No Worker initialization or inference is needed for structural validation.
    adapter = HostedAgentS3Adapter(full, out / 'validation-unused.jsonl'); adapter.close()
    if full['policy'] != read(REFERENCE / 'configs/full_agent_components.json')['policy']:
        raise ValueError('Agent S3 policy changed')
    native = read(out / 'configs/notification_legacy_cpp_detector.json')
    reference = read(REFERENCE / 'configs/notification_legacy_cpp_detector.json')
    allowed = {'planner', 'model_id', 'base_url', 'max_tokens', 'temperature', 'enable_thinking', 'reasoning_effort'}
    if any(native[k] != v for k, v in reference.items() if k not in allowed):
        raise ValueError('Legacy controller behavior changed')
    return {'state': 'passed', 'spec_checks': 3000, 'qualification_trials': 0, 'model_calls': 0}


def freeze(out):
    configure(out); validate(out)
    target = out / 'execution_manifest.json'
    if target.exists():
        raise FileExistsError(target)
    # Immutable executable + source snapshot; every launch verifies the live source
    # too, so a later edit cannot silently change an in-progress experiment.
    from agents3_full import source_fingerprints
    files = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'chronotope/python/chronotope_core').glob('*.py'))
    files += list((ROOT / 'chronotope/src').glob('*')) + [ROOT / 'chronotope/CMakeLists.txt']
    files += [ROOT / 'studies' / name for name in ('run_hosted_notification.py', 'hosted_api.py', 'agents3_full.py',
                                                  'core5_ledger_plan.py', 'run_core5_detector_full_agents3.py', 'legacy_d1_w8.py')]
    files += list((out / 'configs').glob('*.json')) + [out / 'plan.json', out / 'config.json']
    files += [REFERENCE / 'configs/notification_compact_prompt.json', REFERENCE / 'configs/full_agent_components.json',
              REFERENCE / 'service_launch.json', REFERENCE.parent / 'grounder_checkpoint_verification.json',
              REFERENCE.parent / 'completion_audit.json']
    build = Path(read(out / 'config.json')['build'])
    files += [build / 'chronotope', build / 'libchronotope.so']
    hashes = {str(p): digest(p) for p in files if p.is_file()}
    hashes.update(source_fingerprints())
    snapshots = out / 'frozen_source'
    for p in files:
        if p.is_file() and ROOT in p.parents and out not in p.parents and build not in p.parents:
            dest = snapshots / p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    write(target, {'created_at': now(), 'hashes': hashes,
                   'packages': {p: importlib.metadata.version(p) for p in ('openai', 'httpx', 'anthropic')},
                   'validation': validate(out), 'api_calls': 0})


def call_audit(path, arm, model, trace, scenario):
    calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    events = trace.all('decision_invocation')
    if arm == ARMS[0] and len(calls) != len(events):
        raise ValueError('Native call accounting mismatch')
    if arm == ARMS[1] and len(calls) != sum(e['model_calls'] for e in events):
        raise ValueError('Agent S3 call accounting mismatch')
    owners = events if arm == ARMS[0] else [e for e in events for _ in range(e['model_calls'])]
    observations = {o['observation_id']: o for o in trace.all('observation')}
    result = []
    for call, owner in zip(calls, owners):
        if call.get('error') or call.get('http_status') != 200:
            raise RuntimeError('Provider/grounder infrastructure error recorded')
        response = call['response']
        raw = response.get('provider_response', response)
        request = response.get('provider_request', call['request'])
        expected = 'ByteDance-Seed/UI-TARS-1.5-7B' if call['role'] == 'grounder' else model
        if raw['model'] != expected:
            raise ValueError('Observed model differs')
        usage = raw['usage']
        price = estimated_cost(raw['model'], usage)
        if call['role'] == 'planner':
            verify_planner_input(request, arm, model, observations[owner['observation_id']], scenario)
        result.append({'role': call['role'], 'model': raw['model'], 'request_id': raw['id'],
            'input_tokens': usage.get('input_tokens', usage.get('prompt_tokens')) +
                            usage.get('cache_read_input_tokens', 0) + usage.get('cache_creation_input_tokens', 0),
            'output_tokens': usage.get('output_tokens', usage.get('completion_tokens')),
            'usage': usage, 'elapsed_ns': call['end_ns'] - call['start_ns'], 'api_cost_usd_estimate': price,
            'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
            'response_sha256': hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()})
    return result


def verify_planner_input(request, arm, model, observation, scenario):
    from PIL import Image
    from chronotope.png import encode_png
    messages = request.get('input', request.get('messages'))
    images = []
    for message in messages:
        for part in message['content']:
            if part['type'] == 'input_image': images.append(part['image_url'].split(',', 1)[1])
            elif part['type'] == 'image': images.append(part['source']['data'])
    if not images:
        raise ValueError('Planner request omitted its screenshot')
    actual = Image.open(io.BytesIO(base64.b64decode(images[-1], validate=True))).convert('RGB')
    expected = Image.open(io.BytesIO(encode_png(scenario.render(observation['source_time_ns']), 320, 240, 2))).convert('RGB')
    if actual.size != expected.size or actual.tobytes() != expected.tobytes():
        raise ValueError('Planner screenshot does not match the decision observation')
    if arm == ARMS[0]:
        compact = read(REFERENCE / 'configs/notification_compact_prompt.json')
        if model == 'gpt-6.1-sol':
            system = messages[0]['content'][0]['text']; user = messages[1]['content']
        else:
            system = request['system']; user = messages[0]['content']
        if system != compact['planner_system'] or user[-1].get('text') != compact['initial_user_text']:
            raise ValueError('Legacy compact prompt changed')
        if len(images) != 1 or len(user) != 2:
            raise ValueError('Legacy request gained history or additional content')


def estimated_cost(model, usage):
    """Standard-tier estimate, USD; excludes local grounder hardware and taxes.

    Rates checked 2026-11-02. Raw provider usage is retained for invoice reconciliation.
    Long-context OpenAI multipliers are applied when applicable.
    """
    if model == 'ByteDance-Seed/UI-TARS-1.5-7B':
        return 0.0
    ins, outs = usage['input_tokens'], usage['output_tokens']
    if model == 'gpt-6.1-sol':
        cached = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
        rate_in, rate_out = (2, 1.5) if ins > 272000 else (1, 1)
        return ((ins - cached) * 2 * rate_in + cached * 0.1 * rate_in + outs * 10 * rate_out) / 1e6
    if model == 'claude-fable-5-1':
        writes = usage.get('cache_creation_input_tokens', 0)
        details = usage.get('cache_creation', {})
        long_writes = details.get('ephemeral_1h_input_tokens', 0)
        return (ins * 10 + outs * 50 + usage.get('cache_read_input_tokens', 0) * 0.25 +
                (writes - long_writes) * 12.5 + long_writes * 20) / 1e6
    raise ValueError('Unknown pricing model')


def summarize(out, records):
    cells = defaultdict(list)
    for r in records:
        j = r['job']; cells[j['arm'], j['window_s'], j['mode']].append(r)
    rows = []
    for (arm, window, mode), group in sorted(cells.items()):
        calls = [c for r in group for c in r['request_audit']]
        latencies = sorted(r['outcome']['L_EA'] / 1e9 for r in group if r['outcome'].get('L_EA') is not None)
        def percentile(q):
            return latencies[min(len(latencies) - 1, math.ceil(q * len(latencies)) - 1)] if latencies else None
        rows.append({'model': read(out / 'config.json')['model'], 'controller': arm, 'window_s': window,
            'condition': mode, 'completed': len(group), 'planned': 100,
            'successes': sum(r['outcome']['Y'] for r in group),
            'success_rate': sum(r['outcome']['Y'] for r in group) / len(group),
            'model_calls': len(calls), 'input_tokens': sum(c['input_tokens'] for c in calls),
            'output_tokens': sum(c['output_tokens'] for c in calls),
            'api_cost_usd_estimate': sum(c['api_cost_usd_estimate'] for c in calls),
            'event_to_receipt_p50_s': percentile(.5), 'event_to_receipt_p95_s': percentile(.95),
            'latency_sample_count': len(latencies)})
    if rows:
        with (out / 'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def preflight(out):
    configure(out); validate(out)
    frozen = read(out / 'execution_manifest.json')
    for name, expected in frozen['hashes'].items():
        if digest(name) != expected:
            raise ValueError('Frozen artifact changed: ' + name)
    model = read(out / 'config.json')['model']
    predecessor = read(out / 'config.json').get('model_predecessor')
    if predecessor:
        prior = Path(predecessor)
        status, evidence = read(prior / 'status.json'), read(prior / 'completion_audit.json')
        if status['state'] != 'complete' or evidence['state'] != 'passed' or evidence['completed_trials'] != 3000:
            raise RuntimeError('Prior hosted model has not completed and passed its evidence audit')
    if not os.environ.get(MODELS[model]['key_env']):
        raise RuntimeError('Missing ' + MODELS[model]['key_env'] + '; no model calls made')
    # Read-only service identity check, never a screening model invocation.
    grounder = read(out / 'configs/full_agent_components.json')['components']['grounder']
    provenance = read(read(out / 'config.json')['grounder_checkpoint_provenance'])
    if provenance['state'] != 'passed' or provenance['revision'] != grounder['revision']:
        raise ValueError('Grounder checkpoint provenance missing')
    if provenance['configured_runtime_dtype'] != 'BF16' or provenance['configured_kv_cache_dtype'] != 'BF16':
        raise ValueError('Grounder precision differs from accepted BF16 configuration')
    for name, expected in provenance['file_hashes'].items():
        if digest(name) != expected:
            raise ValueError('Grounder checkpoint changed: ' + name)
    service = read(read(out / 'config.json')['grounder_service_provenance'])['components']['grounder']
    proc = Path('/proc') / str(service['pid'])
    # PID reuse and a replacement server with different dtype cannot satisfy this check.
    start_ticks = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[19])
    command = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if start_ticks != service['process_starttime_ticks'] or command != service['command']:
        raise ValueError('Grounder process no longer matches its launch provenance')
    if command[command.index('--dtype') + 1] != 'bfloat16' or service['effective_kv_cache_dtype'] != 'bfloat16':
        raise ValueError('Grounder runtime precision differs')
    with urllib.request.urlopen(grounder['base_url'].rstrip('/') + '/models', timeout=15) as response:
        if grounder['model'] not in [m['id'] for m in json.load(response)['data']]:
            raise ValueError('Grounder service identity differs')
    return model


def audit(out):
    """Re-read all completed evidence without invoking any model."""
    configure(out); validate(out)
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    from chronotope.distributional.manifest import EpisodeRow, build_manifest
    from chronotope.distributional.episode import score_trace, verify_trace_evidence
    plan = read(out / 'plan.json'); cases = load_cases(out, plan)
    frozen = read(out / 'execution_manifest.json')
    for name, expected in frozen['hashes'].items():
        if digest(name) != expected: raise ValueError('Frozen artifact changed: ' + name)
    records = {p.stem: read(p) for p in (out / 'records').glob('*.json')}
    if set(records) != {j['id'] for j in plan['jobs']}:
        raise ValueError('Completed records do not cover the full 3000-trial plan')
    for job in plan['jobs']:
        record = records[job['id']]; trace = Trace.load(out / record['trace'])
        if record['job'] != job or trace.file_hash() != record['trace_sha256']:
            raise ValueError('Record or trace provenance changed')
        case = cases[job['manifest']][job['index']]
        row = EpisodeRow.from_dict(case['distributional_row']); scenario = build_manifest(row)
        verify_trace_evidence(row, scenario, trace, case['distributional_manifest_hash'])
        outcome = score_trace(row, trace, case['distributional_manifest_hash'], controller=job['arm'], repeat_index=0)
        if outcome.to_dict() != record['outcome']:
            raise ValueError('Recomputed score differs')
        if call_audit(out / 'calls' / f'{job["id"]}.jsonl', job['arm'], plan['model'], trace, scenario) != record['request_audit']:
            raise ValueError('Request audit differs')
        if job['mode'] == 'replay':
            live = Trace.load(out / 'trials' / job['cell_id'] / job['arm'] / f'{job["episode_id"]}_live.jsonl')
            if not validate_pair(trace, live).valid: raise ValueError('Replay pair invalid')
    result = {'state': 'passed', 'completed_trials': len(records), 'model_calls': 0,
              'plan_sha256': digest(out / 'plan.json'), 'execution_manifest_sha256': digest(out / 'execution_manifest.json')}
    write(out / 'completion_audit.json', result)
    return result


def run(out):
    # Never resume a partial/failed/stopped run or rewrite its evidence.
    if read(out / 'status.json')['state'] != 'prepared' or list((out / 'records').iterdir()) or list((out / 'trials').iterdir()):
        raise RuntimeError('This runner starts fresh prepared cohorts only; partial runs require explicit recovery')
    model = preflight(out)
    lock = (ROOT / 'runs/legacy_d1_gpu.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from chronotope.controller import CallBudget
    from chronotope.runner import run_trial, PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    from chronotope_core.controller import build as build_native
    from chronotope.distributional.manifest import EpisodeRow, build_manifest
    from chronotope.distributional.episode import verify_trace_evidence, score_trace
    from run_core5_detector_full_agents3 import make_host
    plan = read(out / 'plan.json'); cases = load_cases(out, plan)
    full = read(out / 'configs/full_agent_components.json')
    records = []; current = None

    def publish(state, error=None):
        summarize(out, records)
        write(out / 'status.json', {'state': state, 'updated_at': now(), 'model': model,
              'completed_trials': len(records), 'planned_trials': 3000, 'qualification_trials': 0,
              'pid': os.getpid() if state == 'running' else None, 'current_job': current,
              'error': error, 'automatic_resume': False})

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for current in plan['jobs']:
            publish('running')
            ident, arm, mode = current['id'], current['arm'], current['mode']
            case = cases[current['manifest']][current['index']]
            config = read(out / 'configs' / f'notification_{arm}.json')
            folder = out / 'trials' / current['cell_id'] / arm; folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{current["episode_id"]}_{mode}.jsonl'
            live = Trace.load(folder / f'{current["episode_id"]}_live.jsonl') if mode == 'replay' else None
            spec = make_trial_spec(case, config, mode, live.file_hash() if live else None)
            row = EpisodeRow.from_dict(case['distributional_row']); scenario = build_manifest(row)
            calls = out / 'calls' / f'{ident}.jsonl'; os.environ['CHRONOTOPE_PLANNER_AUDIT'] = str(calls)
            adapter = HostedAgentS3Adapter(full, calls) if arm == ARMS[1] else None
            controller = None; start = time.monotonic()
            try:
                controller = build_native(spec, budget=CallBudget(100)) if arm == ARMS[0] else make_host(spec, 'notification', adapter, {'code_action': 'defer_unsupported'})
                run_trial(spec, path, PayloadStore(out / 'payloads'), live_trace=live, controller=controller, manifest=scenario)
            finally:
                if controller and hasattr(controller, 'close'): controller.close()
                if adapter: adapter.close()
            trace = Trace.load(path)
            if trace.all('decision_failed') or (adapter and adapter.errors):
                raise RuntimeError('Infrastructure error; interrupted trace retained, no scored record')
            request_evidence = call_audit(calls, arm, model, trace, scenario)
            verify_trace_evidence(row, scenario, trace, case['distributional_manifest_hash'])
            outcome = score_trace(row, trace, case['distributional_manifest_hash'], controller=arm, repeat_index=0)
            pair = validate_pair(trace, live) if live else None
            if outcome.invalid_trial or outcome.contract_violations or (pair and not pair.valid):
                raise RuntimeError('Invalid trial contract or Replay pair; evidence retained unscored')
            record = {'job': current, 'outcome': outcome.to_dict(), 'request_audit': request_evidence,
                      'trace': str(path.relative_to(out)), 'trace_sha256': trace.file_hash(),
                      'pair_valid': pair.valid if pair else None, 'wall_seconds': time.monotonic() - start,
                      'completed_at': now()}
            write(out / 'records' / f'{ident}.json', record); records.append(record)
            print(f'{len(records)}/3000 {ident} success={outcome.Y}', flush=True)
        current = None; publish('auditing'); audit(out); publish('complete')
    except BaseException as exc:
        publish('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', f'{type(exc).__name__}: {exc}')
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'freeze', 'preflight', 'run', 'audit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--build', type=Path, default=ROOT / 'runs/hosted_api_notification_20261102/build')
    args = parser.parse_args(); out = args.out.resolve()
    if args.action == 'prepare':
        if not args.model: parser.error('--model required for prepare')
        prepare(out, args.model, args.build)
    else:
        result = globals()[args.action](out)
        if result: print(json.dumps(result))
