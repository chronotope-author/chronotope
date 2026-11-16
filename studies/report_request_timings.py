"""Read saved evaluation evidence and publish request timing coverage.

This performs no inference, modifies no experiment inputs, and never controls
the evaluator. Historical missing metrics stay missing. All timings are wall
clock intervals with the boundaries documented in timing_status.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time


SCHEMA = 'chronotope-request-timing-report-1'
REQUEST_METRICS = (
    'pure_prefill_ms', 'client_ttft_ms', 'engine_ttft_ms', 'decode_ms',
    'queue_ms', 'mean_itl_ms', 'inference_output_tokens_per_second',
    'request_total_ms',
)
DEFINITIONS = {
    'pure_prefill_ms': 'Unavailable in both saved paths; never inferred from TTFT.',
    'client_ttft_ms': 'Native invoke entry to first token callback; includes input setup, image staging, vision, tokenization and initial inference. The raw field is misleadingly named prefill_ms. Not recorded for the nonstreaming host client.',
    'engine_ttft_ms': 'vLLM scheduled_ts to first_token_ts: engine prefill-to-first-token interval, excluding queue and frontend processing. Not interchangeable with native client_ttft_ms, and not isolated GPU prefill.',
    'decode_ms': 'Native: first token callback to handleRequest return, including finalization. vLLM: first output token to last output token, from response.metrics.generation_time_ms. These boundaries differ slightly.',
    'queue_ms': 'vLLM scheduled_ts minus queued_ts.',
    'mean_itl_ms': 'vLLM first-to-last output interval divided by generated tokens minus one; stock API value, not estimated from HTTP total.',
    'inference_output_tokens_per_second': 'vLLM generated tokens divided by scheduled-to-last-token interval, including engine TTFT. This is not decode-only throughput.',
    'request_total_ms': 'Native: trace deliberation_ms (invoke entry to handleRequest return, before response parsing/audit write). Host: saved HTTP end_ns minus start_ns, including HTTP and response processing.',
    'e_to_a_ms': 'Scored event-to-action receipt interval from the completed trial record; one value per trial, missing when there is no receipt.',
}
SCOPES = {
    'all_requests': 'All recorded planner and reflection requests in completed trials.',
    'first_planner_request': 'Exactly the first planner request in each completed trial, when present.',
    'trial': 'One completed evaluation trial; no duplication for multiple model requests.',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'Non-numeric timing value: {value!r}')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid timing value: {value!r}')
    return value


def percentiles(values):
    values = sorted(number(v) for v in values if v is not None)
    return {f'p{p}': values[math.ceil(p / 100 * len(values)) - 1] if values else None
            for p in (50, 95)}


def extract_request_metrics(arm, call, invocation):
    """Keep unavailable timing fields null; native prefill_ms is TTFT."""
    result = dict.fromkeys(REQUEST_METRICS)
    if arm == 'legacy_cpp':
        result.update(client_ttft_ms=number(invocation.get('prefill_ms')),
                      decode_ms=number(invocation.get('decode_ms')),
                      request_total_ms=number(invocation.get('deliberation_ms')))
        revision = 'native-first-token-callback-v1'
    else:
        result['request_total_ms'] = number((call['end_ns'] - call['start_ns']) / 1e6)
        metrics = call.get('response', {}).get('metrics')
        if metrics is not None and not isinstance(metrics, dict):
            raise ValueError('Unexpected response.metrics format')
        mapping = {
            'time_to_first_token_ms': 'engine_ttft_ms',
            'generation_time_ms': 'decode_ms',
            'queue_time_ms': 'queue_ms',
            'mean_itl_ms': 'mean_itl_ms',
            'tokens_per_second': 'inference_output_tokens_per_second',
        }
        for source, dest in mapping.items():
            result[dest] = number((metrics or {}).get(source))
        revision = ('vllm-per-request-metrics-v1' if any(result[k] is not None for k in mapping.values())
                    else 'vllm-historical-metrics-unavailable')
    return result, revision


def load_record(out, path, execution):
    raw_record = path.read_bytes()
    record = json.loads(raw_record)
    job = record['job']
    if job['phase'] != 'evaluation' or job['task'] != 'notification' or job['family'] != 'closing_window':
        raise ValueError(f'Unexpected active experiment task: {path}')
    if job['arm'] not in ('legacy_cpp', 'agents3_stripped') or job['mode'] not in ('live', 'replay', 'paused'):
        raise ValueError(f'Unexpected arm/condition: {path}')
    trace_bytes = (out / record['trace']).read_bytes()
    if hashlib.sha256(trace_bytes).hexdigest() != record['trace_sha256']:
        raise ValueError(f'Trace hash mismatch: {path}')
    events = [json.loads(line) for line in trace_bytes.splitlines()]
    invocations = [event for event in events if event.get('record') == 'decision_invocation']
    calls_path = out / 'calls' / f'{path.stem}.jsonl'
    calls = read_jsonl(calls_path) if calls_path.exists() else []
    mapped = []
    for event in invocations:
        count = 1 if job['arm'] == 'legacy_cpp' else event['planner_calls'] + event['reflection_calls']
        mapped.extend([event] * count)
    if len(calls) != len(mapped):
        raise ValueError(f'Call/invocation count mismatch: {path}')
    base = {
        'trial_id': path.stem,
        'task': 'Disappearing Notification',
        'arm': job['arm'],
        'controller': execution.get('arm_labels', {}).get(job['arm'], job['arm']),
        'model': execution.get('model'),
        'revision': execution.get('revision'),
        'runtime_precision': execution.get('runtime_precision'),
        'window_s': job['window_s'],
        'condition': job['mode'],
        'case_index': job['index'],
    }
    requests = []
    first_planner_seen = False
    for index, (call, event) in enumerate(zip(calls, mapped)):
        role = call.get('request_kind', call.get('role', 'planner'))
        first = role == 'planner' and not first_planner_seen
        first_planner_seen = first_planner_seen or role == 'planner'
        timings, revision = extract_request_metrics(job['arm'], call, event)
        usage = call.get('response', {}).get('usage', {}) if isinstance(call.get('response'), dict) else {}
        requests.append({**base, 'request_index': index, 'role': role,
            'decision_id': event.get('decision_id'), 'first_planner_request': first,
            'input_tokens': call.get('input_tokens', usage.get('prompt_tokens')),
            'output_tokens': call.get('output_tokens', usage.get('completion_tokens')),
            'timing_revision': revision, **timings,
            'source_calls': str(calls_path.relative_to(out)), 'source_call_line': index + 1,
            'source_trace': record['trace'], 'source_record': str(path.relative_to(out))})
    ea = record['outcome'].get('L_EA')
    trial = {**base, 'success': record['outcome']['Y'],
             'receipt': record['outcome']['R'],
             'e_to_a_ms': None if ea is None else number(ea / 1e6),
             'request_count': len(requests),
             'timing_revisions': sorted(set(row['timing_revision'] for row in requests))}
    return raw_record, requests, trial


def summarize(requests, trials):
    groups = defaultdict(list)
    metadata = {}
    for row in requests:
        key = (row['arm'], row['window_s'], row['condition'])
        metadata[key] = row
        groups[key + ('all_requests',)].append(row)
        if row['first_planner_request']:
            groups[key + ('first_planner_request',)].append(row)
    for trial in trials:
        key = (trial['arm'], trial['window_s'], trial['condition'])
        metadata[key] = trial
        groups[key + ('trial',)].append(trial)
        # Even zero-call conditions retain explicit missing request metrics.
        groups[key + ('all_requests',)]
        groups[key + ('first_planner_request',)]
    result = []
    for (arm, window, condition, scope), rows in sorted(groups.items()):
        meta = metadata[(arm, window, condition)]
        own_trials = [t for t in trials if (t['arm'], t['window_s'], t['condition']) == (arm, window, condition)]
        metrics = ('e_to_a_ms',) if scope == 'trial' else REQUEST_METRICS
        for metric in metrics:
            values = [row.get(metric) for row in rows]
            available = sum(value is not None for value in values)
            result.append({k: meta.get(k) for k in ('task', 'arm', 'controller', 'model', 'revision', 'runtime_precision')} |
                {'window_s': window, 'condition': condition, 'scope': scope, 'metric': metric,
                 'unit': 'tokens/s' if metric.endswith('per_second') else 'ms',
                 'trials': len(own_trials), 'successes': sum(t['success'] for t in own_trials),
                 'receipts': sum(t['receipt'] for t in own_trials), 'eligible_samples': len(rows),
                 'measured_samples': available, 'missing_samples': len(rows) - available,
                 **percentiles(values), 'measurement_boundary': DEFINITIONS[metric]})
    return result


def atomic_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def publish(out, cache=None):
    cache = {} if cache is None else cache
    execution = read_json(out / 'execution_manifest.json')
    batch = read_json(out / 'status.json')
    paths = sorted((out / 'records').glob('*.json'))
    if set(cache) - {path.name for path in paths}:
        raise ValueError('Completed records disappeared')
    for path in paths:
        if path.name in cache:
            if path.read_bytes() != cache[path.name][0]:
                raise ValueError(f'Completed record changed: {path}')
        else:
            cache[path.name] = load_record(out, path, execution)
    requests = [row for _, own, _ in cache.values() for row in own]
    trials = [trial for _, _, trial in cache.values()]
    summaries = summarize(requests, trials)
    dest = out / 'results'
    dest.mkdir(exist_ok=True)
    request_fields = list(requests[0]) if requests else ['trial_id', 'arm', 'window_s', 'condition', *REQUEST_METRICS]
    summary_fields = list(summaries[0]) if summaries else ['arm', 'window_s', 'condition', 'scope', 'metric', 'measured_samples', 'p50', 'p95']
    atomic_csv(dest / 'timing_requests.csv', requests, request_fields)
    atomic_csv(dest / 'timing_summary.csv', summaries, summary_fields)
    revisions = defaultdict(int)
    for row in requests:
        revisions[row['timing_revision']] += 1
    status = {
        'schema': SCHEMA, 'state': 'reporting', 'pid': os.getpid(),
        'updated_at': datetime.now(timezone.utc).isoformat(), 'batch_state': batch['state'],
        'completed_trials_read': len(trials), 'recorded_requests': len(requests),
        'timing_revision_request_counts': dict(revisions),
        'source': str(out), 'definitions': DEFINITIONS, 'scopes': SCOPES,
        'percentile_method': 'nearest rank: sorted[ceil(p*n)-1]',
        'historical_missing_metrics': 'Left null; no estimates or backfill.',
        'model_inference_calls': 0, 'input_files_modified': False,
    }
    atomic_json(dest / 'timing_status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=5.0)
    args = parser.parse_args()
    if not 1 <= args.interval <= 15:
        parser.error('--interval must be between 1 and 15 seconds')
    out = args.out.resolve()
    cache = {}
    previous = None
    while True:
        try:
            status = publish(out, cache)
            progress = (status['completed_trials_read'], status['batch_state'], status['timing_revision_request_counts'])
            if progress != previous:
                print(json.dumps({k: status[k] for k in ('updated_at', 'batch_state', 'completed_trials_read', 'recorded_requests', 'timing_revision_request_counts')}), flush=True)
                previous = progress
            if args.once or status['batch_state'] in ('complete', 'completed', 'failed', 'superseded', 'cancelled'):
                status.update(state='finished', pid=None)
                atomic_json(out / 'results/timing_status.json', status)
                return
        except Exception as exc:
            dest = out / 'results'
            dest.mkdir(exist_ok=True)
            atomic_json(dest / 'timing_status.json', {'schema': SCHEMA, 'state': 'failed', 'pid': None,
                'updated_at': datetime.now(timezone.utc).isoformat(), 'error': f'{type(exc).__name__}: {exc}'})
            raise
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
