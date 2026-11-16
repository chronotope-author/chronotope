"""Summarize an already exported Nsight SQLite diagnostic without GPU access.

GPU busy time is an interval union, not a sum across concurrent streams. CUDA
API waits overlap device work and are reported separately. GPU gaps establish
uncovered device time, not their cause. Instrumented timings are diagnostic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import csv
import json
from pathlib import Path
import sqlite3


def intervals_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def aggregate(records, steps):
    groups = defaultdict(list)
    for name, start, end in records:
        groups[name].append(end - start)
    return sorted([{'name': name, 'count': len(durations),
                    'sum_ms': sum(durations) / 1e6,
                    'mean_us': sum(durations) / len(durations) / 1e3,
                    'max_us': max(durations) / 1e3,
                    'sum_ms_per_expected_decode_step': sum(durations) / steps / 1e6}
                   for name, durations in groups.items()], key=lambda r: r['sum_ms'], reverse=True)


def dump_csv(path, records):
    with path.open('w') as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def kernel_family(name):
    """Conservative name grouping; original names remain in the per-kernel CSV."""
    lower = name.lower()
    if 'gdn' in lower or 'gated_delta' in lower:
        return 'Gated-delta recurrent attention'
    if 'gemm' in lower or 'gemv' in lower or 'matmul' in lower:
        if 'cudnn_generated' in lower:
            # The symbol establishes matrix work, not operand precision. The
            # configured FlashInfer FP8 path is corroborated separately.
            return 'cuDNN matrix multiplication'
        if 'block_scaled' in lower or 'nvfp4' in lower or 'e2m1' in lower:
            return 'NVFP4 matrix multiplication'
        if 'e4m3' in lower or 'e5m2' in lower or 'fp8' in lower:
            return 'FP8 matrix multiplication'
        if 'bfloat16' in lower or 'bf16' in lower:
            return 'BF16 matrix multiplication'
        if 'f16' in lower or 'half' in lower:
            return 'FP16 matrix multiplication'
        return 'Other matrix multiplication'
    if ('mha' in lower or 'flashattn' in lower or 'flash_attn' in lower or 'flash_fwd' in lower
            or 'pagedkvcachekernel' in lower or 'persistentvariablelengthmergestateskernel' in lower):
        return 'Attention kernel'
    if 'conv1d' in lower:
        return 'Causal convolution'
    if 'repetition' in lower or 'sampling' in lower or 'topk' in lower:
        return 'Token sampling / repetition penalty'
    return 'Other kernels'


def kernel_families(kernels, steps):
    groups = defaultdict(list)
    for name, start, end in kernels:
        groups[kernel_family(name)].append((start, end))
    return sorted([{'family': family, 'count': len(intervals),
                    'summed_ms': sum(end-start for start,end in intervals)/1e6,
                    'union_ms': sum(end-start for start,end in intervals_union(intervals))/1e6,
                    'union_ms_per_expected_decode_step': sum(end-start for start,end in intervals_union(intervals))/steps/1e6}
                   for family, intervals in groups.items()], key=lambda r:r['union_ms'], reverse=True)


def category_partition(kernels, steps):
    """Partition device time by the exact set of simultaneously active families."""
    events = defaultdict(lambda: defaultdict(int))
    for name, start, end in kernels:
        family = kernel_family(name)
        events[start][family] += 1
        events[end][family] -= 1
    active = defaultdict(int)
    previous = None
    durations = defaultdict(int)
    for instant, deltas in sorted(events.items()):
        if previous is not None and instant > previous:
            families = tuple(sorted(f for f, count in active.items() if count > 0))
            if families:
                durations[families] += instant - previous
        for family, delta in deltas.items():
            active[family] += delta
            assert active[family] >= 0
        previous = instant
    assert all(count == 0 for count in active.values())
    busy = sum(durations.values())
    matrix_ns = sum(duration for families, duration in durations.items()
                    if any('matrix multiplication' in family for family in families))
    families_all = sorted({family for families in durations for family in families})
    intersections = []
    for first, second in combinations(families_all, 2):
        overlap = sum(duration for families, duration in durations.items()
                      if first in families and second in families)
        if overlap:
            intersections.append({'families': [first, second], 'intersection_ms': overlap / 1e6})
    return {
        'kernel_busy_union_ms': busy / 1e6,
        'matrix_kernel_union_ms': matrix_ns / 1e6,
        'matrix_kernel_union_ms_per_expected_decode_step': matrix_ns / steps / 1e6,
        'matrix_fraction_of_kernel_busy_union': matrix_ns / busy,
        'exclusive_family_ms': {family: durations.get((family,), 0) / 1e6 for family in families_all},
        'active_category_partition': sorted([
            {'active_families': list(families), 'union_ms': duration / 1e6,
             'fraction_of_kernel_busy_union': duration / busy}
            for families, duration in durations.items()], key=lambda r:r['union_ms'], reverse=True),
        'pairwise_intersections': sorted(intersections, key=lambda r:r['intersection_ms'], reverse=True),
        'limits': ['The active-category partition is disjoint and sums to kernel busy union.',
                   'Family exclusive time counts only intervals when no other kernel family is active.',
                   'Pairwise intersections may overlap triple intersections; do not sum them for total overlap.',
                   'Kernel names determine categories; original symbols remain available for inspection.'],
    }


def summarize(database, out, steps):
    assert database.is_file() and steps > 0
    out.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    tables = {r['name'] for r in connection.execute("select name from sqlite_master where type='table'")}
    strings = {r['id']: r['value'] for r in connection.execute('select id,value from StringIds')}
    schema = {t: [r['name'] for r in connection.execute(f'pragma table_info("{t}")')]
              for t in sorted(tables) if t.startswith(('CUPTI_ACTIVITY_KIND_', 'NVTX'))}
    records = {}
    gpu_intervals = []
    for table, columns in schema.items():
        if not {'start', 'end'} <= set(columns):
            continue
        name_column = next((name for name in ['demangledName', 'nameId', 'shortName'] if name in columns), None)
        timed = []
        for row in connection.execute(f'select * from "{table}" where end > start'):
            if table == 'NVTX_EVENTS':
                name = row['text'] if 'text' in columns else None
                if not name and 'textId' in columns:
                    name = strings.get(row['textId'])
                name = name or '<unnamed NVTX range>'
            else:
                name = strings.get(row[name_column], str(row[name_column])) if name_column else table
            timed.append((name, int(row['start']), int(row['end'])))
        records[table] = timed
        if table in {'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL',
                     'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'}:
            gpu_intervals.extend((start, end) for _, start, end in timed)
        dump_csv(out / (table.lower() + '.csv'), aggregate(timed, steps))
    merged = intervals_union(gpu_intervals)
    assert merged, 'No timed GPU kernel/memory records in this export'
    first, last = merged[0][0], merged[-1][1]
    busy_ns = sum(end - start for start, end in merged)
    gaps = [{'start_ns': merged[i][1], 'end_ns': merged[i + 1][0],
             'gap_us': (merged[i + 1][0] - merged[i][1]) / 1e3} for i in range(len(merged) - 1)]
    gaps.sort(key=lambda r: r['gap_us'], reverse=True)
    dump_csv(out / 'gpu_gaps.csv', gaps)
    kernel_tables = ['CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL']
    kernels = [r for table in kernel_tables for r in records.get(table, [])]
    runtime = records.get('CUPTI_ACTIVITY_KIND_RUNTIME', [])
    driver = records.get('CUPTI_ACTIVITY_KIND_DRIVER', [])
    nvtx = records.get('NVTX_EVENTS', [])
    summary = {
        'database': str(database), 'expected_decode_steps': steps,
        'scope': 'First to last captured GPU activity; excludes leading/trailing CPU-only time.',
        'gpu_activity_start_ns': first, 'gpu_activity_end_ns': last,
        'gpu_activity_span_ms': (last - first) / 1e6,
        'gpu_busy_union_ms': busy_ns / 1e6,
        'gpu_uncovered_ms': (last - first - busy_ns) / 1e6,
        'gpu_busy_fraction_of_activity_span': busy_ns / (last - first),
        'gpu_activity_span_ms_per_expected_decode_step': (last - first) / steps / 1e6,
        'gpu_busy_union_ms_per_expected_decode_step': busy_ns / steps / 1e6,
        'kernel_count': len(kernels),
        'kernel_sum_ms': sum(end - start for _, start, end in kernels) / 1e6,
        'cuda_graph_launch_calls': sum(name.startswith(('cudaGraphLaunch', 'cuGraphLaunch'))
                                       for name, _, _ in runtime + driver),
        'runtime_api_sum_ms': sum(end - start for _, start, end in runtime) / 1e6,
        'top_kernels': aggregate(kernels, steps)[:30],
        'kernel_family_summary': kernel_families(kernels, steps),
        'kernel_category_partition': category_partition(kernels, steps),
        'top_runtime_apis': aggregate(runtime, steps)[:30],
        'top_driver_apis': aggregate(driver, steps)[:30],
        'generation_nvtx_ranges': aggregate([r for r in nvtx if 'generation' in r[0]], steps),
        'largest_gpu_gaps': gaps[:20],
        'limits': ['Nsight node instrumentation perturbs runtime; these are not evaluation latency observations.',
                   'GPU uncovered intervals alone do not identify CPU, scheduler, graph or memory causes.',
                   'CUDA API duration includes waits overlapping GPU work; do not add it to GPU time.',
                   'Kernel families use names. Family unions may overlap one another; do not sum them as wall time.',
                   'Expected decode count must be validated against profiler audit/NVTX before per-step interpretation.'],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['database', 'kernel_count', 'gpu_activity_span_ms',
                                             'gpu_busy_union_ms', 'gpu_uncovered_ms', 'cuda_graph_launch_calls']}))
    connection.close()
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--decode-steps', type=int, default=32)
    args = parser.parse_args()
    summarize(args.database.resolve(), args.out.resolve(), args.decode_steps)
