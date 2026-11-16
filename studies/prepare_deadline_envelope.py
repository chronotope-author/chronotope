"""Prepare evaluation-only legacy C++ versus stripped Agent S3 deadline cases.
No inference is launched. Missing matching engines/controllers remain explicit.
Paused, Live, and Replay use the same evaluation seeds; no competence screening.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chronotope.distributional.manifest import (
    GenConfig, generate, ManifestSet, build_manifest, expected_event_hash,
    expected_observation_hash, verify_row_against_manifest,
)
from chronotope.hashing import hash_obj

WINDOWS = (.125, .25, .5, 1., 2., 4., 8., 16.)
MODELS = {
    'qwen35_0p8b': {
        'id': 'Qwen/Qwen3.5-0.8B',
        'revision': 'local-sha256:04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696',
        'precision': 'FP16',
        'checkpoint_status': 'local legacy export source verified',
        'legacy_tensorrt_engine_status': 'existing verified FP16 engine',
    },
    'radixark_27b': {
        'id': 'RadixArk/Qwen3.8-27B-NVFP4',
        'revision': '319f741cce68d7914884900c138a1fbb70a42f30',
        'precision': 'NVFP4 MLP/lm_head; FP8 attention; BF16 vision',
        'checkpoint_status': 'downloaded',
        'legacy_tensorrt_engine_status': 'not built for this exact checkpoint',
    },
}
ARMS = ('legacy_cpp', 'agents3_stripped')
MODES = ('live', 'replay', 'paused')


def with_window(row, seconds):
    if row.task_family != 'closing_window':
        raise ValueError('Primary protocol permits Notification only')
    if seconds not in WINDOWS:
        raise ValueError('Window is outside the frozen grid')
    payload = {**row.task_payload, 'episode_horizon_ns': row.event_ns + round(seconds * 1e9) + 500_000_000}
    updated = replace(row, deadline_s=seconds, task_payload=payload)
    manifest = build_manifest(updated)
    updated = replace(updated, expected_event_hash=expected_event_hash(manifest),
                      expected_observation_hash=expected_observation_hash(manifest, None))
    verify_row_against_manifest(updated, manifest)
    assert updated.target_start_position == row.target_start_position
    assert manifest.target_box_at(row.event_ns) == build_manifest(row).target_box_at(row.event_ns)
    return updated


def observed_threshold(points, criterion=.90):
    """Descriptive threshold, never a gate for running or retaining a cell."""
    ps = sorted(points, key=lambda p: p['window_s'])
    if not ps or any(p['n'] != 100 for p in ps):
        raise ValueError('Requires complete 100-trial cells')
    if len({p['window_s'] for p in ps}) != len(ps):
        raise ValueError('Duplicate window')
    if any(not 0 <= p['successes'] <= p['n'] for p in ps):
        raise ValueError('Invalid success count')
    for i, p in enumerate(ps):
        if all(q['successes'] / q['n'] >= criterion for q in ps[i:]):
            return {'window_s': p['window_s'], 'lower_tested_window_s': ps[i-1]['window_s'] if i else None,
                    'status': 'observed' if i else 'below_tested_range'}
    return {'window_s': None, 'lower_tested_window_s': ps[-1]['window_s'], 'status': 'not_reached'}


def assert_model_match(legacy, agents3):
    """Same-model system comparison: checkpoint and component precision must match.

    Runtimes intentionally differ. This does not establish an overhead-only
    intervention or verify checkpoint conversion; those need separate evidence.
    """
    for key in ('checkpoint_sha256', 'precision'):
        if not legacy.get(key) or legacy[key] != agents3.get(key):
            raise ValueError(f'Unmatched or unspecified model: {key}')


def prepare(out, model_key="qwen35_0p8b"):
    models={model_key:MODELS[model_key]}
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / 'manifests').mkdir()
    base = generate(GenConfig(task_family='closing_window', n=100, master_seed=26102832,
                             seed_base=8400000, empirical_windows_s=(16.,), generous_deadline_s=16.))
    hashes, jobs = {}, []
    for w in reversed(WINDOWS):
        rows = [with_window(r, w) for r in base.rows]
        ms = ManifestSet(header={'schema': 'distributional-manifest-1', 'n': 100,
                                'study': 'legacy-vs-agents3-deadline-envelope',
                                'task': 'Disappearing Notification', 'phase': 'evaluation',
                                'window_seconds': w, 'base_geometry_hash': hash_obj([r.to_dict() for r in base.rows])}, rows=rows)
        name = f'evaluation_notification_w{round(w*1e9)}.jsonl'
        ms.freeze(out / 'manifests' / name)
        hashes[name] = ms.frozen_hash
        for index, row in enumerate(rows):
            for model in (list(models) if index % 2 == 0 else list(reversed(models))):
                for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS))):
                    for mode in MODES:
                        jobs.append({'phase': 'evaluation', 'task': 'notification', 'family': 'closing_window',
                                     'model': model, 'arm': arm, 'index': index, 'episode_id': row.episode_id,
                                     'window_s': w, 'mode': mode, 'manifest': name})
    assert len(jobs) == 4800 and all(j['phase'] == 'evaluation' and j['task'] == 'notification' for j in jobs)
    plan = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'state': 'prepared_not_runnable',
        'comparison': 'Identical checkpoint: fresh legacy C++/TensorRT versus stripped Agent S3/vLLM trials',
        'models': models, 'arms': ARMS, 'windows_s': WINDOWS, 'modes': MODES,
        'primary_task': 'Disappearing Notification', 'trials_per_cell': 100,
        'planned_trials': len(jobs), 'jobs': jobs, 'manifest_hashes': hashes,
        'evaluation_policy': 'No separate qualification; retain every configuration and competence failure',
        'success_criterion': 'Descriptive: at least 90/100 Live sustained at wider tested windows; not an eligibility gate',
        'required_setup': ['Matching checkpoint and component precision in both systems',
                           'Matching TensorRT engine and verified controller adapters',
                           'Serial GPU reservation after current batch'],
        'notes': ['No inference or trial has run', 'No separate qualification phase or competence gating',
                  'Paused/Live/Replay share the evaluation cohort; all results are reported',
                  'This primary study covers Notification only',
                  'System comparison; inference and control time must be distinguished for attribution'],
    }
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (out / 'validation.json').write_text(json.dumps({
        'jobs': len(jobs), 'manifests': len(hashes), 'validated_rows': 800, 'scope': ['notification'],
        'same_geometry_across_windows': True, 'separate_qualification_trials': 0,
        'inference_calls': 0, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    with (out / 'planned_cells.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Model role', 'Model', 'Precision', 'Controller', 'Window (s)',
                         'Live n planned', 'Replay n planned', 'Paused n planned', 'Status'])
        for role, model in models.items():
            for arm in ARMS:
                for w in reversed(WINDOWS):
                    writer.writerow(['Disappearing Notification', role, model['id'], model['precision'],
                                     arm, w, 100, 100, 100, 'NOT RUN — controller/engine setup incomplete'])
    print(json.dumps({'out': str(out), 'jobs': len(jobs), 'state': plan['state'], 'inference_calls': 0}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model',choices=list(MODELS),default='qwen35_0p8b')
    args=ap.parse_args();prepare(args.out.resolve(),args.model)
