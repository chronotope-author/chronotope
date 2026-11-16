"""Compare saved Qwen Notification summaries; never run model inference."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/legacy_detector_full_agents3_20261029/experiment'
NEW = ROOT / 'runs/notification_qwen27b_edgellm0111_20261030/experiment'
OUT = ROOT / 'results/qwen0111_reproduction_20261030'


def read_summary(folder):
    path = folder / 'summary.csv'
    if not path.exists():
        return {}
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if row['Task'] != 'Disappearing Notification':
            continue
        window = float(row['Window (s)'])
        if window not in (1, 2, 4, 8, 16):
            continue
        assert row['Model'] == 'nvidia/Qwen3.8-27B-NVFP4'
        assert row['Revision'] == '482ca0f3832238542f8f5295dde86b5f22711d80'
        arm = 'Legacy C++ + detector' if row['Runtime'] == 'TensorRT-Edge-LLM' else 'Agent S3 + grounder'
        key = (window, arm, row['Condition'])
        assert key not in result
        result[key] = row
    return result


def main():
    old, new = read_summary(OLD), read_summary(NEW)
    rows = []
    for window in (16, 8, 4, 2, 1):
        for arm in ('Legacy C++ + detector', 'Agent S3 + grounder'):
            for mode in ('live', 'replay', 'paused'):
                reference = old[window, arm, mode]
                current = new.get((window, arm, mode), {})
                count = int(current.get('n', 0))
                assert int(reference['n']) == 100 and count <= 100
                row = {'Task': 'Disappearing Notification', 'Window (s)': window,
                       'Controller': arm, 'Condition': mode,
                       'New cell status': 'complete' if count == 100 else 'partial' if count else 'pending'}
                for prefix, source in [('Old 0.10.0 study', reference), ('New 0.11.1 study', current)]:
                    for field in ('n', 'Successes', 'SR', 'No receipt', 'Invalid',
                                  'E-to-A p50 (ms)', 'E-to-A p95 (ms)'):
                        row[f'{prefix}: {field}'] = source.get(field, '')
                # Avoid comparing a partial case prefix with the historical full cohort.
                row['SR change (percentage points; complete cells only)'] = (
                    100 * (float(current['SR']) - float(reference['SR'])) if count == 100 else '')
                for percentile in ('p50', 'p95'):
                    field = f'E-to-A {percentile} (ms)'
                    row[f'{field} change (complete cells only)'] = (
                        float(current[field]) - float(reference[field])
                        if count == 100 and current.get(field) and reference.get(field) else '')
                rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, selected in [('comparison.csv', rows), ('live_comparison.csv', [r for r in rows if r['Condition'] == 'live'])]:
        path = OUT / name
        temp = path.with_suffix('.csv.tmp')
        with temp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
        temp.replace(path)
    snapshot = {
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'old_experiment': str(OLD), 'new_experiment': str(NEW),
        'completed_new_trials': sum(int(r.get('n', 0)) for r in new.values()),
        'planned_new_trials': 3000,
        'complete_cells': sum(r['New cell status'] == 'complete' for r in rows),
        'total_cells': len(rows),
        'interpretation': 'Descriptive comparison of the two complete configurations. '
                          'The Edge-LLM version changes only in the native arm. '
                          'Partial cells do not establish reproduction; latency quantiles '
                          'are conditional on an action receipt, with missing receipts reported separately.',
        'model_inference_calls': 0, 'experimental_trials': 0,
    }
    (OUT / 'comparison_status.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    print(json.dumps(snapshot))


def watch():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'comparison.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            summary = NEW / 'summary.csv'
            signature = summary.stat().st_mtime_ns if summary.exists() else 0
            if signature != previous:
                main()
                previous = signature
            state = json.loads((NEW.parent / 'priority_status.json').read_text())['state']
            if state in ('complete', 'failed', 'paused', 'stopped'):
                main()
                return
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Refresh CSVs while this study is active; never launch trials.')
    args = parser.parse_args()
    watch() if args.watch else main()
