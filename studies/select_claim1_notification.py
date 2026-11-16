"""Select the user's measured Agent S3 Qwen0.8B Notification table for Claim 1."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/claim1_existing_20261105/verified_results.csv'
OUT = ROOT / 'results/claim1_paper_20261105'
PAPER = ROOT / 'paper_chronotope_iclr2027/generated'
CONFIG = 'notification_qwen08_edgellm0111_20261031__5e23cf1f89df'


def main():
    rows = [r for r in csv.DictReader(SOURCE.open()) if r['config_id'] == CONFIG]
    rows.sort(key=lambda r: -float(r['setting_s']))
    assert [float(r['setting_s']) for r in rows] == [16, 8, 4, 2, 1]
    for row in rows:
        assert row['action_policy'] == 'Keep action' and row['prl_seed_sets_identical'] == 'True'
        assert row['matched_seed_triplets'] == row['replay_source_hash_verified'] == '100'
        for mode in ('paused', 'replay', 'live'):
            assert row[mode + '_trials'] == '100' and row[mode + '_coverage'] == 'complete'
    OUT.mkdir(exist_ok=True); PAPER.mkdir(exist_ok=True)
    for target in (OUT / 'notification.csv', PAPER / 'claim1_notification_qwen08.csv'):
        with target.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ['% Measured values selected by studies/select_claim1_notification.py.',
             '% Full Agent S3; Qwen3.5-0.8B FP16 planner/vision/KV; UI-TARS BF16; Keep action.']
    markdown = ['# Selected Claim 1 Notification results', '',
        'Full Agent S3, Qwen/Qwen3.5-0.8B FP16 planner/vision and KV, local vLLM; '
        'UI-TARS-1.5-7B BF16 grounder and KV. Keep action. Success evaluated at application receipt.', '',
        '| Task | Setting | Paused | Replay | Live |', '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        setting = int(float(row['setting_s']))
        cells = [row[m + '_cell'] for m in ('paused', 'replay', 'live')]
        lines.append('Disappearing Notification & ' + str(setting) + r'~s & ' + ' & '.join(cells) + r' \\')
        markdown.append(f'| Disappearing Notification | {setting} s | ' + ' | '.join(cells) + ' |')
    (PAPER / 'claim1_notification_qwen08_rows.tex').write_text('\n'.join(lines) + '\n')
    markdown.extend(['', 'Source: `runs/notification_qwen08_edgellm0111_20261031/experiment`.',
        'Planner revision: `2fc06364715b967f1860aea9cf38778875588b17` (BF16 source weights; FP16 inference).',
        'Grounder revision: `683d002dd99d8f95104d31e70391a39348857f4e`.',
        'All cells have the same 100 seeds (8400000–8400099); 1,500 measured Agent S3 trials.',
        'The new other-task evaluation is `runs/claim1_agents3_qwen08_remaining3_20261105/experiment`.',
        'Those rows remain unmeasured until evaluation completes. Their state-change settings are '
        '16/8/4/2/1 s with a separate 32 s task deadline; 100 seeds (1000–1099) per task, setting, and mode.'])
    (OUT / 'notification.md').write_text('\n'.join(markdown) + '\n')
    provenance = dict(selected_config_id=CONFIG, source_csv=str(SOURCE),
        source_csv_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_run='runs/notification_qwen08_edgellm0111_20261031/experiment',
        output_csv_sha256=hashlib.sha256((OUT / 'notification.csv').read_bytes()).hexdigest(),
        selected_trials=1500, qualification_trials_included=0, action_policy='Keep action',
        user_selection='Annotated full Agent S3 Qwen3.5-0.8B Notification table, 2026-11-05',
        verification='results/claim1_existing_20261105/verification.json',
        per_trial_provenance='results/claim1_existing_20261105/trials.csv',
        other_task_results='pending; never filled from illustrative or historical mismatched results')
    (OUT / 'selection.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (PAPER / 'claim1_notification_qwen08_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
