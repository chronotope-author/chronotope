"""CPU-only recovery contracts; no provider, grounder, or evaluation calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies'))
import resume_hosted_notification as recovery


def fixture_jobs():
    return [{'id': f'legacy_case_{mode}', 'mode': mode, 'episode_id': 'case',
             'cell_id': 'notification_w8000000000', 'arm': 'legacy_cpp_detector'}
            for mode in ('live', 'replay', 'paused')]


class HostedResumeContracts(unittest.TestCase):
    def test_completed_failures_are_retained_and_only_missing_jobs_selected(self):
        jobs = fixture_jobs()
        records = {jobs[0]['id']: {'job': jobs[0], 'outcome': {'Y': 0}}}
        self.assertEqual(recovery.select_pending({'jobs': jobs}, records), jobs[1:])
        records[jobs[1]['id']] = {'job': jobs[1], 'outcome': {'Y': 1}}
        self.assertEqual(recovery.select_pending({'jobs': jobs}, records), jobs[2:])
        with self.assertRaises(ValueError):
            recovery.select_pending({'jobs': jobs[1:]}, {})
        with self.assertRaises(ValueError):
            recovery.select_pending({'jobs': jobs}, {'unexpected': records[jobs[0]['id']]})

    def test_replay_archive_preserves_live_evidence_and_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); jobs = fixture_jobs()
            records = {jobs[0]['id']: {'job': jobs[0], 'outcome': {'Y': 0}}}
            for job in jobs[:2]:
                for path in [recovery.trial_path(out, job), out / 'calls' / f'{job["id"]}.jsonl']:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(job['mode'] + ' evidence')
            recovery.base.write(out / 'records' / f'{jobs[0]["id"]}.json', records[jobs[0]['id']])
            kept, interrupted = recovery.evidence_inventory(out, {'jobs': jobs}, records)
            self.assertEqual(len(kept), 3)
            self.assertEqual(len(interrupted), 2)
            folder = out / 'recoveries' / 'fixture'
            recovery.archive_interrupted(out, folder, interrupted)
            recovery.verify_hashes(out, kept)
            recovery.verify_hashes(folder / 'interrupted', interrupted)
            self.assertFalse(recovery.trial_path(out, jobs[1]).exists())
            with self.assertRaises(FileNotFoundError):
                recovery.archive_interrupted(out, folder, interrupted)
            # A replacement trial cannot overwrite the archived first attempt.
            for relative in interrupted:
                (out / relative).write_bytes((folder / 'interrupted' / relative).read_bytes())
            with self.assertRaises(FileExistsError):
                recovery.archive_interrupted(out, folder, interrupted)

    def test_tampered_evidence_prevents_archiving(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); path = out / 'failed.jsonl'; path.write_text('original')
            hashes = {'failed.jsonl': recovery.base.digest(path)}
            path.write_text('changed')
            with self.assertRaises(ValueError):
                recovery.archive_interrupted(out, out / 'recovery', hashes)
            self.assertTrue(path.exists())
            self.assertFalse((out / 'recovery').exists())

    def test_recovery_manifest_cannot_be_launched_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / 'cohort'; (root / 'runs').mkdir()
            (out / 'records').mkdir(parents=True)
            recovery.base.write(out / 'plan.json', {'jobs': fixture_jobs()})
            recovery.base.write(out / 'execution_manifest.json', {})
            recovery.base.write(out / 'status.json', {'state': 'failed'})
            manifest = {'schema': 'hosted-explicit-resume-1', 'cohort': str(out),
                'plan_sha256': recovery.base.digest(out / 'plan.json'),
                'execution_manifest_sha256': recovery.base.digest(out / 'execution_manifest.json'),
                'prior_status_sha256': recovery.base.digest(out / 'status.json'),
                'helper_sha256': recovery.base.digest(recovery.__file__), 'completed_ids': [],
                'pending_ids': [j['id'] for j in fixture_jobs()],
                'retained_evidence_sha256': {}, 'interrupted_evidence_sha256': {}}
            target = out / 'recoveries' / 'fixture' / 'manifest.json'
            recovery.base.write(target, manifest)
            with patch.object(recovery.base, 'ROOT', root), \
                 patch.object(recovery.base, 'preflight', return_value='fixture'), \
                 patch.object(recovery, 'execute_pending') as execute:
                recovery.run(out, target)
                with self.assertRaises(FileExistsError): recovery.run(out, target)
                execute.assert_called_once()


if __name__ == '__main__':
    unittest.main()
