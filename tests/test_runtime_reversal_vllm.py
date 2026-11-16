"""CPU-only checks: no HTTP, model inference, GPU operation, or application trial."""
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('profile_vllm', ROOT / 'studies/runtime_reversal/profile_vllm.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VllmProfileGuards(unittest.TestCase):
    def fixture(self, directory):
        out = Path(directory) / 'capture'
        out.mkdir()
        source = ROOT / 'runs/runtime_reversal_20261029/vllm_qwen08_case7'
        for name in ('launch_plan.json', 'request.json'):
            shutil.copy2(source / name, out / name)
        return out, MODULE.read(out / 'launch_plan.json')

    def test_busy_evaluation_lock_prevents_any_http_or_profile_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, _ = self.fixture(temporary)
            lock = Path(temporary) / 'gpu.lock'
            with lock.open('a+') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(MODULE, 'LOCK', lock), patch.object(MODULE, 'request') as http:
                    with self.assertRaises(BlockingIOError):
                        MODULE.capture(out)
                    http.assert_not_called()
                    self.assertFalse((out / 'capture_result.json').exists())

    def test_exactly_one_model_request_and_explicit_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, plan = self.fixture(temporary)
            paths = []
            def http(url, body=None, **kwargs):
                paths.append(url.rsplit('/', 1)[-1])
                if url.endswith('/models'):
                    return 200, {'data': [{'id': plan['model']}]}
                if url.endswith('/chat/completions'):
                    self.assertEqual(body, MODULE.read(out / 'request.json'))
                    return 200, {'model': plan['model'], 'usage': {'prompt_tokens': 998, 'completion_tokens': 137},
                                 'choices': [{'finish_reason': 'stop', 'message': {'content': 'fixture'}}]}
                return 200, None
            with patch.object(MODULE, 'LOCK', Path(temporary) / 'gpu.lock'), patch.object(MODULE, 'request', side_effect=http):
                result = MODULE.capture(out)
            self.assertEqual(paths, ['models', 'start_profile', 'completions', 'stop_profile'])
            self.assertEqual(result['model_requests_executed'], 1)
            self.assertEqual(result['application_episodes'], 0)
            self.assertEqual(result['qualification_trials'], 0)

    def test_inherited_lock_is_retained_by_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / 'gpu.lock'
            with lock.open('a+') as owner, lock.open('a+') as contender:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(MODULE, 'LOCK', lock):
                    with MODULE.gpu_lock(owner.fileno()):
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_unheld_inherited_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / 'gpu.lock'
            with lock.open('a+') as owner, patch.object(MODULE, 'LOCK', lock):
                with self.assertRaisesRegex(RuntimeError, 'did not already hold'):
                    with MODULE.gpu_lock(owner.fileno()):
                        self.fail('Unheld inherited lock was accepted')

    def test_failed_request_still_stops_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, plan = self.fixture(temporary)
            paths = []
            def http(url, body=None, **kwargs):
                paths.append(url.rsplit('/', 1)[-1])
                if url.endswith('/models'):
                    return 200, {'data': [{'id': plan['model']}]}
                if url.endswith('/chat/completions'):
                    raise RuntimeError('test transport failure')
                return 200, None
            with patch.object(MODULE, 'LOCK', Path(temporary) / 'gpu.lock'), patch.object(MODULE, 'request', side_effect=http):
                with self.assertRaisesRegex(RuntimeError, 'test transport failure'):
                    MODULE.capture(out)
            self.assertEqual(paths[-1], 'stop_profile')
            self.assertEqual(MODULE.read(out / 'capture_result.json')['state'], 'failed')


if __name__ == '__main__':
    unittest.main()
