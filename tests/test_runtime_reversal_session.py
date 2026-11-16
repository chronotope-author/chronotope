"""CPU-only lifecycle checks; no real process signals, HTTP, or GPU calls."""
import importlib.util
import io
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('profile_session', ROOT / 'studies/runtime_reversal/profile_session.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiagnosticLifetime(unittest.TestCase):
    def fixture(self):
        command = ['/venv/bin/vllm', 'serve', '/checkpoint', '--port', '8014']
        wrapper = {'pid': 101, 'command': ['/nsys', 'profile', *command], 'starttime_ticks': 100}
        target = {'pid': 102, 'command': ['/venv/bin/python', *command], 'starttime_ticks': 101}
        return {'process': Mock(), 'identity': wrapper, 'target': None,
                'plan': {'service_command': command, 'base_url': 'http://127.0.0.1:8014/v1', 'model': 'fixture'}}, target

    def test_exact_target_excludes_wrapper_containing_same_command_suffix(self):
        diagnostic, target = self.fixture()
        identities = {101: diagnostic['identity'], 102: target}
        with patch.object(Path, 'iterdir', return_value=[Path('/proc/101'), Path('/proc/102')]), \
             patch.object(MODULE.CONTROL, 'process', side_effect=lambda pid: identities[int(pid)]), \
             patch.object(Path, 'read_text', return_value='102 (python) S 101 102 102'):
            self.assertEqual(MODULE.diagnostic_target(diagnostic), target)

    def test_ready_target_survives_wrapper_exit(self):
        diagnostic, target = self.fixture()
        diagnostic['process'].poll.return_value = 0
        def discover(value):
            value['target'] = target
            return target
        with patch.object(MODULE, 'diagnostic_target', side_effect=discover), \
             patch.object(MODULE.urllib.request, 'urlopen', return_value=io.BytesIO(b'{"data":[{"id":"fixture"}]}')), \
             patch.object(MODULE.CONTROL, 'write'):
            MODULE.wait_diagnostic(diagnostic, Path('/tmp/fixture/service.log'))
        self.assertEqual(diagnostic['target'], target)

    def test_target_stops_before_waiting_for_exporter(self):
        diagnostic, target = self.fixture()
        events = []
        diagnostic['process'].wait.side_effect = lambda **kw: events.append('wait_exporter')
        with patch.object(MODULE, 'diagnostic_target', return_value=target), \
             patch.object(MODULE, 'diagnostic_members', side_effect=[[target, diagnostic['identity']], []]), \
             patch.object(MODULE.CONTROL, 'still_same', return_value=True), \
             patch.object(MODULE.CONTROL, 'write'), \
             patch.object(MODULE.os, 'kill', side_effect=lambda pid, sig: events.append(('signal', pid))):
            MODULE.stop_diagnostic(diagnostic, 'fixture_stop.json')
        self.assertEqual(events, [('signal', target['pid']), 'wait_exporter'])

    def test_adopted_process_tracks_identity_without_inventing_exit_status(self):
        diagnostic, _ = self.fixture()
        adopted = MODULE.AdoptedProcess(diagnostic['identity'])
        with patch.object(MODULE.CONTROL, 'process', side_effect=[diagnostic['identity'], None]):
            self.assertIsNone(adopted.poll())
            self.assertEqual(adopted.wait(timeout=1), 'unavailable_non_child')


if __name__ == '__main__':
    unittest.main()
