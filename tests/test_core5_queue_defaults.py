"""CPU-only checks for conservative queue defaults and successor isolation."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studies'))
import queue_gemma_after_legacy_qwen as queue


def default_choice():
    return {'typed_action_support': {'state': 'conservative_default',
            'selection': 'defer_unsupported', 'user_authorization': None,
            'basis': 'Preserve the accepted controller and continue independent work.'}}


def status(state='complete', count=4800):
    return {'state': state, 'completed_trials': count, 'planned_trials': count,
            'qualification_trials': 0, 'error': None}


class QueueDefaultsTests(unittest.TestCase):
    def test_default_is_not_typing_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue.write(root / 'decisions.json', default_choice())
            self.assertTrue(queue.decisions_gate(root))
            self.assertFalse(queue.action_status(root)['typing_change_authorized'])
            self.assertEqual(queue.action_status(root)['deferred_task'], 'code')
            for update in ({'selection': 'native_type_value_and_host_bridge'},
                           {'user_authorization': 'silence'}, {'basis': ''},
                           {'state': 'pending'}):
                choice = default_choice()
                choice['typed_action_support'].update(update)
                queue.write(root / 'decisions.json', choice)
                self.assertFalse(queue.decisions_gate(root))

    def test_explicit_choices_still_require_actual_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for selection in ('legacy_unsupported_report_failures', 'native_type_value_and_host_bridge'):
                choice = {'typed_action_support': {'state': 'approved', 'selection': selection,
                          'user_authorization': 'Explicit user selection'}}
                queue.write(root / 'decisions.json', choice)
                self.assertTrue(queue.decisions_gate(root))
                choice['typed_action_support']['user_authorization'] = None
                queue.write(root / 'decisions.json', choice)
                self.assertFalse(queue.decisions_gate(root))

    def test_input_configuration_preserves_default_and_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue.write(root / 'decisions.json', default_choice())
            with patch.object(queue, 'QWEN', root / 'qwen/experiment'), \
                 patch.object(queue, 'GEMMA', root / 'gemma/experiment'):
                paths = queue.seal_action_support(root)
                for path in paths:
                    support = queue.read(path)
                    self.assertTrue(support['configuration_reviewed'])
                    self.assertFalse(support['user_authorized'])
                    self.assertEqual(support['code_action'], 'defer_unsupported')
                queue.write(paths[0], {**queue.read(paths[0]), 'user_authorized': True})
                with self.assertRaisesRegex(AssertionError, 'Conflicting action-support'):
                    queue.seal_action_support(root)

    def test_authorized_window_selection_reaches_both_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); decisions = default_choice()
            quote = "Don't do 0.5s for now. We can do that later as needed"
            decisions['window_selection'] = {'deferred_windows_s': [0.5], 'user_authorization': quote}
            queue.write(root / 'decisions.json', decisions)
            with patch.object(queue, 'QWEN', root / 'qwen/experiment'), \
                 patch.object(queue, 'GEMMA', root / 'gemma/experiment'):
                self.assertEqual(queue.study_count(root, queue.QWEN), 10200)
                self.assertEqual(queue.study_count(root, queue.GEMMA), 14400)
                for path in queue.seal_action_support(root):
                    self.assertEqual(queue.read(path)['deferred_windows_s'], [0.5])
                    self.assertEqual(queue.read(path)['window_selection_authorization'], quote)
                    self.assertFalse(queue.read(path)['user_authorized'])
            for windows, authorization in [([1.0], quote), ([0.5], ''), ([0.5, 0.5], quote)]:
                decisions['window_selection'] = {'deferred_windows_s': windows, 'user_authorization': authorization}
                queue.write(root / 'decisions.json', decisions)
                with self.assertRaises(AssertionError):
                    queue.window_selection(root)

    def test_all_subsecond_windows_reserved_for_smaller_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); decisions = default_choice()
            windows = [0.5, 0.25, 0.125]
            quote = 'Yeah. Save the <1s window runs for smaller models <=4B'
            decisions['window_selection'] = {'deferred_windows_s': windows, 'user_authorization': quote}
            queue.write(root / 'decisions.json', decisions)
            with patch.object(queue, 'QWEN', root / 'qwen/experiment'), \
                 patch.object(queue, 'GEMMA', root / 'gemma/experiment'):
                self.assertEqual(queue.study_count(root, queue.QWEN), 10200)
                self.assertEqual(queue.study_count(root, queue.GEMMA), 13200)
                self.assertEqual(queue.study_count(root, queue.QWEN) + queue.study_count(root, queue.GEMMA), 23400)
                for path in queue.seal_action_support(root):
                    self.assertEqual(queue.read(path)['deferred_windows_s'], windows)
                    self.assertEqual(queue.read(path)['window_selection_authorization'], quote)
                scoped = {**status('complete_scoped', 15000), 'completed_trials': 13200,
                          'runnable_trials': 13200, 'deferred_trials': 1800,
                          'deferred_windows_s': sorted(windows), 'user_deferred_window_trials': 1800,
                          'unsupported_action_deferred_trials': 0}
                self.assertTrue(queue.study_completion_gate(root, queue.GEMMA, scoped)[0])
                for change in ({'completed_trials': 14400}, {'deferred_windows_s': [0.5]},
                               {'user_deferred_window_trials': 600}, {'unsupported_action_deferred_trials': 2400},
                               {'state': 'blocked_action_support'}):
                    self.assertFalse(queue.study_completion_gate(root, queue.GEMMA, {**scoped, **change})[0])

    def test_window_only_completion_is_distinct_from_unsupported_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); decisions = default_choice()
            decisions['window_selection'] = {'deferred_windows_s': [0.5], 'user_authorization': 'Defer 0.5s'}
            queue.write(root / 'decisions.json', decisions)
            scoped = {**status('complete_scoped', 15000), 'completed_trials': 14400,
                      'runnable_trials': 14400, 'deferred_trials': 600, 'deferred_windows_s': [0.5],
                      'user_deferred_window_trials': 600, 'unsupported_action_deferred_trials': 0}
            self.assertTrue(queue.study_completion_gate(root, queue.GEMMA, scoped)[0])
            for change in ({'state': 'blocked_action_support'}, {'unsupported_action_deferred_trials': 2400},
                           {'completed_trials': 12000}, {'deferred_windows_s': [0.25]}, {'deferred_trials': 3000}):
                self.assertFalse(queue.study_completion_gate(root, queue.GEMMA, {**scoped, **change})[0])
            self.assertFalse(queue.study_completion_gate(root, queue.QWEN,
                             {**status('blocked_action_support', 10200), 'completed_trials': 7800})[0])

    def test_current_notification_uses_validated_schedule_helpers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(queue, 'planned_count', return_value=4650) as count, \
                 patch.object(queue, 'completion_gate_for', return_value=(True, 'complete_scheduled')) as gate:
                self.assertEqual(queue.study_count(root, queue.CURRENT), 4650)
                s = status('complete_scheduled', 4650)
                self.assertTrue(queue.study_completion_gate(root, queue.CURRENT, s)[0])
                count.assert_called_once_with(queue.CURRENT)
                gate.assert_called_once_with(queue.CURRENT, s)

    def test_new_current_schedule_requires_explicit_queue_repin(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); current = root / 'current'; current.mkdir()
            manifest = {'queue_source': 'source', 'queue_source_sha256': 'hash',
                        'plan_hashes': {}, 'predecessor_execution_sha256': 'hash',
                        'predecessor_schedule_sha256': None}
            with patch.object(queue, 'CURRENT', current), patch.object(queue, 'digest', return_value='hash'), \
                 patch.object(queue, 'verify_hashes'):
                queue.validate_pins(root, manifest)
                queue.write(current / 'continuation_schedule.json', {})
                with self.assertRaisesRegex(AssertionError, 'schedule changed'):
                    queue.validate_pins(root, manifest)

    def test_blocked_or_incomplete_study_cannot_satisfy_successor_gate(self):
        for state in ('blocked_action_support', 'running', 'failed', 'stopped', 'paused'):
            self.assertFalse(queue.completion_gate(status(state, 10200), 10200)[0])
        self.assertFalse(queue.completion_gate({**status(count=10200), 'completed_trials': 7800}, 10200)[0])
        self.assertTrue(queue.completion_gate(status(count=10200), 10200)[0])

    def test_running_predecessor_never_launches_phases_even_with_default(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder); current = root / 'current'
            queue.write(root / 'queue_manifest.json', {'root': str(root)})
            queue.write(root / 'decisions.json', default_choice())
            queue.write(current / 'status.json', status('running'))
            stack.enter_context(patch.object(queue, 'CURRENT', current))
            stack.enter_context(patch.object(queue, 'validate_pins'))
            stack.enter_context(patch.object(queue, 'pipeline_readiness', return_value=('ready', None)))
            stack.enter_context(patch.object(queue.signal, 'signal'))
            stack.enter_context(patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt))
            commands = stack.enter_context(patch.object(queue, 'command'))
            processes = stack.enter_context(patch.object(queue.subprocess, 'Popen'))
            with self.assertRaises(KeyboardInterrupt):
                queue.run(root, 1)
            commands.assert_not_called(); processes.assert_not_called()
            self.assertFalse((root / 'handoff_started.json').exists())

    def test_deferred_qwen_never_builds_or_launches_gemma(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder); current = root / 'current'; qwen = root / 'qwen'; gemma = root / 'gemma'
            for name, value in (('CURRENT', current), ('QWEN', qwen), ('GEMMA', gemma), ('LOCK', root / 'gpu.lock')):
                stack.enter_context(patch.object(queue, name, value))
            queue.write(root / 'queue_manifest.json', {'root': str(root)})
            queue.write(root / 'decisions.json', default_choice())
            queue.write(root / 'pipeline_manifest.json', {'commands': queue.default_commands(),
                        'source_hashes': {str(queue.RUNNER): 'hash'}})
            queue.write(current / 'status.json', status())
            queue.write(qwen / 'status.json', {'state': 'frozen_ready'})
            queue.write(qwen / 'plan.json', {'planned_trials': 10200})
            queue.write(qwen / 'execution_manifest.json', {'planned_trials': 10200,
                        'qualification_trials': 0, 'hashes': {}, 'action_support': {'code_action': 'defer_unsupported'}})
            events = []
            def command(_root, name, _argv, _lock=None):
                events.append(name)
                log = root / (name + '.log')
                if name == 'current_qwen_completion_audit':
                    queue.write(log, {'state': 'passed', 'completed_trials': 4800, 'planned_trials': 4800,
                                      'model_inference_calls': 0, 'experimental_trials': 0,
                                      'qualification_trials': 0, 'plan_sha256': 'hash'})
                    # Audit parser consumes the final JSON line.
                    import json
                    log.write_text(json.dumps(queue.read(log)) + '\n')
                return log
            def popen(_argv, **_kwargs):
                events.append('qwen_runner')
                queue.write(qwen / 'status.json', {**status('blocked_action_support', 10200),
                            'completed_trials': 7800, 'deferred_trials': 2400})
                process = Mock(pid=123); process.poll.return_value = 0
                return process
            stack.enter_context(patch.object(queue, 'validate_pins'))
            stack.enter_context(patch.object(queue, 'verify_hashes'))
            stack.enter_context(patch.object(queue, 'digest', return_value='hash'))
            stack.enter_context(patch.object(queue, 'study_processes', return_value=[]))
            stack.enter_context(patch.object(queue, 'pipeline_readiness', return_value=('ready', None)))
            stack.enter_context(patch.object(queue, 'command', side_effect=command))
            stack.enter_context(patch.object(queue.subprocess, 'Popen', side_effect=popen))
            stack.enter_context(patch.object(queue.signal, 'signal'))
            stack.enter_context(patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt))
            with self.assertRaises(KeyboardInterrupt):
                queue.run(root, 1)
            self.assertEqual(events, ['current_qwen_completion_audit', *queue.QWEN_PHASES, 'qwen_runner'])
            self.assertFalse(any(event.startswith('gemma') for event in events))
            self.assertFalse(queue.read(root / 'queue_status.json')['typing_change_authorized'])


if __name__ == '__main__':
    unittest.main()
