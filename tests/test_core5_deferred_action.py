"""CPU-only scheduler checks; no services, model calls, or benchmark episodes."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studies'))
import run_core5_detector_full_agents3 as runner


def plan(include_notification=False):
    tasks = dict(drift=4, code=4, button=4, trading=5)
    if include_notification:
        tasks = {'notification': 8, **tasks}
    jobs = [dict(id=f'{task}_{setting}_{index}_{arm}_{mode}', task=task,
                 arm=arm, mode=mode, index=index, setting=setting,
                 window_s=(16, 8, 4, 2, 1, 0.5, 0.25, 0.125)[setting]
                          if task == 'notification' else 16)
            for task, settings in tasks.items() for setting in range(settings)
            for index in range(100) for arm in runner.ARMS
            for mode in ('live', 'replay', 'paused')]
    return {'jobs': jobs, 'planned_trials': len(jobs)}


def support(choice='defer_unsupported', authorized=False):
    return {'configuration_reviewed': True, 'code_action': choice, 'user_authorized': authorized}


class DeferredActionTests(unittest.TestCase):
    def test_qwen_unrelated_jobs_proceed_with_original_order_and_full_denominator(self):
        cohort = plan(); original = list(cohort['jobs'])
        ready, deferred = runner.execution_jobs(cohort, support())
        self.assertEqual(Counter(j['task'] for j in ready), {'drift': 2400, 'button': 2400, 'trading': 3000})
        self.assertEqual(len(deferred), 2400)
        self.assertEqual(ready, [j for j in original if j['task'] != 'code'])
        self.assertEqual(cohort['jobs'], original)
        self.assertEqual(cohort['planned_trials'], 10200)
        self.assertEqual({j['arm'] for j in deferred}, set(runner.ARMS))
        self.assertEqual({j['mode'] for j in deferred}, {'live', 'replay', 'paused'})

    def test_gemma_retains_notification_and_all_supported_jobs(self):
        cohort = plan(True)
        ready, deferred = runner.execution_jobs(cohort, support())
        self.assertEqual(Counter(j['task'] for j in ready),
                         {'notification': 4800, 'drift': 2400, 'button': 2400, 'trading': 3000})
        self.assertEqual(len(ready) + len(deferred), 15000)
        completed = {j['id'] for j in ready}
        self.assertEqual(runner.completion_state(cohort, completed, deferred), 'blocked_action_support')
        with self.assertRaisesRegex(AssertionError, 'Runnable jobs remain'):
            runner.completion_state(cohort, completed - {ready[-1]['id']}, deferred)
        with self.assertRaisesRegex(AssertionError, 'must not have scored'):
            runner.completion_state(cohort, completed | {deferred[0]['id']}, deferred)

    def test_deferral_never_grants_typed_action_authorization(self):
        for choice, authorized in [('native_type_value_and_host_bridge', False),
                                   ('legacy_unsupported_report_failures', False),
                                   ('defer_unsupported', True), ('pending', False)]:
            with self.subTest(choice=choice), self.assertRaises(AssertionError):
                runner.execution_jobs(plan(), support(choice, authorized))
        with self.assertRaises(AssertionError):
            runner.execution_jobs(plan(), {**support(), 'configuration_reviewed': False})

    def test_explicit_existing_choices_still_run_full_plan(self):
        cohort = plan()
        for choice in ('native_type_value_and_host_bridge', 'legacy_unsupported_report_failures'):
            with self.subTest(choice=choice):
                ready, deferred = runner.execution_jobs(cohort, support(choice, True))
                self.assertEqual(ready, cohort['jobs'])
                self.assertEqual(deferred, [])
                self.assertEqual(runner.completion_state(cohort, {j['id'] for j in ready}, deferred), 'complete')

    def test_only_half_second_notification_is_deferred(self):
        cohort = plan(True)
        selected = {**support(), 'deferred_windows_s': [0.5],
                    'window_selection_authorization': "Don't do 0.5s for now. We can do that later as needed."}
        ready, deferred = runner.execution_jobs(cohort, selected)
        windows = Counter(job['window_s'] for job in ready if job['task'] == 'notification')
        self.assertEqual(windows, {16: 600, 8: 600, 4: 600, 2: 600, 1: 600, 0.25: 600, 0.125: 600})
        self.assertEqual(Counter(job['task'] for job in ready),
                         {'notification': 4200, 'drift': 2400, 'button': 2400, 'trading': 3000})
        detail = runner.deferral_status(ready, deferred)
        self.assertEqual((detail['runnable_trials'], detail['deferred_trials']), (12000, 3000))
        self.assertEqual(detail['unsupported_action_deferred_trials'], 2400)
        self.assertEqual(detail['user_deferred_window_trials'], 600)
        self.assertEqual(detail['deferred_task'], 'code')
        self.assertEqual(detail['deferred_window_task'], 'notification')
        self.assertEqual(runner.completion_state(cohort, {job['id'] for job in ready}, deferred), 'blocked_action_support')
        self.assertEqual(cohort['planned_trials'], 15000)

    def test_user_window_deferral_alone_finishes_only_the_selected_scope(self):
        cohort = plan(True)
        selected = {**support('native_type_value_and_host_bridge', True), 'deferred_windows_s': [0.5],
                    'window_selection_authorization': "Don't do 0.5s for now."}
        ready, deferred = runner.execution_jobs(cohort, selected)
        self.assertEqual(len(ready), 14400)
        self.assertEqual(len(deferred), 600)
        detail = runner.deferral_status(ready, deferred)
        self.assertEqual(detail['unsupported_action_deferred_trials'], 0)
        self.assertIsNone(detail['deferred_task'])
        self.assertEqual(runner.completion_state(cohort, {job['id'] for job in ready}, deferred), 'complete_scoped')

    def test_window_deferral_requires_direction_and_cannot_expand_to_other_windows(self):
        for fields in ({'deferred_windows_s': [0.5]},
                       {'deferred_windows_s': [1], 'window_selection_authorization': 'Defer subsecond windows'},
                       {'deferred_windows_s': [0.5, 0.5], 'window_selection_authorization': 'Defer 0.5s'}):
            with self.subTest(fields=fields), self.assertRaises(AssertionError):
                runner.execution_jobs(plan(True), {**support(), **fields})

    def test_scoped_audit_requires_exact_authorized_window_only_completion(self):
        cohort = plan(True)
        selected = {**support('native_type_value_and_host_bridge', True), 'deferred_windows_s': [0.5],
                    'window_selection_authorization': "Don't do 0.5s for now."}
        ready, deferred = runner.execution_jobs(cohort, selected)
        status = {'state': 'complete_scoped', 'error': None, 'completed_trials': len(ready),
                  'planned_trials': cohort['planned_trials'], **runner.deferral_status(ready, deferred)}
        self.assertEqual(runner.audit_jobs(cohort, selected, status), ready)
        for changed in ({'completed_trials': 14399}, {'planned_trials': 14400},
                        {'user_deferred_window_trials': 0}, {'unsupported_action_deferred_trials': 1},
                        {'state': 'complete'}, {'state': 'blocked_action_support'}, {'error': 'failed'}):
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                runner.audit_jobs(cohort, selected, {**status, **changed})
        with self.assertRaisesRegex(AssertionError, 'requires user direction'):
            runner.audit_jobs(cohort, {**selected, 'window_selection_authorization': ''}, status)

    def test_scoped_audit_rejects_unsupported_code_even_if_mislabeled_complete(self):
        cohort = plan(True)
        selected = {**support(), 'deferred_windows_s': [0.5],
                    'window_selection_authorization': "Don't do 0.5s for now."}
        ready, deferred = runner.execution_jobs(cohort, selected)
        status = {'state': 'complete_scoped', 'error': None, 'completed_trials': len(ready),
                  'planned_trials': cohort['planned_trials'], **runner.deferral_status(ready, deferred)}
        with self.assertRaisesRegex(AssertionError, 'Only authorized window deferrals'):
            runner.audit_jobs(cohort, selected, status)

    def test_full_audit_still_requires_every_planned_job(self):
        cohort = plan(True); selected = support('native_type_value_and_host_bridge', True)
        status = {'state': 'complete', 'error': None, 'completed_trials': 15000, 'planned_trials': 15000}
        self.assertEqual(runner.audit_jobs(cohort, selected, status), cohort['jobs'])
        with self.assertRaises(AssertionError):
            runner.audit_jobs(cohort, selected, {**status, 'completed_trials': 14999})

    def test_all_selected_subsecond_windows_are_deferred_for_large_model_cohort(self):
        cohort = plan(True)
        selected = {**support(), 'deferred_windows_s': [0.5, 0.25, 0.125],
                    'window_selection_authorization': 'Yeah. Save the <1s window runs for smaller models <=4B'}
        ready, deferred = runner.execution_jobs(cohort, selected)
        self.assertEqual(Counter(job['window_s'] for job in ready if job['task'] == 'notification'),
                         {16: 600, 8: 600, 4: 600, 2: 600, 1: 600})
        detail = runner.deferral_status(ready, deferred)
        self.assertEqual(detail['runnable_trials'], 10800)
        self.assertEqual(detail['deferred_trials'], 4200)
        self.assertEqual(detail['user_deferred_window_trials'], 1800)
        self.assertEqual(detail['unsupported_action_deferred_trials'], 2400)
        self.assertEqual(detail['deferred_windows_s'], [0.125, 0.25, 0.5])
        self.assertEqual(runner.completion_state(cohort, {job['id'] for job in ready}, deferred), 'blocked_action_support')
        self.assertEqual(cohort['planned_trials'], 15000)

        selected.update(code_action='native_type_value_and_host_bridge', user_authorized=True)
        ready, deferred = runner.execution_jobs(cohort, selected)
        self.assertEqual(len(ready), 13200)
        self.assertEqual(len(deferred), 1800)
        status = {'state': 'complete_scoped', 'error': None, 'completed_trials': len(ready),
                  'planned_trials': cohort['planned_trials'], **runner.deferral_status(ready, deferred)}
        self.assertEqual(runner.audit_jobs(cohort, selected, status), ready)


if __name__ == '__main__':
    unittest.main()
