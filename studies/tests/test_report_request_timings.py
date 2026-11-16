"""CPU-only fixtures for read-only timing reports; no evaluation/model calls."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('report_request_timings', Path(__file__).parents[1] / 'report_request_timings.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class RequestTimingTests(unittest.TestCase):
    def test_missing_metrics_stay_missing(self):
        values, revision = report.extract_request_metrics('agents3_stripped',
            {'start_ns': 1_000_000, 'end_ns': 1_101_000_000, 'response': {'metrics': None}}, {})
        self.assertEqual(values['request_total_ms'], 1100)
        self.assertIsNone(values['engine_ttft_ms'])
        self.assertIsNone(values['decode_ms'])
        self.assertIsNone(values['pure_prefill_ms'])
        self.assertEqual(revision, 'vllm-historical-metrics-unavailable')

    def test_native_prefill_field_is_ttft(self):
        values, _ = report.extract_request_metrics('legacy_cpp', {},
            {'prefill_ms': 300, 'decode_ms': 2000, 'deliberation_ms': 2300})
        self.assertEqual(values['client_ttft_ms'], 300)
        self.assertEqual(values['decode_ms'], 2000)
        self.assertEqual(values['request_total_ms'], 2300)
        self.assertIsNone(values['pure_prefill_ms'])
        self.assertIsNone(values['engine_ttft_ms'])

    def test_vllm_boundaries_are_distinct(self):
        values, revision = report.extract_request_metrics('agents3_stripped',
            {'start_ns': 0, 'end_ns': 1_100_000_000, 'response': {'metrics': {
                'time_to_first_token_ms': 100, 'generation_time_ms': 900, 'queue_time_ms': 7,
                'mean_itl_ms': 9, 'tokens_per_second': 101}}}, {})
        self.assertEqual(values['engine_ttft_ms'], 100)
        self.assertEqual(values['decode_ms'], 900)
        self.assertEqual(values['request_total_ms'], 1100)
        self.assertEqual(values['queue_ms'], 7)
        self.assertEqual(values['mean_itl_ms'], 9)
        self.assertIsNone(values['client_ttft_ms'])
        self.assertEqual(revision, 'vllm-per-request-metrics-v1')

    def test_nearest_rank_and_no_duplicate_trial_latency(self):
        self.assertEqual(report.percentiles([None, 1, 2, 30, 40]), {'p50': 2, 'p95': 40})
        base = {'arm': 'legacy_cpp', 'window_s': 16, 'condition': 'live', 'success': 0, 'receipt': 1}
        requests = [base | {'first_planner_request': True, 'request_total_ms': 100},
                    base | {'first_planner_request': False, 'request_total_ms': 200}]
        rows = report.summarize(requests, [base | {'e_to_a_ms': 400}])
        lookup = {(row['scope'], row['metric']): row for row in rows}
        self.assertEqual(lookup['all_requests', 'request_total_ms']['measured_samples'], 2)
        self.assertEqual(lookup['first_planner_request', 'request_total_ms']['measured_samples'], 1)
        self.assertEqual(lookup['trial', 'e_to_a_ms']['measured_samples'], 1)
        self.assertEqual(lookup['all_requests', 'pure_prefill_ms']['missing_samples'], 2)

    def test_invalid_negative_timing_is_rejected(self):
        with self.assertRaises(ValueError):
            report.extract_request_metrics('agents3_stripped', {'start_ns': 2, 'end_ns': 1}, {})


if __name__ == '__main__':
    unittest.main()
