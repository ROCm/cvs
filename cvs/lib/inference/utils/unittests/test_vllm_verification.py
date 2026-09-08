'''Unit tests for vLLM metric verification verdicts.'''

import unittest

from cvs.lib.inference.utils.vllm_verification import active_metric_specs, evaluate_metric_verdicts, metric_definitions


class TestMetricDefinitions(unittest.TestCase):
    def test_definitions_cover_all_metric_families(self):
        definitions = {item['metric']: item for item in metric_definitions()}

        self.assertEqual(definitions['client.output_throughput']['unit'], 'tok/s')
        self.assertEqual(definitions['gpu.gpu_compute_util_pct']['missing_status'], 'skip')
        self.assertEqual(definitions['prom.queue_time_p95_ms']['missing_status'], 'skip')


class TestActiveMetricSpecs(unittest.TestCase):
    def test_record_only_and_info_specs_are_not_active(self):
        thresholds = {
            'client.output_throughput': {'kind': 'min_tok_s', 'value': 100},
            'client.goodput': {'kind': 'info', 'value': 0},
        }

        self.assertEqual(active_metric_specs(thresholds, enforce_thresholds=False), ())
        active = active_metric_specs(thresholds, enforce_thresholds=True)
        self.assertEqual([item['metric'] for item in active], ['client.output_throughput'])


class TestEvaluateMetricVerdicts(unittest.TestCase):
    def test_evaluates_sibling_metrics_after_failure(self):
        thresholds = {
            'client.output_throughput': {'kind': 'min_tok_s', 'value': 100},
            'client.mean_ttft_ms': {'kind': 'max_ms', 'value': 50},
        }
        actuals = {
            'head': {
                'client.output_throughput': 99,
                'client.mean_ttft_ms': 40,
            }
        }

        verdicts = evaluate_metric_verdicts(actuals, thresholds, enforce_thresholds=True)

        self.assertEqual([item['status'] for item in verdicts], ['fail', 'pass'])
        self.assertIn('actual 99.0 tok/s < min 100.0 tok/s', verdicts[0]['reason'])

    def test_missing_client_fails_and_missing_gpu_prom_skip(self):
        thresholds = {
            'client.output_throughput': {'kind': 'min_tok_s', 'value': 100},
            'gpu.gpu_compute_util_pct': {'kind': 'min', 'value': 80},
            'prom.queue_time_p95_ms': {'kind': 'max_ms', 'value': 50},
        }

        verdicts = evaluate_metric_verdicts({'head': {}}, thresholds, enforce_thresholds=True)

        self.assertEqual(
            {item['metric']: item['status'] for item in verdicts},
            {
                'client.output_throughput': 'fail',
                'gpu.gpu_compute_util_pct': 'skip',
                'prom.queue_time_p95_ms': 'skip',
            },
        )

    def test_min_ratio_uses_complete_actuals(self):
        thresholds = {
            'client.output_throughput': {
                'kind': 'min_ratio',
                'reference': 'client.total_token_throughput',
                'value': 0.5,
            }
        }
        actuals = {
            'head': {
                'client.output_throughput': 60,
                'client.total_token_throughput': 100,
            }
        }

        verdicts = evaluate_metric_verdicts(actuals, thresholds, enforce_thresholds=True)

        self.assertEqual(verdicts[0]['status'], 'pass')

    def test_keeps_one_verdict_per_host(self):
        thresholds = {'client.output_throughput': {'kind': 'min_tok_s', 'value': 100}}
        actuals = {
            'head': {'client.output_throughput': 100},
            'worker': {'client.output_throughput': 200},
        }

        verdicts = evaluate_metric_verdicts(actuals, thresholds, enforce_thresholds=True)

        self.assertEqual([verdict['node'] for verdict in verdicts], ['head', 'worker'])


if __name__ == '__main__':
    unittest.main()
