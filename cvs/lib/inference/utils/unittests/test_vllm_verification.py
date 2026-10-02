'''Unit tests for vLLM metric verification verdicts.'''

import math
import unittest

from cvs.lib.inference.utils.vllm_verification import (
    active_metric_specs,
    evaluate_metric_verdicts,
    reportable_metric_specs,
)


class TestMetricSpecs(unittest.TestCase):
    def test_partial_specs_remain_partial_and_registry_ordered(self):
        thresholds = {
            "mean_ttft_ms": {"kind": "max", "value": 50},
            "output_throughput": {"kind": "min", "value": 100},
        }
        self.assertEqual(
            [item["metric"] for item in reportable_metric_specs(thresholds)],
            ["output_throughput", "mean_ttft_ms"],
        )
        self.assertEqual(active_metric_specs(thresholds, enforce_thresholds=False), ())
        self.assertEqual(
            [item["metric"] for item in active_metric_specs(thresholds, enforce_thresholds=True)],
            ["output_throughput", "mean_ttft_ms"],
        )

    def test_empty_partial_cell_has_no_active_specs(self):
        self.assertEqual(reportable_metric_specs({}), ())
        self.assertEqual(active_metric_specs({}, enforce_thresholds=True), ())


class TestEvaluateMetricVerdicts(unittest.TestCase):
    def test_rows_include_all_finite_actuals_and_configured_missing_metrics(self):
        actuals = {
            "head": {
                "request_throughput": 3.0,
                "output_throughput": 99,
                "mean_ttft_ms": None,
                "failed": "bad",
            }
        }
        thresholds = {
            "output_throughput": {"kind": "min", "value": 100},
            "mean_ttft_ms": {"kind": "max", "value": 50},
        }

        verdicts = evaluate_metric_verdicts(actuals, thresholds, enforce_thresholds=False)

        self.assertEqual(
            [(item["metric"], item["status"], item["enforced"]) for item in verdicts],
            [
                ("request_throughput", "record", False),
                ("output_throughput", "record", False),
                ("mean_ttft_ms", "record", False),
            ],
        )

    def test_only_present_enforced_specs_gate(self):
        actuals = {
            "head": {
                "request_throughput": 1,
                "output_throughput": 99,
                "mean_ttft_ms": 40,
            }
        }
        thresholds = {"output_throughput": {"kind": "min", "value": 100}}
        verdicts = evaluate_metric_verdicts(actuals, thresholds, enforce_thresholds=True)

        self.assertEqual(
            [(item["metric"], item["status"], item["enforced"]) for item in verdicts],
            [
                ("request_throughput", "record", False),
                ("output_throughput", "fail", True),
                ("mean_ttft_ms", "record", False),
            ],
        )

    def test_mixed_failure_and_pass_are_both_computed(self):
        thresholds = {
            "output_throughput": {"kind": "min", "value": 100},
            "mean_ttft_ms": {"kind": "max", "value": 50},
        }
        actuals = {"head": {"output_throughput": 99, "mean_ttft_ms": 40}}

        verdicts = evaluate_metric_verdicts(actuals, thresholds, enforce_thresholds=True)

        self.assertEqual([item["status"] for item in verdicts], ["fail", "pass"])

    def test_keeps_one_verdict_per_metric_per_host(self):
        thresholds = {"output_throughput": {"kind": "min", "value": 100}}
        actuals = {
            "head": {"output_throughput": 100},
            "worker": {"output_throughput": 200},
        }

        verdicts = evaluate_metric_verdicts(actuals, thresholds, enforce_thresholds=True)

        self.assertEqual([verdict["node"] for verdict in verdicts], ["head", "worker"])
        self.assertTrue(all(verdict["status"] == "pass" for verdict in verdicts))

    def test_every_invalid_gated_actual_fails(self):
        for actual in (None, True, "1", [], {}, {1}, math.nan, math.inf, -math.inf):
            with self.subTest(actual=actual):
                verdicts = evaluate_metric_verdicts(
                    {"head": {"output_throughput": actual}},
                    {"output_throughput": {"kind": "min", "value": 0}},
                    enforce_thresholds=True,
                )
                self.assertEqual(verdicts[0]["status"], "fail")

    def test_invalid_ungated_values_do_not_gate(self):
        for enforce in (False, True):
            with self.subTest(enforce=enforce):
                verdicts = evaluate_metric_verdicts(
                    {"head": {"output_throughput": "bad", "request_throughput": 1}},
                    {},
                    enforce_thresholds=enforce,
                )
                self.assertEqual(
                    [(item["metric"], item["status"], item["enforced"]) for item in verdicts],
                    [("request_throughput", "record", False)],
                )

    def test_missing_gpu_and_prometheus_gates_fail_never_skip(self):
        thresholds = {
            "gpu_compute_util_pct": {"kind": "min", "value": 80},
            "queue_time_p95_ms": {"kind": "max", "value": 50},
        }

        verdicts = evaluate_metric_verdicts({"head": {}}, thresholds, enforce_thresholds=True)

        self.assertEqual(
            {item["metric"]: item["status"] for item in verdicts},
            {
                "gpu_compute_util_pct": "fail",
                "queue_time_p95_ms": "fail",
            },
        )
        self.assertNotIn("skip", {item["status"] for item in verdicts})


if __name__ == "__main__":
    unittest.main()
