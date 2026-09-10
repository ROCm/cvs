'''Unit tests for the canonical vLLM metric contract.'''

import json
import math
import unittest
from pathlib import Path

from cvs.lib.inference.utils.vllm_metrics import (
    METRIC_CATEGORIES,
    METRIC_NAMES,
    METRIC_REGISTRY,
    METRIC_UNITS,
    PROM_METRICS,
    VLLM_GPU_METRICS,
    is_finite_number,
    merge_metric_sources,
    metric_verdict,
    project_vllm_metrics,
)
from cvs.lib.utils.gpu import GPU_METRICS


EXPECTED_NAMES = (
    "max_concurrency",
    "max_concurrent_requests",
    "num_prompts",
    "completed",
    "failed",
    "success_rate",
    "duration",
    "request_throughput",
    "goodput",
    "output_throughput",
    "total_token_throughput",
    "per_gpu_throughput",
    "decode_throughput_p50",
    "max_output_tokens_per_s",
    "rtfx",
    "total_input_tokens",
    "total_output_tokens",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "normalized_ttft_ms_per_tok",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "decode_latency_ratio",
    "mean_e2el_ms",
    "median_e2el_ms",
    "std_e2el_ms",
    "p50_e2el_ms",
    "p90_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "peak_gpu_memory_mb",
    "model_load_memory_mb",
    "model_load_s",
    "gpu_bandwidth_util_pct",
    "gpu_compute_util_pct",
    "queue_time_p50_ms",
    "queue_time_p95_ms",
    "prefill_time_p50_ms",
    "prefill_time_p95_ms",
)


def _widened_fixture():
    path = Path(__file__).resolve().parents[2] / "unittests/fixtures/vllm_results_widened.json"
    return json.loads(path.read_text())


class TestMetricRegistry(unittest.TestCase):
    def test_exact_ordered_inventory(self):
        self.assertEqual(METRIC_NAMES, EXPECTED_NAMES)
        self.assertEqual(len(METRIC_REGISTRY), 56)
        self.assertEqual(len(METRIC_NAMES), len(set(METRIC_NAMES)))
        self.assertTrue(all("." not in name for name in METRIC_NAMES))

    def test_every_definition_is_complete_and_immutable(self):
        self.assertEqual(set(METRIC_UNITS), set(EXPECTED_NAMES))
        self.assertEqual(
            METRIC_CATEGORIES,
            ("run_health", "throughput", "ttft", "tpot", "itl", "e2el", "gpu", "prometheus"),
        )
        for definition in METRIC_REGISTRY:
            with self.subTest(metric=definition.name):
                self.assertTrue(definition.unit)
                self.assertIn(definition.datasource, {"client", "gpu", "prometheus"})
                self.assertTrue(definition.category)
                self.assertIn(definition.direction, {"min", "max"})
                self.assertTrue(definition.derivation)
                self.assertTrue(definition.required_inputs)
                with self.assertRaises(AttributeError):
                    definition.direction = "other"

    def test_gpu_and_prometheus_views_come_from_registry(self):
        self.assertEqual(VLLM_GPU_METRICS, tuple(GPU_METRICS))
        self.assertEqual(
            tuple(name for name, _unit in PROM_METRICS),
            (
                "queue_time_p50_ms",
                "queue_time_p95_ms",
                "prefill_time_p50_ms",
                "prefill_time_p95_ms",
            ),
        )

    def test_exact_threshold_directions(self):
        minimum = {
            "max_concurrency",
            "max_concurrent_requests",
            "num_prompts",
            "completed",
            "success_rate",
            "request_throughput",
            "goodput",
            "output_throughput",
            "total_token_throughput",
            "per_gpu_throughput",
            "decode_throughput_p50",
            "max_output_tokens_per_s",
            "rtfx",
            "total_input_tokens",
            "total_output_tokens",
            "gpu_bandwidth_util_pct",
            "gpu_compute_util_pct",
        }
        actual = {definition.name for definition in METRIC_REGISTRY if definition.direction == "min"}
        self.assertEqual(actual, minimum)


class TestBareProjection(unittest.TestCase):
    def test_widened_fixture_projects_exactly_47_client_metrics(self):
        raw = _widened_fixture()
        raw["request_rate"] = 4.5
        raw["burstiness"] = 2.0
        metrics = project_vllm_metrics(raw, tp=8, pp=1, isl=1024)

        self.assertEqual(len(metrics), 47)
        self.assertEqual(set(metrics), set(EXPECTED_NAMES[:47]))
        self.assertEqual(metrics["goodput"], raw["request_goodput"])
        self.assertNotIn("request_goodput", metrics)
        self.assertNotIn("request_rate", metrics)
        self.assertNotIn("burstiness", metrics)

    def test_unknown_finite_numeric_field_names_artifact(self):
        with self.assertRaisesRegex(ValueError, "node0:/tmp/results"):
            project_vllm_metrics(
                {"output_throughput": 1.0, "new_upstream_number": 2},
                tp=1,
                pp=1,
                isl=1,
                artifact_path="node0:/tmp/results",
            )

    def test_unknown_nonnumeric_and_boolean_fields_are_ignored(self):
        metrics = project_vllm_metrics(
            {"output_throughput": 0, "unknown": "text", "flag": True},
            tp=1,
            pp=1,
            isl=1,
        )
        self.assertEqual(metrics, {"output_throughput": 0})

    def test_registered_values_must_be_finite_builtin_numbers(self):
        metrics = project_vllm_metrics(
            {
                "output_throughput": True,
                "mean_ttft_ms": math.nan,
                "mean_tpot_ms": math.inf,
                "rtfx": 0.0,
            },
            tp=1,
            pp=1,
            isl=1,
        )
        self.assertEqual(metrics, {"rtfx": 0.0})

    def test_derived_metrics_preserve_real_zero(self):
        metrics = project_vllm_metrics(
            {
                "completed": 0,
                "failed": 1,
                "total_token_throughput": 0.0,
                "mean_ttft_ms": 0,
                "p99_itl_ms": 0,
                "p50_itl_ms": 2,
            },
            tp=8,
            pp=2,
            isl=128,
        )
        self.assertEqual(metrics["success_rate"], 0.0)
        self.assertEqual(metrics["per_gpu_throughput"], 0.0)
        self.assertEqual(metrics["normalized_ttft_ms_per_tok"], 0.0)
        self.assertEqual(metrics["decode_latency_ratio"], 0.0)


class TestMetricSourceMerge(unittest.TestCase):
    def test_disjoint_sources_preserve_zero(self):
        merged = merge_metric_sources(
            {"output_throughput": 0},
            {"gpu_compute_util_pct": 0.0},
            {"queue_time_p50_ms": 0},
        )
        self.assertEqual(
            merged,
            {
                "output_throughput": 0,
                "gpu_compute_util_pct": 0.0,
                "queue_time_p50_ms": 0,
            },
        )

    def test_rejects_wrong_source_and_collision(self):
        with self.assertRaisesRegex(ValueError, "unknown gpu"):
            merge_metric_sources({}, {"output_throughput": 1}, {})
        with self.assertRaisesRegex(ValueError, "collision"):
            merge_metric_sources(
                {"output_throughput": 1},
                {},
                {"queue_time_p50_ms": 1, "output_throughput": 2},
            )


class TestStrictMetricVerdict(unittest.TestCase):
    def test_min_boundaries_cover_int_float_zero_and_negative(self):
        cases = [
            (-2, -1, "fail"),
            (-1, -1, "pass"),
            (0, -1, "pass"),
            (0.5, 1.0, "fail"),
            (1.0, 1.0, "pass"),
            (2, 1.0, "pass"),
        ]
        for actual, target, expected in cases:
            with self.subTest(actual=actual, target=target):
                status, _reason = metric_verdict(
                    "output_throughput",
                    actual,
                    {"kind": "min", "value": target},
                )
                self.assertEqual(status, expected)

    def test_max_boundaries_cover_int_float_zero_and_negative(self):
        cases = [
            (-2, -1, "pass"),
            (-1, -1, "pass"),
            (0, -1, "fail"),
            (0.5, 1.0, "pass"),
            (1.0, 1.0, "pass"),
            (2, 1.0, "fail"),
        ]
        for actual, target, expected in cases:
            with self.subTest(actual=actual, target=target):
                status, _reason = metric_verdict(
                    "mean_ttft_ms",
                    actual,
                    {"kind": "max", "value": target},
                )
                self.assertEqual(status, expected)

    def test_every_invalid_actual_fails(self):
        for actual in (None, True, "1", [], {}, {1}, math.nan, math.inf, -math.inf):
            with self.subTest(actual=actual):
                status, reason = metric_verdict(
                    "output_throughput",
                    actual,
                    {"kind": "min", "value": 0},
                )
                self.assertEqual(status, "fail")
                self.assertIn("finite built-in", reason)

    def test_finite_number_excludes_bool_and_subclasses(self):
        class IntSubclass(int):
            pass

        self.assertTrue(is_finite_number(0))
        self.assertTrue(is_finite_number(-0.5))
        self.assertFalse(is_finite_number(True))
        self.assertFalse(is_finite_number(IntSubclass(1)))


if __name__ == "__main__":
    unittest.main()
