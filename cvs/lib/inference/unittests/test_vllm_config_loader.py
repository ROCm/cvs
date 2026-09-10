'''Unit tests for strict vLLM threshold schema validation.'''

import math
import unittest

from pydantic import ValidationError

from cvs.lib.inference.utils.vllm_config_loader import (
    GATED_GPU_METRICS,
    GATED_PROM_METRICS,
    VariantConfig,
)


CELL = "ISL=128,OSL=2048,TP=8,PP=1,CONC=16"


def _variant(thresholds, enforce=False):
    return VariantConfig(
        enforce_thresholds=enforce,
        threshold_json="threshold.json",
        paths={
            "shared_fs": "/home/x",
            "models_dir": "/home/x/models",
            "log_dir": "/home/x/LOGS",
            "hf_token_file": "/home/x/.hf",
        },
        container={"name": "test", "image": "test", "runtime": {"name": "docker", "args": {}}},
        server_params={"model": "amd/model", "tensor_parallel_size": 8},
        sweeps={CELL: {}},
        runs=[CELL],
        thresholds=thresholds,
    )


class TestPartialThresholdCells(unittest.TestCase):
    def test_one_spec_cell_loads_with_enforcement_off_and_on(self):
        thresholds = {CELL: {"output_throughput": {"kind": "min", "value": 0}}}
        for enforce in (False, True):
            with self.subTest(enforce=enforce):
                variant = _variant(thresholds, enforce=enforce)
                self.assertEqual(variant.thresholds, thresholds)

    def test_empty_cell_loads_with_enforcement_off_and_on(self):
        for enforce in (False, True):
            with self.subTest(enforce=enforce):
                variant = _variant({CELL: {}}, enforce=enforce)
                self.assertEqual(variant.thresholds[CELL], {})

    def test_enforcement_still_requires_selected_run_cell(self):
        _variant({}, enforce=False)
        with self.assertRaisesRegex(ValidationError, "missing threshold coverage"):
            _variant({}, enforce=True)

    def test_accuracy_is_exempt_from_vllm_metric_validation(self):
        accuracy = {
            "gsm8k": {
                "gsm8k.exact_match__strict-match": {
                    "kind": "min",
                    "value": 0,
                    "tolerance_pct": 5,
                }
            }
        }
        variant = _variant({CELL: {}, "accuracy": accuracy}, enforce=True)
        self.assertEqual(variant.thresholds["accuracy"], accuracy)


class TestStrictThresholdSpecs(unittest.TestCase):
    def assertRejected(self, metric, spec):
        with self.assertRaises(ValidationError):
            _variant({CELL: {metric: spec}})

    def test_rejects_nonfinite_or_non_builtin_values(self):
        for value in (True, "1", None, [], {}, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertRejected(
                    "output_throughput",
                    {"kind": "min", "value": value},
                )

    def test_rejects_missing_unknown_and_legacy_kinds(self):
        for spec in (
            {"value": 1},
            {"kind": "min"},
            {"kind": "info", "value": 1},
            {"kind": "min_tok_s", "value": 1},
            {"kind": "max_ms", "value": 1},
            {"kind": "within", "value": 1},
            {"kind": "min_ratio", "value": 1},
            {"kind": "other", "value": 1},
        ):
            with self.subTest(spec=spec):
                self.assertRejected("output_throughput", spec)

    def test_rejects_direction_mismatch(self):
        self.assertRejected("output_throughput", {"kind": "max", "value": 1})
        self.assertRejected("mean_ttft_ms", {"kind": "min", "value": 1})

    def test_rejects_prefixed_and_unknown_metric_names(self):
        for metric in ("client.output_throughput", "gpu.gpu_compute_util_pct", "prom.queue_time_p50_ms", "new_metric"):
            with self.subTest(metric=metric):
                self.assertRejected(metric, {"kind": "min", "value": 1})

    def test_rejects_extra_spec_fields(self):
        for field, value in (
            ("unit", "tok/s"),
            ("reference", "total_token_throughput"),
            ("tolerance_pct", 5),
            ("extra", 1),
        ):
            with self.subTest(field=field):
                self.assertRejected(
                    "output_throughput",
                    {"kind": "min", "value": 1, field: value},
                )

    def test_accepts_finite_int_float_zero_and_negative(self):
        for value in (0, -1, 1.5):
            with self.subTest(value=value):
                variant = _variant(
                    {CELL: {"output_throughput": {"kind": "min", "value": value}}}
                )
                self.assertEqual(variant.thresholds[CELL]["output_throughput"]["value"], value)


class TestDatasourceViews(unittest.TestCase):
    def test_gpu_and_prometheus_names_are_bare_and_complete(self):
        self.assertEqual(
            GATED_GPU_METRICS,
            {
                "peak_gpu_memory_mb",
                "model_load_memory_mb",
                "model_load_s",
                "gpu_bandwidth_util_pct",
                "gpu_compute_util_pct",
            },
        )
        self.assertEqual(
            GATED_PROM_METRICS,
            {
                "queue_time_p50_ms",
                "queue_time_p95_ms",
                "prefill_time_p50_ms",
                "prefill_time_p95_ms",
            },
        )


if __name__ == "__main__":
    unittest.main()
