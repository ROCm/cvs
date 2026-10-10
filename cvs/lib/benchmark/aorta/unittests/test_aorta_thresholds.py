"""Unit tests for Aorta threshold mapping and parser verdicts."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, call

from cvs.lib.benchmark.aorta import aorta_thresholds
from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.parsers.aorta_report import AortaReportParser
from cvs.parsers.schemas import AortaBenchmarkResult, AortaTraceMetrics
from cvs.parsers.tracelens import TraceLensParser


def _result():
    return AortaBenchmarkResult.from_rank_metrics(
        [
            AortaTraceMetrics(rank=1, total_time_us=12000, compute_time_us=9000, communication_time_us=4000),
            AortaTraceMetrics(rank=0, total_time_us=10000, compute_time_us=8000, communication_time_us=3000),
        ],
        num_nodes=1,
        gpus_per_node=2,
        nccl_channels=112,
        rccl_branch="develop",
    )


class TestAortaThresholds(unittest.TestCase):
    def test_threshold_metrics_cover_config_keys(self):
        keys = {key for key, _metric, _kind in aorta_thresholds.THRESHOLD_METRICS}
        self.assertEqual(
            keys,
            {"max_avg_iteration_ms", "min_compute_ratio", "min_overlap_ratio", "max_time_variance_ratio"},
        )
        for key, _metric, kind in aorta_thresholds.THRESHOLD_METRICS:
            with self.subTest(key=key):
                raw = variant_dict()
                raw["thresholds"] = {"expected_results": {key: 1}}
                self.assertIn(key, AortaVariantConfig.model_validate(raw).thresholds["expected_results"])
                self.assertEqual(kind, "max" if key.startswith("max_") else "min")

    def test_active_thresholds_drops_nulls(self):
        self.assertEqual(aorta_thresholds.active_thresholds(None), {})
        self.assertEqual(aorta_thresholds.active_thresholds({}), {})
        self.assertEqual(aorta_thresholds.active_thresholds({"other": 1}), {})
        self.assertEqual(
            aorta_thresholds.active_thresholds(
                {"expected_results": {"max_avg_iteration_ms": 2, "min_compute_ratio": None}}
            ),
            {"max_avg_iteration_ms": 2},
        )

    def test_threshold_actual(self):
        result = _result()
        self.assertEqual(
            aorta_thresholds.threshold_actual(result, "max_avg_iteration_ms"), result.avg_iteration_time_ms
        )
        self.assertEqual(aorta_thresholds.threshold_actual(result, "min_compute_ratio"), result.avg_compute_ratio)
        self.assertEqual(aorta_thresholds.threshold_actual(result, "min_overlap_ratio"), result.avg_overlap_ratio)
        self.assertEqual(
            aorta_thresholds.threshold_actual(result, "max_time_variance_ratio"),
            result.std_iteration_time_us / result.avg_iteration_time_us,
        )
        zero = AortaBenchmarkResult.from_rank_metrics(
            [AortaTraceMetrics(rank=0, total_time_us=0, compute_time_us=0, communication_time_us=0)],
            num_nodes=1,
            gpus_per_node=1,
        )
        self.assertIsNone(aorta_thresholds.threshold_actual(zero, "max_time_variance_ratio"))
        with self.assertRaises(KeyError):
            aorta_thresholds.threshold_actual(result, "unknown")

    def test_verdicts_use_parser_per_key(self):
        result = _result()
        parser = MagicMock()
        parser.validate_thresholds.side_effect = lambda _result, expected: (
            ["boom"] if "min_compute_ratio" in expected else []
        )
        expected = {"max_avg_iteration_ms": 20, "min_compute_ratio": 1, "min_overlap_ratio": None}
        verdicts = aorta_thresholds.threshold_verdicts(parser, result, expected)
        self.assertEqual(
            parser.validate_thresholds.call_args_list,
            [
                call(result, {"max_avg_iteration_ms": 20}),
                call(result, {"min_compute_ratio": 1}),
            ],
        )
        self.assertEqual([row["status"] for row in verdicts], ["pass", "fail"])
        self.assertEqual([row["message"] for row in verdicts], ["", "boom"])
        self.assertEqual(verdicts[1]["limit"], 1)
        self.assertEqual(verdicts[1]["actual"], result.avg_compute_ratio)
        self.assertEqual(set(verdicts[0]), {"threshold", "metric", "kind", "limit", "actual", "status", "message"})

    def test_verdicts_match_both_parsers_at_boundaries(self):
        result = _result()
        parsers = [TraceLensParser(use_tracelens=False)]
        try:
            parsers.append(AortaReportParser())
        except ImportError:
            pass
        for parser in parsers:
            for key, _metric, _kind in aorta_thresholds.THRESHOLD_METRICS:
                actual = aorta_thresholds.threshold_actual(result, key)
                delta = 1e-9 * max(1, abs(actual))
                for limit in (actual, actual - delta, actual + delta, 0, 1e9):
                    with self.subTest(parser=type(parser).__name__, key=key, limit=limit):
                        failures = parser.validate_thresholds(result, {key: limit})
                        verdict = aorta_thresholds.threshold_verdicts(parser, result, {key: limit})[0]
                        self.assertEqual(verdict["status"] == "fail", bool(failures))
                        self.assertEqual(verdict["message"], "; ".join(failures))

    def test_no_report_imports(self):
        source = Path(aorta_thresholds.__file__).read_text()
        self.assertNotIn("from cvs.lib.report", source)
        self.assertNotIn("import cvs.lib.report", source)
