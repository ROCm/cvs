"""Unit tests for the Aorta Run Deck adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.aorta_rundeck import (
    AortaDeckVariant,
    METRIC_TIERS,
    STAGE_LABELS,
    aorta_cell_dimensions,
    aorta_metric_verdict,
    deck_cell_id,
    deck_results,
    metrics_source,
    record_stage_report,
    threshold_specs,
    tier_metric_specs,
)
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
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


def _variant(result=None, checked=True, thresholds=None, enforce=True):
    raw = variant_dict()
    raw["enforce_thresholds"] = enforce
    if thresholds is not None:
        raw["thresholds"] = {"expected_results": thresholds}
    config = AortaVariantConfig.model_validate(raw)
    lifecycle = SimpleNamespace(
        report={},
        deck_results=deck_results(result, "test") if result else {},
        thresholds_checked=checked,
        benchmark_result=result,
        parser=TraceLensParser(use_tracelens=False),
        failed=False,
    )
    return AortaDeckVariant(config, lifecycle), lifecycle


class TestAortaRundeck(unittest.TestCase):
    def test_deck_results_preserve_aggregate_values(self):
        result = _result()
        raw = deck_results(result, "test")["test"]
        for key in ("avg_iteration_time_ms", "avg_compute_ratio", "avg_comm_ratio", "avg_overlap_ratio"):
            self.assertEqual(raw[key][0], getattr(result, key))
        for key in ("std", "min", "max"):
            self.assertEqual(raw[f"{key}_iteration_time_ms"][0], getattr(result, f"{key}_iteration_time_us") / 1000)
        self.assertEqual(raw["time_variance_ratio"], [result.std_iteration_time_us / result.avg_iteration_time_us])
        self.assertEqual(raw["avg_compute_time_ms"], [8.5])
        self.assertEqual(raw["avg_comm_time_ms"], [3.5])
        self.assertEqual(raw["ranks"], [2])
        self.assertEqual(
            raw["_dimensions"],
            {
                "nodes": "1",
                "gpus_per_node": "2",
                "nccl_channels": "112",
                "rccl_branch": "develop",
            },
        )

    def test_metric_charts_series_sorted_by_rank(self):
        series = deck_results(_result(), "test")["test"]["_metric_charts"]["series"]
        self.assertEqual(len(series), 6)
        self.assertTrue(all([point[0] for point in item["points"]] == [0, 1] for item in series))
        self.assertEqual(series[0]["points"], [[0, 10.0], [1, 12.0]])
        self.assertEqual(series[3]["points"], [[0, 80.0], [1, 75.0]])
        self.assertTrue(all("exposed" not in str(item).lower() for item in series))

    def test_single_rank_omits_metric_charts(self):
        result = AortaBenchmarkResult.from_rank_metrics(
            [AortaTraceMetrics(rank=0, total_time_us=10000, compute_time_us=8000, communication_time_us=3000)],
            num_nodes=1,
            gpus_per_node=1,
        )
        self.assertNotIn("_metric_charts", deck_results(result, "test")["test"])

    def test_zero_average_omits_variance_ratio(self):
        result = AortaBenchmarkResult.from_rank_metrics(
            [AortaTraceMetrics(rank=0, total_time_us=0, compute_time_us=0, communication_time_us=0)],
            num_nodes=1,
            gpus_per_node=1,
        )
        self.assertNotIn("time_variance_ratio", deck_results(result, "test")["test"])

    def test_threshold_specs_and_tiers(self):
        specs = threshold_specs({"max_avg_iteration_ms": 20, "min_compute_ratio": 0.5, "min_overlap_ratio": None})
        self.assertEqual(
            specs,
            {
                "training.avg_iteration_time_ms": {"kind": "max", "value": 20},
                "training.avg_compute_ratio": {"kind": "min", "value": 0.5},
            },
        )
        self.assertEqual(
            tier_metric_specs(specs, "iteration_time"),
            {"training.avg_iteration_time_ms": specs["training.avg_iteration_time_ms"]},
        )
        self.assertEqual(
            tier_metric_specs(specs, "compute_ratio"),
            {"training.avg_compute_ratio": specs["training.avg_compute_ratio"]},
        )
        self.assertEqual(tier_metric_specs(specs, "record"), {})
        self.assertNotIn("training.avg_compute_ratio", METRIC_TIERS["record"])
        self.assertEqual(deck_cell_id(SimpleNamespace(model=SimpleNamespace(id=""))), "default")

    def test_metric_verdict_prefers_stored_verdict(self):
        self.assertEqual(aorta_metric_verdict("m", 5, {"kind": "max", "value": 1, "verdict": "pass"}), ("pass", ""))
        self.assertEqual(
            aorta_metric_verdict("m", 0, {"kind": "min", "value": 1, "verdict": "fail", "reason": "boom"}),
            ("fail", "boom"),
        )
        self.assertEqual(aorta_metric_verdict("m", 1, {"kind": "max", "value": 1}), ("pass", ""))
        self.assertEqual(aorta_metric_verdict("m", 2, {"kind": "max", "value": 1})[0], "fail")
        self.assertEqual(aorta_metric_verdict("m", 0, {"kind": "min", "value": 1})[0], "fail")
        self.assertEqual(aorta_metric_verdict("m", None, {"kind": "max", "value": 1})[0], "na")
        self.assertEqual(aorta_metric_verdict("m", 1, {"kind": "other", "value": 1})[0], "fail")

    def test_deck_variant_states(self):
        result = _result()
        variant, lifecycle = _variant(result)
        self.assertEqual(variant.threshold_state, "enforced")
        self.assertTrue(variant.enforce_thresholds)
        self.assertEqual(variant.thresholds["test"]["training.avg_iteration_time_ms"]["verdict"], "pass")
        self.assertEqual(variant.thresholds["test"]["training.avg_iteration_time_ms"]["reason"], "")
        failed, _ = _variant(result, thresholds={"max_avg_iteration_ms": 1, "min_compute_ratio": 0.01})
        rows = failed.threshold_rows()
        self.assertEqual([row["status"] for row in rows], ["fail", "pass"])
        self.assertEqual(failed.thresholds["test"]["training.avg_iteration_time_ms"]["reason"], rows[0]["message"])
        lifecycle.thresholds_checked = False
        self.assertEqual(variant.threshold_state, "not evaluated")
        self.assertFalse(variant.enforce_thresholds)
        self.assertNotIn("verdict", variant.thresholds["test"]["training.avg_iteration_time_ms"])
        self.assertEqual(variant.threshold_rows()[0]["actual"], result.avg_iteration_time_ms)
        self.assertEqual(variant.threshold_rows()[0]["status"], "not evaluated")
        record, _ = _variant(result, enforce=False)
        self.assertEqual(record.threshold_state, "record-only")
        self.assertEqual(record.threshold_rows()[0]["status"], "record")
        lifecycle.thresholds_checked = True
        lifecycle.benchmark_result = None
        self.assertFalse(variant.gate_ran)

    def test_metrics_source_labels(self):
        self.assertEqual(metrics_source(None), "—")
        self.assertEqual(metrics_source(TraceLensParser(use_tracelens=False)), "Basic raw-trace scan")
        with patch("cvs.parsers.tracelens.TRACELENS_AVAILABLE", True):
            self.assertIn("fallback possible", metrics_source(TraceLensParser(use_tracelens=True)))
        try:
            self.assertEqual(metrics_source(AortaReportParser()), "Aorta TraceLens Excel reports")
        except ImportError:
            pass

    def test_record_stage_report(self):
        lifecycle = SimpleNamespace(report={}, thresholds_checked=False)
        good = pytest.CallInfo.from_call(lambda: None, when="call")
        fail = pytest.CallInfo.from_call(lambda: pytest.fail("x"), when="call")
        crash = pytest.CallInfo.from_call(lambda: 1 / 0, when="call")
        report = SimpleNamespace(when="call", skipped=False, passed=True, failed=False, duration=1.5)
        record_stage_report(lifecycle, "test_run_benchmark", "node::test_run_benchmark", report, good)
        self.assertEqual(lifecycle.report["node::test_run_benchmark"], [("benchmark", 1.5, "s")])
        report.duration = 2.0
        record_stage_report(lifecycle, "test_run_benchmark", "node::test_run_benchmark", report, good)
        self.assertEqual(lifecycle.report["node::test_run_benchmark"], [("benchmark", 2.0, "s")])
        record_stage_report(lifecycle, "unknown", "unknown", report, good)
        self.assertNotIn("unknown", lifecycle.report)
        report.when = "setup"
        record_stage_report(lifecycle, "test_validate_thresholds", "gate", report, good)
        self.assertFalse(lifecycle.thresholds_checked)
        report.when = "call"
        report.skipped = True
        record_stage_report(lifecycle, "test_validate_thresholds", "gate", report, good)
        self.assertFalse(lifecycle.thresholds_checked)
        report.skipped = False
        record_stage_report(lifecycle, "test_validate_thresholds", "gate", report, good)
        self.assertTrue(lifecycle.thresholds_checked)
        lifecycle.thresholds_checked = False
        report.passed = False
        report.failed = True
        record_stage_report(lifecycle, "test_validate_thresholds", "gate", report, fail)
        self.assertTrue(lifecycle.thresholds_checked)
        lifecycle.thresholds_checked = False
        record_stage_report(lifecycle, "test_validate_thresholds", "gate", report, crash)
        self.assertFalse(lifecycle.thresholds_checked)

    def test_record_stage_report_tolerates_lifecycle_without_timings(self):
        lifecycle = SimpleNamespace(thresholds_checked=False)
        good = pytest.CallInfo.from_call(lambda: None, when="call")
        report = SimpleNamespace(when="call", skipped=False, passed=True, failed=False, duration=1.0)
        record_stage_report(lifecycle, "test_validate_thresholds", "gate", report, good)
        self.assertFalse(hasattr(lifecycle, "report"))
        self.assertTrue(lifecycle.thresholds_checked)

    def test_cell_dimensions_hook(self):
        variant, _ = _variant(_result())
        self.assertEqual(aorta_cell_dimensions(variant, "test")["nodes"], "1")
        self.assertEqual(aorta_cell_dimensions(variant, "missing"), {})

    def test_profile_matches_stage_labels(self):
        profile = load_json_profile("aorta")
        self.assertEqual(profile["lifecycle"]["session_labels"], list(STAGE_LABELS.values()))
        self.assertEqual(load_json_profile("aorta_single"), load_json_profile("aorta_distributed"))
        self.assertIs(resolve_report_config(profile).metric_verdict, aorta_metric_verdict)
