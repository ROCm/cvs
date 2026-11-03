"""Unit tests for the Aorta run-card hook."""

import unittest
from types import SimpleNamespace

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.aorta_rundeck import AortaDeckVariant, deck_results
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.lib.report.profiles.hooks.aorta_run_card import aorta_run_card_display
from cvs.parsers.schemas import AortaBenchmarkResult, AortaTraceMetrics
from cvs.parsers.tracelens import TraceLensParser


def _variant(checked=True, enforce=True, result=True, limit=20):
    raw = variant_dict()
    raw["enforce_thresholds"] = enforce
    raw["thresholds"] = {"expected_results": {"max_avg_iteration_ms": limit, "min_compute_ratio": 0.01}}
    config = AortaVariantConfig.model_validate(raw)
    metrics = (
        AortaBenchmarkResult.from_rank_metrics(
            [AortaTraceMetrics(rank=0, total_time_us=10000, compute_time_us=8000, communication_time_us=3000)],
            num_nodes=1,
            gpus_per_node=1,
            nccl_channels=112,
            rccl_branch="develop",
        )
        if result
        else None
    )
    lifecycle = SimpleNamespace(
        report={},
        deck_results=deck_results(metrics, "test") if metrics else {},
        thresholds_checked=checked,
        benchmark_result=metrics,
        parser=TraceLensParser(use_tracelens=False),
        failed=False,
    )
    return AortaDeckVariant(config, lifecycle)


class TestAortaRunCard(unittest.TestCase):
    def test_enforced_variant_and_links(self):
        rows = aorta_run_card_display(_variant(), {"pytest_html_href": "report.html", "log_file_href": "run.log"})
        values = dict((label, value) for label, value, _link in rows)
        for label in (
            "Workload",
            "Framework",
            "Image",
            "Nodes",
            "GPUs/node",
            "NCCL channels",
            "RCCL branch",
            "Metrics source",
            "Pytest stages",
        ):
            self.assertIn(label, values)
        self.assertEqual(values["Thresholds"], "enforced")
        self.assertEqual(
            [label for label, _value, _link in rows if label.startswith(("max_", "min_"))],
            ["max_avg_iteration_ms", "min_compute_ratio"],
        )
        self.assertTrue(any(link for _label, _value, link in rows))

    def test_forced_failure(self):
        rows = dict((label, value) for label, value, _link in aorta_run_card_display(_variant(limit=1), {}))
        self.assertTrue(rows["max_avg_iteration_ms"].startswith("FAIL"))
        self.assertIn("≤", rows["max_avg_iteration_ms"])
        self.assertIn(" ms", rows["max_avg_iteration_ms"])
        self.assertTrue(rows["min_compute_ratio"].startswith("PASS"))
        self.assertIn("≥", rows["min_compute_ratio"])

    def test_not_evaluated_and_record_only(self):
        for variant, state, verdict in (
            (_variant(checked=False), "not evaluated", "NOT EVALUATED"),
            (_variant(enforce=False), "record-only", "RECORD"),
        ):
            with self.subTest(state=state):
                rows = dict((label, value) for label, value, _link in aorta_run_card_display(variant, {}))
                self.assertEqual(rows["Thresholds"], state)
                self.assertTrue(rows["max_avg_iteration_ms"].startswith(verdict))
        empty = dict((label, value) for label, value, _link in aorta_run_card_display(_variant(result=False), {}))
        self.assertIn("n/a", empty["max_avg_iteration_ms"])

    def test_plain_variant_falls_back_to_generic_threshold_row(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="test"), container=SimpleNamespace(image="image"), enforce_thresholds=False
        )
        rows = dict((label, value) for label, value, _link in aorta_run_card_display(variant, {}))
        self.assertEqual(rows["Thresholds"], "record-only")
