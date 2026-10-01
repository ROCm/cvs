"""Offline checks for stage gating, parser compatibility and report generation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.aorta_job import AortaJob
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.parsers.schemas import ParseResult, ParseStatus
from cvs.tests.benchmark.aorta import _common


class TestAortaStages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        raw = variant_dict()
        raw["output_dir"] = self.tmp.name
        orch = MagicMock()
        orch.hosts = ["node-a"]
        self.job = AortaJob(orch, AortaVariantConfig.model_validate(raw))
        self.job.status = "completed"
        self.lifecycle = SimpleNamespace(
            failed=False, torn_down=False, container_started=False, benchmark_result=None, parser=None
        )
        trace_dir = Path(self.tmp.name) / "torch_profiler"
        (trace_dir / "rank0").mkdir(parents=True)
        (trace_dir / "rank0/trace.json").write_text(
            json.dumps({"traceEvents": [{"name": "aten::matmul", "cat": "kernel", "dur": 100}]})
        )
        self.job.artifacts["torch_traces"] = trace_dir
        patcher = patch("cvs.parsers.tracelens.TRACELENS_AVAILABLE", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_real_parsers_accept_job_and_report_preserves_metrics(self):
        _common.parse_results(self.job, self.lifecycle)
        _common.validate_thresholds(self.job, self.lifecycle)
        _common.generate_report(self.job, self.lifecycle)
        report = json.loads(self.job.get_artifact("report").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["cluster"]["nodes"], 1)
        self.assertEqual(report["per_rank_summary"][0]["rank"], 0)
        self.assertGreater(report["performance"]["avg_compute_ratio"], 0)

    def test_threshold_failure_still_generates_report(self):
        _common.parse_results(self.job, self.lifecycle)
        self.job.config.thresholds = {"expected_results": {"max_avg_iteration_ms": 0}}
        with self.assertRaises(pytest.fail.Exception):
            _common.validate_thresholds(self.job, self.lifecycle)
        self.assertTrue(self.lifecycle.failed)
        _common.generate_report(self.job, self.lifecycle)
        report = json.loads(self.job.get_artifact("report").read_text())
        self.assertFalse(report["validation_passed"])

    def test_failed_distributed_run_uses_raw_traces_even_with_excel_reports(self):
        self.job.hosts = ["node-a", "node-b"]
        self.job.status = "failed"
        self.job.error_message = "node-b failed"
        self.lifecycle.failed = True
        reports = Path(self.tmp.name) / "reports/individual_reports"
        reports.mkdir(parents=True)
        (reports / "perf_rank0.xlsx").touch()
        self.job.artifacts["tracelens_analysis"] = reports.parent
        with patch.object(_common, "AortaReportParser") as excel:
            _common.parse_results(self.job, self.lifecycle)
        excel.assert_not_called()
        self.assertIsNotNone(self.lifecycle.benchmark_result)
        _common.generate_report(self.job, self.lifecycle)
        report = json.loads(self.job.get_artifact("report").read_text())
        self.assertEqual(report["status"], "failed")

    def test_empty_excel_results_fall_back_to_raw_traces(self):
        reports = Path(self.tmp.name) / "reports/individual_reports"
        reports.mkdir(parents=True)
        (reports / "perf_rank0.xlsx").touch()
        self.job.artifacts["tracelens_analysis"] = reports.parent
        with patch.object(_common, "AortaReportParser") as excel:
            excel.return_value.parse.return_value = ParseResult(status=ParseStatus.NO_DATA, warnings=["empty report"])
            _common.parse_results(self.job, self.lifecycle)
        excel.return_value.parse.assert_called_once_with(self.job)
        self.assertIsNotNone(self.lifecycle.benchmark_result)

    def test_aggregation_failure_fails_parse_stage(self):
        with patch.object(_common.TraceLensParser, "aggregate", return_value=None):
            with self.assertRaisesRegex(pytest.fail.Exception, "aggregation produced no benchmark result"):
                _common.parse_results(self.job, self.lifecycle)
        self.assertTrue(self.lifecycle.failed)

    def test_setup_failure_gates_dependent_stages(self):
        with (
            patch.object(self.job, "prepare_hosts"),
            patch.object(self.job.orch, "setup_containers", return_value=False),
        ):
            with self.assertRaises(pytest.fail.Exception):
                _common.launch_container(self.job, self.lifecycle)
        self.assertTrue(self.lifecycle.failed)
        self.assertFalse(self.lifecycle.container_started)
        with self.assertRaises(pytest.skip.Exception):
            _common.clone_aorta(self.job, self.lifecycle)

    def test_failed_benchmark_still_stops_and_scans_kernel(self):
        with (
            patch.object(self.job, "build_launch_cmd"),
            patch.object(self.job, "record_kernel_start"),
            patch.object(self.job, "start_job"),
            patch.object(self.job, "poll_for_completion", side_effect=RuntimeError("node failed")),
            patch.object(self.job, "stop_processes") as stop,
            patch.object(self.job, "check_kernel_errors") as scan,
        ):
            with self.assertRaisesRegex(RuntimeError, "node failed"):
                _common.run_benchmark(self.job, self.lifecycle)
        stop.assert_called_once()
        scan.assert_called_once()
        self.assertTrue(self.lifecycle.failed)

    def test_teardown_runs_despite_prior_failure_and_ownership_error(self):
        self.lifecycle.failed = True
        with patch.object(self.job, "teardown", side_effect=RuntimeError("chown failed")):
            with self.assertRaisesRegex(RuntimeError, "chown failed"):
                _common.teardown(self.job, self.lifecycle)
        self.job.orch.teardown_containers.assert_called_once()
        self.assertTrue(self.lifecycle.torn_down)

    def test_optional_stage_skips_do_not_poison_lifecycle(self):
        self.job.config.skip_rccl_build = True
        for operation in (_common.build_rccl, _common.analyze):
            with self.assertRaises(pytest.skip.Exception):
                operation(self.job, self.lifecycle)
            self.assertFalse(self.lifecycle.failed)

    def test_container_teardown_failure_is_reported_and_keeps_fixture_guard(self):
        self.job.orch.teardown_containers.return_value = False
        with patch.object(self.job, "teardown"):
            with self.assertRaisesRegex(pytest.fail.Exception, "container teardown failed"):
                _common.teardown(self.job, self.lifecycle)
        self.assertFalse(self.lifecycle.torn_down)
