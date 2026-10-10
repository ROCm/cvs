"""Offline checks for stage gating, parser compatibility and report generation."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.aorta_job import AortaJob
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.lib.report.benchmark_metric_registry import benchmark_metric_rows_for_nodeid
from cvs.parsers.schemas import ParseResult, ParseStatus
from cvs.tests.benchmark.aorta import _common


class _RecordingSubtests:
    def __init__(self):
        self.outcomes = []

    @contextmanager
    def test(self, msg=None, **kwargs):
        try:
            yield
        except pytest.fail.Exception as exc:
            self.outcomes.append((msg, kwargs, str(exc)))
        else:
            self.outcomes.append((msg, kwargs, None))


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
        self.subtests = _RecordingSubtests()
        trace_dir = Path(self.tmp.name) / "torch_profiler"
        (trace_dir / "rank0").mkdir(parents=True)
        (trace_dir / "rank0/trace.json").write_text(
            json.dumps({"traceEvents": [{"name": "aten::matmul", "cat": "kernel", "dur": 100}]})
        )
        self.job.artifacts["torch_traces"] = trace_dir
        patcher = patch("cvs.parsers.tracelens.TRACELENS_AVAILABLE", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _request(self, test_name):
        node = SimpleNamespace(nodeid=f"{self.id()}::{test_name}", stash={}, user_properties=[])
        return SimpleNamespace(node=node)

    def test_real_parsers_accept_job_and_report_preserves_metrics(self):
        request = self._request("test_validate_thresholds")
        _common.parse_results(self.job, self.lifecycle)
        _common.validate_thresholds(self.job, self.lifecycle, request, self.subtests)
        self.assertEqual(
            self.subtests.outcomes,
            [
                ("threshold", {"metric": "max_avg_iteration_ms"}, None),
                ("threshold", {"metric": "min_compute_ratio"}, None),
            ],
        )
        self.assertEqual(
            request.node.user_properties,
            [
                ("cvs_aorta_subtest", "[threshold] (metric=max_avg_iteration_ms): passed"),
                ("cvs_aorta_subtest", "[threshold] (metric=min_compute_ratio): passed"),
            ],
        )
        _common.generate_report(self.job, self.lifecycle)
        report = json.loads(self.job.get_artifact("report").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["cluster"]["nodes"], 1)
        self.assertEqual(report["per_rank_summary"][0]["rank"], 0)
        self.assertGreater(report["performance"]["avg_compute_ratio"], 0)

    def test_threshold_failure_still_generates_report(self):
        request = self._request("test_validate_thresholds")
        _common.parse_results(self.job, self.lifecycle)
        self.job.config.thresholds = {"expected_results": {"max_avg_iteration_ms": 0}}
        with self.assertRaises(pytest.fail.Exception):
            _common.validate_thresholds(self.job, self.lifecycle, request, self.subtests)
        self.assertEqual(len(self.subtests.outcomes), 1)
        self.assertEqual(self.subtests.outcomes[0][:2], ("threshold", {"metric": "max_avg_iteration_ms"}))
        self.assertIn("Average iteration time", self.subtests.outcomes[0][2])
        self.assertTrue(self.lifecycle.failed)
        _common.generate_report(self.job, self.lifecycle)
        report = json.loads(self.job.get_artifact("report").read_text())
        self.assertFalse(report["validation_passed"])

    def test_each_threshold_reports_its_own_subtest(self):
        request = self._request("test_validate_thresholds")
        self.job.config.thresholds = {
            "expected_results": {"max_avg_iteration_ms": 0, "min_compute_ratio": 0.01, "min_overlap_ratio": None}
        }
        _common.parse_results(self.job, self.lifecycle)
        with self.assertRaisesRegex(pytest.fail.Exception, "Average iteration time"):
            _common.validate_thresholds(self.job, self.lifecycle, request, self.subtests)
        self.assertEqual(len(self.subtests.outcomes), 2)
        self.assertEqual(
            [item[1]["metric"] for item in self.subtests.outcomes], ["max_avg_iteration_ms", "min_compute_ratio"]
        )
        self.assertIn("Average iteration time", self.subtests.outcomes[0][2])
        self.assertIsNone(self.subtests.outcomes[1][2])
        self.assertTrue(self.lifecycle.failed)
        rows = benchmark_metric_rows_for_nodeid(request.node.nodeid)
        self.assertEqual(
            [row["label"] for row in rows],
            ["[threshold] (metric=max_avg_iteration_ms)", "[threshold] (metric=min_compute_ratio)"],
        )
        self.assertEqual([row["status"] for row in rows], ["fail", "pass"])
        self.assertEqual(rows[0]["spec"], {"kind": "max", "value": 0})
        self.assertTrue(request.node.user_properties[0][1].endswith(": failed"))
        self.assertTrue(request.node.user_properties[1][1].endswith(": passed"))

    def test_parser_exception_escapes_before_any_subtest(self):
        request = self._request("test_validate_thresholds")
        _common.parse_results(self.job, self.lifecycle)
        with patch.object(self.lifecycle.parser, "validate_thresholds", side_effect=RuntimeError("parser broke")):
            with self.assertRaisesRegex(RuntimeError, "parser broke"):
                _common.validate_thresholds(self.job, self.lifecycle, request, self.subtests)
        self.assertEqual(self.subtests.outcomes, [])
        self.assertTrue(self.lifecycle.failed)
        self.assertEqual(benchmark_metric_rows_for_nodeid(request.node.nodeid), [])

    def test_disabled_thresholds_emit_no_subtests(self):
        request = self._request("test_validate_thresholds")
        self.job.config.enforce_thresholds = False
        with self.assertRaises(pytest.skip.Exception):
            _common.validate_thresholds(self.job, self.lifecycle, request, self.subtests)
        self.assertEqual(self.subtests.outcomes, [])
        self.assertEqual(benchmark_metric_rows_for_nodeid(request.node.nodeid), [])
        self.assertEqual(request.node.user_properties, [])
        self.assertFalse(self.lifecycle.failed)

    def test_trace_collection_reports_each_attempted_host(self):
        request = self._request("test_collect_traces")
        self.job.hosts = ["node-a", "node-b", "node-c"]
        self.job.started = True

        def collect():
            self.job.trace_trees = {"node-a": ["t"]}
            self.job.collection_errors = {"node-b": "download failed"}

        with patch.object(self.job, "collect_traces", side_effect=collect), patch.object(self.job, "collect_logs"):
            with self.assertRaisesRegex(pytest.fail.Exception, "Trace collection on node-b: download failed"):
                _common.collect_traces(self.job, self.lifecycle, request, self.subtests)
        self.assertEqual([item[1]["host"] for item in self.subtests.outcomes], ["node-a", "node-b"])
        self.assertIsNone(self.subtests.outcomes[0][2])
        self.assertIn("Trace collection on node-b: download failed", self.subtests.outcomes[1][2])
        rows = benchmark_metric_rows_for_nodeid(request.node.nodeid)
        self.assertEqual([row["node"] for row in rows], ["node-a", "node-b"])
        self.assertEqual(
            [row["label"] for row in rows],
            ["[trace collection] (host=node-a)", "[trace collection] (host=node-b)"],
        )
        self.assertTrue(self.lifecycle.failed)

    def test_single_host_trace_collection_passes_one_subtest(self):
        request = self._request("test_collect_traces")
        self.job.started = True

        def collect():
            self.job.trace_trees = {"node-a": ["t"]}

        with patch.object(self.job, "collect_traces", side_effect=collect), patch.object(self.job, "collect_logs"):
            _common.collect_traces(self.job, self.lifecycle, request, self.subtests)
        self.assertEqual(self.subtests.outcomes, [("trace collection", {"host": "node-a"}, None)])
        self.assertFalse(self.lifecycle.failed)

    def test_missing_trace_artifact_fails_without_host_subtest_failure(self):
        request = self._request("test_collect_traces")
        self.job.started = True
        self.job.artifacts.pop("torch_traces")

        def collect():
            self.job.trace_trees = {"node-a": ["t"]}

        with patch.object(self.job, "collect_traces", side_effect=collect), patch.object(self.job, "collect_logs"):
            with self.assertRaisesRegex(pytest.fail.Exception, "No fresh torch_traces artifact"):
                _common.collect_traces(self.job, self.lifecycle, request, self.subtests)
        self.assertEqual(self.subtests.outcomes, [("trace collection", {"host": "node-a"}, None)])
        self.assertTrue(self.lifecycle.failed)

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
