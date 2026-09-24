"""Shared stage bodies for single-node and distributed Aorta benchmarks."""

import json
import logging
from contextlib import contextmanager

import pytest

from cvs.lib import globals
from cvs.lib.utils_lib import fail_test, update_test_result
from cvs.parsers.aorta_report import AortaReportParser
from cvs.parsers.schemas import ParseStatus
from cvs.parsers.tracelens import TraceLensParser

log = logging.getLogger(__name__)


@contextmanager
def _stage(lifecycle, recover=False):
    """Run a stage unless prior failure blocks it; recovery never clears failure state."""
    if lifecycle.failed and not recover:
        pytest.skip("a prior lifecycle stage failed")
    globals.error_list = []
    try:
        yield
        update_test_result()
    except (Exception, pytest.fail.Exception):
        lifecycle.failed = True
        raise


def launch_container(aorta_job, lifecycle):
    with _stage(lifecycle):
        aorta_job.prepare_hosts()
        orch = aorta_job.orch
        if not orch.setup_containers(groups=aorta_job.container_groups()):
            pytest.fail("Aorta container setup failed")
        lifecycle.container_started = True
        name = orch.get_container_name(orch.container_config, orch.container_config["image"])
        if not orch.verify_containers_running(name):
            pytest.fail(f"Aorta container {name} is not running on every node")


def clone_aorta(aorta_job, lifecycle):
    with _stage(lifecycle):
        aorta_job.clone_or_verify_aorta_repo()


def setup_rdma(aorta_job, lifecycle):
    with _stage(lifecycle):
        aorta_job.setup_distributed()


def build_rccl(aorta_job, lifecycle):
    with _stage(lifecycle):
        if aorta_job.config.skip_rccl_build:
            pytest.skip("skip_rccl_build=true")
        aorta_job.build_rccl()


def run_benchmark(aorta_job, lifecycle):
    with _stage(lifecycle):
        aorta_job.build_launch_cmd()
        aorta_job.record_kernel_start()
        try:
            aorta_job.start_job()
            aorta_job.poll_for_completion()
        finally:
            # Cleanup and the kernel scan must both run even after launch/polling
            # failure; fail_test accumulates errors without masking the first one.
            for operation in (aorta_job.stop_processes, aorta_job.check_kernel_errors):
                try:
                    operation()
                except Exception as exc:
                    fail_test(str(exc))


def collect_traces(aorta_job, lifecycle):
    # Surviving ranks' artifacts remain useful after a distributed job fails.
    with _stage(lifecycle, recover=aorta_job.started):
        try:
            aorta_job.collect_traces()
        finally:
            aorta_job.collect_logs()
        for host, error in aorta_job.collection_errors.items():
            fail_test(f"Trace collection on {host}: {error}")
        if not aorta_job.get_artifact("torch_traces"):
            fail_test("No fresh torch_traces artifact was collected")


def analyze(aorta_job, lifecycle):
    with _stage(lifecycle):
        analysis = aorta_job.config.analysis
        if not (analysis.enable_tracelens or analysis.enable_gemm_analysis):
            pytest.skip("Optional TraceLens/GEMM analysis is disabled")
        aorta_job.run_analysis()


def parse_results(aorta_job, lifecycle):
    with _stage(lifecycle, recover=bool(aorta_job.get_artifact("torch_traces"))):
        trace_dir = aorta_job.get_artifact("torch_traces")
        analysis_dir = aorta_job.get_artifact("tracelens_analysis")
        parsed = None
        parser = None
        report_warnings = []
        if len(aorta_job.hosts) == 1 and analysis_dir and analysis_dir.is_dir():
            reports = analysis_dir / "individual_reports"
            if list(reports.glob("perf_rank*.xlsx")) or list(reports.glob("perf_*ch_rank*.xlsx")):
                try:
                    parser = AortaReportParser()
                    parsed = parser.parse(aorta_job)
                    report_warnings = list(parsed.warnings)
                except ImportError as exc:
                    log.warning("Excel parser unavailable: %s", exc)
        if parsed is None or not parsed.has_results:
            if not trace_dir or not trace_dir.is_dir():
                pytest.fail("No torch_traces artifact; cannot parse benchmark metrics")
            parser = TraceLensParser(use_tracelens=True)
            parsed = parser.parse(aorta_job)
            parsed.warnings = report_warnings + list(parsed.warnings)
        for warning in parsed.warnings:
            log.warning("%s", warning)
        if parsed.status == ParseStatus.FAILED or not parsed.has_results:
            pytest.fail(f"Aorta parsing produced no usable metrics: {parsed.errors}")
        config = aorta_job.config
        lifecycle.benchmark_result = parser.aggregate(
            parsed,
            num_nodes=len(aorta_job.hosts),
            gpus_per_node=config.multi_node.nproc_per_node or config.gpus_per_node,
            nccl_channels=int(aorta_job._build_base_env()["NCCL_MAX_NCHANNELS"]),
            rccl_branch=config.rccl.branch,
        )
        if lifecycle.benchmark_result is None:
            pytest.fail("Aorta aggregation produced no benchmark result")
        lifecycle.parser = parser


def validate_thresholds(aorta_job, lifecycle):
    with _stage(lifecycle):
        if not aorta_job.config.enforce_thresholds:
            pytest.skip("enforce_thresholds=false; metrics are recorded without threshold assertions")
        if lifecycle.benchmark_result is None:
            pytest.fail("No benchmark result to validate")
        expected = {
            key: value for key, value in aorta_job.config.thresholds["expected_results"].items() if value is not None
        }
        for failure in lifecycle.parser.validate_thresholds(lifecycle.benchmark_result, expected):
            fail_test(failure)


def generate_report(aorta_job, lifecycle):
    with _stage(lifecycle, recover=lifecycle.benchmark_result is not None):
        result = lifecycle.benchmark_result
        if result is None:
            pytest.skip("No parsed benchmark result available")
        report = {
            "status": aorta_job.status,
            "validation_passed": not lifecycle.failed,
            "error_message": aorta_job.error_message,
            "collection_errors": aorta_job.collection_errors,
            "duration_seconds": aorta_job.duration_seconds,
            "cluster": {
                "nodes": result.num_nodes,
                "gpus_per_node": result.gpus_per_node,
                "total_gpus": result.total_gpus,
            },
            "configuration": {
                "nccl_channels": result.nccl_channels,
                "compute_channels": result.compute_channels,
                "rccl_branch": result.rccl_branch,
            },
            "performance": {
                "avg_iteration_time_ms": result.avg_iteration_time_ms,
                "std_iteration_time_us": result.std_iteration_time_us,
                "avg_compute_ratio": result.avg_compute_ratio,
                "avg_comm_ratio": result.avg_comm_ratio,
                "avg_overlap_ratio": result.avg_overlap_ratio,
            },
            "per_rank_summary": [
                {"rank": metric.rank, "total_time_us": metric.total_time_us, "compute_ratio": metric.compute_ratio}
                for metric in result.per_rank_metrics
            ],
        }
        aorta_job.output_dir.mkdir(parents=True, exist_ok=True)
        path = aorta_job.output_dir / "aorta_benchmark_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        aorta_job.artifacts["report"] = path
        log.info("Aorta report saved to %s", path)


def teardown(aorta_job, lifecycle):
    with _stage(lifecycle, recover=True):
        try:
            aorta_job.teardown()
        finally:
            if not aorta_job.orch.teardown_containers():
                pytest.fail("Aorta container teardown failed")
            lifecycle.torn_down = True
