"""Shared stage bodies for single-node and distributed Aorta benchmarks."""

import json
import logging
from contextlib import contextmanager

import pytest

from cvs.lib import globals
from cvs.lib.benchmark.aorta.aorta_thresholds import active_thresholds, threshold_verdicts
from cvs.lib.report.artifacts import export_payload
from cvs.lib.report.benchmark_metric_registry import record_benchmark_metric_rows
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


def _case(msg, context, failure="", *, spec=None, actual=None, unit=None, digits=None):
    """Describe one sub-test: its label parts, failure message and optional measurement."""
    return {
        "msg": msg,
        "context": context,
        "failure": failure,
        "spec": spec,
        "actual": actual,
        "unit": unit,
        "digits": digits,
    }


def _case_label(case):
    context = case["context"]
    return f"[{case['msg']}] (" + ", ".join(f"{key}={value}" for key, value in context.items()) + ")"


def _number(value, case):
    if value is None:
        return "n/a"
    digits = 1 if case["digits"] is None else case["digits"]
    return f"{float(value):.{digits}f}" + (f" {case['unit']}" if case["unit"] else "")


def _case_summary(case):
    """One-line JUnit value: label, outcome, measured value and gate, then the failure message."""
    parts = [f"{_case_label(case)}: {'failed' if case['failure'] else 'passed'}"]
    if case["spec"] is not None:
        parts.append(f"actual {_number(case['actual'], case)}")
        parts.append(f"gate {case['spec']['kind']} {_number(case['spec']['value'], case)}")
    if case["failure"]:
        parts.append(case["failure"])
    return "; ".join(parts)


def _report_cases(request, subtests, cases):
    """Report cases as sub-tests and preserve their results in HTML and JUnit."""
    if not cases:
        return
    record_benchmark_metric_rows(
        request.node,
        [
            {
                "node": case["context"].get("host", ""),
                "metric": case["context"].get("metric", case["msg"]),
                "label": _case_label(case),
                "status": "fail" if case["failure"] else "pass",
                "actual": case["actual"],
                "unit": case["unit"],
                "digits": case["digits"],
                "spec": case["spec"],
                "reason": case["failure"],
            }
            for case in cases
        ],
    )
    for case in cases:
        request.node.user_properties.append(("cvs_aorta_subtest", _case_summary(case)))
        with subtests.test(msg=case["msg"], **case["context"]):
            if case["failure"]:
                pytest.fail(case["failure"])
        # With pytest < 9 and the pytest-subtests plugin, a failed sub-test leaves
        # its parent passed. fail_test fails the parent stage on every pytest. On
        # pytest >= 9 the parent fails either way, and this gives its <failure> the
        # reason instead of "contains N failed subtests"; JUnit counts are the same.
        if case["failure"]:
            fail_test(case["failure"])


def launch_container(aorta_job, lifecycle):
    with _stage(lifecycle):
        aorta_job.prepare_hosts()
        orch = aorta_job.orch
        if not orch.setup_containers(groups=aorta_job.container_groups()):
            error = getattr(orch.runtime, "last_setup_error", None) or {}
            image = error.get("image") or orch.container_config.get("image")
            reason = error.get("reason", "unknown reason")
            failed_hosts = error.get("failed_hosts", [])
            pytest.fail(f"Aorta container setup failed for image {image!r}: {reason} on hosts {failed_hosts}")
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


def collect_traces(aorta_job, lifecycle, request, subtests):
    # Surviving ranks' artifacts remain useful after a distributed job fails.
    with _stage(lifecycle, recover=aorta_job.started):
        try:
            aorta_job.collect_traces()
        finally:
            aorta_job.collect_logs()
        cases = []
        for host in aorta_job.hosts:
            if host in aorta_job.collection_errors:
                failure = f"Trace collection on {host}: {aorta_job.collection_errors[host]}"
            elif host in aorta_job.trace_trees:
                failure = ""
            else:
                continue
            cases.append(_case("trace collection", {"host": host}, failure))
        # The HTML panel replaces the parent row's log, so every failure of this
        # stage must be a sub-test row or its message is lost from the report.
        if not aorta_job.get_artifact("torch_traces"):
            cases.append(
                _case("trace collection", {"artifact": "torch_traces"}, "No fresh torch_traces artifact was collected")
            )
        _report_cases(request, subtests, cases)


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


def validate_thresholds(aorta_job, lifecycle, request, subtests):
    with _stage(lifecycle):
        if not aorta_job.config.enforce_thresholds:
            pytest.skip("enforce_thresholds=false; metrics are recorded without threshold assertions")
        if lifecycle.benchmark_result is None:
            pytest.fail("No benchmark result to validate")
        expected = active_thresholds(aorta_job.config.thresholds)
        cases = []
        for verdict in threshold_verdicts(lifecycle.parser, lifecycle.benchmark_result, expected):
            # Same precision as the gate's own messages: milliseconds, or a ratio.
            in_ms = verdict["threshold"] == "max_avg_iteration_ms"
            cases.append(
                _case(
                    "threshold",
                    {"metric": verdict["threshold"]},
                    verdict["message"],
                    spec={"kind": verdict["kind"], "value": verdict["limit"]},
                    actual=verdict["actual"],
                    unit="ms" if in_ms else None,
                    digits=2 if in_ms else 4,
                )
            )
        _report_cases(request, subtests, cases)


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
        path.write_text(json.dumps(export_payload(report), indent=2, default=str) + "\n")
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
