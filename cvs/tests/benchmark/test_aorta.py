"""
Pytest tests for Aorta benchmark using runner-parser architecture.

Usage:
    pytest cvs/tests/benchmark/test_aorta.py \
        --cluster_file input/cluster_file/cluster.json \
        --config_file input/config_file/aorta/aorta_benchmark.yaml

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import logging
from pathlib import Path

import pytest

from cvs.runners.aorta import (
    AortaRunner,
    AortaConfig,
    AortaDockerConfig,
    RcclConfig,
    AortaEnvironment,
    AortaAnalysisConfig,
)
from cvs.runners._base_runner import RunStatus
from cvs.parsers.aorta_report import AortaReportParser
from cvs.parsers.tracelens import TraceLensParser
from cvs.parsers.schemas import (
    ParseStatus,
    # Config validation schemas
    ClusterConfigFile,
    AortaBenchmarkConfigFile,
    validate_config_file,
)

from cvs.lib import globals
from cvs.lib.utils_lib import fail_test, update_test_result, resolve_cluster_config_placeholders

log = logging.getLogger(__name__)


# =============================================================================
# Fixtures - Config Loading with Early Validation (Fail Fast)
# =============================================================================


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """Path to cluster configuration file."""
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """Path to benchmark configuration file."""
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def validated_cluster_config(cluster_file) -> ClusterConfigFile:
    """
    Load and validate cluster configuration.

    Fails fast with clear error messages if config is invalid.
    Resolves {user-id} placeholders before validation.
    """
    import json as json_module

    log.info(f"Validating cluster config: {cluster_file}")

    try:
        # Load raw JSON and resolve placeholders first
        with open(cluster_file) as f:
            raw_config = json_module.load(f)

        # Resolve {user-id} placeholders
        resolved_config = resolve_cluster_config_placeholders(raw_config)
        log.info(f"Resolved cluster config placeholders for user: {resolved_config.get('username')}")

        # Validate with Pydantic schema
        config = ClusterConfigFile.model_validate(resolved_config)
        log.info(f"Cluster config valid: {len(config.node_dict)} nodes configured")
        return config
    except Exception as e:
        pytest.fail(f"CLUSTER CONFIG VALIDATION FAILED:\n{e}")


@pytest.fixture(scope="module")
def validated_aorta_config(config_file) -> AortaBenchmarkConfigFile:
    """
    Load and validate Aorta benchmark configuration.

    Fails fast with clear error messages if config is invalid.
    """
    log.info(f"Validating Aorta config: {config_file}")

    try:
        config = validate_config_file(config_file, config_type="aorta")
        log.info(f"Aorta config valid: image={config.docker.image}")

        # Additional path validation
        path_errors = config.validate_paths_exist()
        if path_errors:
            error_msg = "Path validation failed:\n" + "\n".join(f"  - {e}" for e in path_errors)
            pytest.fail(f"AORTA CONFIG PATH VALIDATION FAILED:\n{error_msg}")

        return config
    except Exception as e:
        pytest.fail(f"AORTA CONFIG VALIDATION FAILED:\n{e}")


@pytest.fixture(scope="module")
def aorta_runner_config(
    validated_cluster_config: ClusterConfigFile, validated_aorta_config: AortaBenchmarkConfigFile
) -> AortaConfig:
    """
    Build AortaConfig from validated cluster and aorta configs.

    This bridges the validated Pydantic models to the runner's dataclass config.
    """
    # Extract node list from validated cluster config
    node_list = list(validated_cluster_config.node_dict.keys())

    # Build Docker config from validated aorta config
    docker_config = AortaDockerConfig(
        image=validated_aorta_config.docker.image,
        container_name=validated_aorta_config.docker.container_name,
        shm_size=validated_aorta_config.docker.shm_size,
        network_mode=validated_aorta_config.docker.network_mode,
        privileged=validated_aorta_config.docker.privileged,
    )

    # Build RCCL config
    rccl_config = RcclConfig(
        clone_url=validated_aorta_config.rccl.clone_url,
        branch=validated_aorta_config.rccl.branch,
        build_path=validated_aorta_config.rccl.build_path,
    )

    # Build environment config
    env = validated_aorta_config.environment
    env_config = AortaEnvironment(
        NCCL_MAX_NCHANNELS=env.NCCL_MAX_NCHANNELS,
        NCCL_MAX_P2P_NCHANNELS=env.NCCL_MAX_P2P_NCHANNELS,
        NCCL_DEBUG=env.NCCL_DEBUG,
        TORCH_NCCL_HIGH_PRIORITY=env.TORCH_NCCL_HIGH_PRIORITY,
        OMP_NUM_THREADS=env.OMP_NUM_THREADS,
        RCCL_MSCCL_ENABLE=env.RCCL_MSCCL_ENABLE,
    )

    # Build analysis config
    analysis_cfg = validated_aorta_config.analysis
    analysis_config = AortaAnalysisConfig(
        enable_tracelens=analysis_cfg.enable_tracelens,
        enable_gemm_analysis=analysis_cfg.enable_gemm_analysis,
        tracelens_script=analysis_cfg.tracelens_script,
        skip_if_exists=analysis_cfg.skip_if_exists,
    )

    # Build full runner config
    return AortaConfig(
        nodes=node_list,
        username=validated_cluster_config.username,
        pkey=validated_cluster_config.priv_key_file,
        aorta_path=Path(validated_aorta_config.aorta_path),
        aorta_auto_clone=getattr(validated_aorta_config, "aorta_auto_clone", False),
        aorta_clone_url=getattr(validated_aorta_config, "aorta_clone_url", None),
        container_mount_path=validated_aorta_config.container_mount_path,
        base_config=validated_aorta_config.base_config,
        training_overrides=validated_aorta_config.training_overrides,
        docker=docker_config,
        rccl=rccl_config,
        environment=env_config,
        analysis=analysis_config,
        build_script=validated_aorta_config.build_script,
        experiment_script=validated_aorta_config.experiment_script,
        gpus_per_node=validated_aorta_config.gpus_per_node,
        skip_rccl_build=validated_aorta_config.skip_rccl_build,
        timeout_seconds=validated_aorta_config.timeout_seconds,
    )


# =============================================================================
# Tests
# =============================================================================


class TestAortaBenchmark:
    """Test suite for Aorta benchmark."""

    # Store results between tests (parser used for parse + validation)
    run_result = None
    parse_result = None
    benchmark_result = None
    _parser = None  # Parser that produced benchmark_result (for validate_thresholds)

    def test_validate_runner_config(self, aorta_runner_config):
        """
        Validate the runner configuration before executing.

        This is a secondary validation after the config file validation.
        Checks runtime requirements like Docker availability.
        """
        globals.error_list = []

        runner = AortaRunner(aorta_runner_config)
        errors = runner.validate_config()

        for error in errors:
            fail_test(f"Runner config validation error: {error}")

        if not errors:
            log.info("Runner configuration validated successfully")

        update_test_result()

    def test_run_benchmark(self, aorta_runner_config):
        """Execute the Aorta benchmark."""
        globals.error_list = []

        runner = AortaRunner(aorta_runner_config)

        # Execute full lifecycle
        result = runner.execute()

        # Store for subsequent tests
        TestAortaBenchmark.run_result = result

        # Check status
        if result.status != RunStatus.COMPLETED:
            fail_test(f"Benchmark run failed: {result.error_message}")

        log.info(f"Benchmark completed in {result.duration_seconds:.1f}s")
        log.info(f"Artifacts: {list(result.artifacts.keys())}")

        update_test_result()

    def test_parse_results(self, aorta_runner_config, validated_aorta_config):
        """Parse benchmark results on host from artifacts (container reports if present, else raw traces)."""
        globals.error_list = []

        if TestAortaBenchmark.run_result is None:
            pytest.skip("No run result available - run test_run_benchmark first")

        run_result = TestAortaBenchmark.run_result
        trace_dir = run_result.get_artifact("torch_traces")

        # Optional: try container-generated Excel reports first (if present and valid)
        analysis_dir = run_result.get_artifact("tracelens_analysis") or (
            trace_dir.parent / "tracelens_analysis" if trace_dir else None
        )
        has_valid_reports = False
        if analysis_dir and analysis_dir.exists():
            reports_dir = analysis_dir / "individual_reports"
            if reports_dir.exists():
                report_files = list(reports_dir.glob("perf_rank*.xlsx")) or list(
                    reports_dir.glob("perf_*ch_rank*.xlsx")
                )
                has_valid_reports = len(report_files) > 0

        parse_result = None
        parser = None
        report_warnings = []

        if has_valid_reports:
            try:
                log.info("Container reports present; trying AortaReportParser (host parse path)")
                report_parser = AortaReportParser()
                parse_result = report_parser.parse(run_result)
                parser = report_parser
                report_warnings = list(parse_result.warnings)
                if parse_result.has_results:
                    log.info("Using metrics from container Excel reports")
            except ImportError as e:
                log.warning(f"AortaReportParser unavailable ({e}), using host raw-trace parsing")
                parse_result = None

        # Primary path: host parsing from torch_traces. When trace_dir exists we always attempt
        # host parsing if we don't have metrics yet, so NO_DATA only occurs when both report
        # and raw-trace parsing yield no metrics.
        if not trace_dir or not trace_dir.exists():
            if parse_result is None:
                fail_test("No torch_traces artifact; cannot parse on host")
            # else we already have parse_result from reports
        else:
            if parse_result is None or not parse_result.has_results:
                if parse_result is not None and parse_result.status == ParseStatus.NO_DATA:
                    log.info("No Excel metrics; falling back to host raw-trace parsing")
                else:
                    log.info("Parsing raw traces on host (TraceLensParser)")
                host_parser = TraceLensParser(use_tracelens=True)
                host_result = host_parser.parse(run_result)
                host_result.warnings = report_warnings + list(host_result.warnings)
                parse_result = host_result
                parser = host_parser

        if parse_result is None:
            fail_test("Parse step produced no result")
        TestAortaBenchmark.parse_result = parse_result

        if parse_result.status == ParseStatus.FAILED:
            for error in parse_result.errors:
                fail_test(f"Parse error: {error}")
        elif parse_result.status == ParseStatus.NO_DATA and not parse_result.has_results:
            # NO_DATA only when both report and raw-trace parsing yielded no metrics
            log.info("No parsed metrics available; raw traces may still be inspected in Perfetto")

        # Log warnings (for all statuses)
        for warning in parse_result.warnings:
            log.warning(warning)

        log.info(f"Parsed {len(parse_result.results)} rank metrics")

        # Aggregate results
        if parse_result.has_results:
            benchmark_result = parser.aggregate(
                parse_result,
                num_nodes=len(aorta_runner_config.nodes),
                gpus_per_node=aorta_runner_config.gpus_per_node,
                nccl_channels=aorta_runner_config.environment.NCCL_MAX_NCHANNELS,
                rccl_branch=aorta_runner_config.rccl.branch,
            )
            TestAortaBenchmark.benchmark_result = benchmark_result
            TestAortaBenchmark._parser = parser

            if benchmark_result:
                log.info("Aggregated results:")
                log.info(f"  Avg iteration time: {benchmark_result.avg_iteration_time_ms:.2f}ms")
                log.info(f"  Compute ratio: {benchmark_result.avg_compute_ratio:.2%}")
                log.info(f"  Comm ratio: {benchmark_result.avg_comm_ratio:.2%}")
                log.info(f"  Overlap ratio: {benchmark_result.avg_overlap_ratio:.2%}")

        update_test_result()

    def test_validate_thresholds(self, validated_aorta_config):
        """Validate results against expected thresholds."""
        globals.error_list = []

        if TestAortaBenchmark.benchmark_result is None:
            pytest.skip("No benchmark result available - run test_parse_results first")

        # Use same parser that produced benchmark_result (both have validate_thresholds)
        parser = TestAortaBenchmark._parser or TraceLensParser(use_tracelens=True)

        # Get expected results from validated config
        expected = {
            "max_avg_iteration_ms": validated_aorta_config.expected_results.max_avg_iteration_ms,
            "min_compute_ratio": validated_aorta_config.expected_results.min_compute_ratio,
            "min_overlap_ratio": validated_aorta_config.expected_results.min_overlap_ratio,
            "max_time_variance_ratio": validated_aorta_config.expected_results.max_time_variance_ratio,
        }
        # Filter out None values
        expected = {k: v for k, v in expected.items() if v is not None}

        failures = parser.validate_thresholds(TestAortaBenchmark.benchmark_result, expected)

        for failure in failures:
            fail_test(failure)

        if not failures:
            log.info("All threshold validations passed")

        update_test_result()

    def test_generate_report(self, aorta_runner_config):
        """Generate benchmark report."""
        globals.error_list = []

        if TestAortaBenchmark.benchmark_result is None:
            pytest.skip("No benchmark result available")

        result = TestAortaBenchmark.benchmark_result
        run_result = TestAortaBenchmark.run_result

        # Build report
        report = {
            "status": "completed",
            "duration_seconds": run_result.duration_seconds if run_result else 0,
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
                {
                    "rank": m.rank,
                    "total_time_us": m.total_time_us,
                    "compute_ratio": m.compute_ratio,
                }
                for m in result.per_rank_metrics
            ],
        }

        # Save report
        output_dir = aorta_runner_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "aorta_benchmark_report.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        log.info(f"Report saved to {report_path}")

        update_test_result()
