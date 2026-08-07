'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

JAX MaxText training suite — single test file for both single-node and distributed.

The mode is determined by the config file passed at runtime via --config_file.
The config's `training.distributed` field drives skipping of distributed-only
stages (RDMA, NIC setup).
'''

import time

import pytest

from cvs.lib import globals
from cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib import MaxTextTrainingJob
from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import (
    TRAINING_METRICS,
    TRAINING_METRIC_UNITS,
    compute_scaling_efficiency,
    compute_convergence,
    sample_loss_curve,
    evaluate_loss_decreasing,
)
from cvs.lib.training.jaxmaxtext.utils.loss_curve import render_loss_curve_png
from cvs.lib.utils.verdict import evaluate_all, ThresholdViolation

import uuid as _uuid
from pathlib import Path as _Path

import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location("_training_shared", _pl.Path(__file__).with_name("_shared.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
test_print_results_table = _mod.test_print_results_table  # noqa: F841

log = globals.log


def pytest_generate_tests(metafunc):
    """Parametrize test_metric over TRAINING_METRICS."""
    if "metric" in metafunc.fixturenames:
        ids = [short for short, _ in TRAINING_METRICS]
        metafunc.parametrize("metric", ids, ids=ids)


def test_launch_container(orch, variant_config, lifecycle, request):
    """Stage 1: launch the container. Verify it is running."""
    t = time.monotonic()
    ok = orch.setup_containers()
    lifecycle.record(request.node.nodeid, "container_launch", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        name = orch.get_container_name(orch.container_config, orch.container_config["image"])
        pytest.fail(f"setup_containers() returned False for {name}")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


def test_setup_rdma(orch, variant_config, hf_token, lifecycle, request):
    """Stage 2: copy RDMA library into container (distributed + thor2 NIC only)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not variant_config.training.distributed:
        pytest.skip("single-node: RDMA not needed")
    if not variant_config.training.nic_type or "thor" not in variant_config.training.nic_type.lower():
        pytest.skip(f"nic_type={variant_config.training.nic_type}: RDMA lib copy not needed")
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.setup_rdma_lib()
    lifecycle.record(request.node.nodeid, "rdma_setup", time.monotonic() - t)


def test_setup_nic(orch, variant_config, hf_token, lifecycle, request):
    """Stage 3: run NIC setup scripts (distributed only)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not variant_config.training.distributed:
        pytest.skip("single-node: NIC setup not needed")
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.exec_nic_setup_scripts()
    lifecycle.record(request.node.nodeid, "nic_setup", time.monotonic() - t)


def test_setup_tokenizer(orch, variant_config, hf_token, lifecycle, request):
    """Stage 4: download HF tokenizer into models dir."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.setup_tokenizer()
    lifecycle.record(request.node.nodeid, "tokenizer_setup", time.monotonic() - t)


def test_training_run(orch, variant_config, hf_token, training_res_dict, lifecycle, request):
    """Stage 5: build training command, start training, poll for completion, parse results."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    try:
        job.setup_training_env()
        job.build_training_cmd()
        t = time.monotonic()
        job.start_training()
        job.poll_for_completion()
        wall_time = time.monotonic() - t
        results = job.parse_results()
    except Exception:
        lifecycle.failed = True
        raise

    results["training.wall_time_seconds"] = wall_time
    results["training.convergence_steps"] = variant_config.training.steps
    results["training.convergence_wall_time"] = wall_time

    # Scaling efficiency % vs the configured 1-node throughput baseline. Cross-run
    # metric: it needs num_nodes and the reference throughput, which the pure log
    # parser does not have, so it is computed here after parse_results().
    baseline = variant_config.training.scaling_baseline
    results["training.scaling_efficiency_pct"] = compute_scaling_efficiency(
        results.get("training.tokens_per_sec_total"),
        job.num_nodes,
        baseline.tokens_per_sec_total,
        baseline.num_nodes,
    )

    # Convergence / time-to-target-accuracy (row 33). Cross-series metric: it
    # needs the configured target and both the per-step and eval loss series,
    # so it is computed here rather than in the pure log parser.
    conv = variant_config.training.convergence
    steps_to_target, time_to_target = compute_convergence(
        job.step_metrics,
        job.eval_metrics,
        conv.target_metric,
        conv.target_value,
    )
    results["training.steps_to_target"] = steps_to_target
    results["training.time_to_target_seconds"] = time_to_target

    training_res_dict["results"] = results
    training_res_dict["step_metrics"] = job.step_metrics
    training_res_dict["num_nodes"] = job.num_nodes


def _format_expected(spec):
    """Human-readable expected-threshold string for the console log + summary file."""
    if not spec:
        return "-"
    kind = spec.get("kind")
    value = spec.get("value")
    if kind in ("min", "min_tok_s"):
        return f">= {value}"
    if kind == "max":
        return f"<= {value}"
    if kind == "max_ms":
        return f"<= {value} ms"
    if kind == "within":
        return f"{value} +/-{spec.get('tolerance_pct')}%"
    if kind == "min_ratio":
        return f">= {value} x {spec.get('reference')}"
    return str(spec)


def _format_value(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def test_metric(metric, training_res_dict, variant_config, lifecycle, request):
    """One test (row) per training metric.

    Logs `expected | actual | PASS/FAIL` to the console and collects the same
    into `training_res_dict['metric_rows']` (rendered as a single metric-results
    HTML file by test_print_results_table and linked from every metric row).
    PASS/FAIL is threshold-driven: a metric with a threshold spec and a non-None
    value is asserted via `evaluate_all`; a metric with no value produced this
    run is skipped (N/A), and gating is a no-op unless `enforce_thresholds`.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    results = training_res_dict.get("results")
    if not results:
        pytest.skip("no training results (training_run did not complete)")

    full = "training." + metric
    value = results.get(full)
    unit = TRAINING_METRIC_UNITS.get(metric, "-")
    num_nodes = training_res_dict.get("num_nodes", 1)
    cell = variant_config.cell_key(num_nodes=num_nodes)
    spec = (variant_config.thresholds.get(cell) or {}).get(full)
    expected = _format_expected(spec)
    actual = _format_value(value)

    rows = training_res_dict.setdefault("metric_rows", [])

    def _record(status):
        rows.append(
            {"metric": metric, "expected": expected, "actual": actual, "unit": unit, "status": status}
        )

    # Metric not produced this run (feature disabled, rampup, etc.) -> not a failure.
    if value is None:
        log.info("[metric] %-24s | expected %-14s | actual None | %s -> N/A", metric, expected, unit)
        _record("N/A")
        pytest.skip(f"{metric}: no value produced this run")

    # No threshold, or gating disabled -> record the value without asserting.
    if spec is None or not variant_config.enforce_thresholds:
        log.info("[metric] %-24s | expected %-14s | actual %s | %s -> RECORD", metric, expected, actual, unit)
        _record("RECORD")
        return

    try:
        evaluate_all(results, {full: spec})
    except ThresholdViolation as e:
        log.error("[metric] %-24s | expected %-14s | actual %s | %s -> FAIL", metric, expected, actual, unit)
        _record("FAIL")
        pytest.fail(str(e))
    else:
        log.info("[metric] %-24s | expected %-14s | actual %s | %s -> PASS", metric, expected, actual, unit)
        _record("PASS")


def test_loss_curve(training_res_dict, variant_config, lifecycle, request):
    """Row 32: sample the training loss, render a PNG, and gate on a downward trend.

    Samples per-step loss (every N steps + milestone steps), fits a least-squares
    slope, renders a PNG linked in this row, and fails when the curve is not
    decreasing (unless loss_curve.enforce is False, or there are too few points).
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    step_metrics = training_res_dict.get("step_metrics")
    if not step_metrics:
        pytest.skip("no step metrics (training_run did not complete)")

    cfg = variant_config.training.loss_curve
    points = sample_loss_curve(step_metrics, cfg.sample_every, cfg.milestone_steps)
    verdict = evaluate_loss_decreasing(points, cfg.max_slope)

    # Render the PNG into the HTML report bundle dir when reporting is enabled,
    # else into the run's log_dir. The clickable link is attached by the conftest
    # makereport hook from the artifact stashed on `lifecycle`.
    mgr = getattr(request.config, "_html_report_manager", None)
    if mgr is not None and getattr(mgr, "is_enabled", False):
        out_dir = mgr.log_dir
    else:
        out_dir = _Path(variant_config.paths.log_dir)
    png_path = None
    try:
        _Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = f"loss_curve_{variant_config.model.id}_{str(_uuid.uuid4()).split('-')[-1]}.png"
        abs_path = _Path(out_dir) / fname
        title = f"Training Loss Curve — {variant_config.model.id}"
        png_path = render_loss_curve_png(points, abs_path, title=title)
    except Exception as e:  # noqa: BLE001 - plotting must never break the verdict
        log.warning("loss curve: could not prepare PNG output (%s)", e)

    if png_path and mgr is not None and getattr(mgr, "is_enabled", False):
        try:
            rel_path = str(_Path(png_path).relative_to(mgr.htmlpath.parent))
            lifecycle.add_artifact(request.node.nodeid, "Loss Curve", rel_path, str(png_path))
        except Exception as e:  # noqa: BLE001
            log.warning("loss curve: could not register report link (%s)", e)

    if verdict is not None:
        _decreasing, _slope, detail = verdict
        log.info("loss curve: %s", detail)

    if verdict is None:
        pytest.skip(f"loss curve needs >= 2 sampled points (got {len(points)})")
    decreasing, _slope, detail = verdict
    if cfg.enforce and not decreasing:
        pytest.fail(f"training loss is not decreasing: {detail}")


def test_teardown(orch, lifecycle, request):
    """Final stage: explicit container teardown."""
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
