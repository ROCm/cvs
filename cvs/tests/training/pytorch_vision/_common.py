"""Shared stage implementations for the PyTorch Vision training suites.

Both ``pytorch_vision_single`` and ``pytorch_vision_distributed`` are thin
skeletons that declare the report rows and delegate here, so single-node and
multi-node runs execute identical logic and differ only in topology. This
module is a helper, not a runnable suite.
"""

import html as _html
import time
import uuid as _uuid
from pathlib import Path as _Path

import pytest
from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.training.pytorch_vision.job import PyTorchVisionJob
from cvs.lib.training.pytorch_vision.utils.metrics import (
    GATED_METRICS,
    METRICS,
    METRIC_UNITS,
    gradient_accumulation_overhead_pct,
    required_metrics_for_run,
    throughput_overhead_pct,
)
from cvs.lib.utils.loss_curve import render_loss_curve_png
from cvs.lib.utils.verdict import evaluate_all


log = globals.log


def _mode(variant_config):
    """ "distributed" or "single" - used in report titles and artifact names."""
    return "distributed" if variant_config.training.distributed else "single"


def _ga_group(sweep, gpus_per_node):
    return (
        sweep.model,
        sweep.backend,
        sweep.precision,
        sweep.image_size,
        sweep.batch_size * sweep.gradient_accumulation_steps * gpus_per_node,
        sweep.data_mode,
        sweep.rocal_device,
        sweep.augmentation,
    )


def _refresh_ga_overhead(variant_config, training_results):
    sweeps = variant_config.training.enabled_sweeps()
    baselines = {
        _ga_group(sweep, variant_config.training.gpus_per_node): sweep
        for sweep in sweeps
        if sweep.gradient_accumulation_steps == 1 and sweep.name in training_results
    }
    for sweep in sweeps:
        if sweep.name not in training_results:
            continue
        baseline = baselines.get(_ga_group(sweep, variant_config.training.gpus_per_node))
        if baseline is None:
            continue
        for host, actuals in training_results[sweep.name].items():
            baseline_actuals = training_results[baseline.name].get(host)
            if baseline_actuals is None:
                continue
            actuals["training.gradient_accumulation_overhead_pct"] = gradient_accumulation_overhead_pct(
                baseline_actuals["training.step_time_ms_mean"],
                actuals["training.step_time_ms_mean"],
            )


def _pipeline_group(sweep):
    return (
        sweep.model,
        sweep.backend,
        sweep.precision,
        sweep.image_size,
        sweep.batch_size,
        sweep.gradient_accumulation_steps,
    )


def _refresh_pipeline_overheads(variant_config, training_results):
    sweeps = variant_config.training.enabled_sweeps()
    baselines = {
        _pipeline_group(sweep): sweep
        for sweep in sweeps
        if sweep.data_mode == "rocal"
        and sweep.rocal_device == "gpu"
        and sweep.augmentation == "standard"
        and sweep.name in training_results
    }
    for sweep in sweeps:
        if sweep.name not in training_results or sweep.data_mode != "rocal":
            continue
        baseline = baselines.get(_pipeline_group(sweep))
        if baseline is None:
            continue
        for host, actuals in training_results[sweep.name].items():
            baseline_actuals = training_results[baseline.name].get(host)
            if baseline_actuals is None:
                continue
            baseline_ips = baseline_actuals["training.data_loader_images_per_sec"]
            candidate_ips = actuals["training.data_loader_images_per_sec"]
            if sweep.rocal_device == "gpu" and sweep.augmentation in {"standard", "heavy"}:
                actuals["training.augmentation_overhead_pct"] = throughput_overhead_pct(
                    baseline_ips,
                    candidate_ips,
                )
            if sweep.augmentation == "standard" and sweep.rocal_device in {"gpu", "cpu"}:
                actuals["training.rocal_cpu_overhead_pct"] = throughput_overhead_pct(
                    baseline_ips,
                    candidate_ips,
                )


def launch_container(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    start = time.monotonic()
    if not orch.setup_containers():
        lifecycle.failed = True
        pytest.fail(f"failed to launch container {name}")
    lifecycle.record(request.node.nodeid, "container_launch", time.monotonic() - start)
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} is not running after launch")


def verify_environment(orch, variant_config, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    sweep_name = variant_config.training.enabled_sweeps()[0].name
    job = PyTorchVisionJob(orch, variant_config, sweep_name)
    start = time.monotonic()
    try:
        summary = job.verify_environment()
    except Exception:
        lifecycle.failed = True
        raise
    lifecycle.record(request.node.nodeid, "environment_verification", time.monotonic() - start)
    log.info("PyTorch Vision environment: %s", summary)


def _execute_sweep(  # noqa: PLR0913
    orch, variant_config, sweep_name, training_results, loss_series
):
    job = PyTorchVisionJob(orch, variant_config, sweep_name)
    globals.error_list = []
    try:
        try:
            job.stage_benchmark()
            job.run_benchmark()
            job.verify_training_log()
            host_results = job.parse_results()
            training_results[sweep_name] = host_results
            loss_series[sweep_name] = job.loss_series
            _refresh_ga_overhead(variant_config, training_results)
            _refresh_pipeline_overheads(variant_config, training_results)
        finally:
            try:
                if job.training_start_time:
                    job.scan_dmesg_for_errors()
            finally:
                job.stop_training_processes()
        if globals.error_list:
            pytest.fail("dmesg verification found errors: " + "; ".join(globals.error_list))
    except Exception:
        raise
    return job


def real_data_smoke(
    orch,
    variant_config,
    sweep_name,
    training_results,
    loss_series,
    lifecycle,
    request,
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    sweep = variant_config.sweep(sweep_name)
    if variant_config.training.run_mode != "smoke" or sweep.data_mode != "rocal":
        pytest.skip("real-data smoke stage applies to rocAL smoke profiles")
    start = time.monotonic()
    _execute_sweep(orch, variant_config, sweep_name, training_results, loss_series)
    lifecycle.record(request.node.nodeid, "real_data_smoke", time.monotonic() - start)


def training(  # noqa: PLR0913
    orch, variant_config, sweep_name, training_results, loss_series, lifecycle, request
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    start = time.monotonic()
    if sweep_name not in training_results:
        _execute_sweep(orch, variant_config, sweep_name, training_results, loss_series)
    lifecycle.record(request.node.nodeid, "training", time.monotonic() - start)


def rocal_overhead_comparisons(variant_config, training_results):
    sweeps = variant_config.training.enabled_sweeps()
    _refresh_pipeline_overheads(variant_config, training_results)
    for sweep in sweeps:
        if sweep.name not in training_results or sweep.data_mode != "rocal":
            continue
        _host, actuals = next(iter(training_results[sweep.name].items()))
        if sweep.rocal_device == "gpu" and sweep.augmentation == "heavy":
            if "training.augmentation_overhead_pct" not in actuals:
                pytest.fail(f"{sweep.name} has no matching GPU-standard augmentation baseline")
        if sweep.rocal_device == "cpu" and sweep.augmentation == "standard":
            if "training.rocal_cpu_overhead_pct" not in actuals:
                pytest.fail(f"{sweep.name} has no matching GPU-standard rocAL baseline")


def metric(  # noqa: PLR0913
    sweep_name, metric, variant_config, training_results, metric_rows, lifecycle, request
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if sweep_name not in training_results:
        pytest.skip(f"no results recorded for {sweep_name}")

    host, actuals = next(iter(training_results[sweep_name].items()))
    name = f"training.{metric}"
    required = required_metrics_for_run(
        variant_config.training.run_mode,
        variant_config.sweep(sweep_name).data_mode,
        variant_config.training.codecarbon.enabled,
    )
    if name not in actuals:
        cell = variant_config.cell_key(sweep_name)
        if name in (variant_config.thresholds.get(cell) or {}):
            pytest.fail(f"declared metric was not produced for {cell}: {name}")
        if metric in required:
            pytest.fail(f"required metric was not produced: {name}")
        pytest.skip(f"metric is not applicable to this sweep: {name}")
    value = actuals[name]
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", METRIC_UNITS[metric]))
    lifecycle.record(request.node.nodeid, "metric_evaluation", 0.0)
    log.info("%s %s=%s %s", host, name, value, METRIC_UNITS[metric])

    label = variant_config.sweep(sweep_name).label
    cell = variant_config.cell_key(sweep_name)
    spec = (variant_config.thresholds.get(cell) or {}).get(name)
    unit = METRIC_UNITS[metric]

    if not variant_config.enforce_thresholds or spec is None:
        _record_metric_row(metric_rows, label, name, spec, value, unit, "RECORD")
        if variant_config.enforce_thresholds and metric in GATED_METRICS:
            pytest.fail(f"threshold is missing for {cell}: {name}")
        return

    try:
        evaluate_all(actuals, {name: spec})
    except Exception:
        _record_metric_row(metric_rows, label, name, spec, value, unit, "FAIL")
        raise
    _record_metric_row(metric_rows, label, name, spec, value, unit, "PASS")


def loss_curve(  # noqa: PLR0913
    sweep_name, variant_config, training_results, loss_series, lifecycle, request
):
    if variant_config.training.run_mode in {"smoke", "perf"}:
        pytest.skip("loss-curve checks do not apply to performance-only phases")
    if sweep_name not in training_results:
        pytest.skip(f"no results recorded for {sweep_name}")
    _host, actuals = next(iter(training_results[sweep_name].items()))
    points = actuals.get("training.loss_curve_points")
    decreased = actuals.get("training.loss_curve_decreased")
    if points is None or decreased is None:
        pytest.fail("training artifact is missing loss-curve evidence")
    _attach_loss_curve_png(sweep_name, variant_config, loss_series, lifecycle, request)
    if points < variant_config.training.loss_curve.minimum_points:
        pytest.fail(f"loss curve has only {points} points")
    if variant_config.training.loss_curve.require_decrease and decreased != 1.0:
        pytest.fail("training loss curve did not decrease")


def _attach_loss_curve_png(sweep_name, variant_config, loss_series, lifecycle, request):
    """Render the per-sweep loss curve and link it from this test's report row.

    Rendering is best-effort: the loss-curve verdict above is computed from the
    artifact, so a missing plot never changes the result.
    """
    points = loss_series.get(sweep_name) or []
    mgr = getattr(request.config, "_html_report_manager", None)
    if not points or mgr is None or not getattr(mgr, "is_enabled", False):
        return
    label = variant_config.sweep(sweep_name).label
    try:
        out_dir = mgr.log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"loss_curve_{_mode(variant_config)}_{label}_{_uuid.uuid4().hex[:12]}.png"
        abs_path = out_dir / fname
        title = f"Training Loss - {variant_config.gpu_arch} / {_mode(variant_config)} / {label}"
        if render_loss_curve_png(points, abs_path, title=title) is None:
            return
        rel_path = str(_Path(abs_path).relative_to(mgr.htmlpath.parent))
        lifecycle.add_artifact(
            request.node.nodeid,
            f"Loss Curve [{_mode(variant_config)}/{label}]",
            rel_path,
            str(abs_path),
        )
    except Exception as exc:  # noqa: BLE001 - reporting must never break the run
        log.warning("could not attach loss curve PNG for %s: %s", label, exc)


def convergence(sweep_name, variant_config, training_results):
    config = variant_config.training.convergence
    if not config.enabled:
        pytest.skip("convergence target is not configured")
    if sweep_name not in training_results:
        pytest.skip(f"no results recorded for {sweep_name}")
    _host, actuals = next(iter(training_results[sweep_name].items()))
    if "training.convergence_step" not in actuals or "training.convergence_time_seconds" not in actuals:
        pytest.fail("configured convergence target was not reached")


_STATUS_COLORS = {"PASS": "#137333", "FAIL": "#c5221f", "RECORD": "#5f6368"}


def _format_expected(spec):
    """Render a threshold spec the way the metric-results page shows it."""
    if not spec:
        return "record-only"
    kind = spec.get("kind")
    value = spec.get("value")
    if kind == "info" or value is None:
        return "record-only"
    symbol = {"min": ">=", "max": "<=", "max_ms": "<= (ms)"}.get(kind, kind)
    return f"{symbol} {value}"


def _record_metric_row(metric_rows, sweep_label, name, spec, value, unit, status):
    metric_rows.append(
        {
            "sweep": sweep_label,
            "metric": name,
            "expected": _format_expected(spec),
            "actual": f"{value:.4f}" if isinstance(value, float) else str(value),
            "unit": unit,
            "status": status,
        }
    )


def _write_metric_results_html(metric_rows, variant_config, request):
    """Write every metric verdict to ONE HTML page in the report bundle dir."""
    mgr = getattr(request.config, "_html_report_manager", None)
    if not metric_rows or mgr is None or not getattr(mgr, "is_enabled", False):
        return
    try:
        out_dir = mgr.log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        body = ""
        for row in metric_rows:
            color = _STATUS_COLORS.get(row["status"], "#000000")
            cells = "".join(
                f"<td>{_html.escape(str(row[key]))}</td>" for key in ("sweep", "metric", "expected", "actual", "unit")
            )
            body += f"<tr>{cells}<td style=\"color:{color};font-weight:bold;\">{_html.escape(row['status'])}</td></tr>"
        title = f"PyTorch Vision Metric Results ({variant_config.gpu_arch} / {_mode(variant_config)})"
        doc = (
            f"<html><head><meta charset='utf-8'><title>{_html.escape(title)}</title></head>"
            f"<body><h2>{_html.escape(title)}</h2>"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Sweep</th><th>Metric</th><th>Expected</th>"
            "<th>Actual</th><th>Unit</th><th>Status</th></tr>"
            f"{body}</table></body></html>"
        )
        (out_dir / "metric_results.html").write_text(doc, encoding="utf-8")
        log.info("wrote metric results HTML: %s", out_dir / "metric_results.html")
    except Exception as exc:  # noqa: BLE001 - reporting must never break the run
        log.warning("could not write metric results HTML: %s", exc)


def print_results_table(variant_config, training_results, metric_rows, request):
    _write_metric_results_html(metric_rows, variant_config, request)
    if not training_results:
        log.info("No PyTorch Vision results to print")
        return
    headers = ["Cell", "Host", *[name for name, _unit in METRICS]]
    rows = []
    for sweep_name, host_results in training_results.items():
        for host, actuals in host_results.items():
            rows.append(
                [
                    variant_config.cell_key(sweep_name),
                    host,
                    *[actuals.get(f"training.{name}", "-") for name, _unit in METRICS],
                ]
            )
    log.info("\n%s", tabulate(rows, headers=headers, tablefmt="github", floatfmt=".3f"))


def teardown(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    start = time.monotonic()
    if not orch.teardown_containers():
        pytest.fail(f"failed to tear down container {name}")
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - start)
    status = orch.runtime.is_running(name)
    running = [host for host, info in status.items() if info.get("running")]
    probe_failures = [host for host, info in status.items() if info.get("exit_code") != 0]
    if running or probe_failures:
        pytest.fail(f"container teardown verification failed: running={running}, probe_failures={probe_failures}")
    lifecycle.torn_down = True
