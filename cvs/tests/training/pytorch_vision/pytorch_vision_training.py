"""W1 ResNet-50 training performance suite.

When pytest HTML is enabled, the matching
``cvs.lib.report.presets.pytorch_vision_training`` preset emits the run deck,
JSON payload, and CI summary beside the standard report.
"""

import json
import os
import time

import pytest
from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.training.pytorch_vision.job import PyTorchVisionJob
from cvs.lib.training.pytorch_vision.utils.config_loader import validate_sweep_selector
from cvs.lib.training.pytorch_vision.utils.metrics import (
    GATED_METRICS,
    METRICS,
    METRIC_UNITS,
    gradient_accumulation_overhead_pct,
)
from cvs.lib.utils.verdict import evaluate_all


log = globals.log


def _ga_group(sweep, gpus_per_node):
    return (
        sweep.model,
        sweep.backend,
        sweep.precision,
        sweep.image_size,
        sweep.batch_size * sweep.gradient_accumulation_steps * gpus_per_node,
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


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as stream:
        raw = json.load(stream)

    training = raw.get("training", {})
    sweeps = training.get("sweeps", [])
    by_name = {sweep["name"]: sweep for sweep in sweeps}
    enabled = training.get("enabled_sweep_list") or list(by_name)
    validate_sweep_selector(
        by_name.keys(),
        enabled,
        [sweep["label"] for sweep in sweeps],
    )

    if "metric" in metafunc.fixturenames:
        cases = [(sweep_name, metric) for sweep_name in enabled for metric, _unit in METRICS]
        metafunc.parametrize(
            "sweep_name,metric",
            cases,
            ids=[f"{by_name[sweep_name]['label']}-{metric}" for sweep_name, metric in cases],
        )
    elif "sweep_name" in metafunc.fixturenames:
        metafunc.parametrize(
            "sweep_name",
            enabled,
            ids=[by_name[sweep_name]["label"] for sweep_name in enabled],
        )


def test_launch_container(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    start = time.monotonic()
    if not orch.setup_containers():
        lifecycle.failed = True
        pytest.fail(f"failed to launch container {name}")
    lifecycle.record(request.node.nodeid, "container_launch", time.monotonic() - start)
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} is not running after launch")


def test_verify_environment(orch, variant_config, lifecycle, request):
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


def test_training(orch, variant_config, sweep_name, training_results, inf_res_dict, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    job = PyTorchVisionJob(orch, variant_config, sweep_name)
    globals.error_list = []
    start = time.monotonic()
    try:
        try:
            job.stage_benchmark()
            job.run_benchmark()
            job.verify_training_log()
            host_results = job.parse_results()
            training_results[sweep_name] = host_results
            sweep = variant_config.sweep(sweep_name)
            report_key = (
                sweep.model,
                variant_config.gpu_arch,
                (
                    f"W1-ROCAL-{sweep.precision}-R{sweep.image_size}"
                    if sweep.data_mode == "rocal"
                    else f"W1-{sweep.precision}-R{sweep.image_size}"
                ),
                sweep.image_size,
                f"GA{sweep.gradient_accumulation_steps}",
                sweep.batch_size,
            )
            inf_res_dict[report_key] = host_results
            _refresh_ga_overhead(variant_config, training_results)
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
    lifecycle.record(request.node.nodeid, "training", time.monotonic() - start)


def test_metric(sweep_name, metric, variant_config, training_results, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if sweep_name not in training_results:
        pytest.skip(f"no results recorded for {sweep_name}")

    host, actuals = next(iter(training_results[sweep_name].items()))
    name = f"training.{metric}"
    if name not in actuals:
        if metric in GATED_METRICS:
            pytest.fail(f"required metric was not produced: {name}")
        pytest.skip(f"metric is not applicable to this sweep: {name}")
    value = actuals[name]
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", METRIC_UNITS[metric]))
    lifecycle.record(request.node.nodeid, "metric_evaluation", 0.0)
    log.info("%s %s=%s %s", host, name, value, METRIC_UNITS[metric])

    if variant_config.enforce_thresholds:
        cell = variant_config.cell_key(sweep_name)
        spec = (variant_config.thresholds.get(cell) or {}).get(name)
        if spec is None:
            if metric in GATED_METRICS:
                pytest.fail(f"threshold is missing for {cell}: {name}")
            return
        evaluate_all(actuals, {name: spec})


def test_print_results_table(variant_config, training_results):
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


def test_teardown(orch, lifecycle, request):
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
