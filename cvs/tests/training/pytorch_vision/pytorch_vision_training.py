"""W1 ResNet-50 training performance suite."""

import json
import os
import time

import pytest
from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.training.pytorch_vision.job import PyTorchVisionJob
from cvs.lib.training.pytorch_vision.utils.config_loader import validate_sweep_selector
from cvs.lib.training.pytorch_vision.utils.metrics import GATED_METRICS, METRICS, METRIC_UNITS
from cvs.lib.utils.verdict import evaluate_all


log = globals.log


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as stream:
        raw = json.load(stream)

    sweep = raw.get("sweep", {})
    combinations = sweep.get("combinations", {})
    runs = sweep.get("runs", [])
    validate_sweep_selector(combinations.keys(), runs)

    if "metric" in metafunc.fixturenames:
        cases = [(combo_key, metric) for combo_key in runs for metric, _unit in METRICS]
        metafunc.parametrize(
            "combo_key,metric",
            cases,
            ids=[f"{combo_key}-{metric}" for combo_key, metric in cases],
        )
    elif "combo_key" in metafunc.fixturenames:
        metafunc.parametrize("combo_key", runs, ids=runs)


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
    combo_key = variant_config.sweep.runs[0]
    job = PyTorchVisionJob(orch, variant_config, combo_key)
    start = time.monotonic()
    try:
        summary = job.verify_environment()
    except Exception:
        lifecycle.failed = True
        raise
    lifecycle.record(request.node.nodeid, "environment_verification", time.monotonic() - start)
    log.info("PyTorch Vision environment: %s", summary)


def test_training(orch, variant_config, combo_key, training_results, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    job = PyTorchVisionJob(orch, variant_config, combo_key)
    globals.error_list = []
    start = time.monotonic()
    try:
        try:
            job.stage_benchmark()
            job.run_benchmark()
            training_results[combo_key] = job.parse_results()
        finally:
            try:
                if job.training_start_time:
                    job.scan_dmesg_for_errors()
            finally:
                job.stop_training_processes()
        if globals.error_list:
            pytest.fail("dmesg verification found errors: " + "; ".join(globals.error_list))
    except Exception:
        lifecycle.failed = True
        raise
    lifecycle.record(request.node.nodeid, "training", time.monotonic() - start)


def test_metric(combo_key, metric, variant_config, training_results, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if combo_key not in training_results:
        pytest.skip(f"no results recorded for {combo_key}")

    host, actuals = next(iter(training_results[combo_key].items()))
    name = f"training.{metric}"
    value = actuals[name]
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", METRIC_UNITS[metric]))
    log.info("%s %s=%s %s", host, name, value, METRIC_UNITS[metric])

    if variant_config.enforce_thresholds:
        cell = variant_config.cell_key(combo_key)
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
    for combo_key, host_results in training_results.items():
        for host, actuals in host_results.items():
            rows.append(
                [
                    variant_config.cell_key(combo_key),
                    host,
                    *[actuals[f"training.{name}"] for name, _unit in METRICS],
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
