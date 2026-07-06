'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Parametrized pytorch_vision single-node benchmark suite.

Lifecycle-as-tests (each stage is a timed, pass/fail HTML row):
  test_launch_container -> test_setup_sshd -> test_verify_env
    -> test_vision_benchmark (per cell) -> test_metric (per metric per cell)
    -> test_print_results_table -> test_teardown

The container image (see container.image in the config) is pulled and launched by
the shared ContainerOrchestrator in test_launch_container; there is no
suite-specific docker/registry handling here.
'''

import json
import os
import time

import pytest

from cvs.lib import globals
from cvs.lib.utils.verdict import evaluate_all
from cvs.lib.vision.utils.vision_config_loader import validate_model_sweep
from cvs.lib.vision.utils.vision_parsing import VISION_METRICS as _METRICS, VISION_METRIC_UNITS as _METRIC_UNITS
from cvs.lib.vision.vision_job import VisionJob

import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location("_vision_shared", _pl.Path(__file__).with_name("_shared.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
test_print_results_table = _mod.test_print_results_table  # exported as a sibling test  # noqa: F841

log = globals.log


def pytest_generate_tests(metafunc):
    """Parametrize test_vision_benchmark + test_metric from the sweep selector.

    Lives in the suite module (not conftest) because it parametrizes fixtures
    only these tests consume. Runs at collection time, before fixtures exist, so
    it reads the raw config_file JSON directly (it cannot use the typed loader).

    The sweep lists `model_combinations` (each with a `name`) once and a `runs`
    array of `{combo, batch_size}` pairs; one case is emitted per run. No NxM
    cartesian -- exactly the cells `runs` enumerates.
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)
    sweep = raw.get("sweep", {})
    combos = sweep.get("model_combinations", [])
    runs = sweep.get("runs", [])
    # Mirror the typed Sweep validator here (this path reads raw JSON before
    # load_variant runs) via the shared rule so the two cannot drift: a duplicate
    # combo name or a run referencing an unknown combo must fail collection.
    validate_model_sweep([c["name"] for c in combos], [r["combo"] for r in runs])
    by_name = {c["name"]: c for c in combos}
    cases = []
    ids = []
    for run in runs:
        combo = by_name[run["combo"]]
        bs = run["batch_size"]
        cases.append((combo, bs))
        ids.append(run["combo"] + "-bs" + str(bs))
    if "metric" in metafunc.fixturenames:
        if cases:
            metric_cases = []
            metric_ids = []
            for (combo, bs), cid in zip(cases, ids):
                for short, _unit in _METRICS:
                    metric_cases.append((combo, bs, short))
                    metric_ids.append(cid + "-" + short)
            metafunc.parametrize("model_combo,batch_size,metric", metric_cases, ids=metric_ids)
    elif "model_combo" in metafunc.fixturenames and "batch_size" in metafunc.fixturenames and cases:
        metafunc.parametrize("model_combo,batch_size", cases, ids=ids)


def _cell_result_key(variant_config, combo, batch_size):
    """The res_dict key for one cell. Shared by benchmark + metric tests so the
    two look up the same cell; also the tuple _shared.py unpacks for the table."""
    return (
        combo["arch"],
        variant_config.gpu_arch,
        combo.get("precision", "fp16"),
        combo.get("input_size", "224"),
        combo.get("name", "default"),
        batch_size,
    )


def test_launch_container(orch, variant_config, lifecycle, request):
    """Stage 1: launch the container (pulls the image if not present locally).

    Asserts the container is independently observed running afterwards.
    """
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


def test_setup_sshd(orch, lifecycle, request):
    """Stage 2: start sshd in the container (multinode only; single-node skips it)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    t = time.monotonic()
    ok = orch.setup_sshd()
    lifecycle.record(request.node.nodeid, "sshd_setup", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        pytest.fail("setup_sshd() returned False")
    if len(orch.hosts) > 1:
        probe = orch.exec("bash -c 'ss -ltn 2>/dev/null | grep -q :2224 && echo OK || echo NO'")
        if not any("OK" in (v or "") for v in (probe or {}).values()):
            lifecycle.failed = True
            pytest.fail("sshd not listening on 2224 after setup_sshd()")


def test_verify_env(orch, variant_config, lifecycle, request):
    """Stage 3: prove the pulled image is usable -- torch + GPU + torchvision.

    Replaces the serving suites' model-fetch stage (a torchvision benchmark uses
    random weights, so there is nothing to download). Also stages the env script
    and the benchmark script into the container for the cells that follow.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    # Any combo works to stage scripts / probe; use the first run's combo shape.
    probe_job = VisionJob(orch=orch, variant=variant_config, arch="resnet50", precision="fp16", input_size="224", batch_size="1")
    t = time.monotonic()
    try:
        probe_job.stage_scripts()
        summary = probe_job.verify_env()
    except Exception:
        lifecycle.failed = True
        raise
    lifecycle.record(request.node.nodeid, "verify_env", time.monotonic() - t)
    log.info("container env: %s", summary)


def test_vision_benchmark(orch, variant_config, model_combo, batch_size, res_dict, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    job = VisionJob(
        orch=orch,
        variant=variant_config,
        arch=model_combo["arch"],
        precision=model_combo.get("precision", "fp16"),
        input_size=model_combo.get("input_size", "224"),
        batch_size=batch_size,
    )
    # A failure mid-sweep flips lifecycle.failed so the remaining cells skip
    # cleanly AND the orch leak-guard finalizer still tears the container down.
    try:
        job.stage_scripts()
        t = time.monotonic()
        job.run_benchmark()
        lifecycle.record(request.node.nodeid, "benchmark", time.monotonic() - t)
        results = job.parse_results()
    except Exception:
        lifecycle.failed = True
        raise

    res_dict[_cell_result_key(variant_config, model_combo, batch_size)] = results


def test_metric(model_combo, batch_size, metric, res_dict, variant_config, lifecycle, request):
    """One pytest test (= one HTML row) per metric per cell.

    Reads a single cached metric recorded by test_vision_benchmark and surfaces
    it as its own pass/fail row. Verdict: when enforce_thresholds is true AND a
    spec exists for this cell+metric the value is asserted via the shared
    evaluate_all; otherwise the row is a record-only PASS that displays the number.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    key = _cell_result_key(variant_config, model_combo, batch_size)
    if key not in res_dict:
        pytest.skip(f"no recorded results for cell {key!r} (benchmark did not run)")
    host_dict = res_dict[key]
    _host, actuals = next(iter(host_dict.items()))
    full = "vision." + metric
    value = actuals.get(full)
    unit = _METRIC_UNITS.get(metric, "-")
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", unit))

    if not variant_config.enforce_thresholds:
        return
    cell = variant_config.cell_key(
        model_combo["arch"],
        model_combo.get("precision", "fp16"),
        model_combo.get("input_size", "224"),
        batch_size,
    )
    spec = (variant_config.thresholds.get(cell) or {}).get(full)
    if spec is None:
        return
    evaluate_all(actuals, {full: spec})


def test_teardown(orch, lifecycle, request):
    """Final stage: explicit container teardown, timed, asserting it is gone.

    Sets lifecycle.torn_down so the orch fixture's leak-guard finalizer no-ops.
    Runs even if an earlier stage failed -- teardown must happen regardless.
    """
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
