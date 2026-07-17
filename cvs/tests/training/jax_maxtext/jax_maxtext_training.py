'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

JAX MaxText training suite — single test file for both single-node and distributed.

The mode is determined by the config file passed at runtime via --config_file.
The config's `training.distributed` field drives skipping of distributed-only
stages (sshd, RDMA, NIC setup).
'''

import time

import pytest

from cvs.lib import globals
from cvs.lib.training.jax_maxtext_training_lib import MaxTextTrainingJob
from cvs.lib.training.utils.maxtext_parsing import TRAINING_METRICS, TRAINING_METRIC_UNITS
from cvs.lib.utils.verdict import evaluate_all

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


def test_setup_sshd(orch, variant_config, lifecycle, request):
    """Stage 2: start sshd in the container (distributed only)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not variant_config.training.distributed:
        pytest.skip("single-node: sshd not needed")
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


def test_setup_rdma(orch, variant_config, hf_token, lifecycle, request):
    """Stage 3: copy RDMA library into container (distributed + thor2 NIC only)."""
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
    """Stage 4: run NIC setup scripts (distributed only)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not variant_config.training.distributed:
        pytest.skip("single-node: NIC setup not needed")
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.exec_nic_setup_scripts()
    lifecycle.record(request.node.nodeid, "nic_setup", time.monotonic() - t)


def test_setup_tokenizer(orch, variant_config, hf_token, lifecycle, request):
    """Stage 5: download HF tokenizer into models dir."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.setup_tokenizer()
    lifecycle.record(request.node.nodeid, "tokenizer_setup", time.monotonic() - t)


def test_training_run(orch, variant_config, hf_token, training_res_dict, lifecycle, request):
    """Stage 6: build training command, start training, poll for completion, parse results."""
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

    training_res_dict["results"] = results
    training_res_dict["step_metrics"] = job.step_metrics
    training_res_dict["num_nodes"] = job.num_nodes


def test_metric(metric, training_res_dict, variant_config, lifecycle, request):
    """One HTML row per training metric. Record-only unless enforce_thresholds."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    results = training_res_dict.get("results")
    if not results:
        pytest.skip("no training results (training_run did not complete)")

    full = "training." + metric
    value = results.get(full)
    unit = TRAINING_METRIC_UNITS.get(metric, "-")
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", unit))

    if not variant_config.enforce_thresholds:
        return
    num_nodes = training_res_dict.get("num_nodes", 1)
    cell = variant_config.cell_key(num_nodes=num_nodes)
    spec = (variant_config.thresholds.get(cell) or {}).get(full)
    if spec is None:
        return
    evaluate_all(results, {full: spec})


def test_teardown(orch, lifecycle, request):
    """Final stage: explicit container teardown."""
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
