'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Parametrized Megatron single-node training suite.
One config per model; sweep.combinations + sweep.runs drive parametrization.
'''

import json
import os
import time

import pytest

from cvs.lib import globals
from cvs.lib.training.megatron import megatron_lib
from cvs.lib.utils_lib import update_test_result

log = globals.log


def pytest_generate_tests(metafunc):
    """Parametrize micro_batch_size and global_batch_size from sweep.combinations filtered by sweep.runs.

    sweep.combinations is a dict of {run_id: {micro_batch_size, global_batch_size, ...}}.
    sweep.runs is a list of run_ids to execute (subset or all).
    One case is emitted per entry in sweep.runs — no cartesian product.
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)

    sweep = raw.get("sweep", {})
    combinations = sweep.get("combinations", {})
    runs = sweep.get("runs", list(combinations.keys()))

    cases = []
    ids = []
    for run_id in runs:
        if run_id not in combinations:
            log.warning("sweep.runs entry '%s' not found in sweep.combinations; skipping", run_id)
            continue
        combo = combinations[run_id]
        mbs = combo["micro_batch_size"]
        gbs = combo["global_batch_size"]
        precision = combo.get("precision", "")
        result_dict = combo.get("result_dict", {})
        cases.append((mbs, gbs, precision, result_dict))
        ids.append(combo.get("name", run_id))

    if "micro_batch_size" in metafunc.fixturenames and "global_batch_size" in metafunc.fixturenames and cases:
        metafunc.parametrize("micro_batch_size,global_batch_size,precision,result_dict", cases, ids=ids)


def test_launch_container(orch, lifecycle, request):
    """Stage 1: launch the container. Asserts it is independently observed running."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

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
    # Single-node runs skip starting the in-container sshd (it exists only for
    # inter-node MPI), so only probe 2224 when there is more than one host.
    if len(orch.hosts) > 1:
        probe = orch.exec("bash -c 'ss -ltn 2>/dev/null | grep -q :2224 && echo OK || echo NO'")
        if not any("OK" in (v or "") for v in (probe or {}).values()):
            lifecycle.failed = True
            pytest.fail("sshd not listening on 2224 after setup_sshd()")


def test_training(orch, variant_config, hf_token, micro_batch_size, global_batch_size, precision, result_dict, train_res_dict, lifecycle, request):
    """Run single-node Megatron training for the given micro_batch_size / global_batch_size combo.

    Model-level params (tp, pp, precision, etc.) come from variant_config.model_params.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    globals.error_list = []
    mt_obj = megatron_lib.MegatronTrainingJob(
        orch,
        variant_config,
        hf_token,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        precision=precision,
        result_dict=result_dict,
        distributed_training=False,
        tune_model_params=False,
    )

    try:
        t = time.monotonic()
        mt_obj.exec_nic_setup_scripts()
        mt_obj.build_training_job_cmd()
        mt_obj.start_training_job()
        mt_obj.poll_for_training_completion()
        mt_obj.verify_training_results()
        elapsed = time.monotonic() - t
    except Exception:
        lifecycle.failed = True
        raise

    lifecycle.record(request.node.nodeid, "training", elapsed)
    request.node.user_properties.append(("metric_value", elapsed))
    request.node.user_properties.append(("metric_unit", "s"))

    combo_key = request.node.callspec.id
    train_res_dict[combo_key] = mt_obj.training_results_dict
    update_test_result()


def test_throughput(variant_config, micro_batch_size, global_batch_size, precision, result_dict, train_res_dict, lifecycle, request):
    """Assert each metric in the combo's result_dict threshold spec is met.

    Thresholds are inline per combo in the config file under sweep.combinations.<id>.result_dict.
    Skips cleanly if training did not run for this combo or enforce_thresholds is false.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    combo_key = request.node.callspec.id
    if combo_key not in train_res_dict:
        pytest.skip(f"no recorded results for combo '{combo_key}' (training did not run)")

    if not variant_config.enforce_thresholds:
        log.info("enforce_thresholds=false; recorded metrics for combo '%s', skipping verdict", combo_key)
        return

    if not result_dict:
        log.warning("no thresholds defined for combo '%s'; skipping threshold checks", combo_key)
        return

    actuals = train_res_dict[combo_key]
    for metric, threshold in result_dict.items():
        measured = actuals.get(metric, [])
        if not measured:
            log.warning("metric '%s' not found in training results for combo '%s'", metric, combo_key)
            continue
        for val in measured:
            if float(val) < float(threshold):
                pytest.fail(
                    f"metric '{metric}' below threshold for combo '{combo_key}': "
                    f"expected >= {threshold}, got {val}"
                )


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
