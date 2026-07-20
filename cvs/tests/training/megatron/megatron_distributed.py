'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Parametrized Megatron distributed (multi-node) training suite.
One config per model; sweep.combinations + sweep.runs drive parametrization.

Each sweep combo runs in its OWN freshly-launched container set: launch -> train ->
verify -> save results -> teardown. Combos never share port 6000, log files, or
scripts dir, and each combo's dmesg/verify window is scoped to its own run.
The image is pulled only on the first launch (cached thereafter), so recycling
the containers per combo is cheap.
'''

import json
import os
import re
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


def test_training(orch, variant_config, hf_token, micro_batch_size, global_batch_size, precision, result_dict, train_res_dict, lifecycle, request):
    """Run the full per-combo lifecycle in a dedicated container set.

    Launches fresh containers for this combo, runs distributed Megatron training
    for the given micro_batch_size / global_batch_size across all nodes, verifies
    and stores the results, then ALWAYS tears the containers down (finally) so the
    next combo starts on a clean cluster — freeing port 6000, the training log,
    and the scripts dir. The image is pulled only on the first launch (cached
    afterwards), so relaunch per combo is cheap.

    Model-level params (tp, pp, precision, etc.) come from variant_config.model_params.
    Each container-lifecycle sub-stage is timed via lifecycle.record so it shows
    up in this test's HTML detail panel.
    """
    nodeid = request.node.nodeid
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])

    # A container set is about to exist; the orch leak-guard should own cleanup until
    # this combo's own teardown (finally) confirms it is gone.
    lifecycle.torn_down = False

    try:
        # Stage 0: disable firewall — required for distributed runs to avoid
        # inter-node MPI threads timing out against the Rendezvous endpoint.
        t = time.monotonic()
        out_dict = orch.exec("sudo service ufw status")
        for node, out in (out_dict or {}).items():
            if not re.search("inactive", out or "", re.I):
                orch.exec("sudo service ufw stop")
        out_dict = orch.exec("sudo ufw status")
        for node, out in (out_dict or {}).items():
            if not re.search("inactive|disabled", out or "", re.I):
                pytest.fail(f"failed to disable firewall on node {node}")
        lifecycle.record(nodeid, "firewall_disable", time.monotonic() - t)

        # Stage 1: launch fresh containers for this combo.
        t = time.monotonic()
        ok = orch.setup_containers()
        lifecycle.record(nodeid, "container_launch", time.monotonic() - t)
        if not ok:
            pytest.fail(f"setup_containers() returned False for {name}")
        if not orch.verify_containers_running(name):
            pytest.fail(f"container {name} not running after setup_containers()")

        # Stage 2: start sshd — distributed runs always require inter-node MPI
        # so sshd on 2224 is mandatory on all nodes.
        t = time.monotonic()
        ok = orch.setup_sshd()
        lifecycle.record(nodeid, "sshd_setup", time.monotonic() - t)
        if not ok:
            pytest.fail("setup_sshd() returned False")
        probe = orch.exec("bash -c 'ss -ltn 2>/dev/null | grep -q :2224 && echo OK || echo NO'")
        if not any("OK" in (v or "") for v in (probe or {}).values()):
            pytest.fail("sshd not listening on 2224 after setup_sshd()")

        # Stage 3: training.
        globals.error_list = []
        mt_obj = megatron_lib.MegatronTrainingJob(
            orch,
            variant_config,
            hf_token,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            precision=precision,
            result_dict=result_dict,
            distributed_training=True,
            tune_model_params=False,
            run_label=request.node.callspec.id,
        )

        t = time.monotonic()
        mt_obj.build_training_job_cmd()
        mt_obj.start_training_job()
        mt_obj.poll_for_training_completion()
        mt_obj.verify_training_results()
        elapsed = time.monotonic() - t

        lifecycle.record(nodeid, "training", elapsed)
        request.node.user_properties.append(("metric_value", elapsed))
        request.node.user_properties.append(("metric_unit", "s"))

        combo_key = request.node.callspec.id
        train_res_dict[combo_key] = mt_obj.training_results_dict
        update_test_result()
    finally:
        # Teardown — always recycle the containers so the next combo starts on a
        # clean cluster even if a stage above failed.
        t = time.monotonic()
        orch.teardown_containers()
        lifecycle.record(nodeid, "teardown", time.monotonic() - t)
        if orch.verify_containers_running(name):
            log.error("container %s still running after teardown_containers()", name)
        else:
            lifecycle.torn_down = True


def test_throughput(variant_config, micro_batch_size, global_batch_size, precision, result_dict, train_res_dict, lifecycle, request):
    """Assert each metric in the combo's result_dict threshold spec is met.

    Reads results saved by test_training (containers are already gone; no
    container is needed here). Thresholds are inline per combo in the config file
    under sweep.combinations.<id>.result_dict. Skips cleanly if training did not
    record results for this combo or enforce_thresholds is false.
    """
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
