'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Parametrized TorchTitan single-node training suite.
One config per model; sweep.combinations + sweep.runs drive parametrization.

Each sweep combo runs in its OWN freshly-launched container: launch -> train ->
verify -> save results -> teardown. Combos never share port 6000, log files, or
scripts dir, and each combo's dmesg/verify window is scoped to its own run.
The image is pulled only on the first launch (cached thereafter), so recycling
the container per combo is cheap.
'''

import json
import os
import time

import pytest

from cvs.lib import globals
from cvs.lib.training.torchtitan import torchtitan_lib
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
    """Run the full per-combo lifecycle in a dedicated container.

    Launches a fresh container for this combo, runs single-node TorchTitan
    training for the given micro_batch_size / global_batch_size, verifies and
    stores the results, then ALWAYS tears the container down (finally) so the
    next combo starts on a clean node — freeing port 6000, the training log, and
    the scripts dir. The image is pulled only on the first launch (cached
    afterwards), so relaunch per combo is cheap.

    Model-level params (tp, pp, precision, etc.) come from variant_config.model_params.
    Each container-lifecycle sub-stage is timed via lifecycle.record so it shows
    up in this test's HTML detail panel.
    """
    nodeid = request.node.nodeid
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])

    # A container is about to exist; the orch leak-guard should own cleanup until
    # this combo's own teardown (finally) confirms it is gone.
    lifecycle.torn_down = False

    try:
        # Stage 1: launch a fresh container for this combo (was test_launch_container).
        t = time.monotonic()
        ok = orch.setup_containers()
        lifecycle.record(nodeid, "container_launch", time.monotonic() - t)
        if not ok:
            pytest.fail(f"setup_containers() returned False for {name}")
        if not orch.verify_containers_running(name):
            pytest.fail(f"container {name} not running after setup_containers()")

        # Stage 2: start sshd (was test_setup_sshd). Single-node runs skip
        # starting the in-container sshd (it exists only for inter-node MPI), so
        # only probe 2224 when there is more than one host.
        t = time.monotonic()
        ok = orch.setup_sshd()
        lifecycle.record(nodeid, "sshd_setup", time.monotonic() - t)
        if not ok:
            pytest.fail("setup_sshd() returned False")
        if len(orch.hosts) > 1:
            probe = orch.exec("bash -c 'ss -ltn 2>/dev/null | grep -q :2224 && echo OK || echo NO'")
            if not any("OK" in (v or "") for v in (probe or {}).values()):
                pytest.fail("sshd not listening on 2224 after setup_sshd()")

        # Stage 3: download HF model assets (TorchTitan-specific).
        # Creates a temporary TorchTitanTrainingJob just for downloading.
        # Idempotent - skips if already present.
        globals.error_list = []
        tt_obj_download = torchtitan_lib.TorchTitanTrainingJob(
            orch,
            variant_config,
            hf_token,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            precision=precision,
            result_dict=result_dict,
            distributed_training=False,
            tune_model_params=False,
            run_label=request.node.callspec.id,
        )

        t = time.monotonic()
        tt_obj_download.download_hf_assets()
        lifecycle.record(nodeid, "model_download", time.monotonic() - t)

        # Stage 4: training.
        globals.error_list = []
        tt_obj = torchtitan_lib.TorchTitanTrainingJob(
            orch,
            variant_config,
            hf_token,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            precision=precision,
            result_dict=result_dict,
            distributed_training=False,
            tune_model_params=False,
            run_label=request.node.callspec.id,
        )

        t = time.monotonic()
        tt_obj.exec_nic_setup_scripts()
        tt_obj.build_training_job_cmd()
        tt_obj.start_training_job()
        tt_obj.poll_for_training_completion()
        tt_obj.verify_training_results()
        elapsed = time.monotonic() - t

        lifecycle.record(nodeid, "training", elapsed)
        request.node.user_properties.append(("metric_value", elapsed))
        request.node.user_properties.append(("metric_unit", "s"))

        combo_key = request.node.callspec.id
        train_res_dict[combo_key] = tt_obj.training_results_dict
        update_test_result()
    finally:
        # Teardown (was test_teardown) — always recycle the container so the next
        # combo starts on a clean node even if a stage above failed.
        t = time.monotonic()
        orch.teardown_containers()
        lifecycle.record(nodeid, "teardown", time.monotonic() - t)
        if orch.verify_containers_running(name):
            log.error("container %s still running after teardown_containers()", name)
        else:
            # This combo's container is gone; suppress the module-end leak-guard
            # so it does not tear down a second time.
            lifecycle.torn_down = True


def test_throughput(variant_config, micro_batch_size, global_batch_size, precision, result_dict, train_res_dict, lifecycle, request):
    """Threshold check using variant_config.cell_key() and threshold_dict.

    Uses cell_key() format: MBS=<mbs>,GBS=<gbs>,PRECISION=<precision>
    Thresholds loaded from external *_threshold.json file via variant_config.threshold_dict
    Supports both new threshold_dict (preferred) and legacy result_dict (backwards compat)
    """
    combo_key = request.node.callspec.id
    if combo_key not in train_res_dict:
        pytest.skip(f"no recorded results for combo {combo_key} (training did not run)")

    if not variant_config.enforce_thresholds:
        log.info("enforce_thresholds=false; recorded metrics for combo %s, skipping verdict", combo_key)
        return

    # Use new cell_key format for threshold lookup
    cell_key = variant_config.cell_key(
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        precision=precision,
    )

    # Prefer external threshold_dict, fallback to legacy result_dict
    if variant_config.threshold_dict:
        if cell_key not in variant_config.threshold_dict:
            log.warning("no threshold entry for cell %s; skipping", cell_key)
            return
        threshold_specs = variant_config.threshold_dict[cell_key]
    elif result_dict:
        # Legacy mode: inline result_dict - convert to threshold spec format
        threshold_specs = {f"training.{k}": {"kind": "min", "value": v} for k, v in result_dict.items()}
    else:
        log.warning("no thresholds defined for combo %s; skipping threshold checks", combo_key)
        return

    # Evaluate thresholds using evaluate_all
    from cvs.lib.utils.verdict import evaluate_all
    actuals = train_res_dict[combo_key]
    evaluate_all(
        threshold_specs,
        actuals,
        cell_key,
        tolerance_pct=5.0,
    )
