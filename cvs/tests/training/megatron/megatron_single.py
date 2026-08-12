'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Unified Megatron training suite for single-node runs.
Topology is determined by the config file:
  framework=megatron_single  -> single-node (distributed_training=False)

Lifecycle (each stage is a separate test):
  test_launch_container  — launch the container once for all sweep combos
  test_smoke             — fixed small cell: model loads and runs N steps without error
  test_training          — parametrized: one test per sweep combo; kills GPU
                           processes in finally so VRAM is free for the next combo
  test_metric            — parametrized: threshold check per combo via evaluate_all
  test_loss_curve        — parametrized: slope-based loss decrease check with PNG render
  test_teardown          — tear down the container once after all combos
'''

import json
import os
import time

import pytest

from cvs.lib import globals
from cvs.lib.training.megatron.megatron_lib import MegatronTrainingJob
from cvs.lib.training.megatron.utils.loss_curve import parse_all_loss_points, sample_loss_curve, evaluate_loss_decreasing
from cvs.lib.training.megatron.utils.loss_curve_plot import render_loss_curve_png
from cvs.lib.training.megatron.utils.scaling import compute_scaling_efficiency
from cvs.lib.utils.verdict import _check_one, ThresholdViolation
from cvs.lib.utils_lib import update_test_result

log = globals.log

# Smoke cell: smallest fixed parameters that confirm the model loads and trains.
_SMOKE_MBS = "1"
_SMOKE_GBS = "8"
_SMOKE_ITERS = "10"
_SMOKE_PRECISION = "BF16"


def pytest_generate_tests(metafunc):
    """Parametrize test_training and test_metric from sweep.combinations filtered by sweep.runs.

    sweep.combinations is a dict of {run_id: {micro_batch_size, global_batch_size, ...}}.
    sweep.runs is a list of run_ids to execute (subset or all).
    One case is emitted per entry in sweep.runs — no cartesian product.
    The pytest parametrize ID is the run_id so that request.node.callspec.id
    can be passed directly to variant_config.cell_key().
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
        cases.append((mbs, gbs, precision))
        ids.append(run_id)

    if "micro_batch_size" in metafunc.fixturenames and "global_batch_size" in metafunc.fixturenames and cases:
        metafunc.parametrize("micro_batch_size,global_batch_size,precision", cases, ids=ids)


def test_launch_container(orch, variant_config, lifecycle, request):
    """Stage 0: launch the container once for all sweep combos."""
    nodeid = request.node.nodeid
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    lifecycle.torn_down = False

    t = time.monotonic()
    ok = orch.setup_containers()
    lifecycle.record(nodeid, "container_launch", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        pytest.fail(f"setup_containers() returned False for {name}")
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


def test_download_tokenizer(orch, variant_config, hf_token, lifecycle, request):
    """Stage 1: download the tokenizer model once if the model family requires it."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    mt_obj = MegatronTrainingJob(
        orch,
        variant_config,
        hf_token=hf_token,
        micro_batch_size="1",
        global_batch_size="1",
        precision="BF16",
        distributed_training=False,
        tune_model_params=False,
        run_label="tokenizer_check",
    )

    if not mt_obj._needs_local_tokenizer():
        lifecycle.tokenizer_path = None
        log.info(
            "test_download_tokenizer: no local tokenizer needed for %s — skipping download",
            variant_config.model_params["tokenizer_model"],
        )
        return

    t = time.monotonic()
    try:
        mt_obj.download_tokenizer_model()
    except Exception:
        lifecycle.failed = True
        raise

    lifecycle.tokenizer_path = mt_obj.local_tokenizer_path
    lifecycle.record(request.node.nodeid, "tokenizer_download", time.monotonic() - t)
    log.info("test_download_tokenizer: tokenizer ready at %s", lifecycle.tokenizer_path)


def test_smoke(orch, variant_config, hf_token, lifecycle, request):
    """Stage 2: smoke-test — model loads and runs _SMOKE_ITERS steps without error.

    Passes if training reaches iteration _SMOKE_ITERS/_SMOKE_ITERS without error.
    No metric assertions — completion without error is the only requirement.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    globals.error_list = []

    mt_obj = MegatronTrainingJob(
        orch,
        variant_config,
        hf_token=hf_token,
        micro_batch_size=_SMOKE_MBS,
        global_batch_size=_SMOKE_GBS,
        precision=_SMOKE_PRECISION,
        distributed_training=False,
        tune_model_params=False,
        run_label="smoke",
    )
    mt_obj.iterations = int(_SMOKE_ITERS)
    mt_obj.local_tokenizer_path = getattr(lifecycle, "tokenizer_path", None)

    t = time.monotonic()
    try:
        mt_obj.build_training_job_cmd()
        mt_obj.start_training_job()
        mt_obj.poll_for_training_completion()
    except Exception:
        lifecycle.failed = True
        raise
    finally:
        mt_obj.stop_training_processes()

    if globals.error_list:
        lifecycle.failed = True
    update_test_result()
    lifecycle.record(request.node.nodeid, "smoke", time.monotonic() - t)
    log.info("smoke PASSED | iters=%s", _SMOKE_ITERS)


def test_training(
    orch, variant_config, hf_token, micro_batch_size, global_batch_size, precision, train_res_dict, lifecycle, request
):
    """Stage 3 (parametrized): run one sweep combo inside the shared container.

    stop_training_processes() runs in a finally block after every combo so GPU
    memory is released before the next combo starts.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    nodeid = request.node.nodeid
    combo_key = request.node.callspec.id
    globals.error_list = []
    mt_obj = MegatronTrainingJob(
        orch,
        variant_config,
        hf_token=hf_token,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        precision=precision,
        distributed_training=False,
        tune_model_params=False,
        run_label=combo_key,
    )

    mt_obj.local_tokenizer_path = getattr(lifecycle, "tokenizer_path", None)

    elapsed = 0
    try:
        t = time.monotonic()
        mt_obj.build_training_job_cmd()
        mt_obj.start_training_job()
        mt_obj.poll_for_training_completion()
        mt_obj.verify_training_results()
        elapsed = time.monotonic() - t
    except Exception:
        lifecycle.failed = True
        raise
    finally:
        mt_obj.stop_training_processes()

    lifecycle.record(nodeid, "training", elapsed)
    request.node.user_properties.append(("metric_value", elapsed))
    request.node.user_properties.append(("metric_unit", "s"))

    train_res_dict[combo_key] = mt_obj.training_results_dict
    train_res_dict[combo_key]["_combo_log_dir"] = mt_obj.combo_log_dir

    tput_per_gpu = train_res_dict[combo_key].get("throughput_per_gpu", [])
    if tput_per_gpu:
        gpus_per_node = 8
        tokens_per_sec_total = float(tput_per_gpu[-1]) * int(mt_obj.nnodes) * gpus_per_node
        baseline = variant_config.scaling_baseline
        efficiency = compute_scaling_efficiency(
            tokens_per_sec_total,
            int(mt_obj.nnodes),
            baseline.tokens_per_sec_total,
            baseline.num_nodes,
        )
        if efficiency is not None:
            train_res_dict[combo_key]["scaling_efficiency_pct"] = [str(efficiency)]
    try:
        tail = mt_obj._read_last_node_log(tail_lines=50)
        train_res_dict[combo_key]["_log_tail"] = tail
        request.node.user_properties.append(("training_log_tail", tail))
    except Exception:
        pass
    update_test_result()


def test_metric(variant_config, micro_batch_size, global_batch_size, precision, train_res_dict, lifecycle, request):
    """Stage 4 (parametrized): compare each combo's metrics against thresholds."""
    combo_key = request.node.callspec.id
    if combo_key not in train_res_dict:
        pytest.skip(f"no recorded results for combo '{combo_key}' (training did not run)")

    if not variant_config.enforce_thresholds:
        log.info("enforce_thresholds=false; skipping verdict for combo '%s'", combo_key)
        return

    cell = variant_config.cell_key(combo_key)
    thresholds = variant_config.thresholds.get(cell)
    if not thresholds:
        log.warning("no thresholds defined for cell '%s'; skipping threshold checks", cell)
        return

    actuals_raw = train_res_dict[combo_key]
    request.node.user_properties.append(("training_log_tail", actuals_raw.get("_log_tail", "")))
    actuals = {f"training.{k}": float(v[-1]) for k, v in actuals_raw.items() if v and not k.startswith("_")}

    log.info("--- Threshold check for combo '%s' ---", combo_key)
    violations = []
    for metric, spec in thresholds.items():
        if metric not in actuals:
            msg = f"{metric}: missing from actuals"
            log.error("  FAILED  %s", msg)
            violations.append(msg)
            continue
        if actuals[metric] is None:
            msg = f"{metric}: value is None (metric unavailable for this run)"
            log.error("  FAILED  %s", msg)
            violations.append(msg)
            continue
        spec_with_actuals = dict(spec)
        if spec.get("kind") == "min_ratio":
            spec_with_actuals["_actuals"] = actuals
        v = _check_one(metric, actuals[metric], spec_with_actuals)
        if v:
            log.error("  FAILED  %s", v)
            violations.append(v)
        else:
            log.info("  PASSED  %s: actual=%s  threshold=%s", metric, actuals[metric], spec)

    if violations:
        summary = "FAILED\n" + "\n".join(violations)
        log.error("--- %d violation(s) for combo '%s' ---", len(violations), combo_key)
        request.node.user_properties.append(("threshold_comparison", summary))
        raise ThresholdViolation(violations)

    log.info("--- All threshold checks PASSED for combo '%s' ---", combo_key)
    request.node.user_properties.append(("threshold_comparison", "PASSED"))


def test_loss_curve(
    orch, variant_config, micro_batch_size, global_batch_size, precision, train_res_dict, lifecycle, request
):
    """Parametrized: slope-based loss curve check with PNG render."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    combo_key = request.node.callspec.id
    if combo_key not in train_res_dict:
        pytest.skip(f"no recorded results for combo '{combo_key}' (training did not run)")

    combo_log_dir = train_res_dict[combo_key].get("_combo_log_dir")
    if not combo_log_dir:
        log.warning("no log dir recorded for combo '%s'; skipping loss curve check", combo_key)
        pytest.skip(f"no log dir recorded for combo '{combo_key}'")

    log_path = f"{combo_log_dir}/out-node0/training.log"
    out_dict = orch.exec(f"cat {log_path}")
    log_text = list(out_dict.values())[-1] or ""

    lc = variant_config.loss_curve
    step_metrics = parse_all_loss_points(log_text)
    points = sample_loss_curve(step_metrics, lc.sample_every, lc.milestone_steps)

    log.info("--- Loss curve check for combo '%s' (%d points sampled) ---", combo_key, len(points))

    mgr = getattr(request.config, "_html_report_manager", None)
    mgr_enabled = mgr is not None and getattr(mgr, "is_enabled", False)
    out_dir = mgr.log_dir if mgr_enabled else "/tmp"
    try:
        from pathlib import Path as _Path
        import uuid as _uuid
        _Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = f"loss_curve_{combo_key}_{str(_uuid.uuid4()).split('-')[-1]}.png"
        png_path = _Path(out_dir) / fname
        title = f"Training Loss Curve — {variant_config.model_params.get('model_name', '')} [{combo_key}]"
        rendered = render_loss_curve_png(points, png_path, title=title)
        if rendered and mgr_enabled:
            rel_path = str(_Path(rendered).relative_to(mgr.htmlpath.parent))
            lifecycle.add_artifact(request.node.nodeid, f"Loss Curve [{combo_key}]", rel_path, rendered)
    except Exception as e:
        log.warning("loss curve: could not render PNG (%s)", e)

    verdict = evaluate_loss_decreasing(points, lc.max_slope)
    if verdict is None:
        pytest.skip(f"loss curve needs >= 2 sampled points (got {len(points)}); increase training_iterations")

    decreasing, slope, detail = verdict
    log.info("loss curve: %s", detail)

    if points:
        request.node.user_properties.append(("metric_value", points[-1][1]))
        request.node.user_properties.append(("metric_unit", "lm_loss"))

    if lc.enforce and not decreasing:
        pytest.fail(f"training loss is not decreasing for combo '{combo_key}': {detail}")


def test_teardown(orch, lifecycle, request):
    """Stage 5: tear down the container once after all combos have run."""
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        log.error("container %s still running after teardown_containers()", name)
    else:
        lifecycle.torn_down = True
