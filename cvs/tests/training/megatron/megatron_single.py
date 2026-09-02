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
  test_download_tokenizer — download HF tokenizer when the model needs a local file
  test_smoke             — fixed small cell: model loads and runs N steps without error
  test_checkpoint        — Primus-only: checkpoint save + resume correctness check
  test_training          — parametrized: one test per sweep combo; kills GPU
                           processes in finally so VRAM is free for the next combo
  test_metric            — parametrized: threshold check per combo via evaluate_all
  test_loss_curve        — parametrized: slope-based loss decrease check with PNG render
  test_teardown          — tear down the container once after all combos
'''

import json
import os
import re
import shlex
import time

import pytest

from cvs.lib import globals
from cvs.lib.training.megatron.megatron_lib import MegatronTrainingJob
from cvs.lib.training.megatron.primus_lib import PrimusTrainingJob, _parse_step_losses
from cvs.lib.training.megatron.utils.checkpoint_io import log_checkpoint_io_times, parse_checkpoint_io_seconds
from cvs.lib.training.megatron.utils.convergence import (
    compute_convergence,
    parse_step_metrics,
)
from cvs.lib.training.megatron.utils.loss_curve import (
    parse_all_loss_points,
    sample_loss_curve,
    evaluate_loss_decreasing,
)
from cvs.lib.training.megatron.utils.loss_curve_plot import render_loss_curve_png
from cvs.lib.training.megatron.utils.scaling import compute_scaling_efficiency
from cvs.lib.utils.verdict import _check_one, ThresholdViolation
from cvs.lib.utils_lib import update_test_result

log = globals.log


def _make_training_job(orch, variant_config, **kwargs):
    """Return PrimusTrainingJob or MegatronTrainingJob based on the container image."""
    if re.search(r'primus', orch.container_config.get("image", ""), re.I):
        return PrimusTrainingJob(orch, variant_config, **kwargs)
    return MegatronTrainingJob(orch, variant_config, **kwargs)


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

    mt_obj = _make_training_job(
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

    mt_obj = _make_training_job(
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


def test_checkpoint(orch, variant_config, hf_token, lifecycle, request):
    """Stage 2.5: checkpoint save + resume — verify step counter and loss continuity.

    Two phases:
      Phase 1 — Save : train checkpoint.save_iters steps, saving every checkpoint.save_interval.
                       Last checkpoint lands at floor(save_iters/interval)*interval.
      Phase 2 — Load : resume from that checkpoint with train_iters=checkpoint.resume_iters;
                       Primus continues from last_ckpt_step+1 to resume_iters.

    Asserts:
      - First logged step of load phase == last_ckpt_step + 1 (step counter restored).
      - resume_losses[last_ckpt_step+1] <= save_losses[last_ckpt_step] + tol (no loss spike).

    Skipped if checkpoint.enforce=false or for non-Primus images.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    ckpt_cfg = variant_config.checkpoint
    if not ckpt_cfg.enforce:
        pytest.skip("checkpoint.enforce=false in config; skipping test_checkpoint")

    if not re.search(r'primus', orch.container_config.get("image", ""), re.I):
        pytest.skip("checkpoint test is Primus-only")

    # PRIMUS_WORKSPACE must be on a host path already volume-mounted into the
    # container so checkpoints survive container teardown.  log_dir is mounted
    # at the same path on both host and container (runtime.args.volumes in the
    # JSON config), so a subdirectory of log_dir satisfies that requirement.
    log_dir = variant_config.config.get("log_dir", "/tmp")
    ckpt_dir = f"{log_dir}/ckpt_primus"
    orch.exec(f"mkdir -p {ckpt_dir}")

    def _run(run_label, iters, checkpoint_dir=None, save_interval=None, load_checkpoint=False):
        job = _make_training_job(
            orch,
            variant_config,
            hf_token=hf_token,
            micro_batch_size=_SMOKE_MBS,
            global_batch_size=_SMOKE_GBS,
            precision=_SMOKE_PRECISION,
            distributed_training=False,
            tune_model_params=False,
            run_label=run_label,
        )
        job.iterations = iters
        job.local_tokenizer_path = getattr(lifecycle, "tokenizer_path", None)
        if checkpoint_dir:
            job.checkpoint_dir = checkpoint_dir
            job.save_interval = save_interval or iters
        job.load_checkpoint = load_checkpoint
        try:
            job.build_training_job_cmd()
            job.start_training_job()
            job.poll_for_training_completion()
        finally:
            job.stop_training_processes()
        return job._read_last_node_log()

    # Phase 1: save — train save_iters steps, writing checkpoint every save_interval steps.
    # save_iters must not be an exact multiple of save_interval so the last checkpoint is
    # not the final step (disable_last_saving=true in the YAML suppresses the last-iter save).
    try:
        save_log = _run(
            "ckpt_save",
            ckpt_cfg.save_iters,
            checkpoint_dir=ckpt_dir,
            save_interval=ckpt_cfg.save_interval,
        )
    except Exception:
        raise

    # Verify the checkpoint was written before attempting resume.
    # With PRIMUS_TEAM/USER/EXP_NAME all empty, Primus writes to:
    #   {PRIMUS_WORKSPACE}/checkpoints/latest_checkpointed_iteration.txt
    ckpt_meta = f"{ckpt_dir}/checkpoints/latest_checkpointed_iteration.txt"
    check = orch.exec(f"test -f {ckpt_meta} && echo FOUND || echo MISSING")
    head_node = orch.hosts[0]
    if "MISSING" in (check or {}).get(head_node, "MISSING"):
        ls_out = orch.exec(f"ls -laR {ckpt_dir} 2>&1 || echo DIR_EMPTY")
        log.error("checkpoint listing:\n%s", (ls_out or {}).get(head_node, ""))
        pytest.fail(
            f"checkpoint not written after save phase; expected: {ckpt_meta}\n"
            f"Check that log_dir ({log_dir}) is volume-mounted into the container."
        )

    # last_ckpt_step = floor(save_iters / interval) * interval
    last_ckpt_step = (ckpt_cfg.save_iters // ckpt_cfg.save_interval) * ckpt_cfg.save_interval
    expected_first = last_ckpt_step + 1

    # Phase 2: load — resume from last checkpoint, train to resume_iters total.
    # Primus reads the checkpoint and starts iterating from last_ckpt_step+1.
    try:
        resume_log = _run(
            "ckpt_resume",
            ckpt_cfg.resume_iters,
            checkpoint_dir=ckpt_dir,
            load_checkpoint=True,
        )
    except Exception:
        raise

    save_losses = _parse_step_losses(save_log)
    resume_losses = _parse_step_losses(resume_log)

    log.info("checkpoint save steps  : %s", sorted(save_losses))
    log.info("checkpoint resume steps: %s", sorted(resume_losses))

    # Checkpoint I/O timing — save & load (seconds).
    save_io_times, _ = parse_checkpoint_io_seconds(save_log)
    _, load_io_seconds = parse_checkpoint_io_seconds(resume_log)
    log_checkpoint_io_times(save_io_times, load_io_seconds)

    # Check 1: step counter restored — first logged step of resume == last_ckpt_step + 1
    if not resume_losses:
        pytest.fail("load phase produced no iteration logs; cannot verify step counter")

    first_step = min(resume_losses)
    log.info(
        "CHECK 1 — step counter: last saved step in checkpoint=%d, first step in load phase=%d",
        last_ckpt_step,
        first_step,
    )
    if first_step != expected_first:
        pytest.fail(
            f"FAILED step counter check: last saved checkpoint step={last_ckpt_step}, "
            f"expected load to start at step {expected_first}, got {first_step}"
        )
    log.info(
        "CHECK 1 PASSED — step counter correctly restored: last checkpoint step=%d, load phase started at step=%d",
        last_ckpt_step,
        first_step,
    )

    # Check 2: loss continuity — save loss at last_ckpt_step ≈ resume loss at first_step
    save_val = save_losses.get(last_ckpt_step)
    resume_val = resume_losses.get(first_step)
    if save_val is None or resume_val is None:
        pytest.fail(
            f"Cannot compare losses: "
            f"save step {last_ckpt_step} loss={save_val}, "
            f"resume step {first_step} loss={resume_val}"
        )

    tol = ckpt_cfg.loss_rtol * max(abs(save_val), 1e-9)
    # Loss may decrease (training continues) but must not increase beyond tolerance.
    increase = resume_val - save_val
    log.info(
        "CHECK 2 — loss boundary: "
        "loss at checkpoint step %d (save)=%.6f, "
        "loss at step %d (load)=%.6f, increase=%.6f, allowed_increase=%.6f",
        last_ckpt_step,
        save_val,
        first_step,
        resume_val,
        increase,
        tol,
    )
    if increase > tol:
        pytest.fail(
            f"FAILED loss boundary check: "
            f"loss increased from save step {last_ckpt_step}={save_val:.6f} "
            f"to load step {first_step}={resume_val:.6f} "
            f"(increase={increase:.6f} exceeds tolerance {tol:.6f} [{ckpt_cfg.loss_rtol * 100:.0f}%])"
        )
    log.info(
        "CHECK 2 PASSED — loss did not increase beyond tolerance across checkpoint boundary: "
        "save step %d loss=%.6f, load step %d loss=%.6f, increase=%.6f within tol=%.6f",
        last_ckpt_step,
        save_val,
        first_step,
        resume_val,
        increase,
        tol,
    )

    avg_save_io = sum(e for _, e in save_io_times) / len(save_io_times) if save_io_times else None
    log.info(
        "test_checkpoint PASSED | "
        "last_ckpt_step=%d first_resume_step=%d "
        "save_loss=%.6f resume_loss=%.6f increase=%.6f | "
        "save_io_avg=%s load_io=%s",
        last_ckpt_step,
        first_step,
        save_val,
        resume_val,
        increase,
        f"{avg_save_io:.2f}s" if avg_save_io is not None else "n/a",
        f"{load_io_seconds:.2f}s" if load_io_seconds is not None else "n/a",
    )
    update_test_result()

    if not request.node.session.testsfailed:
        orch.exec(f"rm -rf {shlex.quote(ckpt_dir)}")
    else:
        log.info("checkpoint dir retained for debugging: %s", ckpt_dir)


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
    mt_obj = _make_training_job(
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
        train_res_dict[combo_key] = None
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

    try:
        conv = variant_config.convergence
        log_text = mt_obj._read_last_node_log()
        step_metrics = parse_step_metrics(log_text)
        steps_to_target, time_to_target = compute_convergence(step_metrics, [], conv.target_metric, conv.target_value)
        if steps_to_target is not None:
            train_res_dict[combo_key]["steps_to_target"] = [str(steps_to_target)]
        if time_to_target is not None:
            train_res_dict[combo_key]["time_to_target_seconds"] = [str(time_to_target)]
        log.info(
            "convergence: steps_to_target=%s time_to_target_seconds=%s",
            steps_to_target,
            time_to_target,
        )
    except Exception:
        pass

    update_test_result()


def test_metric(variant_config, micro_batch_size, global_batch_size, precision, train_res_dict, lifecycle, request):
    """Stage 4 (parametrized): compare each combo's metrics against thresholds."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    combo_key = request.node.callspec.id
    if not train_res_dict.get(combo_key):
        pytest.skip(f"no recorded results for combo '{combo_key}' (training did not run or failed)")

    actuals_raw = train_res_dict[combo_key]
    request.node.user_properties.append(("training_log_tail", actuals_raw.get("_log_tail", "")))
    actuals = {f"training.{k}": float(v[-1]) for k, v in actuals_raw.items() if v and not k.startswith("_")}

    if not variant_config.enforce_thresholds:
        log.info("enforce_thresholds=false; record-only for combo '%s'", combo_key)
        for metric, value in actuals.items():
            log.info("  RECORD  %s: actual=%s", metric, value)
        return

    cell = variant_config.cell_key(combo_key)
    thresholds = variant_config.thresholds.get(cell)
    if not thresholds:
        log.warning("no thresholds defined for cell '%s'; skipping threshold checks", cell)
        return

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
    if not train_res_dict.get(combo_key):
        pytest.skip(f"no recorded results for combo '{combo_key}' (training did not run or failed)")

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
