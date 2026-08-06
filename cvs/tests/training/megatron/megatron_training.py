'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
 
Unified Megatron training suite for both single-node and distributed runs.
Replaces megatron_single.py and megatron_distributed.py with one parametrized
suite. The topology is determined entirely by the config file:
  framework=megatron_single       -> single-node  (distributed_training=False)
  framework=megatron_distributed  -> multi-node   (distributed_training=True)
 
Lifecycle (each stage is a separate test):
  test_launch_container  — launch the container once for all sweep combos
  test_smoke             — fixed small cell: model loads and runs N steps without error
  test_training          — parametrized: one test per sweep combo; kills GPU
                           processes in finally so VRAM is free for the next combo
  test_metric            — parametrized: threshold check per combo via evaluate_all
  test_loss_curve        — parametrized: loss decreases smoothly at steps 100/500/1k/5k
  test_teardown          — tear down the container once after all combos
'''
 
import json
import os
import time
 
import pytest
 
from cvs.lib import globals
from cvs.lib.training.factory import create_training_job
from cvs.lib.training.megatron.utils.loss_curve import parse_loss_at_steps, check_loss_decreasing
from cvs.lib.training.megatron.utils.scaling import compute_scaling_efficiency
from cvs.lib.utils.verdict import evaluate_all
from cvs.lib.utils_lib import update_test_result
 
log = globals.log

# Loss curve checkpoints: iteration numbers at which lm_loss is sampled.
# The run must reach at least the second entry for the test to execute.
_LOSS_CURVE_STEPS = [100, 500, 1000, 5000]

# Smoke cell: smallest fixed parameters that confirm the model loads and trains.
# Concurrency/num_prompts are irrelevant — the smoke never runs the benchmark
# sweep. Runs once before the sweep so a broken model or NCCL setup fails fast.
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
    """Stage 0: launch the container once for all sweep combos.
 
    GPU memory is freed between combos via stop_training_processes() so a fresh
    container launch per combo is not needed.
    """
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
    """Stage 1: download the tokenizer model once if the model family requires it.

    No-op for llama and qwen — their training scripts accept the HF repo ID
    directly and no local file is needed. For deepseek and mixtral, downloads
    tokenizer.model into data_cache_dir inside the container and verifies the
    file is present before any training starts.

    Stores the resolved local path in lifecycle.tokenizer_path so test_smoke
    and test_training can reuse it without re-downloading. A failure here tells
    the user exactly where the problem is (token missing, network unreachable,
    disk full) rather than surfacing as a cryptic training script error.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    distributed = "distributed" in variant_config.framework
    mt_obj = create_training_job(
        orch,
        variant_config,
        hf_token=hf_token,
        micro_batch_size="1",
        global_batch_size="1",
        precision="BF16",
        distributed_training=distributed,
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

    Independent of the sweep. Brings up a short-lived training job with a fixed
    small cell (MBS/GBS above) so a broken model, missing script, or NCCL failure
    is caught fast before burning sweep-scale GPU time. Always stops GPU processes
    in finally (success or failure) so test_training's first combo starts on a
    clean node.

    Passes if training reaches iteration _SMOKE_ITERS/_SMOKE_ITERS without error.
    No metric assertions — completion without error is the only requirement.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    distributed = "distributed" in variant_config.framework
    globals.error_list = []

    mt_obj = create_training_job(
        orch,
        variant_config,
        hf_token=hf_token,
        micro_batch_size=_SMOKE_MBS,
        global_batch_size=_SMOKE_GBS,
        precision=_SMOKE_PRECISION,
        distributed_training=distributed,
        tune_model_params=False,
        run_label="smoke",
    )
    # Override iterations for the smoke cell without mutating variant_config.
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

    update_test_result()
    lifecycle.record(request.node.nodeid, "smoke", time.monotonic() - t)
    log.info("smoke PASSED | iters=%s", _SMOKE_ITERS)


def test_training(orch, variant_config, hf_token, micro_batch_size, global_batch_size, precision, train_res_dict, lifecycle, request):
    """Stage 1 (parametrized): run one sweep combo inside the shared container.
 
    distributed_training is derived from the config framework field:
      megatron_distributed -> True
      megatron_single      -> False
 
    stop_training_processes() runs in a finally block after every combo so GPU
    memory is released before the next combo starts — preventing HIP OOM errors
    when combos run back-to-back in the same container.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
 
    distributed_training = 'distributed' in variant_config.framework
 
    nodeid = request.node.nodeid
    combo_key = request.node.callspec.id
    globals.error_list = []
    mt_obj = create_training_job(
        orch,
        variant_config,
        hf_token=hf_token,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        precision=precision,
        distributed_training=distributed_training,
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
    """Stage 2 (parametrized): compare each combo's metrics against thresholds.
 
    No container interaction — reads results saved by test_training.
    Skips if training did not record results or enforce_thresholds is false.
    Uses variant_config.cell_key(combo_key) to look up the threshold row and
    evaluate_all() to assert every metric in that row.
    """
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
    actuals = {
        f"training.{k}": float(v[-1])
        for k, v in actuals_raw.items()
        if v and not k.startswith("_")
    }
    evaluate_all(actuals, thresholds)
 
 
def test_loss_curve(orch, variant_config, micro_batch_size, global_batch_size, precision, train_res_dict, lifecycle, request):
    """Parametrized: verify lm_loss decreases smoothly at steps 100, 500, 1k, 5k.

    Reads the full training log written by test_training (no container interaction
    needed — the log file already exists on disk). Skips gracefully when the run
    did not produce enough iterations to reach at least two checkpoints.

    Uses parse_loss_at_steps() to extract lm_loss per checkpoint and
    check_loss_decreasing() to validate:
      - Monotonic decrease  : loss at each checkpoint < loss at previous (hard fail)
      - Smoothness          : < 1% drop between any two checkpoints logged as warning
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    combo_key = request.node.callspec.id
    if combo_key not in train_res_dict:
        pytest.skip(f"no recorded results for combo '{combo_key}' (training did not run)")

    combo_log_dir = train_res_dict[combo_key].get("_combo_log_dir")
    if not combo_log_dir:
        pytest.skip(f"no log dir recorded for combo '{combo_key}'")

    n = len(orch.hosts)
    out_dict = orch.exec(f'cat {combo_log_dir}/out-node{n - 1}/training.log')
    log_text = list(out_dict.values())[-1] or ""

    losses = parse_loss_at_steps(log_text, _LOSS_CURVE_STEPS)

    if len(losses) < 2:
        pytest.skip(
            f"fewer than 2 loss checkpoints found in log "
            f"(steps checked: {_LOSS_CURVE_STEPS}) — "
            f"training needs at least {_LOSS_CURVE_STEPS[1]} iterations for this test"
        )

    log.info("loss curve for combo '%s': %s", combo_key, losses)
    request.node.user_properties.append(("metric_value", losses.get(max(losses))))
    request.node.user_properties.append(("metric_unit", "lm_loss"))

    messages = check_loss_decreasing(losses)
    warnings = [m for m in messages if m.startswith("WARN:")]
    failures = [m for m in messages if not m.startswith("WARN:")]

    for w in warnings:
        log.warning("loss curve '%s': %s", combo_key, w)

    if failures:
        pytest.fail(
            f"loss not smoothly decreasing for combo '{combo_key}':\n"
            + "\n".join(failures)
            + f"\nfull curve: {losses}"
        )


def test_teardown(orch, lifecycle, request):
    """Stage 3: tear down the container once after all combos have run."""
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        log.error("container %s still running after teardown_containers()", name)
    else:
        lifecycle.torn_down = True