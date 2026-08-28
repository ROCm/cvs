'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

JAX MaxText Single-Node Training Validation Suite.

Tests performed (in order):
1. test_launch_container      - Launch the training container
2. test_setup_tokenizer       - Download the HuggingFace tokenizer (skipped when
                                dataset_type=synthetic)
3. test_smoke                 - Small fixed run: model loads + trains 10 steps
                                without error_patterns (no metric checks)
4. test_training_run[sweep]   - Run MaxText training per sweep (e.g. BF16, FP8)
5. test_metric[sweep-metric]  - Validate each metric against its threshold
6. test_loss_curve[sweep]     - Render the loss curve and check it decreases
7. test_checkpoint_resume     - Opt-in: checkpoint save+resume correctness +
                                checkpoint I/O timing (skipped unless enabled)
8. test_print_results_table   - Console tables + metric-results HTML + summary
9. test_teardown              - Tear the container down

Metrics validated per sweep (namespace training.*): tflops_per_sec_per_gpu,
tokens_per_sec_per_gpu, tokens_per_sec_total, scaling_efficiency_pct,
step_time_{seconds,mean_ms,p50_ms,p95_ms}, final_loss, loss_decreased,
eval_loss, steps_to_target, time_to_target_seconds.

This is the SINGLE-NODE variant - no RDMA setup stage. Sweeps are parametrized
in conftest.py from the config's `enabled_sweep_list`; the shared stage logic
lives in _common.py.

Example usage:
  cvs run jaxmaxtext_single --cluster_file <cluster>.json \
      --config_file <mi300x_..._single.json> --html log_dir/<out>.html --self-contained-html \
      --log-file=log_dir/log.txt
'''

from cvs.tests.training.jaxmaxtext import _common

log = _common.log


def test_launch_container(orch, variant_config, lifecycle, request):
    """Launch and verify the MaxText training container is running."""
    log.info('Starting Testcase: launch JAX containers')
    return _common.launch_container(orch, variant_config, lifecycle, request)


def test_setup_tokenizer(orch, variant_config, hf_token, lifecycle, request):
    """Download the HuggingFace tokenizer for the model into the models dir."""
    log.info('Starting Testcase: setup tokenizer')
    return _common.setup_tokenizer(orch, variant_config, hf_token, lifecycle, request)


def test_smoke(orch, variant_config, hf_token, lifecycle, request):
    """Smoke test: model loads and trains 10 steps (small fixed batch/seqlen,
    BF16) without any error_pattern firing. No metric/threshold verification."""
    log.info('Starting Testcase: smoke (model loads + runs 10 steps)')
    return _common.smoke(orch, variant_config, hf_token, lifecycle, request)


def test_training_run(orch, variant_config, hf_token, sweep_name, training_res_dict, lifecycle, request):
    """Run one full MaxText training for this sweep, then parse its metrics.

    Parametrized per sweep in conftest.py (e.g. BF16, FP8). A failure is isolated
    to this sweep's row so other sweeps still run.
    """
    log.info('Starting Testcase: training run [%s]', sweep_name)
    return _common.training_run(orch, variant_config, hf_token, sweep_name, training_res_dict, lifecycle, request)


def test_metric(sweep_name, metric, training_res_dict, variant_config, lifecycle, request):
    """One row per (sweep, metric): assert the parsed value against the sweep's
    threshold cell and record PASS / FAIL / N/A / RECORD."""
    log.info('Starting Testcase: metric [%s - %s]', sweep_name, metric)
    return _common.metric(sweep_name, metric, training_res_dict, variant_config, lifecycle, request)


def test_loss_curve(sweep_name, training_res_dict, variant_config, lifecycle, request):
    """Sample the training loss, render a per-sweep PNG, and fail if the curve is
    not decreasing (least-squares slope check)."""
    log.info('Starting Testcase: loss curve [%s]', sweep_name)
    return _common.loss_curve(sweep_name, training_res_dict, variant_config, lifecycle, request)


def test_checkpoint_resume(orch, variant_config, hf_token, training_res_dict, lifecycle, request):
    """Opt-in (training.checkpoint_resume.enabled): save a checkpoint, resume from
    it, and verify the restart restores step + loss (state), plus benchmark
    checkpoint save/load I/O time. Skipped when disabled. No effect on the sweeps."""
    log.info('Starting Testcase: checkpoint save+resume + I/O timing')
    return _common.checkpoint_resume(orch, variant_config, hf_token, training_res_dict, lifecycle, request)


def test_print_results_table(training_res_dict, request):
    """Log per-sweep result tables, write the consolidated metric-results HTML,
    and record the aggregated failure summary for the pytest final summary."""
    log.info('Starting Testcase: print results table')
    return _common.print_results_table(training_res_dict, request)


def test_teardown(orch, lifecycle, request):
    """Tear the container down and verify it is gone."""
    log.info('Starting Testcase: teardown JAX containers')
    return _common.teardown(orch, lifecycle, request)
