'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

PyTorch Training (LLM) Single-Node Validation Suite.

Tests performed (in order):
1. test_launch_container         - Launch the training container
2. test_verify_environment       - Verify torch/transformers/torchao and the GPU count and arch
3. test_prepare_data[sweep]      - Tokenize the corpus once per tokenizer, or reuse the cache
4. test_training[sweep]          - Train one sweep, then parse its metrics
5. test_ga_overhead_pairs        - Require a GA=1 partner for every GA>1 sweep
6. test_metric[sweep-metric]     - Validate each metric against its threshold
7. test_loss_curve[sweep]        - Render the loss curve and check it decreases
8. test_convergence[sweep]       - Steps/time to the configured loss target
9. test_print_results_table      - Console table + metric-results HTML
10. test_teardown                - Tear the container down

Metrics validated per sweep (namespace training.*): tokens_per_sec,
tokens_per_sec_per_gpu, tflops_per_sec_per_gpu, mfu_pct,
step_time_ms_{mean,p50,p95}, peak_memory_{allocated,reserved,used}_mb,
checkpoint_{save,load,io}_seconds, checkpoint_state_match,
checkpoint_loss_delta, gradient_accumulation_overhead_pct, loss_step_{100,500,1000,5000},
eval_loss, perplexity, energy_kwh, tokens_per_kwh, and the lm-eval
checkpoint benchmarks (mmlu_pro_pct, bbh_pct, math_lvl5_pct, gpqa_pct,
musr_pct, gsm8k_pct, hellaswag_pct).

This is the SINGLE-NODE variant: the config must set
``training.distributed: false`` and the cluster file must contain exactly one
node. Sweeps are parametrized in conftest.py from the config's
``enabled_sweep_list``; the shared stage logic lives in _common.py.

Example usage:
  cvs run pytorch_training_single --cluster_file <cluster>.json \
      --config_file <mi325x_pytorch_training_llama-3.1-8b_single_perf_config.json> \
      --html log_dir/<out>.html --self-contained-html --capture=tee-sys
'''

from cvs.tests.training.pytorch_training import _common


def test_launch_container(orch, lifecycle, request):
    """Launch and verify the PyTorch Training container is running."""
    return _common.launch_container(orch, lifecycle, request)


def test_verify_environment(orch, variant_config, lifecycle, request):
    """Verify torch, transformers, torchao, HIP, visible GPU count, and GPU architecture."""
    return _common.verify_environment(orch, variant_config, lifecycle, request)


def test_prepare_data(orch, variant_config, sweep_name, lifecycle, request):
    """Tokenize the corpus with this sweep's tokenizer, reusing a cached copy."""
    return _common.prepare_data(orch, variant_config, sweep_name, lifecycle, request)


def test_training(  # noqa: PLR0913
    orch, variant_config, sweep_name, training_results, loss_series, lifecycle, request
):
    """Run one training sweep and parse its metrics. A failure is isolated to
    this sweep's row so the remaining sweeps still run."""
    return _common.training(orch, variant_config, sweep_name, training_results, loss_series, lifecycle, request)


def test_ga_overhead_pairs(variant_config, training_results):
    """Fail if a GA>1 sweep has no GA=1 sweep at the same micro batch."""
    return _common.ga_overhead_pairs(variant_config, training_results)


def test_metric(  # noqa: PLR0913
    sweep_name, metric, variant_config, training_results, metric_rows, lifecycle, request
):
    """One row per (sweep, metric): assert the parsed value against the sweep's
    threshold cell and record PASS / FAIL / RECORD."""
    return _common.metric(sweep_name, metric, variant_config, training_results, metric_rows, lifecycle, request)


def test_loss_curve(  # noqa: PLR0913
    sweep_name, variant_config, training_results, loss_series, lifecycle, request
):
    """Render a per-sweep loss PNG and fail if the curve is not decreasing."""
    return _common.loss_curve(sweep_name, variant_config, training_results, loss_series, lifecycle, request)


def test_convergence(sweep_name, variant_config, training_results):
    """Assert the configured loss target was reached, recording the step and
    wall-clock time it took."""
    return _common.convergence(sweep_name, variant_config, training_results)


def test_print_results_table(variant_config, training_results, metric_rows, train_res_dict, loss_series, request):
    """Log the per-sweep result table, write metric-results HTML, and fill
    the Run Deck's flat ``train_res_dict``."""
    return _common.print_results_table(
        variant_config, training_results, metric_rows, train_res_dict, loss_series, request
    )


def test_teardown(orch, variant_config, lifecycle, request):
    """Tear the container down and verify it is gone."""
    return _common.teardown(orch, variant_config, lifecycle, request)
