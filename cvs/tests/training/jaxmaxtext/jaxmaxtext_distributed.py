'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

JAX MaxText Distributed (multi-node) Training Validation Suite.

Tests performed (in order):
1. test_launch_container      - Launch the training container on all nodes
2. test_setup_rdma            - Copy the RDMA lib into the container (thor2 NIC)
                                and verify ibv_devinfo
3. test_setup_tokenizer       - Download the HuggingFace tokenizer
4. test_training_run[sweep]   - Run MaxText training per sweep (e.g. BF16, FP8)
5. test_metric[sweep-metric]  - Validate each metric against its threshold
6. test_loss_curve[sweep]     - Render the loss curve and check it decreases
7. test_print_results_table   - Console tables + metric-results HTML + summary
8. test_teardown              - Tear the container down

Metrics validated per sweep (namespace training.*): tflops_per_sec_per_gpu,
tokens_per_sec_per_gpu, tokens_per_sec_total, scaling_efficiency_pct,
step_time_{seconds,mean_ms,p50_ms,p95_ms}, final_loss, loss_decreased,
eval_loss, steps_to_target, time_to_target_seconds.

This is the DISTRIBUTED variant - it adds the RDMA setup stage. Sweeps are
parametrized in conftest.py from the config's `enabled_sweep_list`; the shared
stage logic lives in _common.py. The JAX coordinator is the first cluster node
when `jax_distributed.coordinator_ip` is `auto`.

Example usage:
  cvs run jaxmaxtext_distributed --cluster_file <cluster>.json \
      --config_file <mi325x_..._distributed.json> --html log_dir/<out>.html --self-contained-html \
      --log-file=log_dir/log.txt
'''

from cvs.tests.training.jaxmaxtext import _common


def test_launch_container(orch, variant_config, lifecycle, request):
    """Launch and verify the MaxText training container is running."""
    return _common.launch_container(orch, variant_config, lifecycle, request)


def test_setup_rdma(orch, variant_config, hf_token, lifecycle, request):
    """Distributed-only: copy the host RDMA library into the container (thor2
    NIC workaround) and verify ibv_devinfo reports the expected HCA."""
    return _common.setup_rdma(orch, variant_config, hf_token, lifecycle, request)


def test_setup_tokenizer(orch, variant_config, hf_token, lifecycle, request):
    """Download the HuggingFace tokenizer for the model into the models dir."""
    return _common.setup_tokenizer(orch, variant_config, hf_token, lifecycle, request)


def test_training_run(orch, variant_config, hf_token, sweep_name, training_res_dict, lifecycle, request):
    """Run one full MaxText training for this sweep, then parse its metrics.

    Parametrized per sweep in conftest.py (e.g. BF16, FP8). A failure is isolated
    to this sweep's row so other sweeps still run.
    """
    return _common.training_run(orch, variant_config, hf_token, sweep_name, training_res_dict, lifecycle, request)


def test_metric(sweep_name, metric, training_res_dict, variant_config, lifecycle, request):
    """One row per (sweep, metric): assert the parsed value against the sweep's
    threshold cell and record PASS / FAIL / N/A / RECORD."""
    return _common.metric(sweep_name, metric, training_res_dict, variant_config, lifecycle, request)


def test_loss_curve(sweep_name, training_res_dict, variant_config, lifecycle, request):
    """Sample the training loss, render a per-sweep PNG, and fail if the curve is
    not decreasing (least-squares slope check)."""
    return _common.loss_curve(sweep_name, training_res_dict, variant_config, lifecycle, request)


def test_print_results_table(training_res_dict, request):
    """Log per-sweep result tables, write the consolidated metric-results HTML,
    and record the aggregated failure summary for the pytest final summary."""
    return _common.print_results_table(training_res_dict, request)


def test_teardown(orch, lifecycle, request):
    """Tear the container down and verify it is gone."""
    return _common.teardown(orch, lifecycle, request)
