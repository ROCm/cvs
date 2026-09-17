'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

PyTorch Vision Single-Node Training Validation Suite.

Tests performed (in order):
1. test_launch_container              - Launch the training container
2. test_verify_environment            - Verify torch/torchvision/GPU count and arch
3. test_real_data_smoke[sweep]        - rocAL smoke profiles only: short real-data run
4. test_training[sweep]               - Run one training sweep and parse its metrics
5. test_rocal_overhead_comparisons    - Require the rocAL CPU/GPU and augmentation baselines
6. test_metric[sweep-metric]          - Validate each metric against its threshold
7. test_loss_curve[sweep]             - Render the loss curve and check it decreases
8. test_convergence[sweep]            - Steps/time to the configured accuracy target
9. test_print_results_table           - Console tables + metric-results HTML
10. test_teardown                     - Tear the container down

Metrics validated per sweep (namespace training.*): images_per_sec,
images_per_sec_per_gpu, tflops_per_sec_per_gpu, mfu_pct,
step_time_ms_{mean,p50,p95}, peak_memory_{allocated,reserved,used}_mb,
data_loader_images_per_sec, checkpoint_{save,load}_seconds,
checkpoint_state_match, gradient_accumulation_overhead_pct,
augmentation_overhead_pct, rocal_cpu_overhead_pct, top1_accuracy_pct,
top5_accuracy_pct, eval_loss, loss_step_{100,500,1000,5000}, energy_kwh,
images_per_kwh.

This is the SINGLE-NODE variant - the config must set
``training.distributed: false`` and the cluster file must contain exactly one
node. Sweeps are parametrized in conftest.py from the config's
``enabled_sweep_list``; the shared stage logic lives in _common.py.

Example usage:
  cvs run pytorch_vision_single --cluster_file <cluster>.json \
      --config_file <mi325x_pytorch_vision_resnet50_single.json> \
      --html log_dir/<out>.html --self-contained-html --capture=tee-sys
'''

from cvs.tests.training.pytorch_vision import _common


def test_launch_container(orch, lifecycle, request):
    """Launch and verify the PyTorch Vision training container is running."""
    return _common.launch_container(orch, lifecycle, request)


def test_verify_environment(orch, variant_config, lifecycle, request):
    """Verify torch, torchvision, HIP, visible GPU count, and GPU architecture."""
    return _common.verify_environment(orch, variant_config, lifecycle, request)


def test_real_data_smoke(  # noqa: PLR0913
    orch, variant_config, sweep_name, training_results, loss_series, lifecycle, request
):
    """rocAL smoke profiles only: prove real ImageNet flows end to end before
    any longer profile is trusted. Skipped outside ``run_mode=smoke``."""
    return _common.real_data_smoke(orch, variant_config, sweep_name, training_results, loss_series, lifecycle, request)


def test_training(  # noqa: PLR0913
    orch, variant_config, sweep_name, training_results, loss_series, lifecycle, request
):
    """Run one full training sweep, then parse its metrics.

    Parametrized per sweep in conftest.py. A failure is isolated to this
    sweep's row so the remaining sweeps still run.
    """
    return _common.training(orch, variant_config, sweep_name, training_results, loss_series, lifecycle, request)


def test_rocal_overhead_comparisons(variant_config, training_results):
    """Fail if a heavy-augmentation or CPU-decode sweep has no matching
    GPU-standard baseline to compare against."""
    return _common.rocal_overhead_comparisons(variant_config, training_results)


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
    """Assert the configured accuracy target was reached, recording the step
    and wall-clock time it took."""
    return _common.convergence(sweep_name, variant_config, training_results)


def test_print_results_table(variant_config, training_results, metric_rows, request):
    """Log the per-sweep result table and write the consolidated
    metric-results HTML that every metric row links to."""
    return _common.print_results_table(variant_config, training_results, metric_rows, request)


def test_teardown(orch, lifecycle, request):
    """Tear the container down and verify it is gone."""
    return _common.teardown(orch, lifecycle, request)
