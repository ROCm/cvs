"""Run-deck preset for the PyTorch Vision training suite."""

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.report.presets.builder import (
    make_inference_report_config,
    provenance_link_rows,
    thresholds_run_card_row,
)
from cvs.lib.report.types import ReportChartSeries
from cvs.lib.training.pytorch_vision.utils.metrics import (
    METRIC_TIER_ORDER,
    METRIC_UNITS,
    RESULTS_COLUMNS,
    tier_metric_specs,
)


def _run_card(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    enabled_sweeps = variant.training.enabled_sweeps()
    sweep = enabled_sweeps[0]
    rows: List[Tuple[str, str, bool]] = [
        ("Workload", "W1", False),
        ("Model", sweep.model, False),
        ("GPU", variant.gpu_arch, False),
        ("Topology", f"1 node \u00d7 {variant.training.gpus_per_node} GPUs", False),
        ("Distributed", "DDP", False),
        ("Phase", getattr(variant.training, "phase", "performance"), False),
        ("Run mode", getattr(variant.training, "run_mode", "perf"), False),
        ("Precision", sweep.precision, False),
        (
            "Input",
            (
                f"rocAL GPU ImageNet-1k \u00b7 3\u00d7{sweep.image_size}\u00d7{sweep.image_size}"
                if sweep.data_mode == "rocal"
                else f"synthetic 3\u00d7{sweep.image_size}\u00d7{sweep.image_size}"
            ),
            False,
        ),
        ("rocAL device", getattr(sweep, "rocal_device", "gpu"), False),
        ("Augmentation", getattr(sweep, "augmentation", "standard"), False),
        ("Training FLOPs/image", f"{sweep.training_flops_per_image / 1e9:.1f} GFLOP (provisional)", False),
        ("Peak BF16/GPU", f"{variant.training.peak_tflops_per_gpu:.1f} TFLOPS (provisional)", False),
        ("Checkpoint", "exact load + tolerance-gated resumed step", False),
        ("Sweeps", ", ".join(item.label for item in enabled_sweeps), False),
        ("Image", variant.container.image, False),
        (
            "Result artifacts",
            f"{getattr(getattr(variant, 'paths', None), 'log_dir', '')}/pytorch_vision",
            True,
        ),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


def _cell_nodeid_token(key: tuple) -> str:
    return f"[{key[2]}-MBS{key[5]}-{key[4]}"


def _cell_id(variant: Any, key: tuple) -> str:
    model, _gpu, workload, image_size, ga_label, batch_size = key
    matches = [
        sweep.name
        for sweep in variant.training.enabled_sweeps()
        if sweep.model == model
        and sweep.precision in str(workload)
        and (sweep.data_mode == "rocal") == ("ROCAL" in str(workload))
        and (sweep.data_mode != "rocal" or getattr(sweep, "rocal_device", "gpu").upper() in str(workload))
        and (sweep.data_mode != "rocal" or getattr(sweep, "augmentation", "standard").upper() in str(workload))
        and sweep.image_size == int(image_size)
        and sweep.batch_size == int(batch_size)
        and f"GA{sweep.gradient_accumulation_steps}" == ga_label
    ]
    if len(matches) != 1:
        raise ValueError(f"run-deck cell does not identify exactly one sweep: key={key}, matches={matches}")
    return matches[0]


PYTORCH_VISION_TRAINING_REPORT_CONFIG = make_inference_report_config(
    suite_id="pytorch_vision_training",
    report_basename="pytorch_vision_training_run_deck",
    title="PyTorch Vision W1 Run Deck",
    subtitle="ResNet-50 \u00b7 single-node 8-GPU DDP training performance",
    footer="CVS pytorch_vision_training \u00b7 render-only \u00b7 gates remain owned by pytest",
    link_name="PyTorch Vision W1 Run Deck",
    results_columns=RESULTS_COLUMNS,
    metric_units=METRIC_UNITS,
    tier_metric_specs=tier_metric_specs,
    metric_tier_order=METRIC_TIER_ORDER,
    metric_prefix="training.",
    cell_highlights=(
        ("images_per_sec", "Images/s"),
        ("images_per_sec_per_gpu", "Images/s/GPU"),
        ("tflops_per_sec_per_gpu", "TFLOPS/s/GPU (provisional)"),
        ("mfu_pct", "MFU % (provisional)"),
        ("step_time_ms_p95", "P95 step (ms)"),
        ("device_memory_used_mb_observed", "Observed device used (MB)"),
        ("checkpoint_state_match", "Checkpoint state match"),
        ("gradient_accumulation_overhead_pct", "GA overhead (%)"),
    ),
    chart_series=(
        ReportChartSeries("images_per_sec", "Training throughput", "images/s"),
        ReportChartSeries("tflops_per_sec_per_gpu", "Provisional compute", "TFLOPS/s/GPU"),
        ReportChartSeries("mfu_pct", "Provisional MFU", "%"),
        ReportChartSeries("step_time_ms_mean", "Mean step time", "ms", invert=True),
        ReportChartSeries("step_time_ms_p95", "P95 step time", "ms", invert=True),
        ReportChartSeries("device_memory_used_mb_observed", "Observed device-used memory", "MB", invert=True),
        ReportChartSeries("continuous_peak_device_memory_mb", "Sampled peak device memory", "MB", invert=True),
        ReportChartSeries("top1_accuracy_pct", "Top-1 accuracy", "%"),
        ReportChartSeries("top5_accuracy_pct", "Top-5 accuracy", "%"),
        ReportChartSeries("eval_loss", "Evaluation loss", "-", invert=True),
        ReportChartSeries("convergence_time_seconds", "Time to convergence", "s", invert=True),
        ReportChartSeries("gpu_compute_util_pct", "GPU compute utilization", "%"),
        ReportChartSeries("gpu_bandwidth_util_pct", "GPU bandwidth utilization", "%"),
        ReportChartSeries("energy_kwh", "Training energy", "kWh", invert=True),
        ReportChartSeries("images_per_kwh", "Energy efficiency", "images/kWh"),
        ReportChartSeries("gradient_accumulation_overhead_pct", "GA overhead", "%", invert=True),
    ),
    sweep_throughput_metric="training.images_per_sec",
    sweep_ttft_metric="training.step_time_ms_p95",
    headline_metric="training.images_per_sec",
    inference_test_substring="test_training",
    row_card_extras=False,
    session_lifecycle_labels=(
        "container_launch",
        "environment_verification",
        "training",
        "teardown",
    ),
    cell_lifecycle_labels=("training",),
    run_card_display_builder=_run_card,
    shape_axis_labels=("Workload", "Resolution"),
    sweep_axis_label="MBS/GPU",
    sweep_axis_name="Microbatch-size",
    headline_unit="images/s",
    sweep_latency_label="P95 step",
    cell_nodeid_token_builder=_cell_nodeid_token,
    cell_id_builder=_cell_id,
    interactive_viewer=False,
)
