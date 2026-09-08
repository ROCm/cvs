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
    sweep = variant.training.enabled_sweeps()[0]
    rows: List[Tuple[str, str, bool]] = [
        ("Workload", "W1", False),
        ("Model", sweep.model, False),
        ("GPU", variant.gpu_arch, False),
        ("Topology", f"1 node \u00d7 {variant.training.gpus_per_node} GPUs", False),
        ("Distributed", "DDP", False),
        ("Precision", sweep.precision, False),
        ("Input", f"synthetic 3\u00d7{sweep.image_size}\u00d7{sweep.image_size}", False),
        ("Image", variant.container.image, False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


def _cell_nodeid_token(key: tuple) -> str:
    return f"[{key[2]}"


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
        ("step_time_ms_p95", "P95 step (ms)"),
        ("peak_memory_allocated_mb", "Peak allocated (MB)"),
        ("peak_memory_reserved_mb", "Peak reserved (MB)"),
    ),
    chart_series=(
        ReportChartSeries("images_per_sec", "Training throughput", "images/s"),
        ReportChartSeries("step_time_ms_mean", "Mean step time", "ms", invert=True),
        ReportChartSeries("step_time_ms_p95", "P95 step time", "ms", invert=True),
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
    sweep_axis_label="BS/GPU",
    sweep_axis_name="Batch-size",
    headline_unit="images/s",
    sweep_latency_label="P95 step",
    cell_nodeid_token_builder=_cell_nodeid_token,
    interactive_viewer=False,
)
