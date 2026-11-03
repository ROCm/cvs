'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Loss curve utilities for TorchTitan training log analysis.

parse_all_loss_points  — extract every (step, loss) pair from a training log
sample_loss_curve      — downsample points by stride and milestone steps
evaluate_loss_decreasing — slope-based smooth-decrease check (least-squares)
'''

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from cvs.lib.training.torchtitan.utils.iteration_metrics import (
    parse_iteration_metrics,
    sample_metric_curve,
)


def parse_all_loss_points(log_text: str) -> List[Dict]:
    """Extract every step row that carries a loss from a TorchTitan training log.

    Delegates to ``parse_iteration_metrics``. Rows also include tps, tflops,
    grad_norm, and memory when those fields are on the same ``step:`` line.
    """
    return [row for row in parse_iteration_metrics(log_text) if isinstance(row.get("loss"), (int, float))]


def sample_loss_curve(
    step_metrics: List[Dict],
    sample_every: int = 10,
    milestone_steps: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """Downsample per-step training loss for the slope gate.

    A point is kept when its step is a multiple of ``sample_every``, is one of
    the ``milestone_steps``, or is the first or last step. Warmup steps stay
    in the series: ``evaluate_loss_decreasing`` fits these points, and dropping
    the first 10% would change pass/fail without a change to ``max_slope``.
    """
    return sample_metric_curve(step_metrics, "loss", sample_every, milestone_steps, warmup_frac=0)


def evaluate_loss_decreasing(
    points: List[Tuple[int, float]],
    max_slope: float = 0.0,
) -> Optional[Tuple[bool, float, str]]:
    """Decide whether a sampled loss curve trends downward using linear regression.

    Fits a least-squares line to ``points`` and treats the run as decreasing
    when the slope is below ``max_slope`` (default 0.0, i.e. strictly negative).
    Uses a dependency-free closed form:

        slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx²)

    Args:
        points:    Ordered list of ``(step, loss)`` tuples from
                   ``sample_loss_curve``.
        max_slope: Slope threshold; slope < max_slope is considered decreasing.

    Returns:
        ``(decreasing, slope, detail)`` or ``None`` when fewer than 2 points
        are present or all steps are identical (degenerate case). Never raises.
    """
    if not points or len(points) < 2:
        return None

    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)

    denom = n * sxx - sx * sx
    if denom == 0:
        return None

    slope = (n * sxy - sx * sy) / denom
    decreasing = slope < max_slope
    detail = (
        f"loss slope {slope:.6g}/step over {n} points "
        f"(first={points[0][1]:.4f}@step{points[0][0]}, "
        f"last={points[-1][1]:.4f}@step{points[-1][0]}); "
        f"{'decreasing' if decreasing else 'NOT decreasing'} "
        f"(threshold max_slope={max_slope})"
    )
    return (decreasing, slope, detail)
