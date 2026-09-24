'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Loss curve utilities for Megatron training log analysis.

parse_all_loss_points  — extract every (step, lm_loss) pair from a training log
sample_loss_curve      — downsample points by stride and milestone steps
evaluate_loss_decreasing — slope-based smooth-decrease check (least-squares)
'''

from cvs.lib.training.megatron.utils.iteration_metrics import (  # noqa: F401
    parse_all_loss_points,
    sample_loss_curve,
)


def evaluate_loss_decreasing(points, max_slope=0.0):
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
