'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Loss curve utilities for Megatron training log analysis.

parse_all_loss_points  — extract every (step, lm_loss) pair from a training log
sample_loss_curve      — downsample points by stride and milestone steps
evaluate_loss_decreasing — slope-based smooth-decrease check (least-squares)
'''

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def parse_all_loss_points(log_text: str) -> List[Dict]:
    """Extract every (step, lm_loss) pair from a full Megatron training log.

    Each Megatron log line has the form:
        iteration   <N>/  <total> | ... | lm loss: <value> | ...

    Scans all iteration lines and returns them as a list of
    ``{"step": int, "loss": float}`` dicts in log order. Only lines that
    contain both an iteration number and a numeric ``lm loss`` value are
    included.

    Args:
        log_text: Full training log text.

    Returns:
        List of ``{"step": int, "loss": float}`` dicts, one per parsed line.
    """
    results = []
    pattern = re.compile(
        r'iteration\s+(\d+)\s*/\s*\d+[^\n]*?\blm loss:\s*([0-9.eE+\-]+)',
        re.I,
    )
    for m in pattern.finditer(log_text):
        results.append({"step": int(m.group(1)), "loss": float(m.group(2))})
    return results


def sample_loss_curve(
    step_metrics: List[Dict],
    sample_every: int = 10,
    milestone_steps: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """Downsample per-step training loss for the loss curve check.

    Keeps a point when its step is a multiple of ``sample_every``, is one of
    the ``milestone_steps`` (e.g. 100/500/1k/5k), or is the first/last
    recorded step. The first/last inclusion keeps short runs from producing
    an empty curve.

    Args:
        step_metrics:    List of ``{"step": int, "loss": float}`` dicts as
                         returned by ``parse_all_loss_points``.
        sample_every:    Keep every Nth step (default 10).
        milestone_steps: Additional steps to always include.

    Returns:
        Ordered, de-duplicated list of ``(step, loss)`` tuples. Empty when
        ``step_metrics`` is empty or contains no numeric loss values.
    """
    milestones = set(milestone_steps or [])
    every = sample_every if sample_every and sample_every > 0 else 1

    loss_steps = [
        s for s in (step_metrics or [])
        if s.get("step") is not None and isinstance(s.get("loss"), (int, float))
    ]
    if not loss_steps:
        return []

    first_step = loss_steps[0]["step"]
    last_step = loss_steps[-1]["step"]

    picked: Dict[int, float] = {}
    for s in loss_steps:
        step = s["step"]
        if step % every == 0 or step in milestones or step in (first_step, last_step):
            picked[step] = s["loss"]

    return [(step, picked[step]) for step in sorted(picked)]


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
