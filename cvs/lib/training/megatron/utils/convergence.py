'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Convergence utilities for Megatron training log analysis.

parse_step_metrics  — extract per-step (step, elapsed_time_ms, loss) from a training log
compute_convergence — steps and wall-clock to reach a target loss
'''

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def parse_step_metrics(log_text: str) -> List[Dict]:
    """Extract per-step metrics from a full Megatron training log.

    Each Megatron log line has the form:
        iteration   <N>/  <total> | ... | elapsed time per iteration (ms): <X>[/<avg>] | ... | lm loss: <value> | ...

    elapsed_time_ms captures the instantaneous per-step time (X, not the running
    average). For warmup iterations the value is a single number; for steady-state
    iterations it is X/Y — the regex captures X in both cases.

    Args:
        log_text: Full training log text.

    Returns:
        List of ``{"step": int, "elapsed_time_ms": float, "loss": float}`` dicts
        in log order. Lines missing either elapsed time or lm loss are skipped.
    """
    results = []
    pattern = re.compile(
        r'iteration\s+(\d+)\s*/\s*\d+[^\n]*?'
        r'elapsed time per iteration \(ms\):\s*([0-9.]+)(?:/[0-9.]+)?[^\n]*?'
        r'\blm loss:\s*([0-9.eE+\-]+)',
        re.I,
    )
    for m in pattern.finditer(log_text):
        results.append({
            "step": int(m.group(1)),
            "elapsed_time_ms": float(m.group(2)),
            "loss": float(m.group(3)),
        })
    return results


def compute_convergence(step_metrics, eval_metrics, target_metric="auto", target_value=0.0):
    """Steps and wall-clock to reach a target loss.

    target_metric:
      - "eval_loss":  converge on validation loss (eval_metrics points)
      - "train_loss": converge on per-step training loss (step_metrics)
      - "auto":       use eval_metrics when present, else training loss

    A `target_value <= 0` disables the metric and returns (None, None) so an
    uncalibrated target never gates or misleads.

    Returns (steps_to_target, time_to_target_seconds), where the time is the
    cumulative sum of per-step elapsed_time_ms (converted to seconds) up to and
    including the target step. Returns (None, None) when disabled or the target
    is never reached. Never raises.
    """
    if not target_value or target_value <= 0:
        return (None, None)

    use_eval = target_metric == "eval_loss" or (target_metric == "auto" and bool(eval_metrics))

    # Cumulative training seconds indexed by step, derived from elapsed_time_ms.
    cum = {}
    running = 0.0
    for s in step_metrics or []:
        elapsed_ms = s.get("elapsed_time_ms")
        if isinstance(elapsed_ms, (int, float)):
            running += elapsed_ms / 1000.0
        step = s.get("step")
        if step is not None:
            cum[step] = running

    target_step = None
    if use_eval:
        for e in eval_metrics or []:
            loss = e.get("eval_loss")
            if loss is not None and e.get("step") is not None and loss <= target_value:
                target_step = e.get("step")
                break
    else:
        for s in step_metrics or []:
            loss = s.get("loss")
            if loss is not None and s.get("step") is not None and loss <= target_value:
                target_step = s.get("step")
                break

    if target_step is None:
        return (None, None)

    time_to_target = cum.get(target_step)
    if time_to_target is None and cum:
        prior = [t for st, t in cum.items() if st <= target_step]
        time_to_target = max(prior) if prior else None

    return (target_step, time_to_target)
 