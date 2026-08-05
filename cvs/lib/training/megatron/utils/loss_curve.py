'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Loss curve utilities for Megatron training log analysis.

parse_loss_at_steps  — extract lm_loss at specific iteration numbers
check_loss_decreasing — assert loss decreases monotonically across checkpoints
'''

from __future__ import annotations

import re
from typing import Dict, List


def parse_loss_at_steps(log_text: str, steps: List[int]) -> Dict[int, float]:
    '''Extract lm_loss at specific training iteration numbers from full log text.

    Each Megatron log line has the form:
        iteration   <N>/  <total> | ... | lm loss: <value> | ...

    For each step in `steps`, searches for its iteration line and extracts the
    lm loss value. Steps not present in the log (run did not reach them) are
    omitted from the returned dict so callers can detect under-run configs.

    Args:
        log_text: Full training log text.
        steps:    Iteration numbers to sample, e.g. [100, 500, 1000, 5000].

    Returns:
        {step: lm_loss} for every step whose iteration line was found.
    '''
    results = {}
    for step in steps:
        # Iteration lines have variable whitespace padding around the numbers.
        # Match `lm loss:` specifically — not `load_balancing_loss:` which
        # appears on the same line immediately after.
        m = re.search(
            rf'iteration\s+{step}\s*/\s*\d+[^\n]*?\blm loss:\s*([0-9.eE+\-]+)',
            log_text,
            re.I,
        )
        if m:
            results[step] = float(m.group(1))
    return results


def check_loss_decreasing(losses: Dict[int, float]) -> List[str]:
    '''Validate that lm_loss decreases at every consecutive checkpoint pair.

    Checks two conditions per interval [prev_step → curr_step]:
      1. Monotonic decrease  — loss must be strictly lower (hard violation).
      2. Smoothness          — decrease of < 1 % flags a possible training stall
                               (returned as a warning string prefixed "WARN:").

    Args:
        losses: {step: lm_loss} dict as returned by parse_loss_at_steps.
                Must contain at least 2 entries; caller should skip if fewer.

    Returns:
        List of violation / warning strings. Empty list means the curve is
        healthy. Callers treat "WARN:"-prefixed entries as warnings and all
        others as hard failures.
    '''
    messages = []
    sorted_steps = sorted(losses)
    for i in range(1, len(sorted_steps)):
        prev_step = sorted_steps[i - 1]
        curr_step = sorted_steps[i]
        prev_loss = losses[prev_step]
        curr_loss = losses[curr_step]

        if curr_loss > prev_loss:
            messages.append(
                f'step {prev_step}→{curr_step}: loss increased '
                f'{prev_loss:.6f} → {curr_loss:.6f}'
            )
        elif prev_loss > 0 and (prev_loss - curr_loss) / prev_loss < 0.01:
            messages.append(
                f'WARN: step {prev_step}→{curr_step}: loss nearly flat '
                f'{prev_loss:.6f} → {curr_loss:.6f} '
                f'(<1% decrease — possible training stall)'
            )
    return messages
