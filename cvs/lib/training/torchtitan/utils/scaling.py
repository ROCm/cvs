'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Scaling efficiency utilities for TorchTitan distributed training analysis.
'''

from __future__ import annotations

from typing import Optional


def compute_scaling_efficiency(
    tokens_per_sec_total: Optional[float],
    num_nodes: Optional[int],
    baseline_tokens_per_sec_total: Optional[float],
    baseline_num_nodes: int = 1,
) -> Optional[float]:
    """Scaling efficiency % for a training run.

    efficiency % = throughput_N / ((N / ref_N) * throughput_ref) * 100

    where throughput_N is this run's total tokens/sec on `num_nodes` nodes and
    throughput_ref is the reference (typically 1-node) total tokens/sec measured
    on `baseline_num_nodes` nodes. 100% means perfectly linear scaling; lower
    means communication/straggler overhead is eating into the added nodes.

    Returns None (record-only) when any input is missing or non-positive so an
    uncalibrated baseline never produces a misleading number or a crash.
    """
    if not tokens_per_sec_total or not baseline_tokens_per_sec_total:
        return None
    if not num_nodes or not baseline_num_nodes:
        return None
    ideal = (num_nodes / baseline_num_nodes) * baseline_tokens_per_sec_total
    if ideal <= 0:
        return None
    return tokens_per_sec_total / ideal * 100.0
