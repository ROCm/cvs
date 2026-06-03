"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Benchmarks (PR-Z tier 5): the configured p99 latency thresholds, if
any, passed. Explicitly skips (NOT silently passes) when the run did
not complete -- ``test_bench_completed`` is the canonical
overall-status gate; this tier-5 assertion would otherwise read green
on a broken run.
"""

from __future__ import annotations

import pytest


def test_p99_latency_threshold(workload_run):
    if workload_run.verdicts.overall_status != "complete":
        pytest.skip(
            f"run did not complete ({workload_run.verdicts.overall_status}); percentile thresholds not exercised"
        )
    pct = [v for v in workload_run.verdicts.threshold_verdicts if v.threshold_type == "percentile"]
    failed = [v for v in pct if not v.passed]
    assert not failed, [v.model_dump() for v in failed]
