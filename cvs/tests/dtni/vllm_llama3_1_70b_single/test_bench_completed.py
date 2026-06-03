"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Benchmarks (PR-Z tier 5): the benchmark client finished and the adapter
parsed result scalars.
"""

from __future__ import annotations


def test_bench_completed(workload_run):
    assert workload_run.verdicts.overall_status == "complete", (
        f"run did not complete: status={workload_run.verdicts.overall_status} "
        f"category={workload_run.verdicts.failure_category}"
    )
    assert workload_run.verdicts.scalars, "no result scalars parsed from bench_result.json"
    # ``elapsed_s`` is the canonical proof the bench ran end-to-end.
    assert "elapsed_s" in workload_run.verdicts.scalars, (
        "elapsed_s missing -- bench may have crashed before emitting --save-result"
    )
