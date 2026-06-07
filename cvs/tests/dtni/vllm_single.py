"""Customer-facing pytest entrypoint for the DTNI vllm_single framework.

Drives one vllm single-node workload (specified by --config_file, with
sibling threshold.json) and emits one PASS/FAIL pytest node per threshold
metric. Run via the standard CVS CLI:

    cvs run vllm_single \
        --cluster_file=cluster.json \
        --config_file=cvs/input/dtni/vllm_single/<model>/<variant>/config.json \
        --html=report.html

Phase failures (launch/await/parse) skip every per-metric node with the
underlying error; threshold failures fail their node with actual vs
expected. The parametrize hook and `workload_outcome` fixture live in
this directory's conftest.py and are shared with future framework tests.
"""

from __future__ import annotations

import pytest


def test_threshold(metric: str, workload_outcome):
    """One node per threshold metric. Shows up as `test_threshold[<metric>]`."""
    if metric.startswith("__"):
        pytest.skip(f"workload not collectible: {metric}")

    result = workload_outcome.job_result
    if result.failed_phase and result.failed_phase != "verify":
        pytest.skip(
            f"phase {result.failed_phase!r} failed before verify: {result.message}"
        )

    verdict = next((v for v in result.verdicts if v["metric"] == metric), None)
    if verdict is None:
        pytest.fail(f"threshold {metric!r} defined but no verdict produced")

    assert verdict["passed"], (
        f"{metric}: actual={verdict['actual']} threshold={verdict['threshold']} "
        f"kind={verdict['kind']}"
        + (f" — {verdict['note']}" if verdict.get("note") else "")
    )
