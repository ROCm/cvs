"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Inference (PR-Z tier 3): the server reached HTTP 200 and the run did
not end in either of the "server unavailable" categories. The vLLM
adapter classifies a never-ready server as ``liveness_failure`` (via
the bounded poll in ``_wait_for_server_ready``) and a mid-run predicate
break as ``safety_violation``; both indicate the framework was not
responsive to clients, so both must fail this tier-3 gate.
"""

from __future__ import annotations


def test_framework_responsive(workload_run):
    bad = {"safety_violation", "liveness_failure"}
    assert workload_run.verdicts.failure_category not in bad, (
        f"framework not responsive: failure_category={workload_run.verdicts.failure_category}"
    )
