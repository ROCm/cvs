"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Logistics (PR-Z tier 1): the launch phase ran and did not fail.
"""

from __future__ import annotations


def _phase(manifest, name):
    return next((p for p in manifest.phases if p.phase == name), None)


def test_container_up(workload_run):
    launch = _phase(workload_run, "launch")
    assert launch is not None, "launch phase missing from manifest"
    assert launch.status != "failed", "launch phase failed"
