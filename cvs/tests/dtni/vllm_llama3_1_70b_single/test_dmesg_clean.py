"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Logistics (PR-Z tier 1): no fatal failure pattern (OOM, ECC, kernel
panic, ...) matched any captured log. Vacuous unless the scanner is
wired (B5) -- PR-X's conftest threads ``scanner=FailurePatternScanner()``
into ``Job``.
"""

from __future__ import annotations


def test_dmesg_clean(workload_run):
    fatal = [m for m in workload_run.verdicts.pattern_matches if m.severity == "fatal"]
    assert not fatal, f"fatal patterns matched: {[m.id for m in fatal]}"
