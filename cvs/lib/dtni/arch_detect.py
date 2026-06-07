"""Detect GPU family on head node via rocm-smi.

Explicit allowlist; fail-loud on no match. Returned string is one of:
mi300x, mi325x, mi350x, mi355x, mi308x, mi210, mi250x, mi300a.
"""

from __future__ import annotations

import re

# Order matters: longer/more-specific first.
VALID_ARCHES = frozenset({
    "mi355x", "mi350x", "mi325x", "mi300x", "mi300a", "mi308x", "mi250x", "mi210",
})

_ARCH_PATTERNS = [
    ("mi355x", re.compile(r"\bMI355X\b", re.IGNORECASE)),
    ("mi350x", re.compile(r"\bMI350X\b", re.IGNORECASE)),
    ("mi325x", re.compile(r"\bMI325X\b", re.IGNORECASE)),
    ("mi300x", re.compile(r"\bMI300X\b", re.IGNORECASE)),
    ("mi300a", re.compile(r"\bMI300A\b", re.IGNORECASE)),
    ("mi308x", re.compile(r"\bMI308X\b", re.IGNORECASE)),
    ("mi250x", re.compile(r"\bMI250X\b", re.IGNORECASE)),
    ("mi210",  re.compile(r"\bMI210\b",  re.IGNORECASE)),
]


def detect_arch(rocm_smi_output: str) -> str:
    for arch, pat in _ARCH_PATTERNS:
        if pat.search(rocm_smi_output):
            return arch
    raise ValueError(
        "could not detect AMD GPU arch from rocm-smi --showproductname output; "
        f"first 300 chars:\n{rocm_smi_output[:300]}"
    )


def detect_arch_via(executor) -> str:
    """Run rocm-smi via executor; return arch family."""
    out = executor.exec("rocm-smi --showproductname 2>&1", timeout=30)
    return detect_arch(out)
