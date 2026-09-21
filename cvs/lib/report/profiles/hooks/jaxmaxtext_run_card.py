'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Run-card hook for the JAX MaxText training Run Deck (single + distributed stems).
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row

_MISSING = "\u2014"


def jaxmaxtext_run_card_display(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    """Training run-card rows: model, GPU, topology, sweeps, steps, thresholds."""
    if variant is None:
        return provenance_link_rows(provenance)

    training = getattr(variant, "training", None)
    distributed = bool(getattr(training, "distributed", False))
    sweeps = variant.enabled_sweeps() if hasattr(variant, "enabled_sweeps") else []

    rows: List[Tuple[str, str, bool]] = [
        ("Model", getattr(getattr(variant, "model", None), "id", _MISSING) or _MISSING, False),
        ("GPU", getattr(variant, "gpu_arch", _MISSING) or _MISSING, False),
        ("Mode", "distributed" if distributed else "single-node", False),
        ("Framework", getattr(variant, "framework", "jaxmaxtext"), False),
        ("Sweeps", str(len(sweeps)), False),
    ]

    steps = getattr(training, "steps", None)
    if steps is not None:
        rows.append(("Steps", str(steps), False))

    rows.append(thresholds_run_card_row(variant))
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["jaxmaxtext_run_card_display"]
