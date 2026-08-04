'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX ATOM run-card hook for JSON deck profiles.
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def atom_run_card_display(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    """Build run-card rows for InferenceX ATOM sweep decks."""
    try:
        from cvs.lib.inference.inferencex_atom.inferencex_atom_launch import build_launch_provenance  # noqa: F401

        rc = variant.run_card
        rows: List[Tuple[str, str, bool]] = [
            ("Model", variant.model.id, False),
            ("GPU", variant.gpu_arch, False),
            ("Driver", variant.params.driver, False),
            ("TP", str(variant.params.tensor_parallelism), False),
            thresholds_run_card_row(variant),
        ]
        if rc.upstream_run_url:
            rows.append(("Upstream", rc.upstream_run_url, True))
        rows.extend(provenance_link_rows(provenance))
        return rows
    except (ImportError, AttributeError):
        rows: List[Tuple[str, str, bool]] = [
            ("Model", getattr(getattr(variant, "model", None), "id", "\u2014"), False),
            ("GPU", getattr(variant, "gpu_arch", "\u2014"), False),
            thresholds_run_card_row(variant),
        ]
        params = getattr(variant, "params", None)
        if params is not None and hasattr(params, "tensor_parallelism"):
            rows.insert(3, ("TP", str(params.tensor_parallelism), False))
        rows.extend(provenance_link_rows(provenance))
        return rows


__all__ = ["atom_run_card_display"]
