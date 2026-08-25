'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

ATOM run-card hook for JSON deck profiles.
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def atom_run_card_display(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    rc = variant.run_card
    rows: List[Tuple[str, str, bool]] = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
        ("Driver", variant.params.driver, False),
        ("Image pin", rc.atom_image_pin or "\u2014", False),
        ("TP", str(variant.params.tensor_parallelism), False),
        thresholds_run_card_row(variant),
    ]
    if rc.upstream_run_url:
        rows.append(("Upstream", rc.upstream_run_url, True))
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["atom_run_card_display"]
