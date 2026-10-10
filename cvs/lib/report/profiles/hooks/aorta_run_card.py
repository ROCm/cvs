"""Aorta run-card hook for the shared training Run Deck.

Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

from cvs.lib.report.formatting import fmt_num
from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def _image_name(variant):
    container = getattr(variant, "container", None)
    image = container.get("image") if isinstance(container, dict) else getattr(container, "image", None)
    return str(image or "—")


def _threshold_value(row):
    digits = 2 if row["threshold"] == "max_avg_iteration_ms" else 4
    suffix = " ms" if digits == 2 else ""
    actual = "n/a" if row["actual"] is None else fmt_num(row["actual"], digits=digits) + suffix
    limit = fmt_num(row["limit"], digits=digits) + suffix
    operator = "≤" if row["kind"] == "max" else "≥"
    return f"{row['status'].upper()} · {actual} (limit {operator} {limit})"


def aorta_run_card_display(variant, provenance):
    """Return Aorta run-card rows and provenance links."""
    rows = [
        ("Workload", getattr(getattr(variant, "model", None), "id", None) or "—", False),
        ("Framework", "Aorta", False),
        ("Image", _image_name(variant), False),
    ]
    cell_id = getattr(variant, "cell_id", None)
    dimensions = ((getattr(variant, "results", None) or {}).get(cell_id) or {}).get("_dimensions") or {}
    for key, label in (
        ("nodes", "Nodes"),
        ("gpus_per_node", "GPUs/node"),
        ("nccl_channels", "NCCL channels"),
        ("rccl_branch", "RCCL branch"),
    ):
        if dimensions.get(key):
            rows.append((label, dimensions[key], False))
    if hasattr(variant, "metrics_source"):
        rows.append(("Metrics source", variant.metrics_source, False))
    state = getattr(variant, "threshold_state", None) or thresholds_run_card_row(variant)[1]
    rows.append(("Thresholds", state, False))
    if hasattr(variant, "threshold_rows"):
        rows.extend((row["threshold"], _threshold_value(row), False) for row in variant.threshold_rows())
    if hasattr(variant, "suite_failed"):
        rows.append(("Pytest stages", "failures — see pytest report" if variant.suite_failed else "all passed", False))
    rows.extend(provenance_link_rows(provenance or {}))
    return rows
