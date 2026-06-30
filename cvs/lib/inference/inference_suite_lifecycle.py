'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Reusable lifecycle helpers for DTNI inference suites (trimmed).

This is the render/report-support subset of the upstream module: the
cross-test ``InferenceLifecycle`` state object, the pytest-html helpers that
render per-test lifecycle/metric tables, the item-ordering helper, and the
canonical ``inf_res_dict`` key builder consumed by ``cvs.lib.report``.

The launch / sshd / model-fetch / teardown *stage tests* are intentionally NOT
included here: the unified vLLM suite implements its own stage tests inline, so
nothing collects them from this module. Dropping them also removes the
``cvs.lib.inference.cache_probe.du_bytes`` dependency, which is absent on this
branch.
'''

from __future__ import annotations

try:
    import pytest_html
except ImportError:
    pytest_html = None


class InferenceLifecycle:
    """Cross-test state shared by lifecycle stage tests in one module scope."""

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


def sweep_cell_result_key(variant_config, seq_combo, isl, osl, concurrency):
    """Canonical ``inf_res_dict`` key for one sweep cell."""
    return (
        variant_config.model.id,
        variant_config.gpu_arch,
        isl,
        osl,
        seq_combo.get("name", "default"),
        concurrency,
    )


def sort_lifecycle_items(items, rank):
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


def attach_lifecycle_html_table(item, report):
    if report.when != "call":
        return
    lc = item.funcargs.get("lifecycle")
    rows = getattr(lc, "report", {}).get(item.nodeid) if lc else None
    if not rows:
        return
    if pytest_html is None:
        return
    body = "".join(
        f"<tr><td>{label}</td><td>{value:.1f}</td><td>{unit}</td></tr>"
        for label, value, unit in rows
    )
    html = f"<table><tr><th>stage</th><th>value</th><th>unit</th></tr>{body}</table>"
    extras = getattr(report, "extras", [])
    extras.append(pytest_html.extras.html(html))
    report.extras = extras


def html_metric_table_header(cells):
    cells.insert(-1, "<th>Value</th>")
    cells.insert(-1, "<th>Unit</th>")


def html_metric_table_row(report, cells):
    props = dict(report.user_properties)
    has = "metric_value" in props
    val = props.get("metric_value")
    unit = props.get("metric_unit", "") if has else ""
    if not has:
        shown = ""
    elif val is None:
        shown = "-"
    elif isinstance(val, float):
        shown = f"{val:.3f}"
    else:
        shown = str(val)
    cells.insert(-1, f"<td>{shown}</td>")
    cells.insert(-1, f"<td>{unit}</td>")
