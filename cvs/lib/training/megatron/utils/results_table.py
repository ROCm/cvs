'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Consolidated metric-results table for the Megatron training suites: one row per
(sweep, metric) with Expected/Actual/Unit/Status, rendered to a single HTML file
linked from the pytest report.
'''

import html as _html

from cvs.lib.training.megatron.utils.megatron_metrics import METRIC_UNITS

_STATUS_COLORS = {
    "PASS": "#2e7d32",
    "FAIL": "#c62828",
    "N/A": "#f9a825",
    "RECORD": "#555555",
}


def format_expected(spec):
    """Human-readable expected-threshold string for a threshold spec."""
    if not spec:
        return "-"
    kind = spec.get("kind")
    value = spec.get("value")
    if kind == "info":
        return f"info ({value})" if value is not None else "info"
    if kind in ("min", "min_tok_s"):
        return f">= {value}"
    if kind == "max":
        return f"<= {value}"
    if kind == "max_ms":
        return f"<= {value} ms"
    if kind == "within":
        return f"{value} +/-{spec.get('tolerance_pct')}%"
    if kind == "min_ratio":
        return f">= {value} x {spec.get('reference')}"
    return str(spec)


def format_value(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_display_name(metric):
    """Drop the ``training.`` namespace prefix for display."""
    return metric.split("training.", 1)[-1]


def metric_unit(metric):
    """Unit for a ``training.<short>`` metric name; '-' when unknown."""
    return METRIC_UNITS.get(metric_display_name(metric), "-")


def build_metric_row(sweep, metric, spec, value, status):
    """Assemble one table row dict from a raw (sweep, metric) verdict."""
    return {
        "sweep": sweep,
        "metric": metric_display_name(metric),
        "expected": format_expected(spec),
        "actual": format_value(value),
        "unit": metric_unit(metric),
        "status": status,
    }


def build_benchmark_metric_row(metric, spec, value, status, reason="", enforced=True):
    """Registry-format row for the collapsible per-metric pass/fail panel rendered
    by ``cvs.lib.report.render.perf_metric_table.render_benchmark_metrics_html``.

    ``status`` is one of ``pass``/``fail``/``skip``/``record`` (lower-case, as the
    shared renderer expects). ``node`` is empty because a training sweep verdict is
    cluster-wide, so the renderer drops the node column.
    """
    return {
        "node": "",
        "metric": metric_display_name(metric),
        "label": metric_display_name(metric),
        "status": status,
        "actual": value,
        "unit": metric_unit(metric),
        "spec": spec,
        "reason": reason,
        "enforced": enforced,
    }


def render_metric_results_html(metric_rows, title):
    """Render the Sweep/Metric/Expected/Actual/Unit/Status rows as a full HTML doc."""
    body = ""
    for r in metric_rows:
        color = _STATUS_COLORS.get(r["status"], "#000000")
        body += (
            "<tr>"
            f"<td>{_html.escape(str(r.get('sweep', '-')))}</td>"
            f"<td>{_html.escape(str(r['metric']))}</td>"
            f"<td>{_html.escape(str(r['expected']))}</td>"
            f"<td>{_html.escape(str(r['actual']))}</td>"
            f"<td>{_html.escape(str(r['unit']))}</td>"
            f"<td style=\"color:{color};font-weight:bold;\">{_html.escape(str(r['status']))}</td>"
            "</tr>"
        )
    return (
        f"<html><head><meta charset='utf-8'><title>{_html.escape(title)}</title></head>"
        f"<body><h2>{_html.escape(title)}</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Sweep</th><th>Metric</th><th>Expected</th><th>Actual</th><th>Unit</th><th>Status</th></tr>"
        f"{body}</table></body></html>"
    )
