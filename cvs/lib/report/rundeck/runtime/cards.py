'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Profile-driven card renderers for Run Deck static HTML.
'''

from __future__ import annotations

import html
from typing import Any, Callable

from cvs.lib.report.formatting import fmt_num, link_or_text_html
from cvs.lib.report.rundeck.runtime.theme import render_launch_panel_html
from cvs.lib.report.inference_payload import sweep_has_multi_shape_comparison
from cvs.lib.report.render.cell_card import CellCardConfig, CellCardRenderer
from cvs.lib.report.render.gate_matrix import render_gate_heatmap_html, render_gate_matrix_html
from cvs.lib.report.render.panel_shell import render_results_table_html
from cvs.lib.report.rundeck.context import is_empty, resolve_bind
from cvs.lib.report.rundeck.runtime.sweep_charts import render_sweep_charts_html
from cvs.lib.report.types import DEFAULT_SESSION_LIFECYCLE_LABELS

SESSION_FALLBACK = DEFAULT_SESSION_LIFECYCLE_LABELS


def _render_run_card(payload: dict, card: dict, _data: Any) -> str:
    hero_html = "".join(
        f"<div class='meta-item'><span class='meta-k'>{html.escape(label)}</span>"
        f"<span class='meta-v'>"
        f"{link_or_text_html(value, label) if is_link else html.escape(str(value))}"
        f"</span></div>"
        for label, value, is_link in payload.get("run_card_display", [])
    )
    notes = payload.get("run_card_notes") or ""
    notes_html = f"<p class='notes'>{html.escape(notes)}</p>" if notes else ""
    return f"<div class='meta-grid'>{hero_html}</div>{notes_html}"


def _render_lifecycle(payload: dict, _card: dict, data: Any) -> str:
    lifecycle = data if isinstance(data, dict) else payload.get("lifecycle") or {}
    report = payload.get("report") or {}
    timeline_total = sum(lifecycle.values()) or 1.0
    parts = []
    for lbl in report.get("session_lifecycle_labels", ()) or SESSION_FALLBACK:
        sec = lifecycle.get(lbl, 0.0)
        if sec <= 0:
            continue
        pct = 100.0 * sec / timeline_total
        parts.append(
            f"<div class='tl-seg' style='flex-grow:{pct:.2f}'>"
            f"<span class='tl-lbl'>{html.escape(lbl.replace('_', ' '))}</span>"
            f"<span class='tl-val'>{sec:.1f}s</span></div>"
        )
    return "".join(parts) or "<p class='muted'>No lifecycle timings recorded.</p>"


def _render_sweep_analytics(payload: dict, _card: dict, data: Any) -> str:
    sweep = data if isinstance(data, dict) else {}
    summaries = sweep.get("sweep_summaries") or payload.get("sweep_summaries") or []
    summary_html = (
        "".join(
            f"<article class='summary-card'><h3>ISL={html.escape(str(s['isl']))} "
            f"\u00b7 OSL={html.escape(str(s['osl']))}</h3>"
            f"<div class='summary-stat'>{fmt_num(s['max_output_throughput'])} "
            f"<span class='headline-unit'>tok/s</span></div>"
            f"<div class='summary-meta'>Peak at C={s['conc_at_max_tput']}"
            f" &middot; TTFT {fmt_num(s.get('ttft_at_max_tput'))} ms"
            f"{' &middot; saturated at max C' if s.get('saturated') else ''}</div></article>"
            for s in summaries
        )
        or "<p class='muted'>No sweep summary (no throughput data).</p>"
    )

    cells = payload.get("cells") or []
    summary = payload.get("summary") or {}
    viewer_name = summary.get("viewer_html")
    viewer_banner = ""
    if sweep_has_multi_shape_comparison(cells) and viewer_name:
        viewer_banner = (
            "<div class='viewer-banner'>Cross-shape comparison (grouped bars and scaling trends) "
            f"is in the <a href='{html.escape(viewer_name)}'>interactive viewer</a>.</div>"
        )
    elif sweep_has_multi_shape_comparison(cells):
        viewer_banner = (
            "<div class='viewer-banner'>Cross-shape comparison charts are available in the "
            "interactive viewer sidecar.</div>"
        )

    chart_series = sweep.get("chart_series") or payload.get("chart_series") or {}
    chart_config = sweep.get("chart_config") or payload.get("chart_config") or []
    charts_html = render_sweep_charts_html(chart_config, chart_series)
    hint = (
        "<p class='chart-sweep-hint'>Per-shape bars use a y/x grid; hover a bar for the exact value.</p>"
        if charts_html
        else ""
    )
    return f"<div class='summary-grid'>{summary_html}</div>{viewer_banner}{hint}{charts_html}"


def _render_gate_matrix(payload: dict, _card: dict, data: Any) -> str:
    gate_matrix = data if isinstance(data, list) else payload.get("gate_matrix") or []
    tier_order = (payload.get("report") or {}).get("metric_tier_order") or ()
    return render_gate_matrix_html(gate_matrix, tier_order)


def _render_gate_heatmap(payload: dict, _card: dict, data: Any) -> str:
    gate_matrix = data if isinstance(data, list) else payload.get("gate_matrix") or []
    tier_order = (payload.get("report") or {}).get("metric_tier_order") or ()
    heatmap = render_gate_heatmap_html(gate_matrix, tier_order)
    return f"<div id='heatmap' class='heatmap-section'>{heatmap}</div>" if heatmap else ""


def _render_cell_cards(payload: dict, _card: dict, data: Any) -> str:
    cells = data if isinstance(data, list) else payload.get("cells") or []
    report = payload.get("report") or {}
    tier_order = report.get("metric_tier_order") or ()
    enforce = any(row[1] == "enforced" for row in payload.get("run_card_display", []) if row[0] == "Thresholds")
    cell_lifecycle_labels = tuple(report.get("cell_lifecycle_labels") or ("server_ready", "client_complete"))
    pytest_basename = (payload.get("provenance") or {}).get("pytest_html_href") or (
        (payload.get("provenance") or {}).get("pytest_html_basename", "")
    )
    config = CellCardConfig(
        tier_order=tuple(tier_order),
        headline_metric=report.get("headline_metric", "client.output_throughput"),
        enforce=enforce,
        cell_lifecycle_labels=cell_lifecycle_labels,
        pytest_html_basename=pytest_basename or None,
    )
    renderer = CellCardRenderer(config)
    cards = [renderer.render(c) for c in cells]
    summary = payload.get("summary") or {}
    banner = ""
    viewer_name = summary.get("viewer_html")
    if summary.get("mode") == "truncated" and viewer_name:
        banner = (
            f"<div class='viewer-banner'>Showing {len(cells)} of {summary.get('total_cells', len(cells))} "
            f"cells in this summary. <a href='{html.escape(viewer_name)}'>Open interactive viewer</a> "
            f"for filter and search across all cells.</div>"
        )
    empty_cells = "<p class='muted'>No cells.</p>"
    return f"{banner}<div class='cells'>{''.join(cards) or empty_cells}</div>"


def _render_table(_payload: dict, _card: dict, data: Any) -> str:
    table = data if isinstance(data, dict) else {}
    return render_results_table_html(
        table.get("headers") or [],
        table.get("rows") or [],
        empty_message="No results table rows.",
    )


def _render_launch(_payload: dict, _card: dict, data: Any) -> str:
    return render_launch_panel_html(data or {})


def _render_line_chart(_payload: dict, card: dict, data: Any) -> str:
    from cvs.lib.report.rundeck.runtime.sweep_charts import render_series_chart_html

    series_cfg = card.get("series") or {}
    y_field = series_cfg.get("y_field") or "bus_bw"
    charts = data.get("charts") if isinstance(data, dict) else {}
    raw = charts.get(y_field) if isinstance(charts, dict) else None
    entries: list = []
    if isinstance(raw, dict):
        for series_list in raw.values():
            if isinstance(series_list, list):
                entries.extend(series_list)
            elif isinstance(series_list, dict):
                entries.append(series_list)
    elif isinstance(raw, list):
        entries = raw
    if not entries:
        return "<p class='muted'>No series data.</p>"
    parts = []
    title = card.get("title") or y_field
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        points = entry.get("points") or []
        label = entry.get("label") or title
        part = render_series_chart_html(str(label), points, series_cfg.get("unit") or "GB/s")
        if part:
            parts.append(part)
    return f"<div class='chart-grid'>{''.join(parts)}</div>" if parts else "<p class='muted'>No series data.</p>"


def _render_heatmap(_payload: dict, card: dict, data: Any) -> str:
    rows = data.get("compare_rows") if isinstance(data, dict) else []
    if not rows:
        return ""
    headers = ["Collective", "Size", "Current", "Reference", "Delta %"]
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v) if v is not None else '—')}</td>" for v in row) + "</tr>"
        for row in rows
    )
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    title = card.get("title") or "Compare matrix"
    return f"<h3>{html.escape(title)}</h3><table class='results-table'><tr>{head}</tr>{body}</table>"


def _render_interactivity_viewer(payload: dict, card: dict, _data: Any) -> str:
    summary = payload.get("summary") or {}
    viewer_name = summary.get("viewer_html")
    if not viewer_name:
        return ""
    vc = payload.get("viewer_config") or {}
    inter = vc.get("interactivity") or {}
    if inter.get("enabled") is False:
        return ""
    title = card.get("title") or inter.get("title") or "Interactivity chart"
    hint = inter.get("hint") or ("Interactivity = 1000 / mean TPOT (ms) (tok/s/user) · Y = token throughput per GPU")
    href = html.escape(f"{viewer_name}#interactivity-panel")
    return (
        f"<p class='viewer-banner'><strong>{html.escape(title)}</strong> — "
        f"one line per sweep shape; hover for detail, click to pin. "
        f"<a href='{href}'>Open interactivity chart</a> in the interactive viewer.</p>"
        f"<p class='subsection-hint'>{html.escape(hint)}</p>"
    )


CARD_RENDERERS: dict[str, Callable[[dict, dict, Any], str]] = {
    "run_card": _render_run_card,
    "lifecycle_timeline": _render_lifecycle,
    "sweep_analytics": _render_sweep_analytics,
    "gate_matrix": _render_gate_matrix,
    "gate_heatmap": _render_gate_heatmap,
    "sweep_cell_cards": _render_cell_cards,
    "table": _render_table,
    "launch_panel": _render_launch,
    "line_chart": _render_line_chart,
    "heatmap": _render_heatmap,
    "interactivity_viewer": _render_interactivity_viewer,
}


def render_card(payload: dict, card: dict) -> tuple[str, str, bool]:
    """Return (section_id, html, include_in_nav)."""
    card_type = card.get("type")
    renderer = CARD_RENDERERS.get(str(card_type))
    if renderer is None:
        return "", f"<p class='muted'>Unknown card type: {html.escape(str(card_type))}</p>", False

    bind = card.get("bind") or card_type
    data = resolve_bind(payload, bind) if "." in bind else payload.get(bind)
    if card.get("when_empty") == "hide" and is_empty(data):
        return "", "", False

    html_body = renderer(payload, card, data)
    if not html_body:
        return "", "", False

    section_id = card.get("id") or card_type.replace("_", "-")
    title = card.get("title") or section_id.replace("-", " ").title()
    if card_type == "gate_heatmap":
        return "heatmap", html_body, True
    if card_type == "launch_panel":
        title = card.get("title") or "Launch commands"
        return "launch", f"<section class='panel' id='launch'><h2>{html.escape(title)}</h2>{html_body}</section>", True
    wrapped = f"<section class='panel' id='{html.escape(section_id)}'><h2>{html.escape(title)}</h2>"
    if card_type == "gate_matrix":
        wrapped += f"<div class='matrix-wrap'>{html_body}"
        return section_id, wrapped, True
    if card_type in ("table", "sweep_cell_cards"):
        wrap_class = "results-wrap" if card_type == "table" else ""
        inner = f"<div class='{wrap_class}'>{html_body}</div>" if wrap_class else html_body
        return section_id, f"{wrapped}{inner}</section>", True
    if card_type == "lifecycle_timeline":
        return section_id, f"{wrapped}<div class='tl-row'>{html_body}</div></section>", True
    if card_type == "run_card":
        return section_id, f"{wrapped}{html_body}</section>", True
    if card_type == "gate_heatmap":
        return section_id, html_body, True
    return section_id, f"{wrapped}{html_body}</section>", True


def close_gate_matrix_section(section_html: str, heatmap_html: str) -> str:
    if not section_html:
        return heatmap_html
    if heatmap_html and "</section>" not in section_html:
        return section_html + heatmap_html + "</div></section>"
    if heatmap_html and section_html.endswith("</section>"):
        return section_html[: -len("</section>")] + heatmap_html + "</div></section>"
    return section_html
