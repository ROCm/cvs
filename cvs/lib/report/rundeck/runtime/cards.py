'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Profile-driven card renderers for Run Deck static HTML.
'''

from __future__ import annotations

import html
from typing import Any

from cvs.lib.report.formatting import fmt_num, link_or_text_html
from cvs.lib.report.inference_payload import sweep_has_multi_shape_comparison
from cvs.lib.report.render.cell_card import CellCardConfig, CellCardRenderer
from cvs.lib.report.render.gate_matrix import GateMatrixRenderer
from cvs.lib.report.render.loss_chart import SERIES_COLORS
from cvs.lib.report.render.panel_shell import render_results_table_html
from cvs.lib.report.rundeck.context import is_empty, resolve_bind
from cvs.lib.report.rundeck.runtime.sweep_charts import SweepChartRenderer
from cvs.lib.report.rundeck.runtime.theme import render_launch_panel_html
from cvs.lib.report.types import DEFAULT_SESSION_LIFECYCLE_LABELS

SESSION_FALLBACK = DEFAULT_SESSION_LIFECYCLE_LABELS

_DEFAULT_STATUS_HINT_HTML = (
    "Click a cell's <strong>items</strong> to expand that node × group's item breakdown and artifact links."
)


def _suppressed_when_empty(card, data):
    if card.get("when_empty") != "hide":
        return False
    if is_empty(data):
        return True
    if str(card.get("type")) != "metric_charts" or not isinstance(data, dict):
        return False
    return not any(data.get(key) for key in ("metrics", "series", "heatmaps"))


def _fmt_pass_rate(rate):
    if rate is None:
        return "\u2014"
    pct = 100.0 * float(rate)
    rounded = round(pct, 1)
    if abs(rounded - round(rounded)) < 0.05:
        return f"{int(round(rounded))}%"
    return f"{rounded:.1f}%"


def _threshold_note(metric, threshold):
    if threshold is None:
        return ""
    direction = str(metric.get("direction") or "")
    sense = {"higher": "higher is better", "lower": "lower is better"}.get(direction, "")
    note = f"threshold {fmt_num(threshold)}"
    unit = str(metric.get("unit") or "")
    if unit:
        note = f"{note} {unit}"
    if sense:
        note = f"{note} · {sense}"
    return note


def _heat_style(value, threshold, direction, lo, hi):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "hm-na", ""
    if threshold is not None and direction in ("higher", "lower"):
        ok = value >= threshold if direction == "higher" else value <= threshold
        return ("hm-pass" if ok else "hm-fail"), ""
    span = (hi - lo) or 1.0
    alpha = 0.15 + 0.6 * ((value - lo) / span)
    return "hm-scale", f"background:rgba(107,159,255,{alpha:.2f})"


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_points(metric):
    points = []
    for point in metric.get("points") or []:
        if not isinstance(point, dict):
            continue
        value = _as_float(point.get("value"))
        if value is None:
            continue
        points.append((point, value))
    return points


def _ordered_nodes(points):
    nodes = []
    for point in points:
        node = str(point.get("node") or "")
        if node not in nodes:
            nodes.append(node)
    return nodes


def _shared_prefix(names):
    """Leading word every name shares ("power GPU2", "power GPU3" -> "power"), else ""."""
    if len(names) < 2:
        return ""
    words = [str(name).split(" ", 1) for name in names]
    first = words[0][0]
    if all(len(parts) == 2 and parts[0] == first for parts in words):
        return first
    return ""


def _worst_point(points, direction):
    if direction == "lower":
        return max(points, key=lambda item: item[1])
    return min(points, key=lambda item: item[1])


def _series_points(series):
    points = []
    for point in series.get("points") or []:
        if isinstance(point, dict):
            x, y = point.get("x"), _as_float(point.get("y"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], _as_float(point[1])
        else:
            continue
        if y is None:
            continue
        points.append((x, y))
    return points


class DeckCardRenderer:
    """Profile-driven card renderers for Run Deck static HTML sections."""

    DEFAULT_MAX_LINE_CHART_SERIES = 40
    MAX_BAR_NODES = 16
    MAX_SERIES_LINES = 8
    _TONES = ("tone1", "tone2", "tone3", "tone4", "tone5", "tone6")

    def __init__(
        self,
        *,
        gate_renderer: GateMatrixRenderer | None = None,
        chart_renderer: SweepChartRenderer | None = None,
    ):
        self._gate = gate_renderer or GateMatrixRenderer()
        self._charts = chart_renderer or SweepChartRenderer()

    def render_run_card(self, payload: dict, _card: dict, data: Any) -> str:
        # A card may bind its own run-card rows (e.g. datasets.status_matrix.
        # run_card_display); fall back to the payload-level rows otherwise.
        rows = data if isinstance(data, list) else payload.get("run_card_display", [])
        hero_html = "".join(
            f"<div class='meta-item'><span class='meta-k'>{html.escape(label)}</span>"
            f"<span class='meta-v'>"
            f"{link_or_text_html(value, label) if is_link else html.escape(str(value))}"
            f"</span></div>"
            for label, value, is_link in rows
        )
        notes = payload.get("run_card_notes") or ""
        notes_html = f"<p class='notes'>{html.escape(notes)}</p>" if notes else ""
        return f"<div class='meta-grid'>{hero_html}</div>{notes_html}"

    @staticmethod
    def _cell_stage_times(payload, label):
        times = []
        for cell in payload.get("cells") or []:
            sec = (cell.get("cell_lifecycle") or {}).get(label)
            try:
                sec = float(sec)
            except (TypeError, ValueError):
                continue
            if sec <= 0:
                continue
            name = cell.get("subtitle") or cell.get("label") or cell.get("cell_id") or cell.get("policy") or label
            times.append((str(name), sec))
        return times

    @staticmethod
    def _stage_html(label, sec, pct, tone, per_cell):
        head = (
            f"<span class='tl-lbl'>{html.escape(label.replace('_', ' '))}</span><span class='tl-val'>{sec:.1f}s</span>"
        )
        if not per_cell:
            return f"<div class='tl-seg tl-{tone}' style='flex-grow:{pct:.2f}'>{head}</div>"
        cells = "".join(
            f"<div class='tl-cell tl-{cell_tone}' "
            f"style='flex-grow:{100.0 * cell_sec / sec:.2f}' title='{html.escape(f'{name}: {cell_sec:.1f}s')}'>"
            f"<span class='tl-lbl'>{html.escape(name)}</span>"
            f"<span class='tl-val'>{cell_sec:.1f}s</span></div>"
            for name, cell_sec, cell_tone in per_cell
        )
        return (
            f"<div class='tl-group tl-{tone}' style='flex-grow:{pct:.2f}'>"
            f"<div class='tl-group-head'>{head}</div>"
            f"<div class='tl-group-body'>{cells}</div></div>"
        )

    @staticmethod
    def _plain_timeline(lifecycle, labels):
        """One untoned segment per stage. Inference decks use this path."""
        timeline_total = sum(lifecycle.values()) or 1.0
        parts = []
        for lbl in labels:
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

    def render_lifecycle(self, payload: dict, _card: dict, data: Any) -> str:
        lifecycle = data if isinstance(data, dict) else payload.get("lifecycle") or {}
        report = payload.get("report") or {}
        expand = tuple(report.get("expand_lifecycle_labels") or ())
        labels = report.get("session_lifecycle_labels", ()) or SESSION_FALLBACK
        if not expand:
            return self._plain_timeline(lifecycle, labels)
        # Expanded stages total their per-cell times: sweep cells run back to back, so the
        # session spends the sum, not the longest cell.
        per_cell = {lbl: self._cell_stage_times(payload, lbl) for lbl in expand}
        totals = {
            lbl: (sum(sec for _, sec in per_cell[lbl]) if per_cell.get(lbl) else lifecycle.get(lbl, 0.0))
            for lbl in labels
        }
        timeline_total = sum(totals.values()) or 1.0
        parts = []
        # Stages and sweep cells share one palette cursor so a cell never repeats the colour
        # of a stage sitting next to it.
        tone = 0
        for lbl in labels:
            sec = totals[lbl]
            if sec <= 0:
                continue
            stage_tone = self._TONES[tone % len(self._TONES)]
            tone += 1
            cells = []
            for name, cell_sec in per_cell.get(lbl) or []:
                cells.append((name, cell_sec, self._TONES[tone % len(self._TONES)]))
                tone += 1
            parts.append(self._stage_html(lbl, sec, 100.0 * sec / timeline_total, stage_tone, cells))
        return "".join(parts) or "<p class='muted'>No lifecycle timings recorded.</p>"

    @staticmethod
    def _summary_card_html(summary):
        if summary.get("label"):
            title = str(summary["label"])
            meta = str(summary.get("meta") or "")
            unit = str(summary.get("headline_unit") or "tok/s")
        else:
            title = f"ISL={summary.get('isl', '')} \u00b7 OSL={summary.get('osl', '')}"
            sat = " \u00b7 saturated at max C" if summary.get("saturated") else ""
            meta = (
                f"Peak at C={summary.get('conc_at_max_tput')}"
                f" \u00b7 TTFT {fmt_num(summary.get('ttft_at_max_tput'))} ms{sat}"
            )
            unit = "tok/s"
        return (
            f"<article class='summary-card'><h3>{html.escape(title)}</h3>"
            f"<div class='summary-stat'>{fmt_num(summary['max_output_throughput'])} "
            f"<span class='headline-unit'>{html.escape(unit)}</span></div>"
            f"<div class='summary-meta'>{html.escape(meta)}</div></article>"
        )

    def render_sweep_analytics(self, payload: dict, _card: dict, data: Any) -> str:
        sweep = data if isinstance(data, dict) else {}
        summaries = sweep.get("sweep_summaries") or payload.get("sweep_summaries") or []
        summary_html = (
            "".join(self._summary_card_html(s) for s in summaries)
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
        charts_html = self._charts.render_sweep_charts(chart_config, chart_series)
        hint = (
            "<p class='chart-sweep-hint'>Per-shape bars use a y/x grid; hover a bar for the exact value.</p>"
            if charts_html
            else ""
        )
        return f"<div class='summary-grid'>{summary_html}</div>{viewer_banner}{hint}{charts_html}"

    def render_gate_matrix(self, payload: dict, _card: dict, data: Any) -> str:
        gate_matrix = data if isinstance(data, list) else payload.get("gate_matrix") or []
        tier_order = (payload.get("report") or {}).get("metric_tier_order") or ()
        return self._gate.render_table(gate_matrix, tier_order)

    def render_gate_heatmap(self, payload: dict, _card: dict, data: Any) -> str:
        gate_matrix = data if isinstance(data, list) else payload.get("gate_matrix") or []
        tier_order = (payload.get("report") or {}).get("metric_tier_order") or ()
        heatmap = self._gate.render_heatmap(gate_matrix, tier_order)
        return f"<div id='heatmap' class='heatmap-section'>{heatmap}</div>" if heatmap else ""

    def render_cell_cards(self, payload: dict, _card: dict, data: Any) -> str:
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

    @staticmethod
    def render_table(_payload: dict, _card: dict, data: Any) -> str:
        table = data if isinstance(data, dict) else {}
        # .results-wrap gives the (potentially wide, all-metrics) table a
        # horizontal scrollbar instead of overflowing the card.
        inner = render_results_table_html(
            table.get("headers") or [],
            table.get("rows") or [],
            empty_message="No results table rows.",
        )
        return f"<div class='results-wrap'>{inner}</div>"

    @staticmethod
    def render_status_matrix(payload: dict, _card: dict, data: Any) -> str:
        dataset = data if isinstance(data, dict) else {}
        nodes = dataset.get("nodes") or []
        groups = dataset.get("groups") or []
        grid = dataset.get("grid") or {}
        if not nodes or not groups:
            return "<p class='muted'>No node results recorded.</p>"

        header = (
            "<tr><th class='sm-node'>Node</th>" + "".join(f"<th>{html.escape(str(g))}</th>" for g in groups) + "</tr>"
        )

        body_rows = []
        for node in nodes:
            cells = [f"<td class='sm-node'>{html.escape(str(node))}</td>"]
            for group in groups:
                cell = (grid.get(node) or {}).get(group) or {}
                cells.append(DeckCardRenderer._status_cell_html(cell))
            body_rows.append(f"<tr>{''.join(cells)}</tr>")

        hint = str(_card.get("hint") or "").strip() if isinstance(_card, dict) else ""
        hint_html = html.escape(hint) if hint else _DEFAULT_STATUS_HINT_HTML
        return (
            "<div class='results-wrap'><table class='status-matrix'>"
            f"{header}{''.join(body_rows)}</table></div>"
            f"<p class='muted sm-hint'>{hint_html}</p>"
        )

    @staticmethod
    def _status_item_html(item: dict) -> str:
        status = html.escape(str(item.get("status", "na")))
        name = html.escape(str(item.get("name", "")))
        msg = item.get("message")
        msg_html = (
            f"<span class='sm-imsg'>{html.escape(str(msg))}</span> "
            if str(item.get("status")) == "fail" and msg
            else ""
        )
        return (
            f"<div class='sm-item'><span class='sm-iname'>{name}</span>"
            f"<span class='sm-iright'>{msg_html}<span class='chip chip-{status}'>{status}</span></span></div>"
        )

    @staticmethod
    def _status_cell_html(cell: dict) -> str:
        status = str(cell.get("status") or "na")
        summary = cell.get("items_summary") or ""
        items = cell.get("items") or []

        if status == "na":
            return (
                "<td class='sm-cell sm-na'><span class='chip chip-na'>n/a</span>"
                "<div class='sm-count'>group not on node</div></td>"
            )

        # ``items`` holds only the real failures (build_node_record), so the cell
        # count comes from ANC's own roll-up line (items_summary); len(items)
        # would read as "0 / <#failures>" and hide the passed items.
        count_txt = summary or (f"{len(items)} failed" if items else status)

        item_rows = (
            "".join(DeckCardRenderer._status_item_html(it) for it in items)
            or "<p class='muted'>No per-item detail captured.</p>"
        )

        links = []
        if cell.get("errors_json_href"):
            links.append(f"<a href='{html.escape(str(cell['errors_json_href']))}'>errors.json</a>")
        if cell.get("log_tarball_href"):
            links.append(f"<a href='{html.escape(str(cell['log_tarball_href']))}'>logs.tar.gz</a>")
        links_html = f"<div class='sm-links'>{''.join(links)}</div>" if links else ""

        summary_line = f"<div class='sm-summary muted'>{html.escape(str(summary))}</div>" if summary else ""

        return (
            f"<td class='sm-cell sm-{status}'><details class='sm-details'>"
            f"<summary><span class='chip chip-{status}'>{status}</span>"
            f"<span class='sm-count'>{html.escape(count_txt)}</span>"
            f"<span class='sm-caret'></span></summary>"
            f"<div class='sm-body'>{summary_line}{item_rows}{links_html}</div>"
            f"</details></td>"
        )

    @staticmethod
    def render_status_overview(_payload, _card, data):
        overview = data if isinstance(data, dict) else {}
        counts = overview.get("counts") if isinstance(overview.get("counts"), dict) else {}
        total = sum(int(counts.get(key) or 0) for key in ("pass", "fail", "na"))
        if total <= 0 and not overview.get("failures_by_node") and not overview.get("failures_by_group"):
            return "<p class='muted'>No node results recorded.</p>"
        rate = _fmt_pass_rate(overview.get("pass_rate"))
        passed = int(counts.get("pass") or 0)
        failed = int(counts.get("fail") or 0)
        na_count = int(counts.get("na") or 0)
        share = max(total, 1)
        stack = "".join(
            f"<span class='ov-stack-{key}' style='width:{100.0 * count / share:.1f}%'></span>"
            for key, count in (("pass", passed), ("fail", failed), ("na", na_count))
            if count
        )
        rate_tone = "summary-stat-fail" if failed else ("summary-stat-pass" if passed else "")
        summary = (
            "<div class='summary-grid ov-stats'>"
            "<div class='summary-card'><h3>Pass rate</h3>"
            f"<div class='summary-stat {rate_tone}'>{html.escape(rate)}</div>"
            f"<div class='ov-stack'>{stack}</div>"
            f"<div class='summary-meta'>{passed} passed · {failed} failed · {na_count} n/a</div>"
            "</div>"
            f"<div class='summary-card'><h3>Passed</h3><div class='summary-stat summary-stat-pass'>{passed}</div>"
            "<div class='summary-meta'>node × group</div></div>"
            f"<div class='summary-card'><h3>Failed</h3><div class='summary-stat summary-stat-fail'>{failed}</div>"
            "<div class='summary-meta'>node × group</div></div>"
            f"<div class='summary-card'><h3>Not evaluated</h3><div class='summary-stat summary-stat-na'>{na_count}</div>"
            "<div class='summary-meta'>node × group</div></div>"
            "</div>"
        )
        tables = (
            "<div class='overview-split'>"
            f"<div>{DeckCardRenderer._failure_table('Node', overview.get('failures_by_node') or [], 'node')}</div>"
            f"<div>{DeckCardRenderer._failure_table('Group', overview.get('failures_by_group') or [], 'group')}</div>"
            "</div>"
        )
        return summary + tables

    @staticmethod
    def _failure_table(title, rows, label_key):
        if not rows:
            return "<p class='muted'>No results.</p>"
        max_fail = max(int(row.get("fail") or 0) for row in rows if isinstance(row, dict)) or 1
        body = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            fail = int(row.get("fail") or 0)
            width = 100.0 * fail / max_fail if fail else 0
            body.append(
                "<tr>"
                f"<td>{html.escape(str(row.get(label_key, '')))}</td>"
                f"<td class='ov-fail'>{fail}"
                f"<span class='fail-track'><span class='fail-fill' style='width:{width:.0f}%'></span></span></td>"
                f"<td>{int(row.get('pass') or 0)}</td>"
                f"<td>{int(row.get('na') or 0)}</td>"
                "</tr>"
            )
        head = f"<tr><th>{html.escape(title)}</th><th>Fail</th><th>Pass</th><th>n/a</th></tr>"
        return f"<table class='overview-table'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"

    def render_metric_charts(self, _payload, _card, data):
        charts = data if isinstance(data, dict) else {}
        metrics = [item for item in (charts.get("metrics") or []) if isinstance(item, dict)]
        series = [item for item in (charts.get("series") or []) if isinstance(item, dict)]
        heatmaps = [item for item in (charts.get("heatmaps") or []) if isinstance(item, dict)]
        if not metrics and not series and not heatmaps:
            return "<p class='muted'>No metric data recorded.</p>"
        groups = {}
        for metric in metrics:
            groups.setdefault(str(metric.get("group") or ""), {"metrics": [], "series": []})["metrics"].append(metric)
        for item in series:
            groups.setdefault(str(item.get("group") or ""), {"metrics": [], "series": []})["series"].append(item)
        parts = []
        for group, members in groups.items():
            panels = self._metric_group_panels(members["metrics"]) + self._series_group_panels(members["series"])
            if not panels:
                continue
            title = f"<h3 class='chart-group-title'>{html.escape(group)}</h3>" if group else ""
            parts.append(f"<div class='chart-group'>{title}<div class='chart-grid'>{''.join(panels)}</div></div>")
        heat_html = "".join(self._heatmap_panel(item) for item in heatmaps)
        if heat_html:
            parts.append(
                "<div class='chart-group'><h3 class='chart-group-title'>Heatmaps</h3>"
                f"<div class='chart-grid chart-grid-heat'>{heat_html}</div></div>"
            )
        return "".join(parts) or "<p class='muted'>No metric data recorded.</p>"

    def _metric_group_panels(self, metrics):
        # One panel per (unit, threshold, direction) so metrics that share a gate
        # land on one axis with one threshold line instead of one chart each.
        charts = {}
        settings = []
        for metric in metrics:
            points = _metric_points(metric)
            if not points:
                continue
            threshold = _as_float(metric.get("threshold"))
            unit = str(metric.get("unit") or "")
            if not unit and threshold is None:
                settings.append((metric, points))
                continue
            key = (unit, threshold, str(metric.get("direction") or ""))
            charts.setdefault(key, []).append((metric, points))
        panels = [self._clustered_bar_panel(members) for members in charts.values()]
        if settings:
            panels.append(self._settings_panel(settings))
        return [panel for panel in panels if panel]

    def _clustered_bar_panel(self, members):
        metric0 = members[0][0]
        unit = str(metric0.get("unit") or "")
        threshold = _as_float(metric0.get("threshold"))
        direction = str(metric0.get("direction") or "")
        nodes = _ordered_nodes(point for _metric, points in members for point, _value in points)
        rollup = len(nodes) > self.MAX_BAR_NODES
        columns = []
        for metric, points in members:
            if rollup:
                point, value = _worst_point(points, direction)
                columns.append((str(metric.get("name") or "metric"), [(point, value, f"worst of {len(points)}")]))
            else:
                by_node = {str(point.get("node") or ""): (point, value) for point, value in points}
                columns.append(
                    (
                        str(metric.get("name") or "metric"),
                        [(*by_node[node], node) for node in nodes if node in by_node],
                    )
                )
        values = [value for _name, bars in columns for _point, value, _label in bars]
        if threshold is not None:
            values.append(threshold)
        domain_min, domain_max, ticks = SweepChartRenderer._display_scale(min(values), max(values))

        def pct(value):
            return SweepChartRenderer._value_pct(value, domain_min, domain_max)

        y_labels = "".join(
            f"<span class='chart-ylbl' style='bottom:{pct(tick):.2f}%'>{html.escape(fmt_num(tick))}</span>"
            for tick in ticks
        )
        grid = "".join(f"<span class='chart-hline' style='bottom:{pct(tick):.2f}%'></span>" for tick in ticks)
        prefix = _shared_prefix([name for name, _bars in columns])
        if prefix:
            columns = [(name[len(prefix) + 1 :], bars) for name, bars in columns]
        cols_html = []
        x_ticks = []
        for name, bars in columns:
            bar_html = []
            for point, value, label in bars:
                status = str(point.get("status") or "")
                bar_class = f"chart-bar-{status}" if status in ("pass", "fail", "na") else "chart-bar-accent2"
                who = str(point.get("node") or "") if rollup else label
                full_name = f"{prefix} {name}" if prefix else name
                tip = html.escape(f"{full_name} · {who}: {fmt_num(value)} {unit} ({status or 'n/a'})".strip())
                bar_html.append(
                    f"<div class='chart-bar {bar_class} chart-has-tip' style='height:{pct(value):.1f}%' "
                    f"data-tip='{tip}' tabindex='0' role='img' aria-label='{tip}'></div>"
                )
            cols_html.append(f"<div class='chart-col chart-cluster'>{''.join(bar_html)}</div>")
            x_ticks.append(f"<span class='chart-xlbl'><span class='chart-xlbl-line'>{html.escape(name)}</span></span>")
        marker = ""
        if threshold is not None:
            marker = f"<span class='chart-threshold' style='bottom:{pct(threshold):.2f}%' title='threshold'></span>"
        notes = [_threshold_note(metric0, threshold)]
        if rollup:
            notes.append(f"worst node per bar across {len(nodes)} nodes")
        elif len(nodes) > 1:
            notes.append("bars left→right: " + ", ".join(nodes))
        note_html = "".join(f"<p class='metric-note'>{html.escape(note)}</p>" for note in notes if note)
        if len(columns) == 1:
            title = str(metric0.get("name") or "metric")
        else:
            title = prefix or unit or "value"
        # A few clusters read fine at grid width; only long x-axes need the full row.
        wide = " chart-panel-wide" if len(columns) > 4 else ""
        return (
            f"<div class='chart-panel{wide}'><h3>{html.escape(title)}</h3>{note_html}"
            f"<div class='chart-viz'><div class='chart-ywrap'><div class='chart-ylabels'>{y_labels}</div></div>"
            f"<div class='chart-main'><div class='chart-plotbox'><div class='chart-hgrid' aria-hidden='true'>{grid}</div>"
            f"{marker}<div class='chart-bars'>{''.join(cols_html)}</div></div>"
            f"<div class='chart-xrow chart-xrow-cluster'>{''.join(x_ticks)}</div></div></div>"
            f"<div class='chart-unit'>{html.escape(unit)}</div></div>"
        )

    @staticmethod
    def _settings_panel(settings):
        nodes = _ordered_nodes(point for _metric, points in settings for point, _value in points)
        head = "<tr><th></th>" + "".join(f"<th>{html.escape(node)}</th>" for node in nodes) + "</tr>"
        body = []
        for metric, points in settings:
            by_node = {str(point.get("node") or ""): value for point, value in points}
            cells = "".join(
                f"<td>{html.escape(fmt_num(by_node[node]) if node in by_node else '—')}</td>" for node in nodes
            )
            body.append(f"<tr><th>{html.escape(str(metric.get('name') or 'metric'))}</th>{cells}</tr>")
        return (
            "<div class='chart-panel'><h3>Settings</h3>"
            f"<table class='overview-table'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"
        )

    def _series_group_panels(self, series):
        merged = {}
        for item in series:
            key = (str(item.get("name") or "series"), str(item.get("unit") or ""))
            merged.setdefault(key, []).append(item)
        panels = []
        for (name, unit), items in merged.items():
            lines = []
            for item in items:
                points = _series_points(item)
                if len(points) >= 2:
                    lines.append((str(item.get("node") or ""), points))
            if lines:
                panels.append(self._multi_series_panel(name, unit, lines, items[0]))
        return panels

    def _multi_series_panel(self, name, unit, lines, sample):
        total = len(lines)
        lines = lines[: self.MAX_SERIES_LINES]
        x_values = []
        for _node, points in lines:
            for x, _y in points:
                if x not in x_values:
                    x_values.append(x)
        ys = [y for _node, points in lines for _x, y in points]
        domain_min, domain_max, ticks = SweepChartRenderer._display_scale(min(ys), max(ys))

        # A narrow canvas keeps axis text legible when the panel sits in a three-column grid.
        left, right = 64, 392

        def y_of(value):
            return 180 - 160 * SweepChartRenderer._value_pct(value, domain_min, domain_max) / 100

        def x_of(x):
            return left + (right - left) * x_values.index(x) / max(1, len(x_values) - 1)

        x_label = str(sample.get("x_label") or "")
        svg = []
        for tick in ticks:
            y = y_of(tick)
            svg.append(
                f"<line x1='{left}' x2='{right}' y1='{y:.2f}' y2='{y:.2f}' stroke='var(--border)'/>"
                f"<text x='{left - 6}' y='{y:.2f}' text-anchor='end' dominant-baseline='middle'>"
                f"{html.escape(fmt_num(tick))}</text>"
            )
        step = max(1, (len(x_values) + 5) // 6)
        last = len(x_values) - 1
        for index, x in enumerate(x_values):
            ticked = index % step == 0 and (last - index >= step / 2 or index == last)
            if not ticked and index != last:
                continue
            anchor = "end" if index == last and last > 0 else ("start" if index == 0 and last > 0 else "middle")
            svg.append(f"<text x='{x_of(x):.2f}' y='202' text-anchor='{anchor}'>{html.escape(str(x))}</text>")
        legend = []
        for index, (node, points) in enumerate(lines):
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            coords = " ".join(f"{x_of(x):.2f},{y_of(y):.2f}" for x, y in points)
            svg.append(f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='2'/>")
            for x, y in points:
                tip = html.escape(f"{node} · {x}: {fmt_num(y)} {unit}".strip())
                svg.append(
                    f"<circle cx='{x_of(x):.2f}' cy='{y_of(y):.2f}' r='3' fill='{color}' tabindex='0' "
                    f"role='img' aria-label='{tip}'><title>{tip}</title></circle>"
                )
            legend.append(
                f"<span class='series-key'><span class='series-swatch' style='background:{color}'></span>"
                f"{html.escape(node)}</span>"
            )
        more = (
            f"<span class='muted'>+{total - len(lines)} more nodes in the viewer</span>" if total > len(lines) else ""
        )
        title = f"{name} by {x_label}" if x_label else name
        return (
            f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
            f"<div class='series-legend'>{''.join(legend)}{more}</div>"
            f"<svg viewBox='0 0 400 214' role='img' aria-label='{html.escape(title)}' "
            f"style='width:100%;fill:var(--muted);font-size:11px'>{''.join(svg)}</svg>"
            f"<div class='chart-unit'>{html.escape(unit)}</div></div>"
        )

    @staticmethod
    def _heatmap_panel(heat):
        rows = heat.get("rows") or []
        cols = heat.get("cols") or []
        values = heat.get("values") or []
        if not rows or not cols:
            return ""
        numbers = [
            value
            for row in values
            if isinstance(row, list)
            for value in row
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        lo = min(numbers) if numbers else 0.0
        hi = max(numbers) if numbers else 1.0
        threshold = heat.get("threshold")
        try:
            threshold = float(threshold) if threshold is not None else None
        except (TypeError, ValueError):
            threshold = None
        direction = str(heat.get("direction") or "")
        head = "<tr><th></th>" + "".join(f"<th>{html.escape(str(col))}</th>" for col in cols) + "</tr>"
        body = []
        for index, row_name in enumerate(rows):
            row_values = values[index] if index < len(values) and isinstance(values[index], list) else []
            cells = []
            for col_index, _col in enumerate(cols):
                value = row_values[col_index] if col_index < len(row_values) else None
                css, style = _heat_style(value, threshold, direction, lo, hi)
                shown = "—" if value is None else fmt_num(value)
                cells.append(f"<td class='hm-cell {css}' style='{style}'>{html.escape(shown)}</td>")
            body.append(f"<tr><th>{html.escape(str(row_name))}</th>{''.join(cells)}</tr>")
        title = str(heat.get("name") or "heatmap")
        if heat.get("node"):
            title = f"{title} · {heat['node']}"
        unit = str(heat.get("unit") or "")
        unit_html = f"<div class='chart-unit'>{html.escape(unit)}</div>" if unit else ""
        return (
            f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
            f"<div class='hm-wrap'><table class='hm-table'>{head}{''.join(body)}</table></div>{unit_html}</div>"
        )

    @staticmethod
    def render_launch(_payload: dict, _card: dict, data: Any) -> str:
        return render_launch_panel_html(data or {})

    def render_line_chart(self, _payload: dict, card: dict, data: Any) -> str:
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
        entries = sorted(entries, key=lambda e: str(e.get("label", "")) if isinstance(e, dict) else "")
        total = len(entries)
        max_series = series_cfg.get("max_series", self.DEFAULT_MAX_LINE_CHART_SERIES)
        truncated = bool(max_series) and total > max_series
        if truncated:
            entries = entries[:max_series]
        parts = []
        title = card.get("title") or y_field
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            points = entry.get("points") or []
            label = entry.get("label") or title
            part = self._charts.render_series_chart(
                str(label),
                points,
                series_cfg.get("unit") or "GB/s",
                x_label=series_cfg.get("x_label") or "{x}",
            )
            if part:
                parts.append(part)
        if not parts:
            return "<p class='muted'>No series data.</p>"
        banner = (
            f"<p class='muted'>Showing {len(parts)} of {total} series charts. "
            "Full results remain in the results table and JSON export.</p>"
            if truncated
            else ""
        )
        return f"{banner}<div class='chart-grid'>{''.join(parts)}</div>"

    @staticmethod
    def render_heatmap(_payload: dict, card: dict, data: Any) -> str:
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

    def card_renderers(self) -> dict[str, Any]:
        return {
            "run_card": self.render_run_card,
            "lifecycle_timeline": self.render_lifecycle,
            "sweep_analytics": self.render_sweep_analytics,
            "gate_matrix": self.render_gate_matrix,
            "gate_heatmap": self.render_gate_heatmap,
            "sweep_cell_cards": self.render_cell_cards,
            "table": self.render_table,
            "status_matrix": self.render_status_matrix,
            "status_overview": self.render_status_overview,
            "metric_charts": self.render_metric_charts,
            "launch_panel": self.render_launch,
            "line_chart": self.render_line_chart,
            "heatmap": self.render_heatmap,
        }

    def render_card(self, payload: dict, card: dict) -> tuple[str, str, bool]:
        """Return (section_id, html, include_in_nav)."""
        card_type = card.get("type")
        renderer = self.card_renderers().get(str(card_type))
        if renderer is None:
            return "", f"<p class='muted'>Unknown card type: {html.escape(str(card_type))}</p>", False

        bind = card.get("bind") or card_type
        data = resolve_bind(payload, bind) if "." in bind else payload.get(bind)
        if _suppressed_when_empty(card, data):
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
            return (
                "launch",
                f"<section class='panel' id='launch'><h2>{html.escape(title)}</h2>{html_body}</section>",
                True,
            )
        wrapped = f"<section class='panel' id='{html.escape(section_id)}'><h2>{html.escape(title)}</h2>"
        if card_type == "gate_matrix":
            wrapped += f"<div class='matrix-wrap'>{html_body}"
            return section_id, wrapped, True
        if card_type in ("table", "sweep_cell_cards"):
            wrap_class = "results-wrap" if card_type == "table" else ""
            inner = f"<div class='{wrap_class}'>{html_body}</div>" if wrap_class else html_body
            return section_id, f"{wrapped}{inner}</section>", True
        if card_type == "status_matrix":
            return section_id, f"{wrapped}{html_body}</section>", True
        if card_type == "lifecycle_timeline":
            return section_id, f"{wrapped}<div class='tl-row'>{html_body}</div></section>", True
        if card_type == "run_card":
            return section_id, f"{wrapped}{html_body}</section>", True
        if card_type == "gate_heatmap":
            return section_id, html_body, True
        return section_id, f"{wrapped}{html_body}</section>", True

    @staticmethod
    def close_gate_matrix_section(section_html: str, heatmap_html: str) -> str:
        if not section_html:
            return heatmap_html
        if heatmap_html and "</section>" not in section_html:
            return section_html + heatmap_html + "</div></section>"
        if heatmap_html and section_html.endswith("</section>"):
            return section_html[: -len("</section>")] + heatmap_html + "</div></section>"
        return section_html


_DEFAULT_RENDERER = DeckCardRenderer()

CARD_RENDERERS = _DEFAULT_RENDERER.card_renderers()


def render_card(payload: dict, card: dict) -> tuple[str, str, bool]:
    return _DEFAULT_RENDERER.render_card(payload, card)


def close_gate_matrix_section(section_html: str, heatmap_html: str) -> str:
    return DeckCardRenderer.close_gate_matrix_section(section_html, heatmap_html)
