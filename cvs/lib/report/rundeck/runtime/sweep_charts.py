'''Static sweep bar charts for Run Deck runtime.'''

from __future__ import annotations

import html
from typing import List, Tuple

from cvs.lib.report.formatting import fmt_num


def _chart_group_keys(chart_cfg: list, chart_series: dict) -> List[Tuple[str, str, str]]:
    seen: set[Tuple[str, str]] = set()
    keys: List[Tuple[str, str, str]] = []
    for chart in chart_cfg:
        for entry in chart_series.get(chart["suffix"], []):
            if not isinstance(entry, dict):
                continue
            key = (str(entry["isl"]), str(entry["osl"]))
            if key in seen:
                continue
            seen.add(key)
            keys.append((entry["isl"], entry["osl"], entry.get("label") or f"ISL={key[0]} \u00b7 OSL={key[1]}"))
    return keys


def _chart_y_ticks(min_val: float, max_val: float, *, count: int = 5) -> List[float]:
    if count < 2:
        return [max_val]
    span = max_val - min_val
    if span <= 0:
        return [max_val]
    return [min_val + span * i / (count - 1) for i in range(count)]


def _chart_display_scale(min_val: float, max_val: float) -> Tuple[float, float, List[float]]:
    if max_val > min_val:
        domain_min, domain_max = min_val, max_val
    else:
        pad = max(abs(max_val) * 0.08, 1.0)
        domain_min, domain_max = max_val - pad, max_val + pad
    ticks = _chart_y_ticks(domain_min, domain_max)
    return domain_min, domain_max, ticks


def _chart_value_pct(val: float, domain_min: float, domain_max: float) -> float:
    span = domain_max - domain_min or 1.0
    return max(0.0, min(100.0, 100.0 * (val - domain_min) / span))


def _bar_height_pct(val: float, min_val: float, max_val: float) -> float:
    domain_min, domain_max, _ = _chart_display_scale(min_val, max_val)
    return max(8.0, _chart_value_pct(val, domain_min, domain_max))


def render_bar_chart(
    title: str,
    points: List[Tuple[int, float]],
    unit: str,
    *,
    accent: str = "accent",
) -> str:
    if len(points) < 2:
        return ""
    values = [p[1] for p in points]
    max_val = max(values) or 1.0
    min_val = min(values) or 0.0
    domain_min, domain_max, ticks = _chart_display_scale(min_val, max_val)
    y_labels = "".join(
        f"<span class='chart-ylbl' style='bottom:{_chart_value_pct(t, domain_min, domain_max):.2f}%'>"
        f"{html.escape(fmt_num(t))}</span>"
        for t in ticks
    )
    grid = "".join(
        f"<span class='chart-hline' style='bottom:{_chart_value_pct(t, domain_min, domain_max):.2f}%'></span>"
        for t in ticks
    )
    bars = []
    x_labels = []
    for conc, val in points:
        h = _bar_height_pct(val, min_val, max_val)
        tip = html.escape(f"C={conc}: {fmt_num(val)} {unit}".strip())
        bars.append(
            f"<div class='chart-col'>"
            f"<div class='chart-bar chart-bar-{accent} chart-has-tip' style='height:{h:.1f}%' "
            f"data-tip='{tip}' tabindex='0' role='img' aria-label='{tip}'></div></div>"
        )
        x_labels.append(f"<span class='chart-xlbl'>C={conc}</span>")
    return (
        f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
        f"<div class='chart-viz'>"
        f"<div class='chart-ywrap'><div class='chart-ylabels'>{y_labels}</div></div>"
        f"<div class='chart-main'>"
        f"<div class='chart-plotbox'><div class='chart-hgrid' aria-hidden='true'>{grid}</div>"
        f"<div class='chart-bars'>{''.join(bars)}</div></div>"
        f"<div class='chart-xrow'>{''.join(x_labels)}</div></div></div>"
        f"<div class='chart-unit'>{html.escape(unit)}</div></div>"
    )


def render_sweep_charts_html(chart_cfg: list, chart_series: dict) -> str:
    chart_accent = ("accent", "accent2", "accent3")
    group_keys = _chart_group_keys(chart_cfg, chart_series)
    chart_sections = []
    for isl, osl, label in group_keys:
        chart_parts = []
        for idx, chart in enumerate(chart_cfg):
            entry = next(
                (
                    e
                    for e in chart_series.get(chart["suffix"], [])
                    if isinstance(e, dict) and str(e.get("isl")) == str(isl) and str(e.get("osl")) == str(osl)
                ),
                None,
            )
            if not entry:
                continue
            part = render_bar_chart(chart["title"], entry["points"], chart["unit"], accent=chart_accent[idx % 3])
            if part:
                chart_parts.append(part)
        if not chart_parts:
            continue
        title_html = f"<h3 class='chart-group-title'>{html.escape(label)}</h3>" if len(group_keys) > 1 else ""
        chart_sections.append(
            f"<div class='chart-group'>{title_html}<div class='chart-grid'>{''.join(chart_parts)}</div></div>"
        )
    return (
        "".join(chart_sections)
        if chart_sections
        else "<p class='muted'>Concurrency charts need two or more points per sweep shape.</p>"
    )


def render_series_chart_html(title: str, points: list, unit: str) -> str:
    normalized = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            normalized.append((p[0], p[1]))
    return render_bar_chart(title, normalized, unit, accent="accent2")
