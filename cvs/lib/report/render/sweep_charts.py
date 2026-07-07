'''Cross-shape sweep comparison charts (grouped bars + multi-line trends).'''

from __future__ import annotations

import html
from typing import List

from cvs.lib.report.formatting import fmt_num

_CHART_ACCENTS = ("accent", "accent2", "accent3")


def chart_tooltip_css() -> str:
    return """
.chart-has-tip { position: relative; cursor: pointer; }
.chart-has-tip::before {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  padding: 0.35rem 0.55rem;
  background: #2a3142;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  z-index: 20;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  transition: opacity 0.12s ease, visibility 0.12s ease;
}
.chart-has-tip:hover::before,
.chart-has-tip:focus-visible::before {
  opacity: 1;
  visibility: visible;
}
.chart-bar:hover, .chart-bar-grouped:hover { filter: brightness(1.12); }
.chart-line-point { cursor: pointer; }
.chart-line-point:hover circle:last-child { r: 5; }
.chart-hover-hint { margin: -0.35rem 0 0.75rem; font-size: 0.75rem; color: var(--muted); }
.chart-legend-shared { margin-top: 0.25rem; margin-bottom: 0.5rem; }
"""


def sweep_comparison_css() -> str:
    return chart_tooltip_css() + """
.sweep-subhead { margin: 1.25rem 0 0.75rem; font-size: 0.95rem; font-weight: 600; color: var(--text); }
.sweep-subhead:first-of-type { margin-top: 0.5rem; }
.chart-clusters { display: flex; justify-content: center; align-items: flex-end; gap: 1.5rem;
  min-height: 160px; padding-top: 0.5rem; }
.chart-cluster { display: flex; flex-direction: column; align-items: center; gap: 0.35rem; }
.chart-cluster-bars { display: flex; align-items: flex-end; justify-content: center; gap: 6px;
  height: 140px; width: 100%; }
.chart-bar-grouped { width: 32px; min-height: 4px; border-radius: 6px 6px 2px 2px; position: relative; }
.chart-legend { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; margin-top: 0.75rem;
  font-size: 0.75rem; color: var(--muted); }
.chart-legend-item { display: inline-flex; align-items: center; gap: 0.35rem; }
.chart-legend-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
.chart-line-wrap { padding: 0.5rem 0; }
.chart-line-svg { width: 100%; max-width: 420px; height: auto; display: block; margin: 0 auto; }
.chart-line-legend { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; justify-content: center;
  margin-top: 0.5rem; font-size: 0.75rem; color: var(--muted); }
"""


def _numeric_values(series: list) -> List[float]:
    out: List[float] = []
    for s in series:
        for v in s.get("values") or []:
            if v is not None:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    continue
    return out


def _series_legend_html(series: list, *, line_colors: bool = False) -> str:
    items: List[str] = []
    for i, s in enumerate(series):
        accent = _CHART_ACCENTS[i % len(_CHART_ACCENTS)]
        if line_colors:
            swatch = f"<span class='chart-legend-swatch' style='background:{_line_color(accent)}'></span>"
        else:
            swatch = f"<span class='chart-legend-swatch chart-bar-{accent}'></span>"
        items.append(
            f"<span class='chart-legend-item'>{swatch}{html.escape(str(s.get('label', '')))}</span>"
        )
    return "".join(items)


def render_grouped_comparison_chart(entry: dict, *, show_legend: bool = True) -> str:
    """Grouped bars: x-axis clusters = concurrency, bars = ISL/OSL shapes."""
    title = entry.get("title") or "Metric"
    unit = entry.get("unit") or ""
    concurrencies = entry.get("concurrencies") or []
    series = entry.get("series") or []
    if len(concurrencies) < 2 or len(series) < 2:
        return ""

    numeric = _numeric_values(series)
    if len(numeric) < 2:
        return ""

    max_val = max(numeric) or 1.0
    clusters: List[str] = []
    for ci, conc in enumerate(concurrencies):
        bars: List[str] = []
        for si, s in enumerate(series):
            values = s.get("values") or []
            if ci >= len(values):
                continue
            val = values[ci]
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            h = max(12.0, 100.0 * fval / max_val)
            accent = _CHART_ACCENTS[si % len(_CHART_ACCENTS)]
            tip = html.escape(
                f"{s.get('label', '')} · C={conc}: {fmt_num(fval)} {unit}".strip()
            )
            bars.append(
                f"<div class='chart-bar-grouped chart-bar-{accent} chart-has-tip' "
                f"style='height:{h:.0f}%' data-tip='{tip}' tabindex='0' "
                f"role='img' aria-label='{tip}'></div>"
            )
        if not bars:
            continue
        clusters.append(
            f"<div class='chart-cluster'><div class='chart-cluster-bars'>{''.join(bars)}</div>"
            f"<div class='chart-x'>C={conc}</div></div>"
        )

    if not clusters:
        return ""

    legend = f"<div class='chart-legend'>{_series_legend_html(series)}</div>" if show_legend else ""
    return (
        f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
        f"<div class='chart-clusters'>{''.join(clusters)}</div>"
        f"{legend}"
        f"<div class='chart-unit'>{html.escape(unit)}</div></div>"
    )


def _line_color(accent: str) -> str:
    return {
        "accent": "#ff6b35",
        "accent2": "#6b9fff",
        "accent3": "#c77dff",
    }.get(accent, "#ff6b35")


def render_line_comparison_chart(entry: dict, *, show_legend: bool = True) -> str:
    """Multi-line trend: x-axis = concurrency, one line per ISL/OSL shape."""
    title = entry.get("title") or "Metric"
    unit = entry.get("unit") or ""
    concurrencies = entry.get("concurrencies") or []
    series = entry.get("series") or []
    if len(concurrencies) < 2 or len(series) < 2:
        return ""

    numeric = _numeric_values(series)
    if len(numeric) < 2:
        return ""

    min_val = min(numeric)
    max_val = max(numeric) or 1.0
    span = max_val - min_val or max_val or 1.0

    width, height, pad = 400, 160, 28
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    n = len(concurrencies)
    x_at = lambda i: pad + (plot_w * i / max(n - 1, 1))

    def y_at(val: float) -> float:
        return pad + plot_h - (plot_h * (val - min_val) / span)

    grid_lines = []
    for i in range(n):
        x = x_at(i)
        grid_lines.append(
            f"<line x1='{x:.1f}' y1='{pad}' x2='{x:.1f}' y2='{pad + plot_h}' "
            f"stroke='#2a2f3d' stroke-width='1'/>"
        )

    polylines: List[str] = []
    dots: List[str] = []
    for si, s in enumerate(series):
        accent = _CHART_ACCENTS[si % len(_CHART_ACCENTS)]
        color = _line_color(accent)
        pts: List[str] = []
        values = s.get("values") or []
        for ci, conc in enumerate(concurrencies):
            if ci >= len(values) or values[ci] is None:
                continue
            try:
                fval = float(values[ci])
            except (TypeError, ValueError):
                continue
            x, y = x_at(ci), y_at(fval)
            pts.append(f"{x:.1f},{y:.1f}")
            tip = (
                f"{html.escape(str(s.get('label', '')))} · C={conc}: "
                f"{fmt_num(fval)} {html.escape(unit)}"
            )
            dots.append(
                f"<g class='chart-line-point'>"
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='10' fill='transparent'>"
                f"<title>{tip}</title></circle>"
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}' pointer-events='none'/>"
                f"</g>"
            )
        if len(pts) >= 2:
            polylines.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2' "
                f"points='{' '.join(pts)}'/>"
            )
        return ""

    x_labels = "".join(
        f"<text x='{x_at(i):.1f}' y='{height - 6}' text-anchor='middle' "
        f"fill='#9aa3b5' font-size='11'>C={conc}</text>"
        for i, conc in enumerate(concurrencies)
    )
    svg = (
        f"<svg class='chart-line-svg' viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{html.escape(title)} trend'>"
        f"{''.join(grid_lines)}{''.join(polylines)}{''.join(dots)}{x_labels}</svg>"
    )
    legend = (
        f"<div class='chart-line-legend'>{_series_legend_html(series, line_colors=True)}</div>"
        if show_legend
        else ""
    )
    return (
        f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
        f"<div class='chart-line-wrap'>{svg}</div>"
        f"{legend}"
        f"<div class='chart-unit'>{html.escape(unit)}</div></div>"
    )


def render_comparison_charts_grid(
    chart_comparison: dict,
    *,
    render_fn,
    shared_legend: bool = False,
) -> str:
    parts: List[str] = []
    legend_series: list | None = None
    line_legend = render_fn is render_line_comparison_chart
    for _suffix, entry in sorted(chart_comparison.items()):
        if not isinstance(entry, dict):
            continue
        if legend_series is None and entry.get("series"):
            legend_series = entry.get("series")
        part = render_fn(entry, show_legend=not shared_legend)
        if part:
            parts.append(part)
    if not parts:
        return ""
    shared = ""
    if shared_legend and legend_series:
        legend_html = _series_legend_html(legend_series, line_colors=line_legend)
        shared = f"<div class='chart-legend chart-legend-shared'>{legend_html}</div>"
    return f"{shared}<div class='chart-grid'>{''.join(parts)}</div>"
