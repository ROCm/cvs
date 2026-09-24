'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Static lm_loss vs step line chart for Run Deck training cards.
'''

from __future__ import annotations

import html

SERIES_COLORS = ("#ff6b35", "#6b9fff", "#c77dff", "#3dd68c", "#f0c040", "#ff5c6a")


def loss_chart_css() -> str:
    return """
.loss-chart svg { display: block; width: 100%; height: auto; }
.loss-grid { stroke: rgba(42, 47, 61, 0.95); stroke-width: 1; }
.loss-axis { stroke: rgba(154, 163, 181, 0.55); stroke-width: 1; }
.loss-line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.loss-ylbl, .loss-xlbl, .loss-axis-title { fill: var(--muted); font-size: 11px; }
.loss-ylbl { text-anchor: end; }
.loss-xlbl, .loss-axis-title { text-anchor: middle; }
.loss-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; margin-top: 0.75rem;
  font-size: 0.75rem; color: var(--muted); }
.loss-key { display: inline-flex; align-items: center; gap: 0.35rem; }
.loss-swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
"""


class LossChartRenderer:
    """Render sampled ``loss_curve`` cell points as an inline SVG line chart."""

    _WIDTH = 760
    _HEIGHT = 282
    _LEFT = 58
    _RIGHT = 744
    _TOP = 16
    _BOTTOM = 234
    _Y_TICKS = 5
    _X_TICKS = 5

    @staticmethod
    def _points(curve):
        points = []
        for entry in curve or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            try:
                points.append((float(entry[0]), float(entry[1])))
            except (TypeError, ValueError):
                continue
        return sorted(points)

    def build_series(self, cells):
        """Return one drawable series per cell that carries at least two loss points."""
        series = []
        for cell in cells or []:
            points = self._points(cell.get("loss_curve"))
            if len(points) < 2:
                continue
            label = cell.get("subtitle") or cell.get("label") or cell.get("cell_id") or "lm_loss"
            series.append(
                {
                    "label": str(label),
                    "points": points,
                    "color": SERIES_COLORS[len(series) % len(SERIES_COLORS)],
                }
            )
        return series

    def _x(self, value, lo, hi):
        span = (hi - lo) or 1.0
        return self._LEFT + (self._RIGHT - self._LEFT) * (value - lo) / span

    def _y(self, value, lo, hi):
        span = (hi - lo) or 1.0
        return self._BOTTOM - (self._BOTTOM - self._TOP) * (value - lo) / span

    @staticmethod
    def _domain(values):
        lo, hi = min(values), max(values)
        if hi > lo:
            return lo, hi
        pad = max(abs(hi) * 0.05, 0.1)
        return hi - pad, hi + pad

    def _y_axis(self, lo, hi):
        parts = []
        for i in range(self._Y_TICKS):
            value = lo + (hi - lo) * i / (self._Y_TICKS - 1)
            y = self._y(value, lo, hi)
            parts.append(f"<line class='loss-grid' x1='{self._LEFT}' y1='{y:.1f}' x2='{self._RIGHT}' y2='{y:.1f}'/>")
            parts.append(f"<text class='loss-ylbl' x='{self._LEFT - 8}' y='{y + 3.5:.1f}'>{value:.2f}</text>")
        return "".join(parts)

    def _x_axis(self, lo, hi):
        parts = []
        for i in range(self._X_TICKS):
            value = lo + (hi - lo) * i / (self._X_TICKS - 1)
            x = self._x(value, lo, hi)
            parts.append(f"<text class='loss-xlbl' x='{x:.1f}' y='{self._BOTTOM + 18}'>{value:,.0f}</text>")
        mid_x = (self._LEFT + self._RIGHT) / 2
        parts.append(f"<text class='loss-axis-title' x='{mid_x:.1f}' y='{self._HEIGHT - 6}'>step</text>")
        return "".join(parts)

    def _frame(self):
        return (
            f"<line class='loss-axis' x1='{self._LEFT}' y1='{self._TOP}' "
            f"x2='{self._LEFT}' y2='{self._BOTTOM}'/>"
            f"<line class='loss-axis' x1='{self._LEFT}' y1='{self._BOTTOM}' "
            f"x2='{self._RIGHT}' y2='{self._BOTTOM}'/>"
        )

    def render(self, cells) -> str:
        series = self.build_series(cells)
        if not series:
            return ""
        x_lo, x_hi = self._domain([x for s in series for x, _ in s["points"]])
        y_lo, y_hi = self._domain([y for s in series for _, y in s["points"]])

        lines = []
        legend = []
        for entry in series:
            coords = " ".join(
                f"{self._x(x, x_lo, x_hi):.1f},{self._y(y, y_lo, y_hi):.1f}" for x, y in entry["points"]
            )
            lines.append(f"<polyline class='loss-line' stroke='{entry['color']}' points='{coords}'/>")
            legend.append(
                f"<span class='loss-key'><span class='loss-swatch' style='background:{entry['color']}'></span>"
                f"{html.escape(entry['label'])}</span>"
            )

        return (
            "<p class='chart-sweep-hint'>Sampled lm_loss vs step &middot; one series per sweep cell.</p>"
            f"<div class='loss-chart'><svg viewBox='0 0 {self._WIDTH} {self._HEIGHT}' role='img' "
            f"aria-label='lm_loss versus step'>"
            f"{self._y_axis(y_lo, y_hi)}{self._frame()}{''.join(lines)}{self._x_axis(x_lo, x_hi)}"
            f"</svg><div class='loss-legend'>{''.join(legend)}</div></div>"
        )


_DEFAULT_RENDERER = LossChartRenderer()


def render_loss_chart_html(cells) -> str:
    return _DEFAULT_RENDERER.render(cells)
