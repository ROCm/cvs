'''Per-cell card HTML used in suite reports and pytest-html row extras.'''

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Literal, Optional

from cvs.lib.report.formatting import fmt_num, pytest_row_link_html


@dataclass(frozen=True)
class CellCardConfig:
    """Configuration for cell card rendering (immutable)."""

    tier_order: tuple[str, ...] = ()
    headline_metric: str = ""
    enforce: bool = False
    cell_lifecycle_labels: tuple[str, ...] = ()
    compact: bool = False
    highlight_metric: Optional[str] = None
    pytest_html_basename: Optional[str] = None
    theme: Literal["pytest", "report"] = "pytest"


_THEME_TOKENS: dict[str, dict[str, str]] = {
    "pytest": {
        "card_bg": "#1a1d27",
        "border": "#2a2f3d",
        "text": "#e8eaef",
        "accent": "#ff6b35",
        "muted": "#9aa3b5",
        "pass": "#3dd68c",
        "fail": "#ff5c6a",
        "record": "#6b9fff",
        "na": "#5c6370",
        "card_font": 'font-family: "Segoe UI", system-ui, sans-serif; color: #e8eaef;',
    },
    "report": {
        "card_bg": "var(--panel)",
        "border": "var(--border)",
        "text": "inherit",
        "accent": "var(--accent)",
        "muted": "var(--muted)",
        "pass": "var(--pass)",
        "fail": "var(--fail)",
        "record": "var(--record)",
        "na": "var(--na)",
        "card_font": "",
    },
}


class CellCardRenderer:
    """Renders cell cards with consistent configuration, eliminating parameter passing."""

    def __init__(self, config: CellCardConfig):
        self.config = config
        self._theme_tokens = _THEME_TOKENS[config.theme]
        self._cell = None  # Current cell being rendered

    def render(self, cell: dict) -> str:
        """Render a complete cell card HTML."""
        self._cell = cell  # Store cell data for method access
        try:
            return (
                f"<article class='{self._card_class()}'>"
                f"{self._render_header()}"
                f"{self._render_timeline()}"
                f"{self._render_headline()}"
                f"{self._render_tiers()}"
                f"{self._render_metrics()}"
                f"{self._render_footer()}"
                f"</article>"
            )
        finally:
            self._cell = None  # Clear cell data after rendering

    def get_css(self) -> str:
        """Generate CSS for this renderer configuration."""
        return self._generate_css()

    def _card_class(self) -> str:
        return "cell-card cell-card-compact" if self.config.compact else "cell-card"

    def _render_header(self) -> str:
        return (
            f"<header><div class='cell-title'>{html.escape(str(self._cell['policy']))}</div>"
            f"<div class='cell-sub'>ISL={self._cell['isl']} OSL={self._cell['osl']} &middot; C={self._cell['concurrency']}</div></header>"
        )

    def _render_timeline(self) -> str:
        if self.config.compact:
            return ""

        cell_lifecycle = self._cell.get("cell_lifecycle") or {}
        if not cell_lifecycle:
            return ""

        total = sum(cell_lifecycle.values()) or 1.0
        parts = []

        for lbl in self.config.cell_lifecycle_labels:
            sec = cell_lifecycle.get(lbl, 0.0)
            if sec <= 0:
                continue
            pct = 100.0 * sec / total
            parts.append(
                f"<div class='cell-mini-seg' style='flex-grow:{pct:.2f}'>"
                f"<span class='tl-lbl'>{html.escape(lbl.replace('_', ' '))}</span>"
                f"<span class='tl-val'>{sec:.1f}s</span></div>"
            )

        return f"<div class='cell-mini-tl'>{''.join(parts)}</div>" if parts else ""

    def _render_headline(self) -> str:
        headline = next((m for m in self._cell["metrics"] if m["metric"] == self.config.headline_metric), None)
        headline_val = fmt_num(headline["actual"]) if headline else "\u2014"

        headline_margin_html = ""
        if headline and headline.get("margin"):
            hm_cls = "headline-margin-fail" if headline.get("status") == "fail" else "headline-margin"
            headline_margin_html = f"<div class='{hm_cls}'>{html.escape(headline['margin'])}</div>"

        return (
            f"<div class='headline'>{headline_val}<span class='headline-unit'>tok/s</span></div>{headline_margin_html}"
        )

    def _render_tiers(self) -> str:
        tier_chips = "".join(self._tier_chip(self._cell["tiers"].get(t, "na"), t) for t in self.config.tier_order)
        return f"<div class='tiers'>{tier_chips}</div>"

    def _render_metrics(self) -> str:
        metric_rows = []
        for m in self._cell["metrics"]:
            if m["actual"] is None:
                continue

            row_cls = "metric-row"
            if self.config.highlight_metric and m["metric"] == self.config.highlight_metric:
                row_cls += " metric-row-highlight"

            bar = self._render_metric_bar(m)
            target = self._render_metric_target(m)
            margin, margin_col = self._render_metric_margin(m)

            if self.config.compact:
                metric_rows.append(
                    f"<div class='{row_cls}'><div class='metric-label'>{html.escape(m['label'])}</div>"
                    f"<div class='metric-val'>{fmt_num(m['actual'])} {html.escape(m['unit'])}</div>"
                    f"{bar}{target}{margin}</div>"
                )
            else:
                na_margin = "<span class='metric-margin-col'>\u2014</span>"
                metric_rows.append(
                    f"<div class='{row_cls} metric-row-grid'>"
                    f"<div class='metric-label'>{html.escape(m['label'])}</div>"
                    f"<div class='metric-val'>{fmt_num(m['actual'])} {html.escape(m['unit'])}</div>"
                    f"{margin_col or na_margin}"
                    f"<div class='metric-extra' style='grid-column:1/-1'>{bar}{target}</div></div>"
                )

        return f"<div class='metrics'>{''.join(metric_rows)}</div>"

    def _render_footer(self) -> str:
        host_line = f" &middot; {html.escape(str(self._cell['host']))}" if self._cell.get("show_host_in_label") else ""

        pytest_link = ""
        pytest_nid = self._cell.get("pytest_metrics_nodeid") or self._cell.get("pytest_inference_nodeid")
        if self.config.pytest_html_basename and pytest_nid:
            pytest_link = " &middot; " + pytest_row_link_html(self.config.pytest_html_basename, pytest_nid)

        return f"<footer class='cell-foot'>{html.escape(self._cell['cell_id'])}{host_line}{pytest_link}</footer>"

    def _tier_chip(self, status: str, label: str) -> str:
        return f'<span class="chip chip-{html.escape(status)}">{html.escape(label)}</span>'

    def _render_metric_bar(self, metric: dict) -> str:
        if metric["bar_pct"] is None:
            return ""
        return (
            f"<div class='bar-track'><div class='bar-fill bar-{metric['status']}' "
            f"style='width:{metric['bar_pct']:.0f}%'></div></div>"
        )

    def _render_metric_target(self, metric: dict) -> str:
        if metric["spec"] is None:
            return ""
        gate_label = "gate" if self.config.enforce else "floor"
        return f"<span class='target'>{gate_label} {fmt_num(metric['spec'].get('value'))}</span>"

    def _render_metric_margin(self, metric: dict) -> tuple[str, str]:
        if not metric.get("margin"):
            return "", ""

        cls = "margin-fail" if metric["status"] == "fail" else "margin"
        margin = f"<span class='{cls}'>{html.escape(metric['margin'])}</span>"
        margin_col = f"<span class='metric-margin-col {cls}'>{html.escape(metric['margin'])}</span>"
        return margin, margin_col

    def _generate_css(self) -> str:
        """Generate CSS for this renderer's theme and configuration."""
        t = self._theme_tokens
        text_rule = f" color: {t['text']};" if t["text"] != "inherit" else ""
        card_font = t["card_font"]
        grid_rule = (
            ".cells { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }\n"
            if self.config.theme == "report"
            else ""
        )
        compact_rules = ""
        if self.config.compact:
            compact_rules = """
.cell-card-compact { padding: 0.85rem 1rem; gap: 0.5rem; font-size: 0.85rem; }
.cell-card-compact .headline { font-size: 1.5rem; }
"""
        chip_margin = " margin-right: 0.25rem;" if self.config.theme == "pytest" else ""
        return (
            grid_rule
            + f"""
.cell-card {{ background: {t['card_bg']}; border: 1px solid {t['border']}; border-radius: 12px;
  padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem;{text_rule} {card_font} }}
{compact_rules}.headline {{ font-size: 2.25rem; font-weight: 700; color: {t['accent']}; line-height: 1; }}
.headline-unit {{ font-size: 0.9rem; color: {t['muted']}; margin-left: 0.35rem; }}
.cell-sub {{ font-size: 0.8rem; color: {t['muted']}; }}
.cell-mini-tl {{ display: flex; gap: 4px; min-height: 36px; border-radius: 6px; overflow: hidden; font-size: 0.65rem; }}
.cell-mini-seg {{ display: flex; flex-direction: column; justify-content: center; align-items: center;
  padding: 0.25rem; background: rgba(255,255,255,0.05); min-width: 40px; }}
.tl-lbl {{ font-size: 0.65rem; color: {t['muted']}; text-align: center; }}
.tl-val {{ font-size: 0.8rem; font-weight: 600; color: {t['accent']}; }}
.chip {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 0.2rem 0.5rem;
  border-radius: 999px; display: inline-block;{chip_margin} }}
.chip-pass {{ background: rgba(61,214,140,0.15); color: {t['pass']}; }}
.chip-fail {{ background: rgba(255,92,106,0.15); color: {t['fail']}; }}
.chip-record {{ background: rgba(107,159,255,0.12); color: {t['record']}; }}
.chip-na {{ background: rgba(92,99,112,0.2); color: {t['na']}; }}
.bar-track {{ height: 4px; background: {t['border']}; border-radius: 2px; margin-top: 0.35rem; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 2px; }}
.bar-pass {{ background: {t['pass']}; }} .bar-fail {{ background: {t['fail']}; }} .bar-record {{ background: {t['record']}; }}
.target, .metric-label, .cell-foot {{ font-size: 0.7rem; color: {t['muted']}; }}
.metric-row-highlight {{ outline: 1px solid {t['record']}; border-radius: 6px; padding: 0.25rem; }}
.margin {{ font-size: 0.7rem; color: {t['pass']}; display: block; }}
.margin-fail {{ color: {t['fail']}; }}
.headline-margin {{ font-size: 0.85rem; color: {t['pass']}; margin-top: 0.15rem; }}
.headline-margin-fail {{ color: {t['fail']}; }}
.metric-val {{ font-weight: 600; }}
.metric-margin-col {{ font-size: 0.75rem; color: {t['muted']}; min-width: 5rem; text-align: right; }}
.metric-row-grid {{ display: grid; grid-template-columns: 1fr auto auto; gap: 0.35rem 0.75rem; align-items: baseline; }}
"""
        )
