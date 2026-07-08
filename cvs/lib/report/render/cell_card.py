'''Per-cell card HTML used in suite reports and pytest-html row extras.'''

from __future__ import annotations

import html
from typing import Literal, Mapping, Optional

from cvs.lib.report.formatting import fmt_num, pytest_row_link_html

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


def _cell_card_css(*, theme: Literal["pytest", "report"] = "pytest", compact: bool = False) -> str:
    t = _THEME_TOKENS[theme]
    text_rule = f" color: {t['text']};" if t["text"] != "inherit" else ""
    card_font = t["card_font"]
    grid_rule = (
        ".cells { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }\n"
        if theme == "report"
        else ""
    )
    compact_rules = ""
    if compact:
        compact_rules = """
.cell-card-compact { padding: 0.85rem 1rem; gap: 0.5rem; font-size: 0.85rem; }
.cell-card-compact .headline { font-size: 1.5rem; }
"""
    chip_margin = " margin-right: 0.25rem;" if theme == "pytest" else ""
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


def cell_card_css(*, compact: bool = False) -> str:
    return _cell_card_css(theme="pytest", compact=compact)


def cell_card_report_css() -> str:
    """Cell card rules using report theme CSS variables (static inference HTML)."""
    return _cell_card_css(theme="report")


def render_cell_lifecycle_html(
    cell_lifecycle: Mapping[str, float],
    labels: tuple[str, ...],
) -> str:
    if not cell_lifecycle:
        return ""
    total = sum(cell_lifecycle.values()) or 1.0
    parts = []
    for lbl in labels:
        sec = cell_lifecycle.get(lbl, 0.0)
        if sec <= 0:
            continue
        pct = 100.0 * sec / total
        parts.append(
            f"<div class='cell-mini-seg' style='flex-grow:{pct:.2f}'>"
            f"<span class='tl-lbl'>{html.escape(lbl.replace('_', ' '))}</span>"
            f"<span class='tl-val'>{sec:.1f}s</span></div>"
        )
    if not parts:
        return ""
    return f"<div class='cell-mini-tl'>{''.join(parts)}</div>"


def _tier_chip(status: str, label: str) -> str:
    return f'<span class="chip chip-{html.escape(status)}">{html.escape(label)}</span>'


def render_cell_card_html(
    cell: dict,
    *,
    tier_order: tuple[str, ...],
    headline_metric: str,
    enforce: bool,
    cell_lifecycle_labels: tuple[str, ...],
    compact: bool = False,
    highlight_metric: Optional[str] = None,
    pytest_html_basename: Optional[str] = None,
) -> str:
    tier_chips = "".join(
        _tier_chip(cell["tiers"].get(t, "na"), t) for t in tier_order
    )
    metric_rows = []
    for m in cell["metrics"]:
        if m["actual"] is None:
            continue
        row_cls = "metric-row"
        if highlight_metric and m["metric"] == highlight_metric:
            row_cls += " metric-row-highlight"
        bar = ""
        if m["bar_pct"] is not None:
            bar = (
                f"<div class='bar-track'><div class='bar-fill bar-{m['status']}' "
                f"style='width:{m['bar_pct']:.0f}%'></div></div>"
            )
        target = ""
        if m["spec"] is not None:
            gate_label = "gate" if enforce else "floor"
            target = f"<span class='target'>{gate_label} {fmt_num(m['spec'].get('value'))}</span>"
        margin = ""
        margin_col = ""
        if m.get("margin"):
            cls = "margin-fail" if m["status"] == "fail" else "margin"
            margin = f"<span class='{cls}'>{html.escape(m['margin'])}</span>"
            margin_col = f"<span class='metric-margin-col {cls}'>{html.escape(m['margin'])}</span>"
        na_margin = "<span class='metric-margin-col'>\u2014</span>"
        if compact:
            metric_rows.append(
                f"<div class='{row_cls}'><div class='metric-label'>{html.escape(m['label'])}</div>"
                f"<div class='metric-val'>{fmt_num(m['actual'])} {html.escape(m['unit'])}</div>"
                f"{bar}{target}{margin}</div>"
            )
        else:
            metric_rows.append(
                f"<div class='{row_cls} metric-row-grid'>"
                f"<div class='metric-label'>{html.escape(m['label'])}</div>"
                f"<div class='metric-val'>{fmt_num(m['actual'])} {html.escape(m['unit'])}</div>"
                f"{margin_col or na_margin}"
                f"<div class='metric-extra' style='grid-column:1/-1'>{bar}{target}</div></div>"
            )

    headline = next((m for m in cell["metrics"] if m["metric"] == headline_metric), None)
    headline_val = fmt_num(headline["actual"]) if headline else "\u2014"
    headline_margin_html = ""
    if headline and headline.get("margin"):
        hm_cls = "headline-margin-fail" if headline.get("status") == "fail" else "headline-margin"
        headline_margin_html = (
            f"<div class='{hm_cls}'>{html.escape(headline['margin'])}</div>"
        )
    mini_tl = render_cell_lifecycle_html(cell.get("cell_lifecycle") or {}, cell_lifecycle_labels)
    card_cls = "cell-card cell-card-compact" if compact else "cell-card"
    host_line = f" &middot; {html.escape(str(cell['host']))}" if cell.get("show_host_in_label") else ""
    pytest_nid = cell.get("pytest_metrics_nodeid") or cell.get("pytest_inference_nodeid")
    pytest_link = ""
    if pytest_html_basename and pytest_nid:
        pytest_link = " &middot; " + pytest_row_link_html(pytest_html_basename, pytest_nid)

    return (
        f"<article class='{card_cls}'>"
        f"<header><div class='cell-title'>{html.escape(str(cell['policy']))}</div>"
        f"<div class='cell-sub'>ISL={cell['isl']} OSL={cell['osl']} &middot; C={cell['concurrency']}</div></header>"
        f"{mini_tl if not compact else ''}"
        f"<div class='headline'>{headline_val}<span class='headline-unit'>tok/s</span></div>"
        f"{headline_margin_html}"
        f"<div class='tiers'>{tier_chips}</div>"
        f"<div class='metrics'>{''.join(metric_rows)}</div>"
        f"<footer class='cell-foot'>{html.escape(cell['cell_id'])}{host_line}{pytest_link}</footer>"
        f"</article>"
    )
