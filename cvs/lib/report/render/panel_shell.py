'''Shared results-table HTML for static reports.'''

from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

TABLE_PAGE_SIZE = 25


def paged_table_html(table_id, table_class, header_html, body_rows, wrap_class="results-wrap", wrap_style=""):
    """One page of body rows. Shorter tables render without a pager."""
    rows = list(body_rows)
    shown = _hide_overflow_rows(rows, TABLE_PAGE_SIZE)
    table = (
        f"<table id='{html.escape(table_id)}' class='{html.escape(table_class)}'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{''.join(shown)}</tbody></table>"
    )
    style = f" style='{html.escape(wrap_style)}'" if wrap_style else ""
    return (
        f"<div class='{html.escape(wrap_class)}'{style}>{table}</div>"
        f"{_pager_html(table_id, len(rows), TABLE_PAGE_SIZE)}"
    )


def _pager_html(table_id, total, page_size):
    if total <= page_size:
        return ""
    end = min(page_size, total)
    safe_id = html.escape(table_id)
    return (
        f"<div class='rd-pager' data-table='{safe_id}' data-page-size='{page_size}'>"
        f"<button type='button' data-dir='-1'>Previous</button>"
        f"<span class='rd-pager-count'>1\u2013{end} of {total}</span>"
        f"<button type='button' data-dir='1'>Next</button></div>"
        "<script>(function(pager){"
        "var size=Number(pager.getAttribute('data-page-size'));"
        "var rows=document.getElementById(pager.getAttribute('data-table')).tBodies[0].rows;"
        "var page=0;var label=pager.querySelector('.rd-pager-count');"
        "function show(){var start=page*size;var end=Math.min(start+size,rows.length);"
        "for(var i=0;i<rows.length;i++){rows[i].hidden=i<start||i>=start+size;}"
        "label.textContent=(start+1)+'\\u2013'+end+' of '+rows.length;}"
        "pager.addEventListener('click',function(event){"
        "var button=event.target.closest('button');if(!button){return;}"
        "var pages=Math.ceil(rows.length/size);"
        "page=Math.max(0,Math.min(pages-1,page+Number(button.getAttribute('data-dir'))));"
        "show();});"
        "})(document.currentScript.previousElementSibling);</script>"
    )


def _hide_overflow_rows(row_html, page_size):
    if len(row_html) <= page_size:
        return row_html
    paged = []
    for index, row in enumerate(row_html):
        if index >= page_size and row.startswith("<tr"):
            row = "<tr hidden" + row[3:]
        paged.append(row)
    return paged


def results_table_parts(headers, rows, units=None, align=None):
    """Header cells and body rows. A unit sits beside its header; align is left or right per column."""
    units = list(units or [])
    align = list(align or [])
    header_html = "".join(
        f"<th>{html.escape(str(header))}"
        + (f" <span class='th-unit'>({html.escape(str(units[i]))})</span>" if i < len(units) and units[i] else "")
        + "</th>"
        for i, header in enumerate(headers)
    )
    cell_open = [f"<td class='al-{side}'>" if side in ("left", "right") else "<td>" for side in align]
    body = [
        "<tr>"
        + "".join(
            f"{cell_open[i] if i < len(cell_open) else '<td>'}{html.escape(str(value))}</td>"
            for i, value in enumerate(row)
        )
        + "</tr>"
        for row in rows
    ]
    return header_html, body


def render_results_table_html(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    table_class: str = "results-table",
    empty_message: str = "No rows.",
    units=None,
    align=None,
) -> str:
    """``units`` optionally gives a unit per column; it renders beside the header in its own case.

    ``align`` optionally gives "left" or "right" per column, overriding the default column alignment.
    """
    row_list = list(rows)
    if not row_list:
        return f"<p class='muted'>{html.escape(empty_message)}</p>"
    header_html, body = results_table_parts(headers, row_list, units, align)
    return (
        f"<table class='{html.escape(table_class)}'><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )
