'''Shared results-table HTML for static reports.'''

from __future__ import annotations

import html
import re
from typing import Any, Iterable, Sequence

TABLE_PAGE_SIZE = 25
_TABLE_RE = re.compile(r"<table\b([^>]*)>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_PAGER_RE = re.compile(
    r"<div class='rd-pager'[^>]*>.*?</div>\s*<script>\(function\(pager\)\{.*?</script>",
    re.DOTALL,
)


def paged_table_html(table_id, table_class, header_html, body_rows, wrap_class="results-wrap"):
    """One page of body rows. Shorter tables render without a pager."""
    rows = list(body_rows)
    table = (
        f"<table id='{html.escape(table_id)}' class='{html.escape(table_class)}'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    return page_long_tables(f"<div class='{html.escape(wrap_class)}'>{table}</div>")


def page_long_tables(document):
    """Give every table longer than one page the same pager.

    Safe to run on HTML that already has pagers: existing pagers are removed and rebuilt.
    """
    if not document or "<table" not in document.lower():
        return document
    stripped = _PAGER_RE.sub("", document).replace("<tr hidden", "<tr")
    counter = {"n": 0}

    def replace(match):
        attrs = match.group(1) or ""
        header, data = _split_table_rows(match.group(2))
        if len(data) <= TABLE_PAGE_SIZE:
            return match.group(0)
        table_id, attrs = _ensure_table_id(attrs, counter)
        shown = _hide_overflow_rows(data, TABLE_PAGE_SIZE)
        thead = f"<thead>{header}</thead>" if header else ""
        table = f"<table{attrs}>{thead}<tbody>{''.join(shown)}</tbody></table>"
        return table + _pager_html(table_id, len(data), TABLE_PAGE_SIZE)

    return _TABLE_RE.sub(replace, stripped)


def _ensure_table_id(attrs, counter):
    found = re.search(r"""\bid=(['"])([^'"]+)\1""", attrs)
    if found:
        return found.group(2), attrs
    counter["n"] += 1
    table_id = f"rd-paged-{counter['n']}"
    return table_id, f"{attrs} id='{table_id}'"


def _split_table_rows(inner):
    thead = re.search(r"<thead\b[^>]*>(.*?)</thead>", inner, re.IGNORECASE | re.DOTALL)
    tbody = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", inner, re.IGNORECASE | re.DOTALL)
    if thead or tbody:
        header_rows = _TR_RE.findall(thead.group(1)) if thead else []
        data = _TR_RE.findall(tbody.group(1)) if tbody else []
        return (header_rows[0] if header_rows else ""), data
    rows = _TR_RE.findall(inner)
    if rows and re.search(r"<th\b", rows[0], re.IGNORECASE):
        return rows[0], rows[1:]
    return "", rows


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


def render_results_table_html(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    table_class: str = "results-table",
    empty_message: str = "No rows.",
) -> str:
    row_list = list(rows)
    if not row_list:
        return f"<p class='muted'>{html.escape(empty_message)}</p>"
    header_html = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body_html = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in row_list)
    return f"<table class='{html.escape(table_class)}'><tr>{header_html}</tr>{body_html}</table>"
