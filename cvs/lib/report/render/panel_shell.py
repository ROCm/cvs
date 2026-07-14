'''Shared results-table HTML for static reports.'''

from __future__ import annotations

import html
from typing import Any, Iterable, Sequence


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
