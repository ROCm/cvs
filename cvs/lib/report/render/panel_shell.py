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
    units=None,
    align=None,
) -> str:
    """``units`` optionally gives a unit per column; it renders beside the header in its own case.

    ``align`` optionally gives "left" or "right" per column, overriding the default column alignment.
    """
    row_list = list(rows)
    if not row_list:
        return f"<p class='muted'>{html.escape(empty_message)}</p>"
    units = list(units or [])
    header_html = "".join(
        f"<th>{html.escape(str(h))}"
        + (f" <span class='th-unit'>({html.escape(str(units[i]))})</span>" if i < len(units) and units[i] else "")
        + "</th>"
        for i, h in enumerate(headers)
    )
    align = list(align or [])
    cell_open = [f"<td class='al-{side}'>" if side in ("left", "right") else "<td>" for side in align]
    body_html = "".join(
        "<tr>"
        + "".join(
            f"{cell_open[i] if i < len(cell_open) else '<td>'}{html.escape(str(v))}</td>" for i, v in enumerate(row)
        )
        + "</tr>"
        for row in row_list
    )
    return (
        f"<table class='{html.escape(table_class)}'><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody></table>"
    )
