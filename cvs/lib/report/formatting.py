'''Shared HTML formatting helpers for suite reports.'''

from __future__ import annotations

import html
import re
from typing import Any


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "\u2014"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(f) >= 1000:
        return f"{f:,.{digits}f}"
    return f"{f:.{digits}f}"


def status_badge_html(status: str) -> str:
    labels = {"pass": "PASS", "fail": "FAIL", "record": "RECORD", "na": "N/A"}
    return (
        f'<span class="status-badge status-{html.escape(status)}">'
        f"{html.escape(labels.get(status, status.upper()))}</span>"
    )


def link_or_text_html(url: str, label: str) -> str:
    if not url:
        return "\u2014"
    safe_url = html.escape(url)
    safe_label = html.escape(label)
    if re.match(r"^https?://", url):
        return f'<a href="{safe_url}" target="_blank" rel="noopener">{safe_label}</a>'
    return f"<span>{safe_label}</span>"


def pytest_row_href(pytest_html_basename: str, nodeid: str) -> str:
    """Fragment link to a pytest-html result row in the sibling report file."""
    if not pytest_html_basename or not nodeid:
        return ""
    from urllib.parse import quote

    return f"{pytest_html_basename}#{quote(nodeid, safe='')}"


def pytest_row_link_html(pytest_html_basename: str, nodeid: str, *, label: str = "pytest row") -> str:
    href = pytest_row_href(pytest_html_basename, nodeid)
    if not href:
        return ""
    return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'
