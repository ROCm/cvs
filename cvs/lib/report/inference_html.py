'''Static HTML rendering for inference suite reports (shim to unified Run Deck runtime).'''

from __future__ import annotations

from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.rundeck.runtime.theme import render_launch_panel_html, report_css

__all__ = ["render_report_html", "report_css", "render_launch_panel_html"]


def render_report_html(payload: dict) -> str:
    return render_rundeck_html(payload)
