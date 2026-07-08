'''HTML render primitives shared by suite reports, pytest row extras, and viewers.'''

from cvs.lib.report.render.cell_card import (
    cell_card_css,
    render_cell_card_html,
    render_cell_lifecycle_html,
)

__all__ = [
    "cell_card_css",
    "render_cell_card_html",
    "render_cell_lifecycle_html",
]
