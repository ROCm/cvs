'''Table pager shared by the sweep viewer and the health viewer.'''

from pathlib import Path

from cvs.lib.report.render.panel_shell import TABLE_PAGE_SIZE

_SOURCE = Path(__file__).with_name("page_tables.js")


def page_tables_js():
    source = _SOURCE.read_text(encoding="utf-8")
    return source.replace("__TABLE_PAGE_SIZE__", str(TABLE_PAGE_SIZE))
