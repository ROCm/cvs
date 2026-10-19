'''The sweep and health viewers share one table pager.'''

import unittest

from cvs.lib.report.render.panel_shell import TABLE_PAGE_SIZE
from cvs.lib.report.viewer.paging import page_tables_js


class TestPageTablesJs(unittest.TestCase):
    def test_page_size_comes_from_the_static_deck(self):
        script = page_tables_js()
        self.assertEqual(script.count("function pageLongTables"), 1)
        self.assertIn(f"var pageSize = {TABLE_PAGE_SIZE};", script)
        self.assertNotIn("__TABLE_PAGE_SIZE__", script)


if __name__ == "__main__":
    unittest.main()
