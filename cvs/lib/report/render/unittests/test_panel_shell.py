'''Unit tests for the shared paged table used by every Run Deck table.'''

import unittest

from cvs.lib.report.render.panel_shell import page_long_tables, paged_table_html


class TestPagedTable(unittest.TestCase):
    def test_short_table_has_no_pager(self):
        html = paged_table_html("rd-sample", "results-table", "<th>Node</th>", ["<tr><td>n1</td></tr>"])
        self.assertNotIn("rd-pager", html)
        self.assertIn("n1", html)
        self.assertIn("<thead>", html)

    def test_long_table_hides_rows_past_the_first_page(self):
        rows = [f"<tr><td>n{index}</td></tr>" for index in range(26)]
        html = paged_table_html("rd-sample", "results-table", "<th>Node</th>", rows)
        self.assertIn("rd-pager", html)
        self.assertIn("1–25 of 26", html)
        self.assertIn("<tr hidden><td>n25</td>", html)
        self.assertNotIn("<tr hidden><td>n0</td>", html)
        self.assertEqual(html.count("class='rd-pager'"), 1)


class TestPageLongTables(unittest.TestCase):
    def _plain_table(self, table_id, count):
        rows = "".join(f"<tr><td>n{index}</td></tr>" for index in range(count))
        ident = f" id='{table_id}'" if table_id else ""
        return f"<table{ident}><tr><th>Node</th></tr>{rows}</table>"

    def test_plain_tables_longer_than_one_page_get_the_same_pager(self):
        document = self._plain_table("left", 26) + self._plain_table("", 30)
        html = page_long_tables(document)
        self.assertEqual(html.count("class='rd-pager'"), 2)
        self.assertIn("1–25 of 26", html)
        self.assertIn("1–25 of 30", html)
        self.assertIn("id='rd-paged-1'", html)
        self.assertIn("<thead><tr><th>Node</th></tr></thead>", html)

    def test_short_plain_table_is_left_alone(self):
        document = self._plain_table("short", 4)
        self.assertEqual(page_long_tables(document), document)

    def test_running_the_pass_twice_keeps_one_pager(self):
        once = page_long_tables(self._plain_table("only", 26))
        twice = page_long_tables(once)
        self.assertEqual(twice.count("class='rd-pager'"), 1)
        self.assertEqual(twice.count("<script>"), 1)
        self.assertIn("<tr hidden><td>n25</td>", twice)


if __name__ == "__main__":
    unittest.main()
