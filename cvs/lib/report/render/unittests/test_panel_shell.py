'''Unit tests for the shared paged table used by every Run Deck table.'''

import unittest

from cvs.lib.report.render.panel_shell import paged_table_html


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
        self.assertEqual(html.count("<script>"), 1)


if __name__ == "__main__":
    unittest.main()
