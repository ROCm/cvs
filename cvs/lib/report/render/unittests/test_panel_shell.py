'''Unit tests for the shared results table and its pager.'''

import unittest

from cvs.lib.report.render.panel_shell import (
    artifact_link_records,
    paged_table_html,
    render_artifact_links,
    render_results_table_html,
)


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


class TestRenderResultsTableHtml(unittest.TestCase):
    def test_header_row_is_in_thead_and_rows_in_tbody(self):
        out = render_results_table_html(["Test", "BW <GB/s>"], [["write", 1.5], ["read", None]])
        self.assertEqual(
            out,
            "<table class='results-table'><thead><tr><th>Test</th><th>BW &lt;GB/s&gt;</th></tr></thead>"
            "<tbody><tr><td>write</td><td>1.5</td></tr><tr><td>read</td><td>None</td></tr></tbody></table>",
        )

    def test_units_render_beside_headers_outside_the_uppercase_text(self):
        out = render_results_table_html(["Test", "Latency"], [["write", 1.5]], units=[None, "µs"])
        self.assertIn("<th>Test</th><th>Latency <span class='th-unit'>(µs)</span></th>", out)

    def test_align_sets_a_class_per_column(self):
        out = render_results_table_html(["Test", "BW", "Note"], [["write", 1.5, "x"]], align=["left", "right"])
        self.assertIn("<tr><td class='al-left'>write</td><td class='al-right'>1.5</td><td>x</td></tr>", out)

    def test_artifact_links_share_one_anchor_for_both_surfaces(self):
        cell = {
            "errors_json_href": "n1_errors.json",
            "log_tarball_href": "a<b>'c\".tar.gz",
        }
        self.assertEqual(
            artifact_link_records(cell),
            [
                {"label": "errors.json", "href": "n1_errors.json"},
                {"label": "logs.tar.gz", "href": "a<b>'c\".tar.gz"},
            ],
        )
        self.assertEqual(
            render_artifact_links(cell),
            "<a href='n1_errors.json'>errors.json</a><a href='a&lt;b&gt;&#x27;c&quot;.tar.gz'>logs.tar.gz</a>",
        )
        self.assertEqual(render_artifact_links({}), "")

    def test_link_records_render_as_anchors_and_other_cells_stay_escaped(self):
        links = [
            {"label": "errors.json", "href": "n1_errors.json"},
            {"label": "logs.tar.gz", "href": "a<b>'c\".tar.gz"},
        ]
        out = render_results_table_html(["Message", "Logs"], [["<fail>", links]])
        self.assertIn("<td>&lt;fail&gt;</td>", out)
        self.assertIn("<a href='n1_errors.json'>errors.json</a>", out)
        self.assertIn("<a href='a&lt;b&gt;&#x27;c&quot;.tar.gz'>logs.tar.gz</a>", out)
        self.assertNotIn("<fail>", out)

    def test_no_rows_renders_the_empty_message_instead_of_a_table(self):
        self.assertEqual(
            render_results_table_html(["Test"], iter([]), empty_message="No <rows>."),
            "<p class='muted'>No &lt;rows&gt;.</p>",
        )


if __name__ == "__main__":
    unittest.main()
