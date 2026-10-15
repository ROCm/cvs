'''Unit tests for cvs/lib/report/render/panel_shell.py.'''

import unittest

from cvs.lib.report.render.panel_shell import render_results_table_html


class TestRenderResultsTableHtml(unittest.TestCase):
    def test_header_row_is_in_thead_and_rows_in_tbody(self):
        out = render_results_table_html(["Test", "BW <GB/s>"], [["write", 1.5], ["read", None]])
        self.assertEqual(
            out,
            "<table class='results-table'><thead><tr><th>Test</th><th>BW &lt;GB/s&gt;</th></tr></thead>"
            "<tbody><tr><td>write</td><td>1.5</td></tr><tr><td>read</td><td>None</td></tr></tbody></table>",
        )

    def test_no_rows_renders_the_empty_message_instead_of_a_table(self):
        self.assertEqual(
            render_results_table_html(["Test"], iter([]), empty_message="No <rows>."),
            "<p class='muted'>No &lt;rows&gt;.</p>",
        )


if __name__ == "__main__":
    unittest.main()
