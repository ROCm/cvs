'''Tests for the Chart.js libraries inlined into viewer pages.'''

import unittest
from unittest.mock import patch

from cvs.lib.report.viewer import chart_libs


class TestChartScripts(unittest.TestCase):
    def test_chart_js_alone(self):
        scripts = chart_libs.chart_scripts()
        self.assertTrue(scripts.startswith("<script>"))
        self.assertIn("Chart.js v4.4.1", scripts)
        self.assertNotIn("chartjs-plugin-zoom", scripts)
        self.assertEqual(scripts.count("</script>"), 1)

    def test_chart_js_with_the_zoom_plugin_after_it(self):
        scripts = chart_libs.chart_scripts(zoom=True)
        self.assertLess(scripts.index("Chart.js v4.4.1"), scripts.index("chartjs-plugin-zoom v2.0.1"))
        self.assertEqual(scripts.count("</script>"), 2)

    def test_closing_tag_inside_a_library_is_escaped(self):
        with patch.object(chart_libs.Path, "read_text", return_value='var s = "</script>";'):
            self.assertEqual(chart_libs.chart_scripts(), '<script>var s = "<\\/script>";</script>')

    def test_licenses_ship_with_the_libraries(self):
        for name in ("LICENSE-chart.js.md", "LICENSE-chartjs-plugin-zoom.md"):
            with self.subTest(name=name):
                self.assertIn("MIT License", (chart_libs.VENDOR_DIR / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
