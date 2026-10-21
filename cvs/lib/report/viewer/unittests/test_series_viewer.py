'''Tests for the series Run Deck viewer.'''

import tempfile
import unittest
from pathlib import Path

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.viewer.series import write_series_viewer


class TestSeriesViewer(unittest.TestCase):
    def test_embeds_payload_and_series_deck_controls(self):
        profile = load_json_profile("rccl")
        chart_title = next(c["title"] for c in profile["cards"] if c.get("type") == "line_chart")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rccl_run_deck_viewer.html"
            write_series_viewer(
                out,
                json_basename="rccl_run_deck.json",
                title="RCCL Run Deck",
                subtitle="RCCL viewer",
                embed_payload={
                    "deck_profile": profile,
                    "run_card_display": [["Collectives", "all_reduce_perf", False]],
                    "datasets": {
                        "series": {
                            "charts": {
                                "bus_bw": {
                                    "all_reduce_perf": [
                                        {"label": "all_reduce_perf", "points": [["1M", 22.3], ["2M", 36.5]]}
                                    ]
                                }
                            }
                        }
                    },
                    "results_table": {
                        "headers": ["Collective", "Message size", "Bus BW (GB/s)"],
                        "rows": [["all_reduce_perf", 1048576, 22.3]],
                    },
                },
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("rccl_run_deck.json", text)
            self.assertIn("RCCL Run Deck", text)
            self.assertIn('<a class="to-top" href="#top">Go to top</a>', text)
            self.assertIn("function pageLongTables", text)
            self.assertNotIn("__PAGE_TABLES_JS__", text)
            self.assertNotIn("__EMBEDDED_JSON__", text)
            self.assertIn("embedded-report-json", text)
            self.assertIn(chart_title, text)
            self.assertIn('data-viewer="series"', text)
            self.assertIn("renderSeriesDeck", text)
            self.assertIn("all_reduce_perf", text)
            self.assertNotIn("metricLabels", text)
            self.assertNotIn("chart.js", text)

    def test_missing_embed_still_fetches_sibling_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rccl_run_deck_viewer.html"
            write_series_viewer(out, json_basename="rccl_run_deck.json", title="RCCL viewer")
            text = out.read_text(encoding="utf-8")
            self.assertNotIn('<script type="application/json" id="embedded-report-json">', text)
            self.assertIn("fetch(JSON_PATH)", text)


if __name__ == "__main__":
    unittest.main()
