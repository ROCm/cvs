'''Tests for the status-matrix health viewer.'''

import tempfile
import unittest
from pathlib import Path

from cvs.lib.report.viewer.status_matrix import write_status_matrix_viewer


class TestStatusMatrixViewer(unittest.TestCase):
    def test_embeds_payload_and_health_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "health_run_deck_viewer.html"
            write_status_matrix_viewer(
                out,
                json_basename="health_run_deck.json",
                title="Health viewer",
                subtitle="Node results",
                embed_payload={
                    "datasets": {
                        "status_matrix": {
                            "nodes": ["n1"],
                            "groups": ["a2a"],
                            "overview": {"pass_rate": 1.0, "counts": {"pass": 1, "fail": 0, "na": 0}},
                            "metric_charts": {
                                "metrics": [{"name": "rtotal", "group": "a2a", "points": [{"node": "n1", "value": 1}]}],
                                "series": [],
                                "heatmaps": [],
                            },
                            "grid": {"n1": {"a2a": {"status": "pass", "items": []}}},
                        }
                    }
                },
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("health_run_deck.json", text)
            self.assertIn("Health viewer", text)
            self.assertIn("embedded-report-json", text)
            self.assertIn('"rtotal"', text)
            self.assertIn("chart.js", text)
            self.assertIn('data-viewer="status-matrix"', text)
            self.assertIn("loadReportData", text)
            self.assertIn("renderMetricCharts", text)
            self.assertIn("renderHeatmaps", text)
            self.assertIn("renderDetails", text)
            self.assertIn('id="f-node"', text)
            self.assertIn('id="f-group"', text)
            self.assertIn('id="f-metric"', text)
            self.assertIn('id="f-search"', text)
            self.assertIn("No metric data recorded.", text)
            self.assertIn("No heatmap data recorded.", text)
            self.assertNotIn("buildInteractivityChart", text)
            self.assertNotIn("loss-panel", text)
            self.assertNotIn("tradeoff-block", text)

    def test_missing_embed_still_fetches_sibling_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "health_run_deck_viewer.html"
            write_status_matrix_viewer(out, json_basename="health_run_deck.json", title="Health viewer")
            text = out.read_text(encoding="utf-8")
            self.assertNotIn('<script type="application/json" id="embedded-report-json">', text)
            self.assertIn("fetch(JSON_PATH)", text)
            self.assertIn("statusMatrix", text)


if __name__ == "__main__":
    unittest.main()
