'''Tests for the series Run Deck viewer.'''

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders import series  # noqa: F401
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.unittests.test_rccl_payload import _graph
from cvs.lib.report.viewer.series import write_series_viewer


class TestSeriesViewer(unittest.TestCase):
    def _rccl_payload(self):
        profile = load_json_profile("rccl")
        store = {
            "cvs_results_dict": _graph(),
            "variant_config": SimpleNamespace(
                framework="rccl",
                nnodes=2,
                mpi_params={"no_of_nodes": "2", "no_of_local_ranks": "8"},
                rccl_test_params={
                    "rccl_collective": ["all_reduce_perf"],
                    "start_msg_size": "1024",
                    "end_msg_size": "16g",
                },
                cvs_params={"nic_model": "thor"},
                enforce_thresholds=False,
            ),
        }
        return profile, build_rundeck_payload(profile=profile, store=store, cvs_version="1.0.0")

    def test_embeds_payload_and_prerendered_deck_sections(self):
        profile, payload = self._rccl_payload()
        chart_title = next(c["title"] for c in profile["cards"] if c.get("type") == "line_chart")
        self.assertIn("viewer_deck_html", payload)
        self.assertIn(chart_title, payload["viewer_deck_html"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rccl_run_deck_viewer.html"
            write_series_viewer(
                out,
                json_basename="rccl_run_deck.json",
                title="RCCL Run Deck",
                subtitle="RCCL viewer",
                embed_payload=payload,
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("rccl_run_deck.json", text)
            self.assertIn("RCCL Run Deck", text)
            self.assertIn('<a class="to-top" href="#top">Go to top</a>', text)
            self.assertIn("function pageLongTables", text)
            self.assertNotIn("__PAGE_TABLES_JS__", text)
            self.assertNotIn("__DECK_SECTIONS__", text)
            self.assertNotIn("__EMBEDDED_JSON__", text)
            self.assertIn("embedded-report-json", text)
            self.assertIn(f"<h2>{chart_title}</h2>", text)
            self.assertIn("class='chart-panel'", text)
            self.assertGreaterEqual(text.count("<polyline "), 2)
            self.assertIn('data-viewer="series"', text)
            self.assertIn("mountSeriesDeck", text)
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
