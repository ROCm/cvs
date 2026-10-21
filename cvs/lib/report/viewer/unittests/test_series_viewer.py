'''Tests for the series Run Deck viewer.'''

import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders import series  # noqa: F401
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.unittests.test_rccl_payload import _graph
from cvs.lib.report.viewer import series as series_viewer
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

    def _write(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rccl_run_deck_viewer.html"
            write_series_viewer(out, json_basename="rccl_run_deck.json", **kwargs)
            return out.read_text(encoding="utf-8")

    def test_fills_scaffold_and_embeds_payload_with_profile_cards(self):
        profile, payload = self._rccl_payload()
        chart_title = next(c["title"] for c in profile["cards"] if c.get("type") == "line_chart")
        self.assertNotIn("viewer_deck_html", payload)
        self.assertNotIn("viewer_deck_nav", payload)
        text = self._write(title="RCCL Run Deck", subtitle="RCCL viewer", embed_payload=payload)
        self.assertIn("rccl_run_deck.json", text)
        self.assertIn("RCCL Run Deck", text)
        self.assertIn('<a class="to-top" href="#top">Go to top</a>', text)
        self.assertIn('data-viewer="series"', text)
        for placeholder in ("__TITLE__", "__SUBTITLE__", "__JSON_PATH__", "__EMBEDDED_JSON__", "__PAGE_TABLES_JS__"):
            self.assertNotIn(placeholder, text)
        self.assertIn('id="embedded-report-json"', text)
        self.assertIn(chart_title, text)
        self.assertIn("all_reduce_perf", text)
        self.assertNotIn("metricLabels", text)

    def test_loads_same_chartjs_and_zoom_scripts_as_interactive_viewer(self):
        interactive = (Path(series_viewer.__file__).with_name("interactive.html")).read_text(encoding="utf-8")
        text = self._write(title="RCCL")
        for script in re.findall(r'<script src="[^"]*(?:chart\.umd|chartjs-plugin-zoom)[^"]*"></script>', interactive):
            self.assertIn(script, text)
        self.assertEqual(text.count('<script src="https://cdn.jsdelivr.net/npm/'), 2)

    def test_draws_zoomable_chartjs_line_charts_not_static_svg(self):
        text = self._write(title="RCCL")
        self.assertIn("new Chart(", text)
        self.assertIn("pan: { enabled: true, mode: 'x' }", text)
        self.assertIn("zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' }", text)
        self.assertIn("Reset zoom", text)
        self.assertIn("resetZoom", text)
        self.assertIn("borderDash: style.dash", text)
        self.assertIn("Math.floor(index / CHART_COLORS.length)", text)
        self.assertNotIn("<polyline", text)
        self.assertNotIn("renderSeriesChart", text)

    def test_chart_inputs_come_from_card_and_series_dataset(self):
        text = self._write(title="RCCL")
        self.assertIn("(card.series || {}).y_field", text)
        self.assertIn("seriesCfg.unit", text)
        self.assertIn("data.charts", text)
        for hardcoded in ("'bus_bw'", "'alg_bw'", "all_reduce_perf"):
            self.assertNotIn(hardcoded, text)

    def test_results_table_is_paged_once_by_shared_pager(self):
        _, payload = self._rccl_payload()
        text = self._write(title="RCCL", embed_payload=payload)
        self.assertIn("function pageLongTables", text)
        self.assertEqual(text.count("pageLongTables(document)"), 1)
        self.assertNotIn("data-page-size", text)
        self.assertIn('<div id="deck-sections"></div>', text)

    def test_esc_escapes_single_quotes_used_in_href_attribute(self):
        text = self._write(title="RCCL")
        self.assertIn(".replace(/'/g, '&#39;')", text)

    def test_missing_embed_still_fetches_sibling_json(self):
        text = self._write(title="RCCL viewer")
        self.assertNotIn('<script type="application/json" id="embedded-report-json">', text)
        self.assertIn("fetch(JSON_PATH)", text)


if __name__ == "__main__":
    unittest.main()
