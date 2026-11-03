'''Publisher tests for status-matrix decks and the health viewer.'''

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cvs.lib.report.auto_register import try_auto_register_suite_report
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.generate_rundeck import generate_rundeck


def _session(root, suite):
    html_path = root / f"{suite}.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    config = SimpleNamespace(
        _suite_name=suite,
        _test_html_dir=f"{suite}_html",
        _suite_report_config=None,
        option=SimpleNamespace(htmlpath=str(html_path), log_file=None, self_contained_html=True),
    )
    return SimpleNamespace(config=config)


class TestStatusMatrixPublish(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(
            patch("cvs.lib.report.rundeck.generate_rundeck.build_inference_report_provenance", return_value={})
        )
        self.patches.enter_context(patch("cvs.lib.report.rundeck.generate_rundeck.cvs_version", return_value="test"))

    def test_status_matrix_publishes_deck_without_viewer_until_enabled(self):
        session = _session(self.root, "anc_test_cpu")
        self.assertTrue(try_auto_register_suite_report(session.config))
        store = {"cvs_results_dict": {"groups": {"g": {"nodes": {"n1": {"status": "pass"}}}}}}
        with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store):
            artifacts = generate_rundeck(session, None)
        self.assertTrue(artifacts["html"].is_file())
        self.assertTrue(artifacts["json"].is_file())
        self.assertNotIn("viewer", artifacts)
        self.assertNotIn("summary", artifacts)
        self.assertNotIn("buildInteractivityChart", artifacts["html"].read_text(encoding="utf-8"))

    def test_status_matrix_interactive_viewer_uses_health_template(self):
        session = _session(self.root, "anc_test_cpu")
        self.assertTrue(try_auto_register_suite_report(session.config))
        profile = load_json_profile("anc_test_cpu")
        profile["interactive_viewer"] = True
        profile["cards"] = [
            {
                "type": "run_card",
                "id": "run-card",
                "title": "Run card",
                "bind": "datasets.status_matrix.run_card_display",
            },
            {
                "type": "status_overview",
                "id": "overview",
                "title": "Health overview",
                "bind": "datasets.status_matrix.overview",
            },
            {
                "type": "metric_charts",
                "id": "metrics",
                "title": "Metrics",
                "bind": "datasets.status_matrix.metric_charts",
            },
            {
                "type": "status_matrix",
                "id": "results",
                "title": "Full results",
                "bind": "datasets.status_matrix",
                "hint": "Preset breakdown",
            },
        ]
        store = {
            "cvs_results_dict": {
                "_meta": {"cluster": "c1", "version": "1.0", "suite": "anc_test_cpu"},
                "groups": {
                    "a2a": {
                        "nodes": {
                            "n1": {
                                "status": "pass",
                                "metrics": [
                                    {
                                        "name": "rtotal",
                                        "value": 420,
                                        "unit": "GB/s",
                                        "threshold": 400,
                                        "direction": "higher",
                                        "status": "pass",
                                    }
                                ],
                            }
                        }
                    }
                },
            }
        }
        with (
            patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store),
            patch("cvs.lib.report.rundeck.generate_rundeck.get_resolved_profile", return_value=profile),
        ):
            artifacts = generate_rundeck(session, None)
        self.assertNotIn("summary", artifacts)
        self.assertTrue(artifacts["html"].is_file())
        self.assertTrue(artifacts["json"].is_file())
        self.assertEqual(artifacts["viewer"].name, "anc_test_cpu_run_deck_viewer.html")
        deck = artifacts["html"].read_text(encoding="utf-8")
        self.assertIn("Health overview", deck)
        self.assertIn("Preset breakdown", deck)
        self.assertIn("anc_test_cpu_run_deck_viewer.html", deck)
        viewer = artifacts["viewer"].read_text(encoding="utf-8")
        self.assertIn('data-viewer="status-matrix"', viewer)
        self.assertIn("embedded-report-json", viewer)
        self.assertIn("chart.js", viewer)
        self.assertIn('id="f-node"', viewer)
        self.assertIn('id="f-group"', viewer)
        self.assertIn('id="f-metric"', viewer)
        self.assertIn('id="f-search"', viewer)
        self.assertIn("renderHealth", viewer)
        self.assertIn("renderHeatmaps", viewer)
        self.assertIn('"rtotal"', viewer)
        self.assertNotIn("buildInteractivityChart", viewer)
        self.assertNotIn("loss-panel", viewer)
        payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
        self.assertAlmostEqual(payload["datasets"]["status_matrix"]["overview"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
