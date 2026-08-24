'''Tests for interactive viewer scaffold.'''

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cvs.lib.report.cell_build import select_summary_cells
from cvs.lib.report.inference import write_report
from cvs.lib.report.testing.fixtures import (
    generic_inference_report_config,
    generic_variant,
    two_cell_inf_res,
)
from cvs.lib.report.viewer.scaffold import write_interactive_viewer


class TestViewerScaffold(unittest.TestCase):
    def test_select_summary_cells_prefers_failures(self):
        cells = [
            {"tiers": {"throughput": "pass"}, "cell_id": "a"},
            {"tiers": {"throughput": "fail"}, "cell_id": "b"},
            {"tiers": {"throughput": "pass"}, "cell_id": "c"},
        ]
        picked = select_summary_cells(cells, 2, gated_tiers=("throughput",))
        self.assertEqual([c["cell_id"] for c in picked], ["b", "a"])

    def test_write_interactive_viewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite_viewer.html"
            write_interactive_viewer(
                out,
                json_basename="suite_report.json",
                title="Suite viewer",
                tier_order=("throughput", "record"),
                embed_payload={"schema_version": 1, "cells": [{"cell_id": "c1", "host": "h1"}]},
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("suite_report.json", text)
            self.assertIn("embedded-report-json", text)
            self.assertIn('"cell_id"', text)
            self.assertIn("chart.js", text)
            self.assertIn("comparison-grid", text)
            self.assertIn("interactivity-block", text)
            self.assertIn("interactivity-panel", text)
            self.assertIn("interactivity-chart-wrap", text)
            self.assertIn("buildInteractivityChart", text)
            self.assertIn("viewerConfig", text)
            self.assertIn("initViewerUi", text)
            self.assertIn("interactivityExternalTooltip", text)
            self.assertNotIn("tradeoff-block", text)
            self.assertNotIn("per-shape-block", text)
            self.assertNotIn("percentile-block", text)
            self.assertIn('id="overview"', text)

    def test_viewer_written_when_interactive_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = generic_inference_report_config()
            artifacts = write_report(
                tmp_path / "test_inference_suite_report.html",
                config=cfg,
                variant_config=generic_variant(),
                inf_res_dict=two_cell_inf_res(),
                lifecycle_report={},
            )
            self.assertIsNotNone(artifacts.get("viewer"))
            self.assertTrue(artifacts["viewer"].is_file())
            viewer_text = artifacts["viewer"].read_text(encoding="utf-8")
            self.assertIn("embedded-report-json", viewer_text)
            self.assertGreater(len(viewer_text), 500)

    def test_summary_mode_truncates_static_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = replace(
                generic_inference_report_config(),
                viewer_cell_threshold=2,
                interactive_viewer=True,
            )
            inf_res = {}
            for conc in (64, 128, 256):
                key = ("org/example-model", "mi300x", "1024", "1024", "default", conc)
                inf_res[key] = {
                    "10.0.0.1": {
                        "client.output_throughput": float(conc * 10),
                        "client.mean_ttft_ms": 100.0,
                    }
                }
            artifacts = write_report(
                tmp_path / "test_inference_suite_report.html",
                config=cfg,
                variant_config=generic_variant(),
                inf_res_dict=inf_res,
                lifecycle_report={},
            )
            self.assertIsNotNone(artifacts.get("viewer"))
            self.assertTrue(artifacts["viewer"].is_file())
            html_doc = artifacts["html"].read_text(encoding="utf-8")
            self.assertIn("Open interactive viewer", html_doc)
            self.assertEqual(html_doc.count("<article class='cell-card'>"), 2)
            payload = artifacts["payload"]
            self.assertEqual(payload["summary"]["mode"], "truncated")
            self.assertEqual(payload["summary"]["total_cells"], 3)


if __name__ == "__main__":
    unittest.main()
