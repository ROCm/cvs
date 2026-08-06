'''Unit tests for cvs.lib.report inference suite reports.'''

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cvs.lib.report.chart_presets import DEFAULT_PERF_CHART_SERIES
from cvs.lib.report.inference import (
    build_inference_report_payload,
    render_report_html,
    write_report,
)
from cvs.lib.report.inference_payload import sweep_has_multi_shape_comparison
from cvs.lib.report.testing.fixtures import (
    generic_inference_report_config,
    generic_variant,
    multi_shape_inf_res,
    two_cell_inf_res,
)


class TestInferenceReport(unittest.TestCase):
    def test_build_inference_report_payload_uses_config(self):
        cfg = generic_inference_report_config()
        payload = build_inference_report_payload(
            config=cfg,
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
            cvs_version="1.0.0",
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["suite_id"], "test_inference_suite")
        self.assertEqual(payload["report"]["title"], "Test inference suite report")
        self.assertEqual(len(payload["cells"]), 2)
        self.assertEqual(len(payload["sweep_summaries"]), 1)
        self.assertEqual(len(payload["chart_series"]["output_throughput"]), 1)
        self.assertEqual(len(payload["chart_series"]["output_throughput"][0]["points"]), 2)
        self.assertEqual(payload["chart_series"]["output_throughput"][0]["label"], "ISL=1024 · OSL=1024")
        self.assertEqual(len(payload["chart_config"]), 1)
        self.assertEqual(payload["chart_config"][0]["suffix"], "output_throughput")
        self.assertNotIn("chart_comparison", payload)

    def test_render_report_html_from_payload(self):
        cfg = generic_inference_report_config()
        payload = build_inference_report_payload(
            config=cfg,
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
            pytest_html_path="/out/ix_atom_run.html",
            log_file_path="/out/ix_atom_run.log",
            provenance={
                "pytest_html_href": "../ix_atom_run.html",
                "log_file_href": "ix_atom_run.log",
            },
        )
        doc = render_report_html(payload)
        self.assertIn("Test inference suite report", doc)
        self.assertIn("report-nav", doc)
        self.assertIn("Gate matrix", doc)
        self.assertIn("heatmap", doc)
        self.assertIn("Full results", doc)
        self.assertIn('<a href="../ix_atom_run.html">Pytest report</a>', doc)
        self.assertIn('<a href="ix_atom_run.log">Run log</a>', doc)
        self.assertNotIn("class='notes'", doc)

    def test_build_chart_series_groups_by_isl_osl(self):
        cfg = replace(generic_inference_report_config(), chart_series=DEFAULT_PERF_CHART_SERIES)
        payload = build_inference_report_payload(
            config=cfg,
            variant_config=generic_variant(),
            inf_res_dict=multi_shape_inf_res(),
            lifecycle_report={},
        )
        output_groups = payload["chart_series"]["output_throughput"]
        self.assertEqual(len(output_groups), 2)
        self.assertEqual(
            {g["label"] for g in output_groups},
            {"ISL=1024 · OSL=1024", "ISL=8192 · OSL=1024"},
        )
        doc = render_report_html(payload)
        self.assertEqual(doc.count("<h3 class='chart-group-title'>"), 2)
        self.assertIn("P99 ITL", doc)
        self.assertNotIn("Compare shapes at each concurrency", doc)
        self.assertIn("interactive viewer", doc)

    def test_sweep_has_multi_shape_comparison(self):
        cfg = generic_inference_report_config()
        multi = build_inference_report_payload(
            config=cfg,
            variant_config=generic_variant(),
            inf_res_dict=multi_shape_inf_res(),
            lifecycle_report={},
        )
        single = build_inference_report_payload(
            config=cfg,
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
        )
        self.assertTrue(sweep_has_multi_shape_comparison(multi["cells"]))
        self.assertFalse(sweep_has_multi_shape_comparison(single["cells"]))

    def test_payload_omits_variant_run_card_notes(self):
        variant = generic_variant()
        variant.run_card = type("RunCard", (), {"notes": "demo note that should not render"})()
        payload = build_inference_report_payload(
            config=generic_inference_report_config(),
            variant_config=variant,
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
        )
        self.assertEqual(payload.get("run_card_notes"), "")

    def test_write_report_writes_html_and_json(self):
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
            self.assertTrue(artifacts["html"].is_file())
            self.assertTrue(artifacts["json"].is_file())
            self.assertIn(
                '"suite_id": "test_inference_suite"',
                artifacts["json"].read_text(encoding="utf-8"),
            )

    def test_lifecycle_populates_cell_and_session_aggregates(self):
        cfg = generic_inference_report_config()
        lifecycle_report = {
            "cvs/tests/x.py::test_inference[combo-128]": [
                ("server_ready", 12.5, "s"),
                ("client_complete", 45.0, "s"),
            ],
            "cvs/tests/x.py::test_inference[combo-256]": [
                ("server_ready", 8.0, "s"),
                ("client_complete", 30.0, "s"),
            ],
        }
        payload = build_inference_report_payload(
            config=cfg,
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report=lifecycle_report,
        )
        session = payload["lifecycle"]
        self.assertEqual(session["server_ready"], 12.5)
        self.assertEqual(session["client_complete"], 45.0)

        cell_128 = next(c for c in payload["cells"] if c["concurrency"] == 128)
        self.assertEqual(cell_128["cell_lifecycle"]["client_complete"], 45.0)


if __name__ == "__main__":
    unittest.main()
