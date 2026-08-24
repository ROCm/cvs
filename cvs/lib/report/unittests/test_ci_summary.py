'''Unit tests for CI summary HTML.'''

import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.report.ci_summary import render_ci_summary_html, worst_cells, write_inference_ci_summary
from cvs.lib.report.inference import build_inference_report_payload
from cvs.lib.report.testing.fixtures import generic_inference_report_config, generic_variant, two_cell_inf_res


class TestCiSummary(unittest.TestCase):
    def test_worst_cells_prefers_failures(self):
        payload = build_inference_report_payload(
            config=generic_inference_report_config(),
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
        )
        payload["cells"][0]["tiers"]["throughput"] = "fail"
        worst = worst_cells(payload, generic_inference_report_config(), limit=3)
        self.assertEqual(worst[0]["tiers"]["throughput"], "fail")

    def test_render_ci_summary_includes_overall_status(self):
        payload = build_inference_report_payload(
            config=generic_inference_report_config(),
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
        )
        html = render_ci_summary_html(
            payload,
            generic_inference_report_config(),
            full_report_basename="test_inference_suite_report",
        )
        self.assertIn("CI summary", html)
        self.assertIn("Cells to review", html)
        self.assertIn("test_inference_suite_report.html", html)

    def test_write_ci_summary_includes_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            payload = build_inference_report_payload(
                config=generic_inference_report_config(),
                variant_config=generic_variant(),
                inf_res_dict=two_cell_inf_res(),
                lifecycle_report={},
            )
            (tmp_path / "inference_parity_report.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {"compare": {"compare.vllm.output_throughput_ratio": 0.9}},
                            {"compare": {"compare.vllm.output_throughput_ratio": 1.0}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = write_inference_ci_summary(
                payload,
                generic_inference_report_config(),
                tmp_path,
            )
            text = out.read_text(encoding="utf-8")
            self.assertEqual(out.name, "test_inference_suite_report_summary.html")
            self.assertIn("CI summary", text)
            self.assertIn("Framework parity", text)
            self.assertIn("inference_parity_report.html", text)


if __name__ == "__main__":
    unittest.main()
