'''Unit tests for CI summary HTML.'''

import json

from cvs.lib.report.ci_summary import render_ci_summary_html, worst_cells, write_inference_ci_summary
from cvs.lib.report.inference import build_inference_report_payload
from cvs.lib.report.unittests._fixtures import generic_inference_report_config, generic_variant, two_cell_inf_res


def test_ci_summary_worst_cells_prefers_failures():
    payload = build_inference_report_payload(
        config=generic_inference_report_config(),
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
    )
    payload["cells"][0]["tiers"]["throughput"] = "fail"
    worst = worst_cells(payload, generic_inference_report_config(), limit=3)
    assert worst[0]["tiers"]["throughput"] == "fail"


def test_render_ci_summary_includes_overall_status():
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
    assert "CI summary" in html
    assert "Cells to review" in html
    assert "test_inference_suite_report.html" in html


def test_write_ci_summary_includes_parity(tmp_path):
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
    assert out.name == "test_inference_suite_report_summary.html"
    assert "CI summary" in text
    assert "Framework parity" in text
    assert "inference_parity_report.html" in text
