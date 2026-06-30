'''Unit tests for CI summary and gate heatmap rendering.'''

from cvs.lib.report.ci_summary import render_ci_summary_html, worst_cells, write_inference_ci_summary
from cvs.lib.report.render.gate_matrix import render_gate_heatmap_html, render_gate_matrix_html
from cvs.lib.report.unittests._fixtures import generic_inference_report_config, generic_variant, two_cell_inf_res
from cvs.lib.report.inference import build_inference_report_payload


def test_gate_heatmap_renders_colored_cells():
    matrix = [
        {"label": "w1 · C=128", "tiers": {"throughput": "pass", "record": "record"}},
        {"label": "w1 · C=256", "tiers": {"throughput": "fail", "record": "record"}},
    ]
    html = render_gate_heatmap_html(matrix, ("throughput", "record"))
    assert "heatmap" in html
    assert "heat-fail" in html
    assert "heat-pass" in html
    table = render_gate_matrix_html(matrix, ("throughput", "record"))
    assert "fail" in table


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


def test_write_ci_summary(tmp_path):
    payload = build_inference_report_payload(
        config=generic_inference_report_config(),
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
    )
    out = write_inference_ci_summary(
        payload,
        generic_inference_report_config(),
        tmp_path,
    )
    text = out.read_text(encoding="utf-8")
    assert "CI summary" in text
    assert "Open full suite report" in text
