'''Unit tests for cvs.lib.report inference suite reports.'''

from cvs.lib.report.inference import (
    build_inference_report_payload,
    render_report_embed_html,
    render_report_html,
    write_report,
)
from cvs.lib.report.unittests._fixtures import (
    generic_inference_report_config,
    generic_variant,
    two_cell_inf_res,
)


def test_build_inference_report_payload_uses_config():
    cfg = generic_inference_report_config()
    payload = build_inference_report_payload(
        config=cfg,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
        cvs_version="1.0.0",
    )
    assert payload["schema_version"] == 1
    assert payload["suite_id"] == "test_inference_suite"
    assert payload["report"]["title"] == "Test inference suite report"
    assert len(payload["cells"]) == 2
    assert len(payload["sweep_summaries"]) == 1
    assert len(payload["chart_series"]["output_throughput"]) == 2
    assert len(payload["chart_config"]) == 1
    assert payload["chart_config"][0]["suffix"] == "output_throughput"


def test_render_report_html_from_payload():
    cfg = generic_inference_report_config()
    payload = build_inference_report_payload(
        config=cfg,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
    )
    doc = render_report_html(payload)
    assert "Test inference suite report" in doc
    assert "report-nav" in doc
    assert "Gate matrix" in doc
    assert "Full results" in doc


def test_write_report_writes_html_and_json(tmp_path):
    cfg = generic_inference_report_config()
    artifacts = write_report(
        tmp_path / "test_inference_suite_report.html",
        config=cfg,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
    )
    assert artifacts["html"].is_file()
    assert artifacts["json"].is_file()
    assert '"suite_id": "test_inference_suite"' in artifacts["json"].read_text(encoding="utf-8")


def test_render_embed_snippet():
    payload = build_inference_report_payload(
        config=generic_inference_report_config(),
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
    )
    assert "<iframe" in render_report_embed_html(payload)


def test_multi_host_cells():
    inf_res = {
        ("org/example-model", "mi300x", "1024", "1024", "default", 128): {
            "10.0.0.1": {"client.output_throughput": 1000.0},
            "10.0.0.2": {"client.output_throughput": 1100.0},
        }
    }
    payload = build_inference_report_payload(
        config=generic_inference_report_config(),
        variant_config=generic_variant(),
        inf_res_dict=inf_res,
        lifecycle_report={},
    )
    assert len(payload["cells"]) == 2
    assert all(c["show_host_in_label"] for c in payload["cells"])
    assert "scaling" in payload.get("panels", {})
