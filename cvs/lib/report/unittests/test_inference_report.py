'''Unit tests for cvs.lib.report inference suite reports.'''

from dataclasses import replace

from cvs.lib.report.inference import (
    build_inference_report_payload,
    render_report_html,
    write_report,
)
from cvs.lib.report.inference_payload import sweep_has_multi_shape_comparison
from cvs.lib.report.chart_presets import DEFAULT_PERF_CHART_SERIES
from cvs.lib.report.unittests._fixtures import (
    generic_inference_report_config,
    generic_variant,
    multi_shape_inf_res,
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
    assert len(payload["chart_series"]["output_throughput"]) == 1
    assert len(payload["chart_series"]["output_throughput"][0]["points"]) == 2
    assert payload["chart_series"]["output_throughput"][0]["label"] == "ISL=1024 · OSL=1024"
    assert len(payload["chart_config"]) == 1
    assert payload["chart_config"][0]["suffix"] == "output_throughput"
    assert "chart_comparison" not in payload


def test_render_report_html_from_payload():
    cfg = generic_inference_report_config()
    payload = build_inference_report_payload(
        config=cfg,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
        pytest_html_path="/out/ix_atom_run.html",
        log_file_path="/out/ix_atom_run.log",
    )
    payload["run_card_display"] = [
        ("Pytest report", "/out/ix_atom_run.html", True),
        ("Run log", "/out/ix_atom_run.log", True),
    ]
    doc = render_report_html(payload)
    assert "Test inference suite report" in doc
    assert "report-nav" in doc
    assert "Gate matrix" in doc
    assert "heatmap" in doc
    assert "Full results" in doc
    assert '<a href="ix_atom_run.html">Pytest report</a>' in doc
    assert '<a href="ix_atom_run.log">Run log</a>' in doc


def test_build_chart_series_groups_by_isl_osl():
    cfg = replace(generic_inference_report_config(), chart_series=DEFAULT_PERF_CHART_SERIES)
    payload = build_inference_report_payload(
        config=cfg,
        variant_config=generic_variant(),
        inf_res_dict=multi_shape_inf_res(),
        lifecycle_report={},
    )
    output_groups = payload["chart_series"]["output_throughput"]
    assert len(output_groups) == 2
    assert {g["label"] for g in output_groups} == {
        "ISL=1024 · OSL=1024",
        "ISL=8192 · OSL=1024",
    }
    doc = render_report_html(payload)
    assert doc.count("<h3 class='chart-group-title'>") == 2
    assert "P99 ITL" in doc
    assert "Compare shapes at each concurrency" not in doc
    assert "interactive viewer" in doc


def test_sweep_has_multi_shape_comparison():
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
    assert sweep_has_multi_shape_comparison(multi["cells"])
    assert not sweep_has_multi_shape_comparison(single["cells"])


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
