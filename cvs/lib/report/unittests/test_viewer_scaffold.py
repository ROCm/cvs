'''Tests for cell card render primitives and interactive viewer scaffold.'''

from dataclasses import replace

from cvs.lib.report.cell_build import build_all_cells, select_summary_cells
from cvs.lib.report.inference import write_report
from cvs.lib.report.render.cell_card import render_cell_card_html
from cvs.lib.report.unittests._fixtures import (
    generic_inference_report_config,
    generic_variant,
    two_cell_inf_res,
)
from cvs.lib.report.viewer.scaffold import write_interactive_viewer


def test_render_cell_card_html_compact():
    cfg = generic_inference_report_config()
    cells = build_all_cells(
        cfg,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
    )
    html_out = render_cell_card_html(
        cells[0],
        tier_order=cfg.metric_tier_order,
        headline_metric=cfg.headline_metric,
        enforce=True,
        cell_lifecycle_labels=cfg.cell_lifecycle_labels,
        compact=True,
    )
    assert "cell-card-compact" in html_out
    assert "ISL=1024" in html_out


def test_select_summary_cells_prefers_failures():
    cells = [
        {"tiers": {"throughput": "pass"}, "cell_id": "a"},
        {"tiers": {"throughput": "fail"}, "cell_id": "b"},
        {"tiers": {"throughput": "pass"}, "cell_id": "c"},
    ]
    picked = select_summary_cells(cells, 2, gated_tiers=("throughput",))
    assert [c["cell_id"] for c in picked] == ["b", "a"]


def test_write_interactive_viewer(tmp_path):
    out = tmp_path / "suite_viewer.html"
    write_interactive_viewer(
        out,
        json_basename="suite_report.json",
        title="Suite viewer",
        tier_order=("throughput", "record"),
        embed_payload={"schema_version": 1, "cells": [{"cell_id": "c1", "host": "h1"}]},
    )
    text = out.read_text(encoding="utf-8")
    assert "suite_report.json" in text
    assert "embedded-report-json" in text
    assert '"cell_id"' in text
    assert "chart.js" in text
    assert "comparison-grid" in text
    assert 'id="overview"' in text


def test_viewer_written_when_interactive_enabled(tmp_path):
    cfg = generic_inference_report_config()
    artifacts = write_report(
        tmp_path / "test_inference_suite_report.html",
        config=cfg,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
    )
    assert artifacts.get("viewer") is not None
    assert artifacts["viewer"].is_file()
    viewer_text = artifacts["viewer"].read_text(encoding="utf-8")
    assert "embedded-report-json" in viewer_text
    assert len(viewer_text) > 500


def test_summary_mode_truncates_static_html(tmp_path):
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
    assert artifacts.get("viewer") is not None
    assert artifacts["viewer"].is_file()
    html_doc = artifacts["html"].read_text(encoding="utf-8")
    assert "Open interactive viewer" in html_doc
    assert html_doc.count("<article class='cell-card'>") == 2
    payload = artifacts["payload"]
    assert payload["summary"]["mode"] == "truncated"
    assert payload["summary"]["total_cells"] == 3
