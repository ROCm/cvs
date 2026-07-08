'''Unit tests for cell record building.'''

from dataclasses import replace
from types import SimpleNamespace

from cvs.lib.report.cell_build import build_cell_record, resolve_pytest_nodeids_for_cell, bar_pct
from cvs.lib.report.formatting import pytest_row_href
from cvs.lib.report.unittests._fixtures import generic_inference_report_config


def test_margin_shown_when_record_only_and_spec_present():
    variant = SimpleNamespace(
        enforce_thresholds=False,
        thresholds={
            "ISL=1024,OSL=1024,TP=8,CONC=128": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
            },
        },
        cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
    )
    key = ("org/example-model", "mi300x", "1024", "1024", "default", 128)
    cell = build_cell_record(
        generic_inference_report_config(),
        key=key,
        host="10.0.0.1",
        actuals={"client.output_throughput": 1200.0},
        variant_config=variant,
        lifecycle_report={},
        multi_host=False,
    )
    throughput = next(m for m in cell["metrics"] if m["metric"] == "client.output_throughput")
    assert throughput["status"] == "record"
    assert throughput["margin"] is not None
    assert "above gate" in throughput["margin"]


def test_resolve_pytest_nodeids_for_cell():
    config = generic_inference_report_config()
    lifecycle = {
        "cvs/tests/x.py::test_inference[combo-128]": [],
        "cvs/tests/x.py::test_cell_metrics[tier0-128]": [],
    }
    ids = resolve_pytest_nodeids_for_cell(config, lifecycle, 128)
    assert "test_inference" in ids["pytest_inference_nodeid"]
    assert "test_cell_metrics" in ids["pytest_metrics_nodeid"]


def test_pytest_row_href_encodes_nodeid():
    href = pytest_row_href("run.html", "cvs/tests/x.py::test_metric[a-128]")
    assert href.startswith("run.html#")
    assert "test_metric" in href


def test_build_cell_record_marks_enforced_failures():
    def _tier_specs(thresholds_cell, tier):
        if tier != "throughput":
            return {}
        return dict(thresholds_cell)

    cfg = replace(generic_inference_report_config(), tier_metric_specs=_tier_specs)
    variant = SimpleNamespace(
        enforce_thresholds=True,
        thresholds={
            "ISL=1024,OSL=1024,TP=8,CONC=128": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 5000.0},
            },
        },
        cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
    )
    key = ("org/example-model", "mi300x", "1024", "1024", "default", 128)
    cell = build_cell_record(
        cfg,
        key=key,
        host="10.0.0.1",
        actuals={"client.output_throughput": 1200.0},
        variant_config=variant,
        lifecycle_report={},
        multi_host=False,
    )
    throughput = next(m for m in cell["metrics"] if m["metric"] == "client.output_throughput")
    assert throughput["status"] == "fail"
    assert cell["tiers"]["throughput"] == "fail"


def test_bar_pct_branches():
    assert bar_pct(50.0, {"kind": "min_tok_s", "value": 100.0}) == 50.0
    assert bar_pct(10.0, {"kind": "max_ms", "value": 100.0}) == 100.0
    assert bar_pct(0.0, {"kind": "max_ms", "value": 100.0}) == 100.0
    assert bar_pct(1.0, {"kind": "within", "value": 1.0}) == 100.0
    assert bar_pct(1.0, {"kind": "unknown", "value": 1.0}) == 50.0
    assert bar_pct(1.0, {"kind": "min", "value": 0.0}) == 0.0
