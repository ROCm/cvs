'''Unit tests for cell record building.'''

from types import SimpleNamespace

from cvs.lib.report.cell_build import build_cell_record
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
