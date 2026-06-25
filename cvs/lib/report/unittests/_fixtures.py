'''Shared generic fixtures for report unit tests (not suite-specific).'''

from types import SimpleNamespace

from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries


def generic_variant():
    return SimpleNamespace(
        model=SimpleNamespace(id="org/example-model"),
        gpu_arch="mi300x",
        enforce_thresholds=True,
        thresholds={
            "ISL=1024,OSL=1024,TP=8,CONC=128": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
            },
            "ISL=1024,OSL=1024,TP=8,CONC=256": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
            },
        },
        cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
        params=SimpleNamespace(nnodes=1),
    )


def generic_inference_report_config() -> InferenceReportConfig:
    return InferenceReportConfig(
        suite_id="test_inference_suite",
        report_basename="test_inference_suite_report",
        title="Test inference suite report",
        subtitle="generic unit test",
        footer="render-only",
        link_name="Suite report",
        embed_summary="Suite report",
        results_columns=(
            ("Model", "model"),
            ("GPU", "gpu"),
            ("ISL", "isl"),
            ("OSL", "osl"),
            ("Policy", "policy"),
            ("Concurrency", "concurrency"),
            ("Host", "host"),
            ("Output tok/s", "client.output_throughput"),
        ),
        metric_tier_order=("throughput", "record"),
        tier_metric_specs=lambda _cell, tier: (
            {"client.output_throughput": {"kind": "min_tok_s", "value": 1000.0}}
            if tier == "throughput"
            else {}
        ),
        metric_units={"output_throughput": "tok/s"},
        cell_highlights=(("output_throughput", "Output tok/s"),),
        chart_series=(ReportChartSeries("output_throughput", "Output tok/s", "tok/s"),),
        inference_test_substring="test_inference",
    )


def two_cell_inf_res():
    inf_res = {}
    for conc, tput in ((128, 4000.0), (256, 6000.0)):
        key = ("org/example-model", "mi300x", "1024", "1024", "default", conc)
        inf_res[key] = {
            "10.0.0.1": {
                "client.output_throughput": tput,
                "client.mean_ttft_ms": 100.0,
                "client.mean_tpot_ms": 28.0,
            }
        }
    return inf_res
