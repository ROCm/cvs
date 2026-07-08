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
        results_columns=(
            ("Model", None),
            ("GPU", None),
            ("ISL", None),
            ("OSL", None),
            ("Policy", None),
            ("Concurrency", None),
            ("Host", None),
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


def multi_shape_inf_res():
    inf_res = {}
    shapes = (
        ("1024", "1024", "c16_1k1k", 16, 741.0),
        ("1024", "1024", "c32_1k1k", 32, 1305.0),
        ("8192", "1024", "c16_8k1k", 16, 546.0),
        ("8192", "1024", "c32_8k1k", 32, 787.0),
    )
    for isl, osl, policy, conc, tput in shapes:
        key = ("amd/Kimi-K2.5-W4A8", "mi300x", isl, osl, policy, conc)
        inf_res[key] = {
            "10.32.81.141": {
                "client.output_throughput": tput,
                "client.total_token_throughput": tput * 2,
                "client.mean_ttft_ms": 200.0,
                "client.mean_tpot_ms": 25.0,
                "client.p99_ttft_ms": 500.0,
                "client.p99_tpot_ms": 30.0,
                "client.p99_itl_ms": 80.0,
            }
        }
    return inf_res
