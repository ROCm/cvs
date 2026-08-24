'''Shared generic fixtures for report unit tests (not suite-specific).'''

from types import SimpleNamespace

from cvs.lib.report.rundeck.config_builder import _default_run_card
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries


def generic_variant():
    return SimpleNamespace(
        model=SimpleNamespace(id="org/example-model"),
        gpu_arch="mi300x",
        enforce_thresholds=True,
        run_card=SimpleNamespace(upstream_run_url=""),
        params=SimpleNamespace(driver="6.2", tensor_parallelism=8, nnodes=1),
        thresholds={
            "ISL=1024,OSL=1024,TP=8,CONC=128": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
            },
            "ISL=1024,OSL=1024,TP=8,CONC=256": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
            },
        },
        cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
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
            {"client.output_throughput": {"kind": "min_tok_s", "value": 1000.0}} if tier == "throughput" else {}
        ),
        metric_units={"output_throughput": "tok/s"},
        cell_highlights=(("output_throughput", "Output tok/s"),),
        chart_series=(ReportChartSeries("output_throughput", "Output tok/s", "tok/s"),),
        inference_test_substring="test_inference",
        run_card_display_builder=_default_run_card,
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


def generic_tier_metric_specs(cell, tier):
    if tier != "throughput":
        return {}
    return {"client.output_throughput": {"kind": "min_tok_s", "value": 1000.0}}


GENERIC_METRIC_UNITS = {
    "output_throughput": "tok/s",
    "mean_ttft_ms": "ms",
    "mean_tpot_ms": "ms",
}


def generic_sweep_profile() -> dict:
    """Minimal sweep deck profile for engine unit tests (not a shipped suite profile)."""
    return {
        "schema_version": 1,
        "profile_id": "generic_sweep_rundeck",
        "suite_id": "test_inference_suite",
        "report_basename": "test_inference_suite_run_deck",
        "title": "Test Sweep Run Deck",
        "subtitle": "generic unit test profile",
        "footer": "render-only",
        "link_name": "Test Run Deck",
        "dataset_builder": "sweep",
        "interactive_viewer": True,
        "sources": {
            "results": "inf_res_dict",
            "variant": "variant_config",
            "lifecycle": "lifecycle",
        },
        "hooks": {
            "tier_metric_specs": "cvs.lib.report.testing.fixtures:generic_tier_metric_specs",
            "metric_units": "cvs.lib.report.testing.fixtures:GENERIC_METRIC_UNITS",
        },
        "sweep": {
            "metric_prefix": "client.",
            "tier_order": ["throughput", "record"],
            "throughput_metric": "client.output_throughput",
            "ttft_metric": "client.mean_ttft_ms",
            "headline_metric": "client.output_throughput",
            "results_table_columns": [
                ["Model", None],
                ["GPU", None],
                ["ISL", None],
                ["OSL", None],
                ["Policy", None],
                ["Conc", None],
                ["Host", None],
                ["Output tok/s", "client.output_throughput"],
                ["Mean TTFT (ms)", "client.mean_ttft_ms"],
                ["Mean TPOT (ms)", "client.mean_tpot_ms"],
            ],
            "cell_highlights": [
                ["output_throughput", "Output tok/s"],
                ["mean_ttft_ms", "Mean TTFT (ms)"],
                ["mean_tpot_ms", "Mean TPOT (ms)"],
            ],
            "chart_series": [
                {"metric_suffix": "output_throughput", "title": "Output tok/s", "unit": "tok/s", "invert": False},
                {"metric_suffix": "mean_ttft_ms", "title": "Mean TTFT", "unit": "ms", "invert": True},
                {"metric_suffix": "mean_tpot_ms", "title": "Mean TPOT", "unit": "ms", "invert": True},
            ],
        },
        "cards": [
            {"type": "run_card", "id": "run-card", "title": "Run card", "bind": "run_card_display"},
            {"type": "sweep_analytics", "id": "sweep", "title": "Sweep analytics", "bind": "datasets.sweep"},
            {"type": "interactivity_viewer", "id": "interactivity", "title": "Interactivity", "bind": "viewer_config"},
            {"type": "gate_matrix", "id": "gates", "title": "Gate matrix", "bind": "gate_matrix"},
            {"type": "table", "id": "results", "title": "Full results", "bind": "results_table"},
        ],
        "viewer": {
            "interactivity": {
                "enabled": True,
                "tpot_metric": "client.mean_tpot_ms",
                "output_throughput_metric": "client.output_throughput",
                "total_throughput_metric": "client.total_token_throughput",
            }
        },
    }
