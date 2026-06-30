'''Tests for inference report preset builder.'''

from cvs.lib.report.presets.builder import make_inference_report_config


def test_make_inference_report_config_defaults():
    cfg = make_inference_report_config(
        suite_id="demo_suite",
        results_columns=(
            ("Model", None),
            ("Output tok/s", "client.output_throughput"),
            ("Mean TTFT (ms)", "client.mean_ttft_ms"),
        ),
        metric_units={"output_throughput": "tok/s", "mean_ttft_ms": "ms"},
        tier_metric_specs=lambda _c, _t: {},
    )
    assert cfg.suite_id == "demo_suite"
    assert cfg.report_basename == "demo_suite_report"
    assert cfg.inference_test_substring == "test_demo_suite"
    assert cfg.cell_highlights
    assert cfg.chart_series
    assert cfg.interactive_viewer is True


def test_make_inference_report_config_overrides():
    cfg = make_inference_report_config(
        suite_id="x",
        results_columns=(),
        metric_units={},
        tier_metric_specs=lambda _c, _t: {},
        inference_test_substring="test_custom_inference",
        report_basename="custom_report",
    )
    assert cfg.inference_test_substring == "test_custom_inference"
    assert cfg.report_basename == "custom_report"
