from cvs.lib.inference.inference_suite_results_table import SGLANG_DISAGG_RESULTS_COLUMNS  # you define
from cvs.lib.inference.utils.sglang_disagg_parsing import (
    CLIENT_METRIC_UNITS,
    METRIC_TIER_ORDER,
    tier_metric_specs,
)
from cvs.lib.report.chart_presets import DEFAULT_PERF_CHART_SERIES
from cvs.lib.report.presets.builder import (
    make_inference_report_config,
    provenance_link_rows,
    thresholds_run_card_row,
)

SGLANG_DISAGG_REPORT_CONFIG = make_inference_report_config(
    suite_id="sglang_disagg_distributed",
    report_basename="sglang_disagg_run_deck",
    title="SGLang Disagg Run Deck",
    subtitle="SGLang disaggregated PD · lab performance summary",
    results_columns=SGLANG_DISAGG_RESULTS_COLUMNS,
    metric_units=CLIENT_METRIC_UNITS,
    tier_metric_specs=tier_metric_specs,
    metric_tier_order=METRIC_TIER_ORDER,
    metric_prefix="client.",
    chart_series=DEFAULT_PERF_CHART_SERIES,  # or custom including MFU
    inference_test_substring="test_run_performance_benchmark",
    row_card_extras=False,
    row_card_test_names=("test_cell_metrics",),  # if you add gate rows
    run_card_display_builder=_sglang_run_card,  # like atom’s _atom_run_card_display
)