'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Per-suite inference report presets. **Reference:** ``inferencex_atom.py``.
Import from suite ``conftest.py`` only when overriding auto-discovery; otherwise add
``presets/<cvs_run_stem>.py`` and root ``cvs/conftest.py`` loads it automatically.
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.inference.utils.inference_suite_results_table import INFERENCEX_ATOM_RESULTS_COLUMNS
from cvs.lib.inference.inferencex_atom.inferencex_atom_parsing import (
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


def _atom_run_card_display(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    rc = variant.run_card
    rows: List[Tuple[str, str, bool]] = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
        ("Driver", variant.params.driver, False),
        ("Image pin", rc.atom_image_pin or "\u2014", False),
        ("TP", str(variant.params.tensor_parallelism), False),
        thresholds_run_card_row(variant),
    ]
    if rc.upstream_run_url:
        rows.append(("Upstream", rc.upstream_run_url, True))
    rows.extend(provenance_link_rows(provenance))
    return rows


INFERENCEX_ATOM_REPORT_CONFIG = make_inference_report_config(
    suite_id="inferencex_atom",
    report_basename="inferencex_atom_run_deck",
    title="IX Run Deck",
    subtitle="InferenceX ATOM \u00b7 lab performance summary",
    footer="CVS inferencex_atom_single \u00b7 render-only \u00b7 does not affect gates",
    link_name="IX Run Deck",
    results_columns=INFERENCEX_ATOM_RESULTS_COLUMNS,
    metric_tier_order=METRIC_TIER_ORDER,
    tier_metric_specs=tier_metric_specs,
    metric_units=CLIENT_METRIC_UNITS,
    metric_prefix="client.",
    cell_highlights=(
        ("output_throughput", "Output tok/s"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("p95_tpot_ms", "P95 TPOT (ms)"),
    ),
    chart_series=DEFAULT_PERF_CHART_SERIES,
    inference_test_substring="test_inferencex_atom_inference",
    row_card_extras=False,
    row_card_test_names=("test_cell_metrics",),
    viewer_cell_threshold=16,
    run_card_display_builder=_atom_run_card_display,
)
