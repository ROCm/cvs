'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

**Copy this file** to ``cvs/lib/report/presets/<cvs_run_stem>.py``.

The filename must match the pytest module stem from ``cvs run <stem>`` (e.g.
``inferencex_atom`` → ``presets/inferencex_atom.py``).

**Reference:** ``inferencex_atom.py`` (IX-atom preset; auto-loaded when ``cvs run inferencex_atom``).
See ``cvs/lib/report/README.md`` for the IX-atom end-to-end example.

Suite owners fill in the TODOs below, keep collecting ``inf_res_dict`` during tests,
and run with ``--html``. No conftest changes required.
'''

from __future__ import annotations

# TODO: column preset + parsing helpers from your suite
# from cvs.lib.inference.utils.inference_suite_results_table import MY_SUITE_RESULTS_COLUMNS
# from cvs.lib.inference.utils.my_parsing import CLIENT_METRIC_UNITS, tier_metric_specs, METRIC_TIER_ORDER
from cvs.lib.report.presets.builder import make_inference_report_config

MY_SUITE_REPORT_CONFIG = make_inference_report_config(
    suite_id="my_suite",  # TODO: short id (used in JSON)
    results_columns=(),  # TODO: MY_SUITE_RESULTS_COLUMNS
    metric_units={},  # TODO: CLIENT_METRIC_UNITS
    tier_metric_specs=lambda _cell, tier: {},  # TODO: tier_metric_specs
    metric_tier_order=("throughput", "latency", "health", "record"),  # TODO: METRIC_TIER_ORDER
    inference_test_substring="test_my_suite_inference",  # TODO: workload test name fragment
    row_card_test_names=("test_metric", "test_cell_metrics"),  # TODO: gate test(s)
)
