'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Generic pytest wiring for inference suite reports.

Suite conftest should call:

1. ``configure_inference_suite_report(config, PRESET)`` in ``pytest_configure``
2. ``bind_inference_suite_report_session(...)`` from an autouse module fixture
3. ``attach_inference_suite_report_row_extra(item, report)`` from ``pytest_runtest_makereport``

Session-end HTML/JSON generation is handled by root ``conftest.py`` via
``HtmlReportManager.generate_suite_reports`` (no per-suite lifecycle test).
'''

from __future__ import annotations

from cvs.lib.report.pytest_extras import attach_inference_cell_row_extra
from cvs.lib.report.registry import bind_session_results, register_suite_report
from cvs.lib.report.types import InferenceReportConfig


def configure_inference_suite_report(pytest_config, preset: InferenceReportConfig) -> None:
    """Register a suite report preset (inference only)."""
    register_suite_report(pytest_config, preset)


def bind_inference_suite_report_session(
    *,
    inf_res_dict,
    variant_config,
    lifecycle,
) -> None:
    """Capture module-scoped results for session-end report generation."""
    bind_session_results(
        inf_res_dict=inf_res_dict,
        variant_config=variant_config,
        lifecycle=lifecycle,
    )


def attach_inference_suite_report_row_extra(item, report) -> None:
    """Attach compact cell cards to metric test rows in pytest-html."""
    attach_inference_cell_row_extra(item, report)
