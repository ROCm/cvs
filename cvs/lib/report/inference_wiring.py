'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Generic pytest wiring for inference suite reports.

**Suite owners** — add ``cvs/lib/report/profiles/<cvs_run_stem>.json`` (see
``cvs/lib/report/README.md``). Root ``cvs/conftest.py`` auto-registers the profile,
binds session data, and attaches HTML row extras when ``--html`` is set.

Optional explicit registration with a resolved ``InferenceReportConfig``::

    from cvs.lib.report.inference_wiring import configure_inference_suite_report
    from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
    from cvs.lib.report.profile import load_json_profile

    def pytest_configure(config):
        profile = load_json_profile("my_suite")
        configure_inference_suite_report(config, build_inference_config_from_profile(profile))
'''

from __future__ import annotations

from cvs.lib.inference.utils.inference_suite_lifecycle import attach_lifecycle_html_table

from cvs.lib.report.pytest_extras import attach_inference_cell_row_extra
from cvs.lib.report.registry import bind_session_results, register_suite_report
from cvs.lib.report.types import InferenceReportConfig


def configure_inference_suite_report(pytest_config, preset: InferenceReportConfig) -> None:
    """Register a suite report preset (overrides auto-discovery)."""
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


def attach_inference_suite_lifecycle_table(item, report) -> None:
    """Attach per-test lifecycle timing table to pytest-html rows."""

    attach_lifecycle_html_table(item, report)
