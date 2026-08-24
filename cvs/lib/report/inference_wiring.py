'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Optional pytest wiring helpers for sweep suite reports.

Root ``cvs/conftest.py`` auto-registers ``profiles/<stem>.json``, binds session
fixtures, and attaches HTML row extras when ``--html`` is set. Suite owners add
JSON profiles only; use ``configure_deck_profile`` to override auto-discovery.
'''

from __future__ import annotations

from typing import Any

from cvs.lib.report.pytest_extras import attach_inference_cell_row_extra
from cvs.lib.report.registry import bind_session_results, register_deck_profile


def configure_deck_profile(pytest_config, profile: dict[str, Any]) -> None:
    """Register a JSON deck profile (overrides stem auto-discovery)."""
    register_deck_profile(pytest_config, profile)


def configure_inference_suite_report(pytest_config, preset) -> None:
    """Deprecated alias — pass a JSON profile dict to ``configure_deck_profile``."""
    register_deck_profile(pytest_config, preset)


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
    """Attach per-test lifecycle timing table when the inference suite module is present."""
    try:
        from cvs.lib.inference.utils.inference_suite_lifecycle import attach_lifecycle_html_table
    except ImportError:
        return
    attach_lifecycle_html_table(item, report)
