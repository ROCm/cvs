'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

CVS Run Deck reporting: HTML/JSON dashboards bundled with pytest-html output.
'''

from __future__ import annotations

from typing import Any

from cvs.lib.report.registry import (
    bind_session_results,
    get_session_results,
    get_suite_report_config,
    register_suite_report,
    resolve_suite_report_config,
)
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries

_LAZY = {
    "build_inference_report_payload",
    "render_report_html",
    "write_report",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from cvs.lib.report import inference as _inference

        return getattr(_inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InferenceReportConfig",
    "ReportChartSeries",
    "bind_session_results",
    "build_inference_report_payload",
    "get_session_results",
    "get_suite_report_config",
    "register_suite_report",
    "resolve_suite_report_config",
    "render_report_html",
    "write_report",
]
