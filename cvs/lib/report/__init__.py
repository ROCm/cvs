'''

Copyright 2025 Advanced Micro Devices, Inc.

All rights reserved.



CVS inference suite reports: HTML/JSON dashboards bundled with pytest-html output.

Per-suite bindings belong in ``presets/`` and are registered automatically from the
``cvs run`` stem (see ``auto_register.py``).

'''

from cvs.lib.report.inference import (
    build_inference_report_payload,
    publish_inference_suite_report,
    render_report_html,
    write_report,
)
from cvs.lib.report.registry import (
    bind_session_results,
    get_session_results,
    get_suite_report_config,
    register_suite_report,
)
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries

__all__ = [
    "InferenceReportConfig",
    "ReportChartSeries",
    "bind_session_results",
    "build_inference_report_payload",
    "get_session_results",
    "get_suite_report_config",
    "publish_inference_suite_report",
    "register_suite_report",
    "render_report_html",
    "write_report",
]
