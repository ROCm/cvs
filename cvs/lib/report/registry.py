'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Session-scoped store and pytest config registration for suite reports.
'''

from __future__ import annotations

from typing import Any, Optional, Union

from cvs.lib.report.types import InferenceReportConfig, TrainingReportConfig

_SESSION: dict[str, Any] = {
    "inf_res_dict": None,
    "training_res_dict": None,
    "variant_config": None,
    "lifecycle_report": None,
}


def register_suite_report(
    pytest_config,
    report_config: Union[InferenceReportConfig, TrainingReportConfig],
) -> None:
    """Register a per-suite report preset on the active pytest config."""
    pytest_config._suite_report_config = report_config


def get_suite_report_config(
    pytest_config,
) -> Optional[Union[InferenceReportConfig, TrainingReportConfig]]:
    return getattr(pytest_config, "_suite_report_config", None)


def bind_session_results(
    *,
    inf_res_dict=None,
    training_res_dict=None,
    variant_config=None,
    lifecycle=None,
) -> None:
    """Capture module-scoped suite state for session-end report generation."""
    if inf_res_dict is not None:
        _SESSION["inf_res_dict"] = inf_res_dict
    if training_res_dict is not None:
        _SESSION["training_res_dict"] = training_res_dict
    if variant_config is not None:
        _SESSION["variant_config"] = variant_config
    if lifecycle is not None:
        _SESSION["lifecycle_report"] = getattr(lifecycle, "report", lifecycle)


def get_session_results() -> dict[str, Any]:
    return dict(_SESSION)


def clear_session_results() -> None:
    _SESSION["inf_res_dict"] = None
    _SESSION["training_res_dict"] = None
    _SESSION["variant_config"] = None
    _SESSION["lifecycle_report"] = None
