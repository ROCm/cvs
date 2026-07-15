'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Session-scoped store and pytest config registration for suite reports.
'''

from __future__ import annotations

from typing import Any, Optional

from cvs.lib.report.types import InferenceReportConfig

_SESSION: dict[str, Any] = {
    "inf_res_dict": None,
    "variant_config": None,
    "lifecycle_report": None,
    "runtime_provenance": None,
}


def register_suite_report(pytest_config, report_config: InferenceReportConfig) -> None:
    """Register a per-suite report preset on the active pytest config."""
    pytest_config._suite_report_config = report_config


def get_suite_report_config(pytest_config) -> Optional[InferenceReportConfig]:
    return getattr(pytest_config, "_suite_report_config", None)


def bind_session_results(
    *,
    inf_res_dict=None,
    variant_config=None,
    lifecycle=None,
) -> None:
    """Capture module-scoped suite state for session-end report generation."""
    if inf_res_dict is not None:
        existing = _SESSION.get("inf_res_dict")
        if isinstance(existing, dict) and isinstance(inf_res_dict, dict) and existing:
            merged = dict(existing)
            merged.update(inf_res_dict)
            _SESSION["inf_res_dict"] = merged
        else:
            _SESSION["inf_res_dict"] = inf_res_dict
    if variant_config is not None:
        _SESSION["variant_config"] = variant_config
    if lifecycle is not None:
        lifecycle_report = getattr(lifecycle, "report", lifecycle)
        existing = _SESSION.get("lifecycle_report")
        if isinstance(existing, dict) and isinstance(lifecycle_report, dict) and existing:
            merged = dict(existing)
            merged.update(lifecycle_report)
            _SESSION["lifecycle_report"] = merged
        else:
            _SESSION["lifecycle_report"] = lifecycle_report


def bind_runtime_provenance(**fields: str) -> None:
    """Capture host/runtime metadata (e.g. resolved container image digest)."""
    if not fields:
        return
    existing = _SESSION.get("runtime_provenance")
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update({k: str(v) for k, v in fields.items() if v})
    _SESSION["runtime_provenance"] = merged


def get_session_results() -> dict[str, Any]:
    return dict(_SESSION)


def clear_session_results() -> None:
    _SESSION["inf_res_dict"] = None
    _SESSION["variant_config"] = None
    _SESSION["lifecycle_report"] = None
    _SESSION["runtime_provenance"] = None
