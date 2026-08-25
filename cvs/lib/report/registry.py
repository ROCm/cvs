'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Session-scoped store and pytest config registration for suite reports.
'''

from __future__ import annotations

from typing import Any, Optional

from cvs.lib.report.profile import (
    DEFAULT_SOURCES,
    DeckProfile,
    load_json_profile,
    sources_for_profile,
)
from cvs.lib.report.types import InferenceReportConfig


class ReportSessionStore:
    """Session-scoped capture of suite results, lifecycle, and provenance."""

    _EMPTY: dict[str, Any] = {
        "cvs_results_dict": None,
        "inf_res_dict": None,
        "variant_config": None,
        "lifecycle_report": None,
        "runtime_provenance": None,
        "golden_results": None,
        "reference_results": None,
    }

    def __init__(self) -> None:
        self._session: dict[str, Any] = dict(self._EMPTY)

    def bind_results(
        self,
        *,
        inf_res_dict=None,
        cvs_results_dict=None,
        variant_config=None,
        lifecycle=None,
    ) -> None:
        results = cvs_results_dict if cvs_results_dict is not None else inf_res_dict
        if results is not None:
            existing = self._session.get("cvs_results_dict") or self._session.get("inf_res_dict")
            if isinstance(existing, dict) and isinstance(results, dict) and existing:
                merged = dict(existing)
                merged.update(results)
                self._session["cvs_results_dict"] = merged
                self._session["inf_res_dict"] = merged
            else:
                self._session["cvs_results_dict"] = results
                self._session["inf_res_dict"] = results
        if variant_config is not None:
            self._session["variant_config"] = variant_config
        if lifecycle is not None:
            lifecycle_report = getattr(lifecycle, "report", lifecycle)
            existing = self._session.get("lifecycle_report")
            if isinstance(existing, dict) and isinstance(lifecycle_report, dict) and existing:
                merged = dict(existing)
                merged.update(lifecycle_report)
                self._session["lifecycle_report"] = merged
            else:
                self._session["lifecycle_report"] = lifecycle_report

    def bind_from_sources(
        self,
        *,
        results=None,
        variant_config=None,
        lifecycle=None,
        reference=None,
        golden=None,
    ) -> None:
        self.bind_results(
            cvs_results_dict=results,
            variant_config=variant_config,
            lifecycle=lifecycle,
        )
        if reference is not None:
            self._session["reference_results"] = reference
        if golden is not None:
            self._session["golden_results"] = golden

    def bind_runtime_provenance(self, **fields: str) -> None:
        if not fields:
            return
        existing = self._session.get("runtime_provenance")
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update({k: str(v) for k, v in fields.items() if v})
        self._session["runtime_provenance"] = merged

    def get_results(self) -> dict[str, Any]:
        results = dict(self._session)
        if results.get("cvs_results_dict") is None and results.get("inf_res_dict") is not None:
            results["cvs_results_dict"] = results["inf_res_dict"]
        elif results.get("inf_res_dict") is None and results.get("cvs_results_dict") is not None:
            results["inf_res_dict"] = results["cvs_results_dict"]
        return results

    def clear(self) -> None:
        self._session = dict(self._EMPTY)


_STORE = ReportSessionStore()


def register_suite_report(pytest_config, report_config: InferenceReportConfig) -> None:
    """Register a resolved config object (unit tests and internal callers only)."""
    pytest_config._suite_report_config = report_config


def register_deck_profile(pytest_config, profile: DeckProfile) -> None:
    """Register a JSON deck profile on the active pytest config."""
    pytest_config._suite_report_config = profile


def get_suite_report_config(pytest_config) -> Optional[InferenceReportConfig]:
    profile = get_resolved_profile(pytest_config)
    if isinstance(profile, InferenceReportConfig):
        return profile
    return None


def resolve_suite_report_config(pytest_config) -> Optional[InferenceReportConfig]:
    """Resolve JSON deck profiles to ``InferenceReportConfig`` for payload builders."""
    profile = get_resolved_profile(pytest_config)
    if profile is None:
        return None
    if isinstance(profile, InferenceReportConfig):
        return profile
    from cvs.lib.report.rundeck.config_adapter import resolve_report_config

    try:
        return resolve_report_config(profile)
    except (TypeError, ValueError):
        return None


def get_resolved_profile(pytest_config) -> Optional[DeckProfile]:
    return getattr(pytest_config, "_suite_report_config", None)


def get_sources(pytest_config) -> dict[str, str]:
    """Fixture bindings for the active deck profile."""
    profile = get_resolved_profile(pytest_config)
    if profile is None:
        return {}
    return sources_for_profile(profile)


def bind_session_results(
    *,
    inf_res_dict=None,
    cvs_results_dict=None,
    variant_config=None,
    lifecycle=None,
) -> None:
    """Capture module-scoped suite state for session-end report generation."""
    _STORE.bind_results(
        inf_res_dict=inf_res_dict,
        cvs_results_dict=cvs_results_dict,
        variant_config=variant_config,
        lifecycle=lifecycle,
    )


def bind_session_from_sources(
    *,
    results=None,
    variant_config=None,
    lifecycle=None,
    reference=None,
    golden=None,
) -> None:
    """Bind standardized session keys from profile ``sources`` fixture values."""
    _STORE.bind_from_sources(
        results=results,
        variant_config=variant_config,
        lifecycle=lifecycle,
        reference=reference,
        golden=golden,
    )


def bind_runtime_provenance(**fields: str) -> None:
    """Capture host/runtime metadata (e.g. resolved container image digest)."""
    _STORE.bind_runtime_provenance(**fields)


def get_session_results() -> dict[str, Any]:
    return _STORE.get_results()


def clear_session_results() -> None:
    _STORE.clear()


def try_load_json_profile_for_stem(stem: str):
    """Load ``profiles/{stem}.json`` when present (no pytest config mutation)."""
    return load_json_profile(stem)


__all__ = [
    "DEFAULT_SOURCES",
    "ReportSessionStore",
    "bind_runtime_provenance",
    "bind_session_from_sources",
    "bind_session_results",
    "clear_session_results",
    "get_resolved_profile",
    "get_session_results",
    "get_sources",
    "get_suite_report_config",
    "resolve_suite_report_config",
    "register_deck_profile",
    "register_suite_report",
    "try_load_json_profile_for_stem",
]
