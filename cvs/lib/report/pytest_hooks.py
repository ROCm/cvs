'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Pytest hooks for CVS Run Deck session wiring.

Root ``cvs/conftest.py`` delegates here so fixture binding follows deck profile
``sources`` instead of hard-coded inference fixture names.
'''

from __future__ import annotations

import logging

import pytest

from cvs.lib.report.profile import RESULTS_FIXTURE_ALIASES, is_rundeck_profile
from cvs.lib.report.registry import (
    bind_session_from_sources,
    clear_session_results,
    get_resolved_profile,
    get_sources,
)

log = logging.getLogger(__name__)


def _resolve_fixture(request, fixture_name: str):
    """Resolve a fixture by name, accepting legacy results aliases."""
    if fixture_name in RESULTS_FIXTURE_ALIASES:
        last_error: pytest.FixtureLookupError | None = None
        for alias in RESULTS_FIXTURE_ALIASES:
            try:
                return request.getfixturevalue(alias)
            except pytest.FixtureLookupError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    return request.getfixturevalue(fixture_name)


def cvs_rundeck_session_fixture(request):
    """Initialize the session report store when a deck profile is registered."""
    if not is_rundeck_profile(get_resolved_profile(request.config)):
        yield
        return

    clear_session_results()
    yield


def cvs_rundeck_bind_module_fixture(request, _cvs_rundeck_session):
    """Bind module-scoped fixtures into the session store at module teardown."""
    profile = get_resolved_profile(request.config)
    if not is_rundeck_profile(profile):
        yield
        return

    sources = get_sources(request.config)
    if not sources:
        sources = {
            "results": "cvs_results_dict",
            "variant": "variant_config",
            "lifecycle": "lifecycle",
        }

    resolved: dict[str, object] = {}
    optional_roles = {"lifecycle", "reference", "golden", "variant"}
    for role, fixture_name in sources.items():
        try:
            resolved[role] = _resolve_fixture(request, fixture_name)
        except pytest.FixtureLookupError:
            if role in optional_roles:
                resolved[role] = None
                continue
            log.warning(
                "Run Deck profile registered but %s fixture (%s) is missing; session-end report will be skipped",
                role,
                fixture_name,
            )
            yield
            return

    def _bind_at_module_end():
        bind_session_from_sources(
            results=resolved.get("results"),
            variant_config=resolved.get("variant"),
            lifecycle=resolved.get("lifecycle"),
            reference=resolved.get("reference"),
            golden=resolved.get("golden"),
        )

    request.addfinalizer(_bind_at_module_end)
    yield


def attach_rundeck_row_extras(item, report) -> None:
    """Attach sweep row extras when the active profile uses sweep data."""
    from cvs.lib.report.registry import get_resolved_profile, resolve_suite_report_config

    profile = get_resolved_profile(item.config)
    if profile is None:
        return
    if isinstance(profile, dict) and profile.get("dataset_builder") not in (None, "sweep"):
        return
    if resolve_suite_report_config(item.config) is None:
        return

    from cvs.lib.report.inference_wiring import (
        attach_inference_suite_lifecycle_table,
        attach_inference_suite_report_row_extra,
    )

    attach_inference_suite_lifecycle_table(item, report)
    attach_inference_suite_report_row_extra(item, report)
