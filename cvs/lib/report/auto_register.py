'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Auto-load Run Deck profiles for ``cvs run <suite_stem>``.

Auto-load order:
  1. ``profiles/{stem}.json`` when present
  2. Legacy ``presets/{stem}.py`` defining ``*_REPORT_CONFIG``
'''

from __future__ import annotations

import importlib
from typing import Optional

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.registry import get_resolved_profile, register_deck_profile, register_suite_report
from cvs.lib.report.types import InferenceReportConfig


def _find_preset_in_module(module) -> Optional[InferenceReportConfig]:
    named: list[InferenceReportConfig] = []
    for name, value in vars(module).items():
        if isinstance(value, InferenceReportConfig):
            if name.endswith("_REPORT_CONFIG"):
                named.append(value)
    if len(named) == 1:
        return named[0]
    if named:
        return named[0]
    for value in vars(module).values():
        if isinstance(value, InferenceReportConfig):
            return value
    return None


def try_auto_register_suite_report(pytest_config) -> bool:
    """Register a deck profile from JSON or legacy preset when not already configured."""
    if get_resolved_profile(pytest_config) is not None:
        return False

    stem = getattr(pytest_config, "_suite_name", None)
    if not stem:
        return False

    json_profile = load_json_profile(stem)
    if json_profile is not None:
        register_deck_profile(pytest_config, json_profile)
        return True

    module_name = f"cvs.lib.report.presets.{stem}"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False

    preset = _find_preset_in_module(module)
    if preset is None:
        return False

    register_suite_report(pytest_config, preset)
    return True


def try_auto_register_inference_suite_report(pytest_config) -> bool:
    """Backward-compatible alias for existing imports."""
    return try_auto_register_suite_report(pytest_config)
