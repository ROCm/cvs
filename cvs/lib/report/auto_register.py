'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Auto-load Run Deck profiles for ``cvs run <suite_stem>``.

Loads ``profiles/{stem}.json`` when present.
'''

from __future__ import annotations

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.registry import get_resolved_profile, register_deck_profile


def try_auto_register_suite_report(pytest_config) -> bool:
    """Register a deck profile from JSON when not already configured."""
    if get_resolved_profile(pytest_config) is not None:
        return False

    stem = getattr(pytest_config, "_suite_name", None)
    if not stem:
        return False

    json_profile = load_json_profile(stem)
    if json_profile is not None:
        register_deck_profile(pytest_config, json_profile)
        return True

    return False


def try_auto_register_inference_suite_report(pytest_config) -> bool:
    """Backward-compatible alias for existing imports."""
    return try_auto_register_suite_report(pytest_config)
