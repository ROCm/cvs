'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Auto-load inference report presets for ``cvs run <suite_stem>``.

When ``cvs/lib/report/presets/<suite_stem>.py`` exists and defines an
``InferenceReportConfig``, it is registered automatically at session start.
Suite owners can still call ``configure_inference_suite_report`` explicitly.
'''

from __future__ import annotations

import importlib
from typing import Optional

from cvs.lib.report.registry import get_suite_report_config, register_suite_report
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


def try_auto_register_inference_suite_report(pytest_config) -> bool:
    """Register a preset from ``presets.<suite_stem>`` when not already configured."""
    if get_suite_report_config(pytest_config) is not None:
        return False

    stem = getattr(pytest_config, "_suite_name", None)
    if not stem:
        return False

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
