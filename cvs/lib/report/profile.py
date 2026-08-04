'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Deck profile discovery and source resolution for CVS Run Deck.

Auto-load order:
  1. ``profiles/{stem}.json`` when present
  2. Legacy ``presets/{stem}.py`` defining ``*_REPORT_CONFIG`` (handled by auto_register)
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from cvs.lib.report.types import InferenceReportConfig

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

DEFAULT_SOURCES: dict[str, str] = {
    "results": "cvs_results_dict",
    "variant": "variant_config",
    "lifecycle": "lifecycle",
}

# During migration inference suites may still expose ``inf_res_dict``.
RESULTS_FIXTURE_ALIASES: tuple[str, ...] = ("cvs_results_dict", "inf_res_dict")

DeckProfile = Union[dict[str, Any], InferenceReportConfig]


def profile_json_path(stem: str) -> Path:
    return _PROFILES_DIR / f"{stem}.json"


def load_json_profile(stem: str) -> Optional[dict[str, Any]]:
    """Load a JSON deck profile when ``profiles/{stem}.json`` exists."""
    path = profile_json_path(stem)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Deck profile must be a JSON object: {path}")
    return data


def sources_for_profile(profile: DeckProfile) -> dict[str, str]:
    """Return fixture bindings for the active deck profile."""
    if isinstance(profile, dict):
        raw = profile.get("sources")
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        return dict(DEFAULT_SOURCES)
    return dict(DEFAULT_SOURCES)


def is_rundeck_profile(profile: DeckProfile | None) -> bool:
    return profile is not None
