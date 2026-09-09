'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Deck profile discovery and source resolution for CVS Run Deck.

Deck profiles live under ``profiles/{stem}.json`` and are loaded by ``auto_register``.
Profiles may ``extends`` another JSON file (stem without ``.json``) to share sweep/cards
config; the child overlay wins on conflict.
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from cvs.lib.report.types import InferenceReportConfig

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

# Multiple ``cvs run`` stems may share one deck profile (same sweep/report layout).
PROFILE_STEM_ALIASES: dict[str, str] = {
    "sglang_single": "sglang",
    "sglang_distributed": "sglang",
    "sglang_disagg_distributed": "sglang",
    "vllm_single": "vllm",
    "vllm_distributed": "vllm",
}

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


def _deep_merge_profile(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_profile(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_profile(stem: str, *, _stack: Optional[frozenset[str]] = None) -> Optional[dict[str, Any]]:
    """Load a JSON deck profile when ``profiles/{stem}.json`` exists."""
    path = profile_json_path(stem)
    if not path.is_file():
        alias = PROFILE_STEM_ALIASES.get(stem)
        if alias:
            return load_json_profile(alias, _stack=_stack)
        return None

    stack = _stack or frozenset()
    if stem in stack:
        raise ValueError(f"Deck profile extends cycle detected: {stem}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Deck profile must be a JSON object: {path}")

    extends = data.get("extends")
    if extends:
        parent = load_json_profile(str(extends), _stack=stack | {stem})
        if parent is None:
            raise ValueError(f"Deck profile {path.name} extends missing profile {extends!r}")
        data = _deep_merge_profile(parent, data)

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
