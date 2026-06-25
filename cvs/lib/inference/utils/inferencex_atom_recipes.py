'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX ATOM recipe registry for W1+ workloads.

Maps ``ix_recipe_id`` (InferenceX ``amd-master.yaml`` names) to server/client
CLI fragments consumed by :func:`apply_ix_recipe`.
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_RECIPES_PATH = (
    Path(__file__).resolve().parents[3]
    / "input"
    / "config_file"
    / "inference"
    / "inferencex_atom_single"
    / "ix_recipes.json"
)


def recipes_path() -> Path:
    return _RECIPES_PATH


_RECIPES_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def load_recipes() -> Dict[str, Dict[str, Any]]:
    global _RECIPES_CACHE
    if _RECIPES_CACHE is None:
        data = json.loads(_RECIPES_PATH.read_text(encoding="utf-8"))
        _RECIPES_CACHE = {k: v for k, v in data.items() if not k.startswith("_")}
    return _RECIPES_CACHE


def get_recipe(recipe_id: str) -> Dict[str, Any]:
    recipes = load_recipes()
    if recipe_id not in recipes:
        known = ", ".join(sorted(recipes))
        raise ValueError(f"unknown ix_recipe_id {recipe_id!r} (known: {known})")
    return recipes[recipe_id]


def _merge_cli_tokens(base: List[str], extra: List[str]) -> List[str]:
    if not extra:
        return list(base)
    if not base:
        return list(extra)
    return list(base) + list(extra)


def apply_ix_recipe(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Merge recipe ``atom_args`` / ``bench_extra_args`` into a variant config dict."""
    recipe_id = raw.get("ix_recipe_id")
    if not recipe_id:
        return raw

    recipe = get_recipe(recipe_id)
    out = dict(raw)

    if recipe.get("gpu_arch") and out.get("gpu_arch") != recipe["gpu_arch"]:
        raise ValueError(
            f"ix_recipe_id {recipe_id!r} expects gpu_arch={recipe['gpu_arch']!r}, "
            f"got {out.get('gpu_arch')!r}"
        )

    model = out.setdefault("model", {})
    if recipe.get("model_id") and model.get("id") and model["id"] != recipe["model_id"]:
        raise ValueError(
            f"ix_recipe_id {recipe_id!r} expects model.id={recipe['model_id']!r}, "
            f"got {model.get('id')!r}"
        )
    if recipe.get("model_id") and not model.get("id"):
        model = dict(model)
        model["id"] = recipe["model_id"]
        out["model"] = model

    roles = dict(out.get("roles") or {})
    server = dict(roles.get("server") or {})
    config_atom = list(server.get("atom_args") or [])
    server["atom_args"] = _merge_cli_tokens(list(recipe.get("atom_args") or []), config_atom)
    roles["server"] = server
    out["roles"] = roles

    params = dict(out.get("params") or {})
    recipe_bench = (recipe.get("bench_extra_args") or "").strip()
    config_bench = (params.get("bench_extra_args") or "").strip()
    if recipe_bench and config_bench:
        params["bench_extra_args"] = f"{recipe_bench} {config_bench}".strip()
    elif recipe_bench:
        params["bench_extra_args"] = recipe_bench
    out["params"] = params

    return out


def recipe_run_card(recipe_id: str) -> Optional[Dict[str, str]]:
    """Short metadata for logging / HTML run cards."""
    try:
        recipe = get_recipe(recipe_id)
    except ValueError:
        return None
    return {
        "ix_recipe_id": recipe_id,
        "workload": str(recipe.get("workload", "")),
        "model_id": str(recipe.get("model_id", "")),
        "gpu_arch": str(recipe.get("gpu_arch", "")),
    }
