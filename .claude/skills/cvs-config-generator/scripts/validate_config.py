#!/usr/bin/env python3
"""Validate a CVS config JSON using the same loaders as the test suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FRAMEWORKS = (
    "auto",
    "cluster",
    "preflight",
    "aorta",
    "pytorch_xdit_wan",
    "pytorch_xdit_flux",
    "sglang",
    "vllm",
    "atom",
    "megatron",
    "torchtitan",
    "jaxmaxtext",
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "cvs" / "parsers" / "schemas.py").is_file():
            return parent
    raise SystemExit("Could not locate CVS repo root (cvs/parsers/schemas.py)")


def _peek_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as fp:
        raw = json.load(fp)
    if raw is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a JSON object: {config_path}")
    return raw


def detect_framework(raw: dict[str, Any]) -> str:
    """Infer the CVS suite/framework from config JSON shape."""
    if "node_dict" in raw:
        return "cluster"
    if "preflight" in raw:
        return "preflight"
    if "aorta_path" in raw:
        return "aorta"

    framework = raw.get("framework")
    if framework == "vllm":
        return "vllm"
    if framework == "atom":
        return "atom"
    if framework == "sglang_single":
        return "sglang"
    if framework in ("megatron_single", "megatron_distributed"):
        return "megatron"
    if framework in ("torchtitan_single", "torchtitan_distributed"):
        return "torchtitan"
    if framework == "jaxmaxtext":
        return "jaxmaxtext"

    if "config" in raw and "benchmark_params" in raw:
        benchmark = raw.get("benchmark_params") or {}
        if "flux1_dev_t2i" in benchmark:
            return "pytorch_xdit_flux"
        if "wan22_i2v_a14b" in benchmark:
            return "pytorch_xdit_wan"
        return "sglang"

    raise ValueError(
        "Cannot auto-detect framework. Pass --framework with one of: "
        + ", ".join(f for f in FRAMEWORKS if f != "auto")
    )


def _load_megatron(config_path: Path, cluster_dict: dict[str, Any], *, allow_changeme: bool) -> None:
    from cvs.lib.training.megatron.utils.training_config_loader import (
        MegatronVariantConfig,
        _check_no_changeme,
    )
    from cvs.lib.utils.config_loader import substitute_config

    raw, thresholds = substitute_config(config_path, cluster_dict)
    if not allow_changeme:
        _check_no_changeme(raw)
    known = {k: v for k, v in raw.items() if k in MegatronVariantConfig.model_fields}
    known["thresholds"] = thresholds
    MegatronVariantConfig(**known)


def _load_torchtitan(config_path: Path, cluster_dict: dict[str, Any], *, allow_changeme: bool) -> None:
    from cvs.lib.training.torchtitan.training_config_loader import (
        TorchTitanVariantConfig,
        _check_no_changeme,
    )
    from cvs.lib.utils.config_loader import substitute_config

    raw, thresholds = substitute_config(config_path, cluster_dict)
    if not allow_changeme:
        _check_no_changeme(raw)
    known = {k: v for k, v in raw.items() if k in TorchTitanVariantConfig.model_fields}
    known["thresholds"] = thresholds
    TorchTitanVariantConfig(**known)


def validate_config(
    config_path: Path,
    *,
    framework: str = "auto",
    cluster_dict: dict[str, Any] | None = None,
    allow_changeme: bool = True,
) -> str:
    """Validate *config_path* and return the detected/selected framework kind."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"file not found: {config_path}")

    if framework != "auto" and framework not in FRAMEWORKS:
        raise ValueError(f"Unknown --framework {framework!r}; expected one of {FRAMEWORKS}")

    cluster_dict = cluster_dict or {}
    raw = _peek_config(config_path)
    kind = detect_framework(raw) if framework == "auto" else framework

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    config_str = str(config_path)

    if kind in ("cluster", "preflight", "aorta", "pytorch_xdit_wan", "pytorch_xdit_flux"):
        from cvs.parsers.schemas import validate_config_file  # pylint: disable=import-outside-toplevel

        validate_config_file(config_path, config_type=kind)
        return kind

    if kind == "vllm":
        from cvs.lib.inference.utils.vllm_config_loader import load_variant  # pylint: disable=import-outside-toplevel

        load_variant(config_str, cluster_dict)
        return kind

    if kind == "atom":
        from cvs.lib.inference.atom.atom_config_loader import load_variant  # pylint: disable=import-outside-toplevel

        load_variant(config_str, cluster_dict)
        return kind

    if kind == "sglang":
        from cvs.lib.inference.sglang.sglang_config_loader import load_variant  # pylint: disable=import-outside-toplevel

        load_variant(config_str, cluster_dict)
        return kind

    if kind == "megatron":
        _load_megatron(config_path, cluster_dict, allow_changeme=allow_changeme)
        return kind

    if kind == "torchtitan":
        _load_torchtitan(config_path, cluster_dict, allow_changeme=allow_changeme)
        return kind

    if kind == "jaxmaxtext":
        from cvs.lib.training.jaxmaxtext.utils.training_config_loader import (  # pylint: disable=import-outside-toplevel
            load_training_variant,
        )

        load_training_variant(config_str, cluster_dict)
        return kind

    raise ValueError(f"Unsupported framework kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a CVS config JSON (inference, training, cluster, preflight, aorta)."
    )
    parser.add_argument("config", type=Path, help="Path to config.json")
    parser.add_argument(
        "--framework",
        choices=FRAMEWORKS,
        default="auto",
        help="Force framework/suite (default: auto-detect from JSON shape)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject configs that still contain '<changeme>' (Megatron/TorchTitan only)",
    )
    args = parser.parse_args()

    try:
        kind = validate_config(
            args.config,
            framework=args.framework,
            allow_changeme=not args.strict,
        )
    except Exception as exc:  # noqa: BLE001 — CLI surfaces loader/schema errors to user
        print(f"INVALID [{args.framework}]: {exc}", file=sys.stderr)
        return 1

    print(f"OK [{kind}]: {args.config.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
