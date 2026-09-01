'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Load and validate TorchTitan training variant configs from
``cvs/input/config_file/training/torchtitan/``.

Pydantic models live in ``cvs.schema.config_file.training.torchtitan.variant``.
'''

from cvs.lib.utils.config_loader import substitute_config
from cvs.schema.config_file.training.torchtitan.variant import TorchTitanVariantConfig


def _check_no_changeme(node, path="", _offenders=None):
    """Recursively collect config fields whose value still contains '<changeme>'."""
    if _offenders is None:
        _offenders = []
    if isinstance(node, dict):
        for k, v in node.items():
            _check_no_changeme(v, f"{path}.{k}" if path else k, _offenders)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check_no_changeme(v, f"{path}[{i}]", _offenders)
    elif isinstance(node, str) and "<changeme>" in node:
        _offenders.append(path)
    if not path:
        if _offenders:
            raise ValueError(f"config has unfilled placeholder '<changeme>' in: {', '.join(_offenders)}")


def load_training_variant(config_path, cluster_dict) -> TorchTitanVariantConfig:
    """Load and validate a TorchTitan training variant config + its threshold file."""
    raw, thresholds = substitute_config(config_path, cluster_dict)

    _check_no_changeme(raw)

    known = {k: v for k, v in raw.items() if k in TorchTitanVariantConfig.model_fields}
    known["thresholds"] = thresholds
    return TorchTitanVariantConfig(**known)
