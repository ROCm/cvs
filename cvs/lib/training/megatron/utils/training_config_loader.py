'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Load and validate Megatron training variant configs from
``cvs/input/config_file/training/megatron/``.

Pydantic models live in ``cvs.schema.config_file.training.megatron.variant``.
'''

from cvs.lib.utils.config_loader import substitute_config
from cvs.schema.config_file.training.megatron.variant import MegatronVariantConfig


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


def load_training_variant(config_path, cluster_dict) -> MegatronVariantConfig:
    """Load and validate a Megatron training variant config + its threshold file."""
    raw, thresholds = substitute_config(config_path, cluster_dict)

    if not raw.get("checkpoint", {}).get("enforce", False):
        raw.get("checkpoint", {}).pop("checkpoint_dir", None)
        try:
            vols = raw["container"]["runtime"]["args"]["volumes"]
            raw["container"]["runtime"]["args"]["volumes"] = [v for v in vols if "<changeme>" not in v]
        except (KeyError, TypeError):
            pass

    _check_no_changeme(raw)

    known = {k: v for k, v in raw.items() if k in MegatronVariantConfig.model_fields}
    known["thresholds"] = thresholds
    return MegatronVariantConfig(**known)
