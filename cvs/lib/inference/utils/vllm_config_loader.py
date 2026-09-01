'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Load and validate vLLM inference variant configs.

Pydantic models live in ``cvs.schema.config_file.inference.vllm.variant``.
'''

from cvs.lib.inference.utils.vllm_server_metrics import PROM_METRICS
from cvs.lib.utils.config_loader import substitute_config
from cvs.lib.utils.gpu import GPU_METRICS
from cvs.schema.config_file.inference.common.sweep import (
    GoodputSlo,
    Run,
    SeqCombo,
    Sweep,
    validate_sweep_selector,
)
from cvs.schema.config_file.inference.vllm.variant import VariantConfig, VllmRoleServer

GATED_GPU_METRICS = {k for k, _unit in GPU_METRICS}
GATED_PROM_METRICS = {k for k, _unit in PROM_METRICS}

# Backward-compat alias used by server-reuse tests.
RoleServer = VllmRoleServer

__all__ = [
    "GATED_GPU_METRICS",
    "GATED_PROM_METRICS",
    "GoodputSlo",
    "RoleServer",
    "Run",
    "SeqCombo",
    "Sweep",
    "VariantConfig",
    "VllmRoleServer",
    "load_variant",
    "validate_sweep_selector",
]


def load_variant(config_path, cluster_dict):
    raw, thresholds = substitute_config(config_path, cluster_dict)
    known = {k: v for k, v in raw.items() if k in VariantConfig.model_fields}
    known["thresholds"] = thresholds
    return VariantConfig(**known)
