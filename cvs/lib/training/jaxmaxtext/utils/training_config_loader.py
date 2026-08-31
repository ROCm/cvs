'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Load and validate JAX MaxText training variant configs from
``cvs/input/config_file/training/jaxmaxtext/``.

Pydantic models live in ``cvs.schema.config_file.training.jaxmaxtext.variant``.
'''

from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import GATED_METRICS
from cvs.lib.utils.config_loader import substitute_config
from cvs.schema.config_file.training.jaxmaxtext.variant import (
    CheckpointResume,
    Convergence,
    JaxDistributed,
    LossCurve,
    NcclConfig,
    RdmaLib,
    ScalingBaseline,
    SmokeTest,
    Sweep,
    Tokenizer,
    TrainingConfig,
    TrainingVariantConfig,
    validate_thresholds_cover_training as _validate_thresholds_cover_training,
)

__all__ = [
    "CheckpointResume",
    "Convergence",
    "JaxDistributed",
    "LossCurve",
    "NcclConfig",
    "RdmaLib",
    "ScalingBaseline",
    "SmokeTest",
    "Sweep",
    "Tokenizer",
    "TrainingConfig",
    "TrainingVariantConfig",
    "load_training_variant",
    "validate_thresholds_cover_training",
]


def validate_thresholds_cover_training(
    *,
    expected_cells,
    thresholds,
    enforce_thresholds: bool,
    gated_metrics=None,
) -> None:
    """Training threshold coverage check; defaults ``gated_metrics`` to MaxText gated set."""
    if gated_metrics is None:
        gated_metrics = GATED_METRICS
    return _validate_thresholds_cover_training(
        expected_cells=expected_cells,
        thresholds=thresholds,
        enforce_thresholds=enforce_thresholds,
        gated_metrics=gated_metrics,
    )


def load_training_variant(config_path, cluster_dict) -> TrainingVariantConfig:
    """Load and validate a jaxmaxtext variant config + its sibling threshold file."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return TrainingVariantConfig(**raw)
