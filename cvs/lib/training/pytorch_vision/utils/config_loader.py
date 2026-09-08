"""Typed configuration for the PyTorch Vision training suite."""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.lib.training.pytorch_vision.utils.metrics import GATED_METRICS
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config


class VisionRun(_Forbid):
    name: str
    model: str
    backend: Literal["torchvision"] = "torchvision"
    precision: Literal["BF16", "FP16", "FP32"] = "BF16"
    batch_size: int = Field(gt=0)
    image_size: int = Field(default=224, gt=0)


def validate_sweep_selector(combo_keys, run_refs) -> None:
    """Reject duplicate combinations and run selectors that reference no cell."""
    counts = Counter(combo_keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sweep.combinations keys: {duplicates}")

    known = set(counts)
    unknown = sorted(ref for ref in run_refs if ref not in known)
    if unknown:
        raise ValueError(f"sweep.runs references unknown combinations: {unknown} (known: {sorted(known)})")


class VisionSweep(_Forbid):
    combinations: Dict[str, VisionRun]
    runs: List[str]

    @model_validator(mode="after")
    def _validate_runs(self):
        validate_sweep_selector(self.combinations.keys(), self.runs)
        if not self.runs:
            raise ValueError("sweep.runs must contain at least one workload")
        return self


class VisionParams(_Forbid):
    nproc_per_node: int = Field(default=8, gt=0)
    warmup_steps: int = Field(default=10, ge=1)
    measure_steps: int = Field(default=50, ge=1)
    num_classes: int = Field(default=1000, gt=1)
    channels_last: bool = True
    learning_rate: float = Field(default=0.1, gt=0)
    momentum: float = Field(default=0.9, ge=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    timeout_s: int = Field(default=1800, gt=0)
    omp_num_threads: int = Field(default=1, gt=0)
    verify_dmesg: bool = True


class VisionVariantConfig(BaseVariantConfig):
    schema_version: Literal[1]
    framework: Literal["pytorch_vision_training"]
    gpu_arch: str
    params: VisionParams
    env: Dict[str, str] = Field(default_factory=dict)
    sweep: VisionSweep

    def cell_key(self, combo_key: str) -> str:
        combo = self.sweep.combinations[combo_key]
        return (
            f"MODEL={combo.model},PRECISION={combo.precision},RES={combo.image_size},"
            f"BS={combo.batch_size},GPUS={self.params.nproc_per_node}"
        )

    def expected_cells(self) -> List[str]:
        return [self.cell_key(key) for key in self.sweep.runs]

    @model_validator(mode="after")
    def _validate_threshold_coverage(self):
        expected = set(self.expected_cells())
        present = set(self.thresholds)
        problems = []

        missing_cells = sorted(expected - present)
        extra_cells = sorted(present - expected)
        if missing_cells:
            problems.append(f"sweep cells with no threshold entry: {missing_cells}")
        if extra_cells:
            problems.append(f"threshold keys matching no sweep cell: {extra_cells}")

        missing_metrics = {}
        required = [f"training.{name}" for name in sorted(GATED_METRICS)]
        for cell in sorted(expected & present):
            absent = [name for name in required if name not in (self.thresholds.get(cell) or {})]
            if absent:
                missing_metrics[cell] = absent
        if missing_metrics:
            problems.append(f"cells missing gated-metric specs: {missing_metrics}")

        if problems:
            message = "threshold.json does not match the PyTorch Vision sweep; " + "; ".join(problems)
            if self.enforce_thresholds:
                raise ValueError(message)
            warnings.warn(f"{message} (enforce_thresholds=false -> record-only)", stacklevel=3)
        return self


def _check_no_changeme(node: Any, path: str = "", offenders=None) -> None:
    if offenders is None:
        offenders = []
    if isinstance(node, dict):
        for key, value in node.items():
            _check_no_changeme(value, f"{path}.{key}" if path else key, offenders)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _check_no_changeme(value, f"{path}[{index}]", offenders)
    elif isinstance(node, str) and "<changeme>" in node:
        offenders.append(path)
    if not path and offenders:
        raise ValueError(f"config has unfilled placeholder '<changeme>' in: {', '.join(offenders)}")


def load_vision_variant(config_path, cluster_dict) -> VisionVariantConfig:
    """Load, substitute, and validate a PyTorch Vision config/threshold pair."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    _check_no_changeme(raw)
    raw["thresholds"] = thresholds
    return VisionVariantConfig(**raw)
