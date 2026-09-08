"""Typed configuration for the PyTorch Vision training suite."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Any, Dict, List

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.lib.training.pytorch_vision.utils.metrics import GATED_METRICS
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config


class VisionSweep(_Forbid):
    name: str
    label: str
    model: str
    backend: Literal["torchvision"] = "torchvision"
    precision: Literal["BF16", "FP16", "FP32"] = "BF16"
    batch_size: int = Field(gt=0)
    image_size: int = Field(default=224, gt=0)


def validate_sweep_selector(sweep_names, enabled_names, sweep_labels=None) -> None:
    """Reject duplicate sweeps/labels and enabled selectors with no declared run."""
    counts = Counter(sweep_names)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate training.sweeps names: {duplicates}")

    known = set(counts)
    unknown = sorted(ref for ref in enabled_names if ref not in known)
    if unknown:
        raise ValueError(f"training.enabled_sweep_list references unknown sweeps: {unknown} (known: {sorted(known)})")

    if sweep_labels is not None:
        label_counts = Counter(sweep_labels)
        duplicate_labels = sorted(key for key, count in label_counts.items() if count > 1)
        if duplicate_labels:
            raise ValueError(f"duplicate training.sweeps labels: {duplicate_labels}")
        collisions = sorted(known & set(sweep_labels))
        if collisions:
            raise ValueError(f"training sweep names and labels must be distinct: {collisions}")


class VisionTrainingConfig(_Forbid):
    distributed: Literal[False] = False
    gpus_per_node: int = Field(default=8, gt=0)
    warmup_steps: int = Field(default=10, ge=1)
    steps: int = Field(default=50, ge=1)
    num_classes: int = Field(default=1000, gt=1)
    channels_last: bool = True
    learning_rate: float = Field(default=0.1, gt=0)
    momentum: float = Field(default=0.9, ge=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    timeout_s: int = Field(default=1800, gt=0)
    omp_num_threads: int = Field(default=1, gt=0)
    verify_dmesg: bool = True
    env_vars: Dict[str, str] = Field(default_factory=dict)
    error_patterns: Dict[str, str] = Field(default_factory=dict)
    sweeps: List[VisionSweep]
    enabled_sweep_list: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_sweeps(self):
        if not self.sweeps:
            raise ValueError("training.sweeps must contain at least one workload")
        enabled = self.enabled_sweep_list or [sweep.name for sweep in self.sweeps]
        validate_sweep_selector(
            [sweep.name for sweep in self.sweeps],
            enabled,
            [sweep.label for sweep in self.sweeps],
        )
        for name, pattern in self.error_patterns.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"training.error_patterns[{name!r}] is invalid: {exc}") from exc
        return self

    def enabled_sweeps(self) -> List[VisionSweep]:
        by_name = {sweep.name: sweep for sweep in self.sweeps}
        names = self.enabled_sweep_list or list(by_name)
        return [by_name[name] for name in names]


class VisionVariantConfig(BaseVariantConfig):
    schema_version: Literal[1]
    framework: Literal["pytorch_vision_training"]
    gpu_arch: str
    training: VisionTrainingConfig

    def sweep(self, sweep_ref: str) -> VisionSweep:
        for sweep in self.training.sweeps:
            if sweep_ref in (sweep.name, sweep.label):
                return sweep
        raise KeyError(f"unknown PyTorch Vision sweep: {sweep_ref}")

    def cell_key(self, sweep_ref: str, image_size=None, batch_size=None) -> str:
        sweep = self.sweep(sweep_ref)
        if image_size is not None and str(image_size) != str(sweep.image_size):
            raise ValueError(f"report image size {image_size} does not match {sweep.label}: {sweep.image_size}")
        if batch_size is not None and int(batch_size) != sweep.batch_size:
            raise ValueError(f"report batch size {batch_size} does not match {sweep.label}: {sweep.batch_size}")
        return sweep.name

    def expected_cells(self) -> List[str]:
        return [sweep.name for sweep in self.training.sweeps]

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
