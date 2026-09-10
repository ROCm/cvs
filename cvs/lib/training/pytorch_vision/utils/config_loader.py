"""Typed configuration for the PyTorch Vision training suite."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.lib.training.pytorch_vision.utils.metrics import GATED_METRICS
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config


RUN_MODES = (
    "smoke",
    "perf",
    "train_1epoch",
    "train_5k",
    "train_90epoch",
    "soak_24h",
)
LONG_RUN_MODES = {"train_90epoch", "soak_24h"}


class LossCurveConfig(_Forbid):
    enabled: bool = True
    sample_every_steps: int = Field(default=1, ge=1)
    minimum_points: int = Field(default=2, ge=2)
    require_decrease: bool = True


class ConvergenceConfig(_Forbid):
    enabled: bool = False
    target_top1_pct: Optional[float] = Field(default=None, ge=0, le=100)
    target_eval_loss: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_target(self):
        if self.enabled and self.target_top1_pct is None and self.target_eval_loss is None:
            raise ValueError("enabled convergence tracking requires a target")
        return self


class CodeCarbonConfig(_Forbid):
    enabled: bool = False
    required: bool = False
    version: Literal["3.2.4"] = "3.2.4"
    tracker: Literal["amdsmi"] = "amdsmi"
    measure_power_secs: int = Field(default=5, ge=1)
    country_iso_code: Optional[str] = None

    @model_validator(mode="after")
    def _validate_required(self):
        if self.required and not self.enabled:
            raise ValueError("codecarbon.required=true requires codecarbon.enabled=true")
        return self


class VisionSweep(_Forbid):
    name: str
    label: str
    model: str
    backend: Literal["torchvision"] = "torchvision"
    precision: Literal["BF16", "FP16", "FP32"] = "BF16"
    batch_size: int = Field(gt=0)
    image_size: int = Field(default=224, gt=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    training_flops_per_image: float = Field(gt=0)
    data_mode: Literal["synthetic", "rocal"] = "synthetic"
    dataset_path: str = ""
    rocal_device: Literal["cpu", "gpu"] = "gpu"
    augmentation: Literal["standard", "heavy"] = "standard"
    rocal_num_threads: int = Field(default=8, ge=1)
    loader_warmup_steps: int = Field(default=5, ge=1)
    loader_benchmark_steps: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def _validate_data(self):
        if self.data_mode == "rocal" and not self.dataset_path:
            raise ValueError("rocAL sweeps require dataset_path")
        return self


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
    enabled: bool = True
    allow_long_run: bool = False
    phase: Literal["smoke", "performance", "accuracy", "soak"] = "performance"
    run_mode: Literal[
        "smoke",
        "perf",
        "train_1epoch",
        "train_5k",
        "train_90epoch",
        "soak_24h",
    ] = "perf"
    gpus_per_node: int = Field(default=8, gt=0)
    warmup_steps: int = Field(default=10, ge=1)
    steps: int = Field(default=50, ge=1)
    epochs: Optional[int] = Field(default=None, ge=1)
    max_duration_seconds: Optional[int] = Field(default=None, ge=1)
    eval_enabled: bool = False
    eval_every_epochs: int = Field(default=1, ge=1)
    eval_steps: Optional[int] = Field(default=None, ge=1)
    eval_sample_count: int = Field(default=50_000, ge=1)
    milestone_steps: List[int] = Field(default_factory=lambda: [100, 500, 1000, 5000])
    num_classes: int = Field(default=1000, gt=1)
    channels_last: bool = True
    learning_rate: float = Field(default=0.1, gt=0)
    momentum: float = Field(default=0.9, ge=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    timeout_s: int = Field(default=1800, gt=0)
    omp_num_threads: int = Field(default=1, gt=0)
    verify_dmesg: bool = True
    peak_tflops_per_gpu: float = Field(gt=0)
    checkpoint_enabled: Literal[True] = True
    checkpoint_keep_file: bool = False
    checkpoint_loss_tolerance: float = Field(default=1e-5, ge=0)
    loss_curve: LossCurveConfig = Field(default_factory=LossCurveConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    codecarbon: CodeCarbonConfig = Field(default_factory=CodeCarbonConfig)
    gpu_poll_interval_seconds: float = Field(default=5.0, gt=0)
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
        by_name = {sweep.name: sweep for sweep in self.sweeps}
        selected = [by_name[name] for name in enabled]
        for sweep in selected:
            if sweep.gradient_accumulation_steps == 1:
                continue
            effective_batch = sweep.batch_size * sweep.gradient_accumulation_steps * self.gpus_per_node
            baselines = [
                candidate
                for candidate in selected
                if candidate.gradient_accumulation_steps == 1
                and candidate.model == sweep.model
                and candidate.backend == sweep.backend
                and candidate.precision == sweep.precision
                and candidate.image_size == sweep.image_size
                and candidate.batch_size * self.gpus_per_node == effective_batch
            ]
            if len(baselines) != 1:
                raise ValueError(
                    f"enabled GA={sweep.gradient_accumulation_steps} sweep {sweep.name!r} requires exactly one "
                    "enabled GA=1 sweep with the same model, precision, resolution, and effective global batch"
                )
        for name, pattern in self.error_patterns.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"training.error_patterns[{name!r}] is invalid: {exc}") from exc
        if len(set(self.milestone_steps)) != len(self.milestone_steps):
            raise ValueError("training.milestone_steps must not contain duplicates")
        if any(b <= a for a, b in zip(self.milestone_steps, self.milestone_steps[1:])):
            raise ValueError("training.milestone_steps must be strictly increasing")
        if self.run_mode == "train_1epoch" and self.epochs != 1:
            raise ValueError("train_1epoch requires training.epochs=1")
        if self.run_mode == "train_90epoch" and self.epochs != 90:
            raise ValueError("train_90epoch requires training.epochs=90")
        if self.run_mode == "train_5k" and self.steps < 5000:
            raise ValueError("train_5k requires training.steps>=5000")
        if self.run_mode == "soak_24h" and self.max_duration_seconds != 86400:
            raise ValueError("soak_24h requires training.max_duration_seconds=86400")
        expected_phase = {
            "smoke": "smoke",
            "perf": "performance",
            "train_1epoch": "accuracy",
            "train_5k": "accuracy",
            "train_90epoch": "accuracy",
            "soak_24h": "soak",
        }[self.run_mode]
        if self.phase != expected_phase:
            raise ValueError(f"{self.run_mode} requires training.phase={expected_phase!r}")
        if self.enabled and self.run_mode in LONG_RUN_MODES and not self.allow_long_run:
            raise ValueError(
                f"{self.run_mode} is a protected long run; set training.enabled=true and "
                "training.allow_long_run=true explicitly"
            )
        if self.run_mode != "perf" and not self.eval_enabled:
            raise ValueError(f"{self.run_mode} requires training.eval_enabled=true")
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
