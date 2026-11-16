"""Typed configuration for the PyTorch Training (LLM) suite."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from typing_extensions import Literal

from cvs.lib.training.pytorch_training.utils.metrics import BENCHMARK_TASKS, GATED_METRICS
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config


_ALLOWED_GPU_NAMES = ("MI300X", "MI325X", "MI355X")

RUN_MODES = ("smoke", "perf", "train_5k", "finetune", "soak")
LONG_RUN_MODES = {"soak"}
# MX formats need the OCP microscaling units that only MI355X has.
MI355X_ONLY_PRECISIONS = {"MXFP8", "MXFP4"}


class LossCurveConfig(_Forbid):
    enabled: bool = True
    sample_every_steps: int = Field(default=1, ge=1)
    minimum_points: int = Field(default=2, ge=2)
    require_decrease: bool = True


class ConvergenceConfig(_Forbid):
    enabled: bool = False
    target_train_loss: Optional[float] = Field(default=None, gt=0)
    target_eval_loss: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_target(self):
        if self.enabled and self.target_train_loss is None and self.target_eval_loss is None:
            raise ValueError("enabled convergence tracking requires a target")
        return self


class ScalingBaselineConfig(_Forbid):
    """Reference throughput, and optionally eval loss, of the same cell on fewer nodes.

    ``tokens_per_sec_total`` is the total tokens/s measured on ``num_nodes``
    nodes. Leave it at 0.0 to keep scaling efficiency record-only.
    ``eval_loss`` is the reference run's eval loss for scale parity.
    """

    tokens_per_sec_total: float = Field(default=0.0, ge=0)
    num_nodes: int = Field(default=1, gt=0)
    eval_loss: Optional[float] = Field(default=None, gt=0)


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


class CheckpointConfig(_Forbid):
    enabled: Literal[True] = True
    # Container path. A sharded checkpoint of a 70B model with optimizer state
    # is over a terabyte, so it belongs on node-local disk, not a shared share.
    directory: str = "/tmp/cvs_pytorch_training_ckpt"
    keep: bool = False
    # Optimizer steps replayed after the reload; their losses are compared
    # with the uninterrupted run's.
    resume_steps: int = Field(default=2, ge=1)


class DataConfig(_Forbid):
    """Raw text source and where its tokenized copy is cached.

    The prepare stage tokenizes ``source`` once per tokenizer into
    ``cache_dir`` and every later run reuses the cached token files.
    """

    source: str = ""
    text_column: str = "text"
    train_split: str = "train"
    # WikiText-103 perplexity is conventionally reported on its test split.
    eval_split: str = "test"
    cache_dir: str = ""
    # Documents are joined with the tokenizer's EOS token between them.
    append_eos: bool = True


class BenchmarkConfig(_Forbid):
    """lm-evaluation-harness checkpoint benchmarks, run after training."""

    enabled: bool = False
    metrics: List[str] = Field(default_factory=list)
    # Per-task sample cap; None scores the full task.
    limit: Optional[int] = Field(default=None, ge=1)
    batch_size: str = "auto"
    # One model replica per GPU; set false when one GPU cannot hold the model
    # and lm-eval should spread a single replica over the node instead.
    data_parallel: bool = True
    timeout_s: int = Field(default=14400, gt=0)

    @model_validator(mode="after")
    def _validate_metrics(self):
        unknown = sorted(set(self.metrics) - set(BENCHMARK_TASKS))
        if unknown:
            raise ValueError(f"benchmarks.metrics has unknown entries {unknown}; known: {sorted(BENCHMARK_TASKS)}")
        if self.enabled and not self.metrics:
            raise ValueError("enabled benchmarks require at least one metric")
        return self


class TrainingSweep(_Forbid):
    name: str
    label: str
    # Short architecture id recorded in the artifact (llama-3.1-8b, ...).
    model: str
    # Container path holding config.json and the tokenizer, plus the
    # safetensors weights when init=pretrained.
    model_path: str
    init: Literal["random", "pretrained"] = "random"
    precision: Literal["BF16", "FP8", "MXFP8", "MXFP4"] = "BF16"
    seq_len: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    # Exactly one of these: a global batch fixes the tokens per step on every
    # topology, while a fixed accumulation count holds the per-rank recipe.
    global_batch_size: Optional[int] = Field(default=None, gt=0)
    gradient_accumulation_steps: Optional[int] = Field(default=None, ge=1)
    # Recompute every Nth decoder layer in backward (1 = all, 0 = none).
    # Checkpointing fewer layers spends spare HBM on stored activations.
    activation_checkpointing_every: int = Field(default=1, ge=0)
    # fp32 keeps an fp32 master copy and computes in bf16; bf16 halves the
    # parameter, gradient, and optimizer state for models that do not fit.
    master_weights: Literal["fp32", "bf16"] = "fp32"
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = "flash_attention_2"
    # torch.compile each decoder layer.
    compile: bool = True
    data_mode: Literal["tokens", "synthetic"] = "tokens"
    # Overrides the FLOPs/token derived from the model config.
    training_flops_per_token: Optional[float] = Field(default=None, gt=0)
    # The 1-node run of this same cell, for scaling efficiency and parity.
    scaling_baseline: Optional[ScalingBaselineConfig] = None

    @model_validator(mode="after")
    def _validate_batch(self):
        if (self.global_batch_size is None) == (self.gradient_accumulation_steps is None):
            raise ValueError(
                f"sweep {self.name!r} must set exactly one of global_batch_size, gradient_accumulation_steps"
            )
        if self.global_batch_size is not None and self.global_batch_size % self.micro_batch_size:
            raise ValueError(f"sweep {self.name!r} global_batch_size must be a multiple of micro_batch_size")
        return self

    def accumulation_steps(self, world_size):
        """GA for ``world_size`` ranks, or a ValueError when the batch does not divide."""
        if self.gradient_accumulation_steps is not None:
            return self.gradient_accumulation_steps
        per_step = self.micro_batch_size * world_size
        if self.global_batch_size % per_step:
            raise ValueError(
                f"sweep {self.name!r}: global_batch_size {self.global_batch_size} is not a multiple of "
                f"micro_batch_size {self.micro_batch_size} x {world_size} ranks"
            )
        return self.global_batch_size // per_step

    def global_batch(self, world_size):
        return self.micro_batch_size * world_size * self.accumulation_steps(world_size)


def validate_sweep_selector(sweep_names, enabled_names, sweep_labels=None):
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


class TrainingConfig(_Forbid):
    # Scorecard workload id (W1 = DeepSeek-V2-Lite, W2 = Llama-3.1-8B BF16, ...).
    workload: str = Field(default="W2", pattern=r"^W\d+$")
    distributed: bool = False
    master_port: int = Field(default=29500, gt=1024, lt=65536)
    enabled: bool = True
    allow_long_run: bool = False
    phase: Literal["smoke", "performance", "accuracy", "soak"] = "performance"
    run_mode: Literal["smoke", "perf", "train_5k", "finetune", "soak"] = "perf"
    gpus_per_node: int = Field(default=8, gt=0)
    warmup_steps: int = Field(default=5, ge=0)
    steps: int = Field(default=20, ge=1)
    # Wall-clock cap; the soak profile runs to this instead of to `steps`.
    max_duration_seconds: Optional[int] = Field(default=None, ge=1)
    seed: int = 1234
    eval_enabled: bool = False
    # 0 evaluates once, after training.
    eval_every_steps: int = Field(default=0, ge=0)
    eval_batches: int = Field(default=8, ge=1)
    milestone_steps: List[int] = Field(default_factory=lambda: [100, 500, 1000, 5000])
    learning_rate: float = Field(default=3e-4, gt=0)
    min_learning_rate_ratio: float = Field(default=0.1, ge=0, le=1)
    lr_schedule: Literal["constant", "cosine"] = "cosine"
    lr_warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.1, ge=0)
    adam_beta1: float = Field(default=0.9, gt=0, lt=1)
    adam_beta2: float = Field(default=0.95, gt=0, lt=1)
    grad_clip_norm: float = Field(default=1.0, ge=0)
    # full_shard spreads parameters over every rank; hybrid_shard shards within
    # a node and replicates across nodes.
    fsdp_strategy: Literal["full_shard", "hybrid_shard"] = "full_shard"
    timeout_s: int = Field(default=3600, gt=0)
    startup_timeout_s: int = Field(default=300, gt=0)
    startup_retries: int = Field(default=0, ge=0, le=5)
    # Rank 0's weight read and the per-rank meta-model build are awaited
    # through the rendezvous store, outside NCCL's collective timeout.
    setup_timeout_s: int = Field(default=1800, gt=0)
    omp_num_threads: int = Field(default=1, gt=0)
    verify_dmesg: bool = True
    # Dense peak per precision, for MFU. Keys are sweep precisions.
    peak_tflops_per_gpu: Dict[str, float] = Field(default_factory=dict)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    loss_curve: LossCurveConfig = Field(default_factory=LossCurveConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    scaling_baseline: ScalingBaselineConfig = Field(default_factory=ScalingBaselineConfig)
    codecarbon: CodeCarbonConfig = Field(default_factory=CodeCarbonConfig)
    benchmarks: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    error_patterns: Dict[str, str] = Field(default_factory=dict)
    sweeps: List[TrainingSweep]
    enabled_sweep_list: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_training(self):
        if not self.sweeps:
            raise ValueError("training.sweeps must contain at least one workload")
        enabled = self.enabled_sweep_list or [sweep.name for sweep in self.sweeps]
        validate_sweep_selector([s.name for s in self.sweeps], enabled, [s.label for s in self.sweeps])
        for name, pattern in self.error_patterns.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"training.error_patterns[{name!r}] is invalid: {exc}") from exc
        if len(set(self.milestone_steps)) != len(self.milestone_steps):
            raise ValueError("training.milestone_steps must not contain duplicates")
        if any(b <= a for a, b in zip(self.milestone_steps, self.milestone_steps[1:])):
            raise ValueError("training.milestone_steps must be strictly increasing")
        expected_phase = {
            "smoke": "smoke",
            "perf": "performance",
            "train_5k": "accuracy",
            "finetune": "accuracy",
            "soak": "soak",
        }[self.run_mode]
        if self.phase != expected_phase:
            raise ValueError(f"{self.run_mode} requires training.phase={expected_phase!r}")
        if self.run_mode == "train_5k" and self.steps < 5000:
            raise ValueError("train_5k requires training.steps>=5000")
        if self.phase == "accuracy" and self.warmup_steps:
            raise ValueError(
                "accuracy profiles require training.warmup_steps=0 so uncounted optimizer updates "
                "cannot alter the training recipe"
            )
        if self.run_mode not in {"perf", "soak"} and not self.eval_enabled:
            raise ValueError(f"{self.run_mode} requires training.eval_enabled=true")
        if self.run_mode == "soak" and self.max_duration_seconds is None:
            raise ValueError("soak requires training.max_duration_seconds")
        if self.max_duration_seconds is not None and self.max_duration_seconds >= self.timeout_s:
            raise ValueError("training.max_duration_seconds must leave room inside training.timeout_s")
        if self.enabled and self.run_mode in LONG_RUN_MODES and not self.allow_long_run:
            raise ValueError(
                f"{self.run_mode} is a protected long run; set training.enabled=true and "
                "training.allow_long_run=true explicitly"
            )
        if self.lr_schedule == "constant" and self.lr_warmup_steps:
            raise ValueError("constant lr_schedule cannot define lr_warmup_steps")
        needs_data = [s.name for s in self.enabled_sweeps() if s.data_mode == "tokens"]
        if needs_data and (not self.data.source or not self.data.cache_dir):
            raise ValueError(f"token-data sweeps {needs_data} require training.data.source and cache_dir")
        if self.eval_enabled and any(s.data_mode != "tokens" for s in self.enabled_sweeps()):
            raise ValueError("evaluation needs held-out text; synthetic sweeps cannot set eval_enabled")
        for sweep in self.enabled_sweeps():
            if sweep.precision not in self.peak_tflops_per_gpu:
                raise ValueError(f"training.peak_tflops_per_gpu has no {sweep.precision} entry for {sweep.name!r}")
        return self

    def enabled_sweeps(self):
        by_name = {sweep.name: sweep for sweep in self.sweeps}
        names = self.enabled_sweep_list or list(by_name)
        return [by_name[name] for name in names]


class TrainingVariantConfig(BaseVariantConfig):
    schema_version: Literal[1]
    framework: Literal["pytorch_training"]
    gpu_name: str
    training: TrainingConfig

    @field_validator("gpu_name")
    @classmethod
    def _uppercase_gpu_name(cls, value):
        name = value.strip().upper()
        if name not in _ALLOWED_GPU_NAMES:
            raise ValueError(f"gpu_name must be one of {list(_ALLOWED_GPU_NAMES)}, got {value!r}")
        return name

    @model_validator(mode="after")
    def _validate_precision_hardware(self):
        if self.gpu_name == "MI355X":
            return self
        offenders = [s.name for s in self.training.enabled_sweeps() if s.precision in MI355X_ONLY_PRECISIONS]
        if offenders:
            raise ValueError(f"MX precisions run on MI355X only; {self.gpu_name} config enables {offenders}")
        return self

    @property
    def gpu_arch(self):
        """Alias kept for the shared report and orchestration layers."""
        return self.gpu_name

    def sweep(self, sweep_ref):
        for sweep in self.training.sweeps:
            if sweep_ref in (sweep.name, sweep.label):
                return sweep
        raise KeyError(f"unknown PyTorch Training sweep: {sweep_ref}")

    def scaling_baseline_for(self, sweep_ref):
        sweep = self.sweep(sweep_ref)
        if sweep.scaling_baseline is not None:
            return sweep.scaling_baseline
        return self.training.scaling_baseline

    def cell_key(self, sweep_ref):
        return self.sweep(sweep_ref).name

    def expected_cells(self):
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
        required = [f"training.{name}" for name in sorted(GATED_METRICS)]
        missing_metrics = {}
        for cell in sorted(expected & present):
            absent = [name for name in required if name not in (self.thresholds.get(cell) or {})]
            if absent:
                missing_metrics[cell] = absent
        if missing_metrics:
            problems.append(f"cells missing gated-metric specs: {missing_metrics}")
        if problems:
            message = "threshold.json does not match the PyTorch Training sweep; " + "; ".join(problems)
            if self.enforce_thresholds:
                raise ValueError(message)
            warnings.warn(f"{message} (enforce_thresholds=false -> record-only)", stacklevel=3)
        return self


def _check_no_changeme(node, path="", offenders=None):
    if offenders is None:
        offenders = []
    if isinstance(node, dict):
        for key, value in node.items():
            _check_no_changeme(value, f"{path}.{key}" if path else key, offenders)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _check_no_changeme(value, f"{path}[{index}]", offenders)
    elif isinstance(node, str) and "<changeme" in node:
        offenders.append(path)
    if not path and offenders:
        raise ValueError(f"config has unfilled '<changeme...>' placeholders in: {', '.join(offenders)}")


def resolve_node_thresholds(thresholds, num_nodes):
    """Select each spec's ``by_nodes`` entry for this cluster's node count.

    The entry for ``num_nodes`` is merged over the spec's other fields. A node
    count with no entry keeps the spec's own ``value`` if it has one and is
    otherwise record-only, so an uncalibrated topology never gates on another
    topology's numbers.
    """
    resolved = {}
    for cell, specs in thresholds.items():
        if not isinstance(specs, dict):
            resolved[cell] = specs
            continue
        resolved[cell] = {}
        for metric, spec in specs.items():
            if not isinstance(spec, dict) or "by_nodes" not in spec:
                resolved[cell][metric] = spec
                continue
            by_nodes = spec["by_nodes"]
            if not isinstance(by_nodes, dict) or not all(
                str(key).isdigit() and int(key) > 0 and isinstance(entry, dict) for key, entry in by_nodes.items()
            ):
                raise ValueError(f"{cell} {metric}: by_nodes must map node counts to threshold specs")
            merged = {key: value for key, value in spec.items() if key != "by_nodes"}
            merged.update(by_nodes.get(str(num_nodes), {}))
            if "value" not in merged and merged.get("kind") != "info":
                merged = {"kind": "info"}
            resolved[cell][metric] = merged
    return resolved


def load_training_variant(config_path, cluster_dict):
    """Load, substitute, and validate a PyTorch Training config/threshold pair."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    _check_no_changeme(raw)
    num_nodes = len((cluster_dict or {}).get("node_dict") or {}) or 1
    raw["thresholds"] = resolve_node_thresholds(thresholds, num_nodes)
    return TrainingVariantConfig(**raw)
