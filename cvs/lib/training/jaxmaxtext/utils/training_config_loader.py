'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training-specific config schema for the jaxmaxtext suite.

The framework-agnostic machinery (paths/model/container schema, the 3-pass
placeholder substitution, the `enforce_thresholds` gate, and the
`substitute_config` file-read helper) lives in `cvs.lib.utils.config_loader`.
This module holds the training half: the MaxText config, tokenizer, NCCL,
JAX distributed settings, RDMA lib, and `TrainingVariantConfig(BaseVariantConfig)`.

A training suite does not sweep cells the way inference does (no NxM matrix of
ISL/OSL/concurrency). Instead each declared `sweep` is one full training run and
its `name` IS the threshold-file key (also the key `metric()` looks up at
runtime). `expected_cells()` therefore returns the declared sweep names, and the
coverage check validates the threshold file against those names directly.
'''

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal

from pydantic import field_validator

from cvs.lib.utils.config_loader import BaseVariantConfig, _Allow, _Forbid, substitute_config
from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import GATED_METRICS


class Tokenizer(_Forbid):
    hf_model_id: str
    tokenizer_path: str


class NcclConfig(_Allow):
    ib_hca_list: str = ""
    ib_hca: str = ""
    socket_ifname: str = ""
    gloo_socket_ifname: str = ""
    ib_gid_index: str = "3"

    @field_validator("ib_hca_list", "ib_hca", "socket_ifname", "gloo_socket_ifname", "ib_gid_index")
    @classmethod
    def _reject_changeme(cls, v, info):
        """Hard-exit when a cluster-specific RDMA/NIC field is left as '<changeme>'.

        These device/interface names are cluster-specific and shipped as
        '<changeme>' placeholders (see the sibling _example_* values). Running a
        distributed job with them unresolved would silently use the wrong
        NIC/RDMA devices, so fail loudly at config load instead.
        """
        if isinstance(v, str) and "<changeme>" in v.lower():
            raise ValueError(
                f"nccl.{info.field_name} is still '<changeme>'. Set your cluster's RDMA/NIC "
                "device/interface (see the sibling _example_* value) before running distributed training."
            )
        return v


class JaxDistributed(_Forbid):
    coordinator_ip: str = "auto"
    coordinator_port: str = "12346"
    initialization_timeout_seconds: str = "1800"
    heartbeat_timeout_seconds: str = "900"


class RdmaLib(_Allow):
    host_source_file: str = ""
    container_mount_file: str = ""
    container_dest_file: str = ""


class ScalingBaseline(_Allow):
    """Reference (typically 1-node) throughput for scaling-efficiency %.

    `tokens_per_sec_total` is the TOTAL tokens/sec measured on a prior run of
    `num_nodes` nodes (source it from a previous single-node run log). Scaling
    efficiency % = throughput_N / ((N / num_nodes) * tokens_per_sec_total) * 100.

    Leave `tokens_per_sec_total` at 0.0 to disable the metric (it then reports
    record-only as None instead of gating on an uncalibrated baseline).
    """

    tokens_per_sec_total: float = 0.0
    num_nodes: int = 1


class Convergence(_Allow):
    """Target for convergence / time-to-target-accuracy (row 33).

    `target_metric` selects the loss series to converge on:
      - "eval_loss"  : validation loss (requires eval enabled + parseable)
      - "train_loss" : per-step training loss
      - "auto"       : eval loss when eval points exist, else training loss

    `target_value` is the loss threshold to reach; <= 0 disables the metric
    (steps_to_target / time_to_target_seconds report record-only as None).
    """

    target_metric: Literal["auto", "train_loss", "eval_loss"] = "auto"
    target_value: float = 0.0


class LossCurve(_Allow):
    """Loss-curve (row 32) sampling + pass/fail settings.

    `sample_every` and `milestone_steps` control which per-step losses are kept
    for the plotted/asserted curve (keeps short runs non-empty). The verdict is
    the least-squares slope of the sampled curve: the run passes when
    `slope < max_slope` (default 0.0 = strictly decreasing). `enforce` gates the
    test (fail on a non-decreasing curve); set False for record-only.
    """

    sample_every: int = 10
    milestone_steps: List[int] = [100, 500, 1000, 5000]
    max_slope: float = 0.0
    enforce: bool = True


class SmokeTest(_Allow):
    """Smoke test (ENABLED by default). Loads the model and runs `steps` steps
    with a small fixed batch/seqlen in BF16, passing only if no error signature
    fires (no metric/threshold checks). A failure gates the rest of the suite.

    Set `enabled=false` to SKIP it -- e.g. during iterative experiments where you
    don't want the smoke run every time (mirrors checkpoint_resume, but opt-OUT
    rather than opt-in). `steps`/`per_device_batch_size`/`max_target_length` tune
    the smoke run itself.
    """

    enabled: bool = True
    steps: int = 5
    per_device_batch_size: int = 1
    max_target_length: int = 2048


class CheckpointResume(_Allow):
    """Checkpoint save + resume test (opt-in; off by default).

    Runs ONE sweep twice: Phase 1 trains `steps_before_ckpt` steps with
    checkpointing on (a checkpoint is written at `checkpoint_period`); Phase 2
    resumes from that checkpoint and trains `steps_after_resume` more. Passes
    when the resumed run restarts at the checkpoint step and the loss at the
    resume boundary matches Phase 1 within `loss_tolerance` (state restored, not
    reinitialized). Also benchmarks checkpoint I/O: `checkpoint_save_seconds` /
    `checkpoint_load_seconds` are gated against `max_save_seconds` /
    `max_load_seconds` when those are > 0 (else record-only).

    `sweep` selects which sweep to use ("" -> first enabled). `smoke_model_overrides`
    optionally shrinks the model for a fast run WITHOUT changing the tokenizer/
    vocab (e.g. {"base_num_decoder_layers": 4}); empty -> the config's full model
    (real-size checkpoint I/O).

    `delete_ckpt_dir` (default true) removes the checkpoint directory after the
    test to free disk space; set it false to keep the checkpoint files for
    inspection.
    """

    enabled: bool = False
    sweep: str = ""
    steps_before_ckpt: int = 6
    steps_after_resume: int = 6
    checkpoint_period: int = 5
    loss_tolerance: float = 0.1
    max_save_seconds: float = 0.0
    max_load_seconds: float = 0.0
    delete_ckpt_dir: bool = True
    smoke_model_overrides: Dict[str, Any] = {}


class Sweep(_Allow):
    """One sweep entry = one full training run with per-run maxtext overrides.

    `name` is the canonical cell key (also the threshold-file key), e.g.
    "NNODES=2,STEPS=30,PRECISION=BF16,BATCH=3,GBS=48,SEQLEN=8192". Only the
    parameters that actually vary need a `maxtext_overrides` entry (for now just
    precision, e.g. FP8 sets `quantization`); everything else falls back to the
    base `maxtext_config`.
    """

    name: str
    maxtext_overrides: Dict[str, Any] = {}


class TrainingConfig(_Allow):
    distributed: bool = True
    gpus_per_node: int = 8  # do not assume a uniform topology; override per cluster
    # Scan host dmesg (all nodes) for GPU/HW/kernel faults over the training
    # window. Set false on clusters without passwordless sudo for `dmesg`.
    verify_dmesg: bool = True
    steps: int = 30
    enable_checkpointing: bool = False
    # MaxText moved the train entrypoint across versions; list candidates and the
    # job picks whichever exists in the running container (first match wins).
    # v26.4+: .../src/maxtext/trainers/pre_train/train.py
    # v26.3 and earlier: .../src/MaxText/train.py
    train_script_paths: List[str] = [
        "/workspace/maxtext/src/maxtext/trainers/pre_train/train.py",
        "/workspace/maxtext/src/MaxText/train.py",
    ]
    # Deprecated single-path form; kept for backward compatibility and used as a
    # final fallback candidate when train_script_paths is empty.
    train_script: str = "/workspace/maxtext/src/MaxText/train.py"
    maxtext_config: Dict[str, Any] = {}
    tokenizer: Tokenizer
    nic_type: str = "thor2"
    rdma_lib: RdmaLib = RdmaLib()
    env_vars: Dict[str, str] = {}
    xla_flags: Dict[str, str] = {}
    # {name: regex} error signatures scanned in the training log during polling.
    # Empty -> the driver falls back to its built-in default set. Lets users
    # add/remove signatures per config without touching code.
    error_patterns: Dict[str, str] = {}
    nccl: NcclConfig = NcclConfig()
    jax_distributed: JaxDistributed = JaxDistributed()
    scaling_baseline: ScalingBaseline = ScalingBaseline()
    convergence: Convergence = Convergence()
    loss_curve: LossCurve = LossCurve()
    smoke: SmokeTest = SmokeTest()
    checkpoint_resume: CheckpointResume = CheckpointResume()
    sweeps: List[Sweep] = []
    enabled_sweep_list: List[str] = []


def validate_thresholds_cover_training(
    *,
    expected_cells,
    thresholds,
    enforce_thresholds: bool,
    gated_metrics=None,
) -> None:
    """Shared training threshold/cell coverage check."""
    expected = set(expected_cells)
    # Skip "_"-prefixed metadata keys (e.g. "_comment") so they are not mistaken
    # for a threshold cell that matches no training sweep.
    present = {k for k in thresholds.keys() if not str(k).startswith("_")}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    problems = []
    if missing:
        problems.append(f"training cells with no threshold entry: {missing}")
    if extra:
        problems.append(f"threshold keys matching no training cell (typo?): {extra}")
    gated = gated_metrics if gated_metrics is not None else GATED_METRICS
    gated_keys = [f"training.{m}" for m in sorted(gated)]
    gated_gaps = {}
    for cell in sorted(expected & present):
        specs = thresholds.get(cell) or {}
        absent = [k for k in gated_keys if k not in specs]
        if absent:
            gated_gaps[cell] = absent
    if gated_gaps:
        problems.append(f"cells missing gated-metric specs: {gated_gaps}")
    if problems:
        msg = "threshold.json does not match the training config; " + "; ".join(problems)
        if enforce_thresholds:
            raise ValueError(msg)
        warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=3)


class TrainingVariantConfig(BaseVariantConfig):
    framework: Literal["jaxmaxtext"]
    gpu_arch: str
    training: TrainingConfig

    def expected_cells(self):
        """Threshold cell keys this config expects: one per declared sweep.

        The sweep `name` IS the threshold-file key and the key `metric()` looks
        up at runtime (see cvs/tests/training/jaxmaxtext/_common.py::metric), so
        coverage is checked against the declared sweep names directly -- not a
        synthesized key. `enabled_sweep_list` only selects which of these
        actually run; the threshold file still carries an entry per declared
        sweep. A config with no sweeps degrades to a single implicit "default"
        cell.
        """
        names = [s.name for s in self.training.sweeps]
        return names or ["default"]

    def enabled_sweeps(self):
        """Return the Sweep objects selected to run.

        `enabled_sweep_list` (if non-empty) selects a subset by name; otherwise
        every declared sweep runs. A config with no `sweeps` degrades to a single
        implicit sweep named "default" (its threshold cell, if any, is keyed
        "default"), so the suite still runs unparametrized.
        """
        sweeps = self.training.sweeps
        if not sweeps:
            return [Sweep(name="default")]
        by_name = {s.name: s for s in sweeps}
        names = self.training.enabled_sweep_list or [s.name for s in sweeps]
        selected = []
        for n in names:
            if n in by_name:
                selected.append(by_name[n])
            else:
                warnings.warn(f"enabled_sweep_list references unknown sweep '{n}'", stacklevel=2)
        return selected


# ---------- public API (training) ----------


def load_training_variant(config_path, cluster_dict):
    """Load and validate a jaxmaxtext variant config + its sibling threshold file.

    Delegates the file read + placeholder substitution + threshold discovery to
    the generic `substitute_config`, then attaches the thresholds and builds the
    typed `TrainingVariantConfig`.
    """
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return TrainingVariantConfig(**raw)
