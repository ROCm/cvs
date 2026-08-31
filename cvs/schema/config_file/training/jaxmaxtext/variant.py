"""
JAX MaxText training variant config schema.

Mirrors ``cvs/input/config_file/training/jaxmaxtext/``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal

from pydantic import field_validator

from cvs.schema.common.base import BaseVariantConfig
from cvs.schema.base import _Allow, _Forbid


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
    tokens_per_sec_total: float = 0.0
    num_nodes: int = 1


class Convergence(_Allow):
    target_metric: Literal["auto", "train_loss", "eval_loss"] = "auto"
    target_value: float = 0.0


class LossCurve(_Allow):
    sample_every: int = 10
    milestone_steps: List[int] = [100, 500, 1000, 5000]
    max_slope: float = 0.0
    enforce: bool = True


class SmokeTest(_Allow):
    enabled: bool = True
    steps: int = 5
    per_device_batch_size: int = 1
    max_target_length: int = 2048


class CheckpointResume(_Allow):
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
    name: str
    maxtext_overrides: Dict[str, Any] = {}


class TrainingConfig(_Allow):
    distributed: bool = True
    gpus_per_node: int = 8
    verify_dmesg: bool = True
    steps: int = 30
    enable_checkpointing: bool = False
    train_script_paths: List[str] = [
        "/workspace/maxtext/src/maxtext/trainers/pre_train/train.py",
        "/workspace/maxtext/src/MaxText/train.py",
    ]
    train_script: str = "/workspace/maxtext/src/MaxText/train.py"
    maxtext_config: Dict[str, Any] = {}
    tokenizer: Tokenizer
    nic_type: str = "thor2"
    rdma_lib: RdmaLib = RdmaLib()
    env_vars: Dict[str, str] = {}
    xla_flags: Dict[str, str] = {}
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
    present = {k for k in thresholds.keys() if not str(k).startswith("_")}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    problems = []
    if missing:
        problems.append(f"training cells with no threshold entry: {missing}")
    if extra:
        problems.append(f"threshold keys matching no training cell (typo?): {extra}")
    gated = gated_metrics if gated_metrics is not None else set()
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
        names = [s.name for s in self.training.sweeps]
        return names or ["default"]

    def enabled_sweeps(self):
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
