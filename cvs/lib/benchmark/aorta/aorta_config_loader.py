"""Aorta variants built on the shared container and threshold configuration.

Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import math
import re
from pathlib import PurePosixPath

from pydantic import ConfigDict, Field, create_model, model_validator

from cvs.lib.utils.config_loader import BaseVariantConfig, substitute_config


# Runtime field declarations keep these schemas compatible with the repository's
# annotation-free library convention while reusing Pydantic validation.
RcclSpec = create_model(
    "RcclSpec",
    __config__=ConfigDict(extra="forbid"),
    clone_url=(str, "https://github.com/ROCm/rccl"),
    branch=(str, "develop"),
    build_path=(str, "/mnt/rccl"),
)
AnalysisSpec = create_model(
    "AnalysisSpec",
    __config__=ConfigDict(extra="forbid"),
    enable_tracelens=(bool, True),
    enable_gemm_analysis=(bool, False),
    tracelens_script=(str, "scripts/tracelens_single_config/run_tracelens_single_config.sh"),
    gemm_script=(str, "scripts/gemm_analysis/run_tracelens_analysis.sh"),
    skip_if_exists=(bool, False),
)
MultiNodeSpec = create_model(
    "MultiNodeSpec",
    __config__=ConfigDict(extra="forbid"),
    master_launch_mode=(str, Field(default="auto", pattern="^(auto|script|torchrun)$")),
    nproc_per_node=(int, Field(default=0, ge=0)),
    master_port=(int, Field(default=0, ge=0, le=65535)),
    master_addr=(str, ""),
    train_script=(str, "train.py"),
    extra_torchrun_args=(list, Field(default_factory=list)),
    extra_train_args=(list, Field(default_factory=list)),
    extra_env=(dict, Field(default_factory=dict)),
    collect_traces=(bool, True),
)


_CVS_PLACEHOLDER = re.compile(
    r"\{(?:user-id|paths\.[a-zA-Z0-9_.-]+|shared_fs|models_dir|log_dir|hf_token_file|temp_dir|"
    r"home|user|home-mount-dir|node-dir-name)\}"
)
_UNKNOWN_PLACEHOLDER = re.compile(r"\{[a-zA-Z0-9_.-]+\}")
_OPAQUE_FIELDS = {
    ("container", "env"),
    ("multi_node", "extra_env"),
    ("multi_node", "extra_torchrun_args"),
    ("multi_node", "extra_train_args"),
    ("training_overrides",),
}


def _reject_placeholders(value, path=()):
    """Reject unresolved placeholders while preserving opaque pass-through syntax."""
    if isinstance(value, str):
        opaque = any(path[: len(prefix)] == prefix for prefix in _OPAQUE_FIELDS)
        unresolved = _CVS_PLACEHOLDER.search(value) or (not opaque and _UNKNOWN_PLACEHOLDER.search(value))
        if "<changeme>" in value.lower() or unresolved:
            raise ValueError(f"Unresolved configuration placeholder: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_placeholders(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholders(item, path + (index,))


def _reject_config_placeholders(cls, data):
    """Reject unresolved placeholders on the raw input, before field coercion masks them."""
    _reject_placeholders(data)
    return data


def _validate_variant(self):
    """Validate paths, launch settings and thresholds without cluster access."""
    for name, value in (
        ("aorta_path", self.aorta_path),
        ("container_mount_path", self.container_mount_path),
        ("rccl.build_path", self.rccl.build_path),
    ):
        path = PurePosixPath(value)
        if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
            raise ValueError(f"{name} must be an absolute directory other than /")
    for value in (
        self.base_config,
        self.build_script,
        self.experiment_script,
        self.multi_node.train_script,
        self.analysis.tracelens_script,
        self.analysis.gemm_script,
    ):
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Script/config paths must be relative to the Aorta repository: {value!r}")
    volumes = self.container.runtime.args.get("volumes", [])
    writable_mount = False
    mount_endpoints = [PurePosixPath(self.aorta_path), PurePosixPath(self.container_mount_path)]
    for volume in volumes:
        parts = volume.split(":", 2)
        if len(parts) < 2 or [PurePosixPath(parts[0]), PurePosixPath(parts[1])] != mount_endpoints:
            continue
        options = set(parts[2].split(",")) if len(parts) == 3 else set()
        if "ro" not in options:
            writable_mount = True
            break
    if not writable_mount:
        mount = f"{self.aorta_path}:{self.container_mount_path}"
        raise ValueError(f"container.runtime.args.volumes must include the writable mount {mount!r}")
    if self.aorta_auto_clone and not self.aorta_clone_url:
        raise ValueError("aorta_clone_url is required when aorta_auto_clone is true")
    if self.multi_node.master_port and self.multi_node.master_port < 1024:
        raise ValueError("multi_node.master_port must be 0 (automatic) or 1024..65535")
    for args in (self.multi_node.extra_torchrun_args, self.multi_node.extra_train_args):
        if not all(isinstance(arg, str) for arg in args):
            raise ValueError("Extra launch arguments must be strings")
    env = {**self.container.env, **self.multi_node.extra_env}
    for key in self.training_overrides:
        if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", key):
            raise ValueError(f"Invalid training override key: {key!r}")
    for key, value in env.items():
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key) or not isinstance(value, (str, int, float)):
            raise ValueError(f"Invalid environment entry: {key!r}")
    for key in ("NCCL_MAX_NCHANNELS", "NCCL_MAX_P2P_NCHANNELS"):
        if key in env and not 1 <= int(env[key]) <= 256:
            raise ValueError(f"{key} must be in 1..256")
    expected = self.thresholds.get("expected_results", {})
    if set(self.thresholds) - {"expected_results"}:
        raise ValueError("Aorta threshold files must contain an expected_results block")
    if self.enforce_thresholds and not any(value is not None for value in expected.values()):
        raise ValueError("enforce_thresholds=true requires at least one expected_results threshold")
    allowed = {"max_avg_iteration_ms", "min_compute_ratio", "min_overlap_ratio", "max_time_variance_ratio"}
    for key, value in expected.items():
        if key not in allowed:
            raise ValueError(f"Unknown Aorta threshold: {key}")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
        if key in ("min_compute_ratio", "min_overlap_ratio") and value > 1:
            raise ValueError(f"{key} must be in 0..1")
    return self


AortaVariantConfig = create_model(
    "AortaVariantConfig",
    __base__=BaseVariantConfig,
    __validators__={
        "_validate_aorta": model_validator(mode="after")(_validate_variant),
        "_reject_placeholders_early": model_validator(mode="before")(classmethod(_reject_config_placeholders)),
    },
    aorta_path=(str, ...),
    container_mount_path=(str, "/mnt"),
    aorta_auto_clone=(bool, False),
    aorta_clone_url=(str, ""),
    base_config=(str, "config/distributed.yaml"),
    build_script=(str, "scripts/build_rccl.sh"),
    experiment_script=(str, "scripts/rccl_exp.sh"),
    gpus_per_node=(int, Field(gt=0)),
    timeout_seconds=(int, Field(default=3600, gt=0)),
    skip_rccl_build=(bool, False),
    output_dir=(str, "aorta_results"),
    training_overrides=(dict, Field(default_factory=dict)),
    rccl=(RcclSpec, Field(default_factory=RcclSpec)),
    analysis=(AnalysisSpec, Field(default_factory=AnalysisSpec)),
    multi_node=(MultiNodeSpec, Field(default_factory=MultiNodeSpec)),
)


def load_training_variant(config_path, cluster_dict):
    """Load an Aorta JSON variant and its sibling threshold file."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return AortaVariantConfig.model_validate(raw)
