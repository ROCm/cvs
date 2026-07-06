'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Config schema for the pytorch_vision *training* suite.

The framework-agnostic machinery (paths/model/image/container schema, the 3-pass
placeholder substitution, the `enforce_thresholds` gate, and the
`substitute_config` file-read helper) lives in `cvs.lib.utils.config_loader`.
This module holds the training half: a `Training` params block (model, dataset,
precision, parallel strategy, step/epoch budgets) and
`VariantConfig(BaseVariantConfig)`.

This is the SCAFFOLD for the "Training Frameworks" rows of the PyTorch Vision
validation matrix (smoke -> full run -> DDP/FSDP -> multi-node -> checkpoint ->
data-pipeline -> AMP parity -> soak). The schema is intentionally small and
un-gated for now: no sweep, no threshold-coverage validator, because the
per-workload metrics (throughput, TFLOPS, Top-1, mIoU, ...) are not defined yet.
Extend `Training` (and add a sweep / metric vocabulary / coverage check) as each
workload row is implemented -- mirror `vision_config_loader.py` when you do.
'''

from __future__ import annotations

from typing import Dict

from typing_extensions import Literal

from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config


class Training(_Forbid):
    # Common knobs shared by the training rows. All str (rendered as CLI args /
    # env to the training entrypoint), mirroring the vllm/vision Params convention.
    # Extend this as workloads land (optimizer, lr schedule, dataset splits, rocAL
    # pipeline flags, FSDP wrap policy, ...). Every field has a default so a
    # minimal config still loads.
    model_arch: str = "resnet50"          # torchvision/HF model id, e.g. resnet50, vit_b_16
    dataset: str = "imagenet-1k"          # logical dataset name
    dataset_path: str = ""                # on-node path (leave "" until staged)
    precision: Literal["fp32", "bf16", "amp_bf16"] = "amp_bf16"
    strategy: Literal["single", "ddp", "fsdp"] = "ddp"
    nnodes: str = "1"
    nproc_per_node: str = "8"
    global_batch_size: str = "256"
    micro_batch_size: str = "32"
    num_steps: str = "10"                 # smoke default (row #1: 10 steps)
    num_epochs: str = "1"
    lr: str = "0.1"
    checkpoint_dir: str = ""              # for save+resume (row #5)
    train_timeout_s: str = "3600"


class VariantConfig(BaseVariantConfig):
    framework: Literal["pytorch_vision_training"]
    gpu_arch: str
    # Extra env vars exported in the container before training (NCCL/RCCL ifname,
    # MIOPEN flags, HIP_VISIBLE_DEVICES, ...). Merged over orchestrator defaults.
    env: Dict[str, str] = {}
    training: Training = Training()


# ---------- public API (vision training) ----------


def load_variant(config_path, cluster_dict):
    """Load and validate a pytorch_vision_training variant config + threshold file.

    Delegates the file read + placeholder substitution + threshold discovery to
    the generic `substitute_config`, then attaches the thresholds and builds the
    typed `VariantConfig`. Thresholds are unused for now (no metrics gated yet)
    but the sibling `*threshold.json` is still required by `substitute_config`.
    """
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return VariantConfig(**raw)
