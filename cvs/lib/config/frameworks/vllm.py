"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cvs.lib.config.inference import InferenceTestConfig
from cvs.lib.config.loader import register_config
from cvs.lib.config.sweep import SweepParams


class SeqCombo(BaseModel):
    """One input/output sequence-length pairing. ``name`` becomes a cell ID token."""

    model_config = ConfigDict(extra="forbid")

    isl: int = Field(gt=0)
    osl: int = Field(gt=0)
    name: Optional[str] = None


class VllmParams(BaseModel):
    """Static (non-swept) vLLM parameters.

    Framework-specific knobs that used to be hardcoded in the shared inference
    base (the AITER env flags) belong in the top-level ``knobs`` of the config,
    not here; ``params`` is the invariant per-run configuration.
    """

    model_config = ConfigDict(extra="forbid")

    server_script: str
    bench_serv_script: str = "benchmark_serving.py"
    backend: str = "vllm"
    dataset_name: str = "random"
    num_prompts: int = Field(default=3200, gt=0)
    max_model_length: int = Field(default=9216, gt=0)
    request_rate: str = "inf"
    tokenizer_mode: str = "auto"
    percentile_metrics: List[str] = Field(default_factory=lambda: ["ttft", "tpot", "itl", "e2el"])
    metric_percentiles: int = 99
    base_url: str = "http://0.0.0.0"
    port_no: int = 8888
    burstiness: float = 1.0
    random_range_ratio: float = 0.8
    random_prefix_len: int = 0
    container_image: Optional[str] = None


class VllmSweepParams(SweepParams):
    """vLLM sweep axes.

    ``concurrency`` is an inference-only axis; a training config that tried to
    set it would be rejected by its own ``SweepParams`` (extra="forbid").
    """

    concurrency: List[int] = Field(min_length=1)
    sequence_combinations: List[SeqCombo] = Field(min_length=1)
    tensor_parallelism: Optional[List[int]] = None

    @field_validator("concurrency")
    @classmethod
    def _positive_concurrency(cls, values: List[int]) -> List[int]:
        if any(v <= 0 for v in values):
            raise ValueError("concurrency levels must be positive")
        return values


@register_config("vllm")
class VllmConfig(InferenceTestConfig):
    """A single (vLLM, model) workload config."""

    framework: Literal["vllm"] = "vllm"
    params: VllmParams
    sweep: VllmSweepParams

    @model_validator(mode="after")
    def _tp_consistent_with_topology(self) -> "VllmConfig":
        """Reject a TP sweep that the fixed topology cannot satisfy.

        For single-node vLLM serving, tensor parallelism *is* the number of GPUs
        the server occupies, i.e. a role's ``gpus_per_node``. Since the topology
        is fixed at bind time, every swept ``tensor_parallelism`` value must
        equal some role's ``gpus_per_node``; otherwise the cells would demand a
        GPU count the binder never allocates -- a silent mismatch.
        """
        tps = self.sweep.tensor_parallelism
        if tps:
            gpus = {role.gpus_per_node for role in self.topology.roles.values()}
            mismatched = sorted({t for t in tps if t not in gpus})
            if mismatched:
                raise ValueError(
                    f"sweep.tensor_parallelism {mismatched} matches no role gpus_per_node "
                    f"{sorted(gpus)}; topology is fixed, so TP cannot be swept independently "
                    f"of GPU allocation"
                )
        return self
