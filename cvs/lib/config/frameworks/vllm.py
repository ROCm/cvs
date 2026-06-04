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

    The adapter COMPOSES ``vllm serve`` / ``vllm bench serve`` argv from the
    typed knobs below -- the YAML declares values, the adapter owns the CLI
    shape. ``server_script`` / ``bench_serv_script`` are legacy escape
    hatches: when set, the adapter passes the string through verbatim
    (server) or shells out to it as-is (bench), bypassing composition. The
    canonical config leaves them ``None`` so a vLLM CLI rename does not
    cascade into every workload YAML in the tree.

    ``server_extra_args`` / ``bench_extra_args`` are pass-through argv
    lists appended after the composed flags -- the operator escape hatch
    that does not require giving up composition (typo guards, sweep
    integration, deprecation tracking).
    """

    model_config = ConfigDict(extra="forbid")

    # Composition mode (default). When ``server_script`` is None the adapter
    # composes ``vllm serve <model> --tensor-parallel-size N ...``; when
    # ``bench_serv_script`` is None the adapter composes
    # ``vllm bench serve --backend vllm --base-url ... --save-result ...``.
    server_script: Optional[str] = None
    bench_serv_script: Optional[str] = None

    # Server typed knobs (adapter lowers each into a vllm-serve flag).
    max_model_len: int = Field(default=9216, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    quantization: Optional[str] = None  # e.g. "fp8", "awq"; emits --quantization
    dtype: Optional[str] = None  # "auto"/"bfloat16"/...; emits --dtype
    trust_remote_code: bool = False  # emits --trust-remote-code
    download_dir: Optional[str] = None  # HF download directory; emits --download-dir

    # Bench typed knobs.
    backend: str = "vllm"
    dataset_name: str = "random"
    num_prompts: int = Field(default=3200, gt=0)
    request_rate: str = "inf"
    burstiness: float = 1.0
    tokenizer_mode: str = "auto"
    random_range_ratio: float = 0.8
    random_prefix_len: int = 0
    # Measure-only debug metrics: union'd in alongside the (metric, percentile)
    # pairs derived from ``thresholds`` so the bench emits them without
    # requiring a no-op threshold. Empty by default -- the typical config
    # drives ``--percentile-metrics`` entirely from its thresholds list.
    extra_percentile_metrics: List[str] = Field(default_factory=list)

    # Endpoint (composed): adapter emits ``--base-url <base_url>:<port_no>``
    # as a single URL (the new ``vllm bench serve`` CLI shape).
    base_url: str = "http://0.0.0.0"
    port_no: int = 8888

    # Operator escape hatches: appended verbatim after composed flags.
    # Use these for one-off knobs the adapter does not type yet; raise a
    # follow-up to promote anything that recurs across configs.
    server_extra_args: List[str] = Field(default_factory=list)
    bench_extra_args: List[str] = Field(default_factory=list)

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
