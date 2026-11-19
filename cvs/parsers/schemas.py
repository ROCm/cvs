"""
Pydantic schemas for ALL benchmark results AND configuration files.

This is the single source of truth for:
- Result data structures (parsed benchmark output)
- Configuration file schemas (validated before running benchmarks)

All parsers produce instances of these models.
Config validation happens early to fail fast with clear errors.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Dict, Generic, List, Literal, Optional, Tuple, TypeVar, Union
import json
import logging
import math
import re
import warnings

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveInt,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)


# =============================================================================
# Common Types
# =============================================================================


class ParseStatus(Enum):
    """Status of a parse operation."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some results parsed, some failed
    FAILED = "failed"
    NO_DATA = "no_data"  # No data to parse (e.g., TraceLens skipped, Chrome traces disabled)


T = TypeVar('T', bound=BaseModel)


@dataclass
class ParseResult(Generic[T]):
    """
    Generic result container for all parsers.

    Contains validated Pydantic models plus any warnings/errors.
    """

    status: ParseStatus
    results: List[T] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == ParseStatus.SUCCESS

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0


# =============================================================================
# Aorta / TraceLens Schemas
# =============================================================================


class AortaTraceMetrics(BaseModel):
    """
    Per-rank metrics extracted from PyTorch profiler traces.

    Represents a single GPU's performance during distributed training.
    """

    model_config = ConfigDict(frozen=True)

    # Identification
    rank: int = Field(ge=0, description="Global rank ID")
    node: Optional[str] = Field(default=None, description="Node hostname")
    local_rank: Optional[int] = Field(default=None, ge=0, description="Local rank on node")

    # Timing metrics (in microseconds for precision)
    total_time_us: float = Field(ge=0, description="Total iteration time")
    compute_time_us: float = Field(ge=0, description="Time spent in compute kernels")
    communication_time_us: float = Field(ge=0, description="Time spent in NCCL/communication")
    memory_time_us: Optional[float] = Field(default=None, ge=0, description="Time in memory operations")
    idle_time_us: Optional[float] = Field(default=None, ge=0, description="Idle/wait time")

    # Memory metrics
    peak_memory_gb: Optional[float] = Field(default=None, ge=0, description="Peak GPU memory usage")
    allocated_memory_gb: Optional[float] = Field(default=None, ge=0, description="Allocated GPU memory")

    # Kernel counts
    compute_kernel_count: Optional[int] = Field(default=None, ge=0, description="Number of compute kernels")
    comm_kernel_count: Optional[int] = Field(default=None, ge=0, description="Number of NCCL kernels")

    @field_validator('total_time_us', 'compute_time_us', 'communication_time_us')
    @classmethod
    def validate_not_nan(cls, v: float, info) -> float:
        """Ensure timing values are not NaN or Inf."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f'{info.field_name} cannot be NaN or Inf')
        return v

    @property
    def compute_ratio(self) -> float:
        """Fraction of time spent in compute (vs communication)."""
        if self.total_time_us > 0:
            return self.compute_time_us / self.total_time_us
        return 0.0

    @property
    def comm_ratio(self) -> float:
        """Fraction of time spent in communication."""
        if self.total_time_us > 0:
            return self.communication_time_us / self.total_time_us
        return 0.0

    @property
    def compute_comm_overlap(self) -> float:
        """
        Estimated compute-communication overlap.

        If compute + comm > total, there's overlap.
        Returns fraction of comm time that overlaps with compute.
        """
        if self.communication_time_us <= 0:
            return 0.0

        overlap_time = (self.compute_time_us + self.communication_time_us) - self.total_time_us
        overlap_time = max(0, overlap_time)  # Can't have negative overlap

        return overlap_time / self.communication_time_us


class AortaBenchmarkResult(BaseModel):
    """
    Aggregated Aorta benchmark results across all ranks.

    Computed from individual AortaTraceMetrics.
    """

    model_config = ConfigDict(frozen=True)

    # Cluster configuration
    num_nodes: int = Field(gt=0, description="Number of nodes")
    gpus_per_node: int = Field(gt=0, description="GPUs per node")
    total_gpus: int = Field(gt=0, description="Total GPU count")

    # Aggregated timing (mean across ranks, in microseconds)
    avg_iteration_time_us: float = Field(ge=0, description="Mean iteration time")
    std_iteration_time_us: float = Field(ge=0, description="Std dev of iteration time")
    min_iteration_time_us: float = Field(ge=0, description="Minimum iteration time")
    max_iteration_time_us: float = Field(ge=0, description="Maximum iteration time")

    # Aggregated ratios
    avg_compute_ratio: float = Field(ge=0, le=1, description="Mean compute ratio")
    avg_comm_ratio: float = Field(ge=0, le=1, description="Mean communication ratio")
    avg_overlap_ratio: float = Field(ge=0, le=1, description="Mean overlap ratio")

    # Throughput (if available)
    samples_per_second: Optional[float] = Field(default=None, ge=0)
    tokens_per_second: Optional[float] = Field(default=None, ge=0)

    # Per-rank metrics
    per_rank_metrics: List[AortaTraceMetrics] = Field(default_factory=list)

    # Metadata
    nccl_channels: Optional[int] = Field(default=None)
    compute_channels: Optional[int] = Field(default=None)
    rccl_branch: Optional[str] = Field(default=None)

    @property
    def avg_iteration_time_ms(self) -> float:
        """Mean iteration time in milliseconds."""
        return self.avg_iteration_time_us / 1000.0

    @classmethod
    def from_rank_metrics(
        cls, metrics: List[AortaTraceMetrics], num_nodes: int, gpus_per_node: int, **kwargs
    ) -> "AortaBenchmarkResult":
        """
        Aggregate individual rank metrics into a benchmark result.

        Args:
            metrics: List of per-rank metrics
            num_nodes: Number of nodes in cluster
            gpus_per_node: GPUs per node
            **kwargs: Additional metadata fields

        Returns:
            Aggregated benchmark result
        """
        if not metrics:
            raise ValueError("Cannot aggregate empty metrics list")

        import statistics

        times = [m.total_time_us for m in metrics]
        compute_ratios = [m.compute_ratio for m in metrics]
        comm_ratios = [m.comm_ratio for m in metrics]
        overlap_ratios = [m.compute_comm_overlap for m in metrics]

        return cls(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            total_gpus=num_nodes * gpus_per_node,
            avg_iteration_time_us=statistics.mean(times),
            std_iteration_time_us=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_iteration_time_us=min(times),
            max_iteration_time_us=max(times),
            avg_compute_ratio=statistics.mean(compute_ratios),
            avg_comm_ratio=statistics.mean(comm_ratios),
            avg_overlap_ratio=statistics.mean(overlap_ratios),
            per_rank_metrics=metrics,
            **kwargs,
        )


# =============================================================================
# RCCL Schemas (for future use - mirrors existing models/rccl.py patterns)
# =============================================================================

# Note: RCCL schemas already exist in models/rccl.py
# When porting RCCL tests to this architecture, we can either:
# 1. Move those schemas here
# 2. Re-export them from here
# 3. Keep them separate and import as needed


# =============================================================================
# Configuration File Schemas (Input Validation - Fail Fast)
# =============================================================================


class ClusterNodeConfig(BaseModel):
    """Schema for a single node entry in cluster.json node_dict."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields like bmc_ip, rack_id

    vpc_ip: str = Field(description="VPC IP or hostname for inter-node communication")
    bmc_ip: Optional[str] = Field(default=None, description="BMC IP for out-of-band management")


class HeadNodeConfig(BaseModel):
    """Schema for head_node_dict in cluster.json."""

    model_config = ConfigDict(extra="allow")

    mgmt_ip: str = Field(description="Management IP of head node")


class RackConfig(BaseModel):
    """
    Schema for a single rack entry inside the 'racks' block of cluster.json.

    A rack groups compute trays (referenced via node_dict rack_id) and the
    switch trays physically associated with that rack.
    """

    model_config = ConfigDict(extra="allow")

    platform: Optional[str] = Field(default=None, description="ARC platform name, e.g. 'HeliosP' or 'HeliosR'")
    arc_controller: Optional[str] = Field(
        default=None,
        description="IP of the ARC controller node. Defaults to first sorted node_dict entry with matching rack_id.",
    )
    switch_trays: List[str] = Field(
        default_factory=list,
        description="IPs of switch trays in this rack",
    )
    rmc: Optional[str] = Field(default=None, description="IP of the Rack Management Controller")


class RacksBlock(BaseModel):
    """
    Schema for the top-level 'racks' field in cluster.json.

    Holds optional global switch credentials and one RackConfig entry per rack
    (keyed by rack ID, e.g. 'rack-01'). Extra keys (rack IDs) are accepted via
    extra='allow' and retrieved via get_racks().

    Switch credentials are fleet-wide (homogeneous across all racks). Per-rack
    overrides are not supported in the current exec path; add them to RackConfig
    when that need arises.
    """

    model_config = ConfigDict(extra="allow")

    switch_ssh_user: Optional[str] = Field(
        default=None,
        description="SSH username for all switch trays in every rack.",
    )
    switch_ssh_password: Optional[str] = Field(
        default=None,
        description="SSH password for all switch trays. Ignored when switch_ssh_key_file is set.",
    )
    switch_ssh_key_file: Optional[str] = Field(
        default=None,
        description="Path to SSH private key for all switch trays. Takes priority over switch_ssh_password when set.",
    )

    def get_racks(self) -> Dict[str, RackConfig]:
        """Return only the rack entries, excluding credential fields."""
        skip = {'switch_ssh_user', 'switch_ssh_password', 'switch_ssh_key_file'}
        result = {}
        for key, value in (self.__pydantic_extra__ or {}).items():
            if key not in skip and isinstance(value, dict):
                result[key] = RackConfig(**value)
        return result


class ClusterConfigFile(BaseModel):
    """
    Schema for cluster.json configuration file.

    Validates the cluster configuration before running benchmarks.
    Fails fast with clear error messages if required fields are missing.
    """

    model_config = ConfigDict(extra="allow")

    username: str = Field(description="SSH username for cluster nodes")
    priv_key_file: Optional[str] = Field(default=None, description="Path to SSH private key")
    password: Optional[str] = Field(default=None, description="SSH password (if not using key)")

    node_dict: Dict[str, ClusterNodeConfig] = Field(
        description="Dictionary mapping node hostname/IP to node configuration"
    )
    head_node_dict: Optional[HeadNodeConfig] = Field(default=None, description="Head node configuration")

    racks: Optional[RacksBlock] = Field(
        default=None,
        description=(
            "Rack topology block. Contains optional global switch credentials and one entry per rack "
            "(keyed by rack ID) listing switch_trays and platform."
        ),
    )
    rack_groups: Optional[RacksBlock] = Field(
        default=None,
        description="Deprecated alias for 'racks'. Use 'racks' instead.",
    )

    # Optional fields that may be present
    home_mount_dir_name: Optional[str] = Field(default="home")
    node_dir_name: Optional[str] = Field(default="root")

    @model_validator(mode='after')
    def validate_auth_method(self):
        """Ensure at least one authentication method is provided."""
        if not self.priv_key_file and not self.password:
            raise ValueError("Authentication required: provide either 'priv_key_file' or 'password' in cluster config")
        return self

    @model_validator(mode='after')
    def validate_nodes_exist(self):
        """Ensure at least one node is configured."""
        if not self.node_dict:
            raise ValueError("No nodes configured in 'node_dict' - at least one node is required")
        return self

    @model_validator(mode='after')
    def warn_rack_groups_deprecated(self):
        """Emit a deprecation warning when the old 'rack_groups' key is used."""
        import warnings

        if self.rack_groups is not None and self.racks is None:
            warnings.warn(
                "'rack_groups' in cluster.json is deprecated. Rename it to 'racks'.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    def get_racks_block(self) -> Optional[RacksBlock]:
        """Return the active racks block, preferring 'racks' over the deprecated 'rack_groups'."""
        return self.racks if self.racks is not None else self.rack_groups

    @field_validator('username')
    @classmethod
    def validate_username_not_placeholder(cls, v: str) -> str:
        """Check that username is not still a placeholder."""
        if '<changeme>' in v.lower():
            raise ValueError(
                "Username contains placeholder '<changeme>'. Please set a valid username in cluster config."
            )
        return v


# =============================================================================
# PyTorch XDit (WAN/Flux) Schemas
# =============================================================================


class PytorchXditDistributedNcclExamples(BaseModel):
    """Documentation-only example values shipped beside ``<changeme>`` in sample JSON."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    example_nccl_ib_hca: Optional[str] = Field(
        default=None,
        alias="_example_nccl_ib_hca",
        description="Documentation only: example nccl_ib_hca value for this cluster",
    )
    example_nccl_socket_ifname: Optional[str] = Field(
        default=None,
        alias="_example_nccl_socket_ifname",
        description="Documentation only: example nccl_socket_ifname value",
    )
    example_gloo_socket_ifname: Optional[str] = Field(
        default=None,
        alias="_example_gloo_socket_ifname",
        description="Documentation only: example gloo_socket_ifname value",
    )


class PytorchXditContainerConfig(BaseModel):
    """Schema for container_config section in pytorch-xdit configs."""

    model_config = ConfigDict(extra="allow")

    device_list: List[str] = Field(
        default=["/dev/dri", "/dev/kfd"], description="List of device paths to mount in container"
    )
    volume_dict: Dict[str, str] = Field(default_factory=dict, description="Host:container volume mount mappings")
    env_dict: Dict[str, str] = Field(default_factory=dict, description="Environment variables for container")


class PytorchXditExpectedResults(BaseModel):
    """Schema for expected_results in pytorch-xdit WAN benchmark params."""

    model_config = ConfigDict(extra="forbid")

    max_avg_total_time_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum acceptable average total_time in seconds (native/packaged Wan)",
    )
    max_avg_pipe_time_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum acceptable average pipe_time in seconds (xFuser Wan I2V)",
    )
    min_clip_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Minimum CLIPScore (cosine similarity x 100) for saved FLUX images",
    )
    min_mean_frame_clip_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description=("Minimum mean frame CLIPScore (cosine similarity x 100) across frames sampled from a WAN MP4"),
    )

    @model_validator(mode="after")
    def validate_threshold_present(self) -> "PytorchXditExpectedResults":
        if self.max_avg_total_time_s is None and self.max_avg_pipe_time_s is None:
            raise ValueError("expected_results entry must include max_avg_total_time_s and/or max_avg_pipe_time_s")
        return self


class PytorchXditWan22Benchmarks(BaseModel):
    """Schema for wan22_i2v_a14b benchmark parameters."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(description="Text prompt for image-to-video generation")
    model_format: Optional[str] = Field(
        default=None,
        description=(
            "WAN checkpoint layout override: native (Wan2.2-I2V-A14B) or diffusers "
            "(Wan2.2-I2V-A14B-Diffusers). Auto-inferred from model_repo or model_index.json when omitted."
        ),
    )
    size: str = Field(default="720*1280", pattern=r"^\d+\*\d+$", description="Video resolution (format: height*width)")
    frame_num: int = Field(default=81, ge=1, description="Number of frames to generate")
    num_benchmark_steps: int = Field(default=5, ge=1, description="Number of benchmark iterations to run")
    num_inference_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Diffusers denoising steps for /app/Wan/run.py (defaults to 40 when omitted).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for Diffusers Wan runs (defaults to 42 when omitted).",
    )
    wan_diffusers_run_script: Optional[str] = Field(
        default=None,
        description=(
            "In-container Diffusers Wan launcher script. Defaults to /app/Wan/run.py "
            "(shipped in amdsiloai/pytorch-xdit and rocm/pytorch-xdit benchmark images)."
        ),
    )
    wan_diffusers_i2v_image: Optional[str] = Field(
        default=None,
        description=(
            "In-container input image for Diffusers Wan I2V. Omit or set 'auto' to generate a "
            "placeholder JPEG in-container; otherwise bind-mount the host file via volume_dict."
        ),
    )
    wan_xfuser_auto_input_image: Optional[bool] = Field(
        default=None,
        description=(
            "Generate a synthetic I2V input JPEG inside the container for xFuser runs. "
            "Defaults to true when no host image is bind-mounted."
        ),
    )
    wan_xfuser_install_video_deps: bool = Field(
        default=True,
        description="Run pip install imageio imageio-ffmpeg before xFuser video export.",
    )
    wan_diffusers_launcher: Optional[str] = Field(
        default=None,
        description=(
            "Diffusers Wan launcher: packaged (/app/Wan/run.py in pytorch-xdit images) or "
            "xfuser_example (mount cvs .../scripts/wan_i2v_example.py for ufb-private)."
        ),
    )
    warmup_steps: Optional[int] = Field(
        default=None,
        ge=0,
        description="Warmup iterations for xfuser_example launcher (defaults to 1).",
    )
    wan_xfuser_output_type: Optional[str] = Field(
        default=None,
        description="xFuser output_type for xfuser_example (defaults to pil for ufb-private video export).",
    )
    wan_xfuser_model: Optional[str] = Field(
        default=None,
        description="xFuser registry model name. Wan2.1-T2V selects the text-to-video launcher and skips the input image.",
    )
    guidance_scale: Optional[float] = Field(
        default=None,
        gt=0,
        description="Classifier-free guidance scale for WAN text-to-video.",
    )
    flow_shift: Optional[float] = Field(
        default=None,
        description="Scheduler flow shift for WAN text-to-video.",
    )
    attention_backend: Optional[str] = Field(
        default=None,
        description="xFuser attention backend for WAN text-to-video, such as AITER.",
    )
    cross_attention_backend: Optional[str] = Field(
        default=None,
        description="xFuser cross-attention backend for WAN text-to-video.",
    )
    wan_diffusers_save_video_path: Optional[str] = Field(
        default=None,
        description="In-container MP4 path for xfuser_example (default /outputs/results/video_i2v.mp4).",
    )
    wan_diffusers_timing_json_path: Optional[str] = Field(
        default=None,
        description="In-container timing JSON path for xfuser_example (default results/timing.json).",
    )
    wan_diffusers_video_fps: Optional[int] = Field(
        default=None,
        ge=1,
        description="FPS passed to export_to_video for xfuser_example (defaults to 16).",
    )
    require_video_artifact: bool = Field(
        default=True,
        description="Require video.mp4 under the output dir when parsing results.",
    )
    clip_model: Optional[str] = Field(
        default=None,
        description=(
            "CLIP model id or local path used to score sampled WAN video frames. "
            "Defaults to openai/clip-vit-base-patch32 when omitted."
        ),
    )
    fid_nfe_steps: Optional[List[int]] = Field(
        default=None,
        description="Sampling step counts for an optional FID vs NFE curve. Omit to skip test_fid_curve.",
    )
    fid_samples: Optional[int] = Field(
        default=None,
        ge=2,
        description="Images or videos to save at each fid_nfe_steps value. The sweep stops early so the benchmark command stays inside its timeout.",
    )
    fid_reference_dir: Optional[str] = Field(
        default=None,
        description=(
            "Optional in-container directory of real images for the FID reference set. When it is missing "
            "or has fewer than two images, the highest step-count samples from the run are used."
        ),
    )
    fid_inception_path: Optional[str] = Field(
        default=None,
        description="Local Inception-v3 checkpoint. The FID test does not download weights.",
    )
    fid_sampler: Optional[str] = Field(
        default=None,
        description="Sampler name drawn as its own FID series. Omit for a single line labeled by the model.",
    )
    clip_sample_frames: Optional[int] = Field(
        default=None,
        ge=1,
        description="Frames uniformly sampled from the WAN MP4 for CLIPScore. Defaults to 8 when omitted.",
    )
    compile: bool = Field(default=True, description="Whether to use torch.compile for optimization")
    torchrun_nproc: int = Field(default=8, ge=1, description="Number of processes for torchrun (usually num GPUs)")
    ulysses_size: int = Field(default=8, ge=1, description="Ulysses parallelism degree")
    ring_size: int = Field(default=1, ge=1, description="Ring parallelism degree")
    expected_results: Optional[Dict[str, PytorchXditExpectedResults]] = Field(
        default=None,
        description="Legacy inline thresholds by GPU type (auto, mi300x, mi355, etc.)",
    )

    @field_validator('expected_results')
    @classmethod
    def validate_has_auto_or_specific(
        cls, v: Optional[Dict[str, PytorchXditExpectedResults]]
    ) -> Optional[Dict[str, PytorchXditExpectedResults]]:
        """Ensure either 'auto' or a specific GPU type is present when embedded thresholds are used."""
        if v is None:
            return v
        if not v:
            raise ValueError("expected_results must contain at least one GPU type threshold")
        if 'auto' not in v and not any(k in v for k in ['mi300x', 'mi325', 'mi350', 'mi355']):
            raise ValueError("expected_results must contain either 'auto' or a specific GPU type (mi300x, mi325, etc.)")
        return v


class PytorchXditFluxExpectedResults(BaseModel):
    """Schema for expected_results in Flux benchmark params."""

    model_config = ConfigDict(extra="forbid")

    max_avg_pipe_time_s: float = Field(gt=0, description="Maximum acceptable average pipe_time in seconds")
    min_clip_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Minimum CLIPScore (cosine similarity x 100) for saved FLUX images",
    )


class PytorchXditFlux1DevBenchmarks(BaseModel):
    """Schema for flux1_dev_t2i benchmark parameters."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(description="Text prompt for text-to-image generation")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    num_inference_steps: int = Field(default=25, ge=1, description="Number of denoising steps")
    max_sequence_length: int = Field(default=256, ge=1, description="Maximum sequence length for text encoder")
    model_type: Optional[str] = Field(
        default=None,
        description=(
            "FLUX model family override (flux2 for FLUX.2-dev, flux_kontext for FLUX.1-Kontext). "
            "Auto-inferred from model_repo or model_index.json when omitted."
        ),
    )
    guidance_scale: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Classifier-free guidance scale for the FLUX launcher. Defaults to 4.0 for FLUX.2-dev "
            "and 2.5 for FLUX.1-Kontext when omitted; not passed for FLUX.1-dev."
        ),
    )
    no_use_resolution_binning: bool = Field(default=True, description="Disable resolution binning")
    warmup_steps: int = Field(default=1, ge=0, description="Number of warmup steps before benchmarking")
    warmup_calls: int = Field(default=5, ge=0, description="Number of warmup calls")
    num_repetitions: int = Field(default=25, ge=1, description="Number of benchmark repetitions")
    height: int = Field(default=1024, ge=1, description="Output image height in pixels")
    width: int = Field(default=1024, ge=1, description="Output image width in pixels")
    ulysses_degree: int = Field(default=8, ge=1, description="Ulysses parallelism degree")
    ring_degree: int = Field(default=1, ge=1, description="Ring parallelism degree")
    pipefusion_parallel_degree: int = Field(
        default=1, ge=1, description="PipeFusion pipeline-parallel degree (multi-node)"
    )
    tensor_parallel_degree: int = Field(default=1, ge=1, description="Tensor-parallel degree (1 = disabled)")
    data_parallel_degree: int = Field(default=1, ge=1, description="Data-parallel degree (1 = disabled)")
    use_torch_compile: bool = Field(default=True, description="Whether to use torch.compile for optimization")
    torchrun_nproc: int = Field(default=8, ge=1, description="Number of processes for torchrun (usually num GPUs)")
    clip_model: Optional[str] = Field(
        default=None,
        description=(
            "CLIP model id or local path used to score saved FLUX images. "
            "Defaults to openai/clip-vit-base-patch32, the CLIPScore paper model, when omitted."
        ),
    )
    fid_nfe_steps: Optional[List[int]] = Field(
        default=None,
        description="Sampling step counts for an optional FID vs NFE curve. Omit to skip test_fid_curve.",
    )
    fid_samples: Optional[int] = Field(
        default=None,
        ge=2,
        description="Images or videos to save at each fid_nfe_steps value. The sweep stops early so the benchmark command stays inside its timeout.",
    )
    fid_reference_dir: Optional[str] = Field(
        default=None,
        description=(
            "Optional in-container directory of real images for the FID reference set. When it is missing "
            "or has fewer than two images, the highest step-count samples from the run are used."
        ),
    )
    fid_inception_path: Optional[str] = Field(
        default=None,
        description="Local Inception-v3 checkpoint. The FID test does not download weights.",
    )
    fid_sampler: Optional[str] = Field(
        default=None,
        description="Sampler name drawn as its own FID series. Omit for a single line labeled by the model.",
    )
    expected_results: Optional[Dict[str, PytorchXditFluxExpectedResults]] = Field(
        default=None,
        description="Legacy inline thresholds by GPU type (auto, mi300x, mi355, etc.)",
    )

    @field_validator('expected_results')
    @classmethod
    def validate_has_auto_or_specific(
        cls, v: Optional[Dict[str, PytorchXditFluxExpectedResults]]
    ) -> Optional[Dict[str, PytorchXditFluxExpectedResults]]:
        """Ensure either 'auto' or a specific GPU type is present when embedded thresholds are used."""
        if v is None:
            return v
        if not v:
            raise ValueError("expected_results must contain at least one GPU type threshold")
        if 'auto' not in v and not any(k in v for k in ['mi300x', 'mi325', 'mi350', 'mi355']):
            raise ValueError("expected_results must contain either 'auto' or a specific GPU type (mi300x, mi325, etc.)")
        return v


class PytorchXditBenchmarkParams(BaseModel):
    """Schema for benchmark_params section in pytorch-xdit configs."""

    model_config = ConfigDict(extra="forbid")

    wan22_i2v_a14b: Optional[PytorchXditWan22Benchmarks] = Field(
        default=None, description="WAN 2.2 image-to-video A14B benchmark parameters"
    )
    flux1_dev_t2i: Optional[PytorchXditFlux1DevBenchmarks] = Field(
        default=None, description="FLUX.1-dev text-to-image benchmark parameters"
    )


class PytorchXditWanConfigFile(BaseModel):
    """
    Schema for PyTorch XDit WAN microbenchmark configuration file.

    Validates WAN inference config structure and provides fail-fast validation.

    Usage:
        with open("mi300x_wan22_i2v_a14b.json") as f:
            raw = json.load(f)
        config = PytorchXditWanConfigFile.model_validate(raw)
    """

    model_config = ConfigDict(extra="forbid")

    config: 'PytorchXditWanConfig' = Field(description="Main configuration section")
    benchmark_params: PytorchXditBenchmarkParams = Field(description="Benchmark parameters section")

    @model_validator(mode='after')
    def validate_benchmark_present(self):
        """Ensure at least one benchmark is configured."""
        if not self.benchmark_params.wan22_i2v_a14b:
            raise ValueError("No benchmarks configured in 'benchmark_params' - at least wan22_i2v_a14b is required")
        if not self.benchmark_params.wan22_i2v_a14b.expected_results:
            raise ValueError("legacy WAN configs must include benchmark_params.wan22_i2v_a14b.expected_results")
        return self

    @model_validator(mode='after')
    def validate_distributed_parallelism(self):
        """When nnodes >= 2, ensure xDiT parallel degrees match nnodes × torchrun_nproc."""
        wan = self.benchmark_params.wan22_i2v_a14b
        nnodes = self.config.nnodes
        if not wan or not nnodes or nnodes < 2:
            return self

        world_size = nnodes * wan.torchrun_nproc
        product = wan.ulysses_size * wan.ring_size
        if product != world_size:
            raise ValueError(
                f"Parallel degree product {product} != world_size {world_size} "
                f"(nnodes={nnodes} × torchrun_nproc={wan.torchrun_nproc}). "
                f"Adjust ulysses_size and ring_size."
            )
        return self


class PytorchXditWanConfig(PytorchXditDistributedNcclExamples):
    """Schema for config section in pytorch-xdit WAN configs."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    container_image: str = Field(
        default="amdsiloai/pytorch-xdit:v25.11.2", description="Docker image for pytorch-xdit container"
    )
    container_name: str = Field(default="wan22-benchmark", description="Name for the Docker container")
    hf_token_file: str = Field(
        default="",
        description=(
            "Optional path to Hugging Face token file. "
            "Not required when using a pre-staged local model path (recommended) or pre-cached HF snapshots (offline)."
        ),
    )
    hf_home: str = Field(description="Host directory for Hugging Face cache (mounted to /hf_home)")
    output_base_dir: str = Field(description="Host base directory for benchmark outputs")
    model_repo: str = Field(
        default="Wan-AI/Wan2.2-I2V-A14B",
        description=(
            "Model identifier. Prefer an explicit local filesystem path (e.g., /models/Wan-AI/Wan2.2-I2V-A14B) "
            "to avoid any runtime downloads. For backward compatibility, a Hugging Face repo id may be used only if "
            "the snapshot is already cached under hf_home."
        ),
    )
    model_rev: str = Field(
        default="206a9ee1b7bfaaf8f7e4d81335650533490646a3",
        description="Model revision (commit hash). Ignored if model_repo is an explicit local filesystem path.",
    )
    nnodes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Distributed node count for unified multi-node torchrun (omit for single-node / scale-out)",
    )
    server_node_list: Optional[List[str]] = Field(
        default=None,
        description="Ordered server nodes for distributed job; defaults to all cluster nodes",
    )
    master_addr: str = Field(
        default="",
        description="Rank-0 rendezvous address; empty means first server node at runtime",
    )
    master_port: int = Field(
        default=29500,
        ge=1,
        le=65535,
        description="Rank-0 rendezvous port for distributed torchrun",
    )
    nccl_ib_hca: str = Field(
        default="",
        description="NCCL_IB_HCA for multi-node ROCm/NCCL (e.g. rdma0,...,rdma7)",
    )
    nccl_socket_ifname: str = Field(
        default="",
        description="NCCL_SOCKET_IFNAME for multi-node jobs",
    )
    gloo_socket_ifname: str = Field(
        default="",
        description="GLOO_SOCKET_IFNAME for multi-node jobs",
    )
    nccl_ib_gid_index: int = Field(
        default=1,
        ge=0,
        description="NCCL_IB_GID_INDEX for IB/RoCE",
    )
    nccl_debug: str = Field(
        default="INFO",
        description="NCCL_DEBUG level (ERROR, INFO, WARN, ...)",
    )
    container_config: PytorchXditContainerConfig = Field(
        default_factory=PytorchXditContainerConfig, description="Container device/volume/env configuration"
    )

    @field_validator('hf_token_file', 'hf_home', 'output_base_dir')
    @classmethod
    def validate_path_not_placeholder(cls, v: str, info) -> str:
        """Check that paths are not still placeholders."""
        if not v:
            return v
        if '<changeme>' in v.lower():
            raise ValueError(f"{info.field_name} contains placeholder '<changeme>'. Please set a valid path in config.")
        return v


class PytorchXditFluxConfigFile(BaseModel):
    """
    Schema for PyTorch XDit Flux microbenchmark configuration file.

    Validates Flux inference config structure and provides fail-fast validation.

    Usage:
        with open("mi300x_flux1_dev_t2i.json") as f:
            raw = json.load(f)
        config = PytorchXditFluxConfigFile.model_validate(raw)
    """

    model_config = ConfigDict(extra="forbid")

    config: 'PytorchXditFluxConfig' = Field(description="Main configuration section")
    benchmark_params: PytorchXditBenchmarkParams = Field(description="Benchmark parameters section")

    @model_validator(mode='after')
    def validate_benchmark_present(self):
        """Ensure at least one benchmark is configured."""
        if not self.benchmark_params.flux1_dev_t2i:
            raise ValueError("No benchmarks configured in 'benchmark_params' - at least flux1_dev_t2i is required")
        if not self.benchmark_params.flux1_dev_t2i.expected_results:
            raise ValueError("legacy FLUX configs must include benchmark_params.flux1_dev_t2i.expected_results")
        return self

    @model_validator(mode='after')
    def validate_distributed_parallelism(self):
        """When nnodes >= 2, ensure xDiT parallel degrees match nnodes × torchrun_nproc."""
        flux = self.benchmark_params.flux1_dev_t2i
        nnodes = self.config.nnodes
        if not flux or not nnodes or nnodes < 2:
            return self

        world_size = nnodes * flux.torchrun_nproc
        product = (
            flux.ulysses_degree
            * flux.ring_degree
            * flux.pipefusion_parallel_degree
            * flux.tensor_parallel_degree
            * flux.data_parallel_degree
        )
        if product != world_size:
            raise ValueError(
                f"Parallel degree product {product} != world_size {world_size} "
                f"(nnodes={nnodes} × torchrun_nproc={flux.torchrun_nproc}). "
                f"Adjust ulysses/ring/pipefusion/tensor_parallel/data_parallel."
            )
        return self


class PytorchXditFluxConfig(PytorchXditDistributedNcclExamples):
    """Schema for config section in pytorch-xdit Flux configs."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    container_image: str = Field(
        default="amdsiloai/pytorch-xdit:v25.11.2", description="Docker image for pytorch-xdit container"
    )
    container_name: str = Field(default="flux-benchmark", description="Name for the Docker container")
    hf_token_file: str = Field(
        default="",
        description=(
            "Optional path to Hugging Face token file. "
            "Not required when using a pre-staged local model path (recommended) or pre-cached HF snapshots (offline)."
        ),
    )
    hf_home: str = Field(description="Host directory for Hugging Face cache (mounted to /hf_home)")
    output_base_dir: str = Field(description="Host base directory for benchmark outputs")
    model_repo: str = Field(
        default="black-forest-labs/FLUX.1-dev",
        description=(
            "Model identifier. Prefer an explicit local filesystem path (e.g., /models/black-forest-labs/FLUX.1-dev) "
            "to avoid any runtime downloads. For backward compatibility, a Hugging Face repo id may be used only if "
            "the snapshot is already cached under hf_home."
        ),
    )
    model_rev: str = Field(
        default="",
        description=(
            "Model revision (commit hash). Empty means use any available cached snapshot under hf_home. "
            "Ignored if model_repo is an explicit local filesystem path."
        ),
    )
    nnodes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Distributed node count for unified multi-node torchrun (omit for single-node / scale-out)",
    )
    server_node_list: Optional[List[str]] = Field(
        default=None,
        description="Ordered server nodes for distributed job; defaults to all cluster nodes",
    )
    master_addr: str = Field(
        default="",
        description="Rank-0 rendezvous address; empty means first server node at runtime",
    )
    master_port: int = Field(
        default=29500,
        ge=1,
        le=65535,
        description="Rank-0 rendezvous port for distributed torchrun",
    )
    nccl_ib_hca: str = Field(
        default="",
        description="NCCL_IB_HCA for multi-node ROCm/NCCL (e.g. rdma0,...,rdma7)",
    )
    nccl_socket_ifname: str = Field(
        default="",
        description="NCCL_SOCKET_IFNAME for multi-node jobs",
    )
    gloo_socket_ifname: str = Field(
        default="",
        description="GLOO_SOCKET_IFNAME for multi-node jobs",
    )
    nccl_ib_gid_index: int = Field(
        default=1,
        ge=0,
        description="NCCL_IB_GID_INDEX for IB/RoCE",
    )
    nccl_debug: str = Field(
        default="INFO",
        description="NCCL_DEBUG level (ERROR, INFO, WARN, ...)",
    )
    container_config: PytorchXditContainerConfig = Field(
        default_factory=PytorchXditContainerConfig, description="Container device/volume/env configuration"
    )

    @field_validator('hf_token_file', 'hf_home', 'output_base_dir')
    @classmethod
    def validate_path_not_placeholder(cls, v: str, info) -> str:
        """Check that paths are not still placeholders."""
        if not v:
            return v
        if '<changeme>' in v.lower():
            raise ValueError(f"{info.field_name} contains placeholder '<changeme>'. Please set a valid path in config.")
        return v


class PytorchXditUnifiedPaths(BaseModel):
    """Host path block for unified xDiT configs."""

    model_config = ConfigDict(extra="forbid")

    shared_fs: str = Field(description="Cluster-visible home or scratch root")
    models_dir: str = Field(description="Host Hugging Face cache or staged model directory")
    log_dir: str = Field(description="Host directory for benchmark outputs and logs")
    hf_token_file: str = Field(default="", description="Optional Hugging Face token file on the host")


class PytorchXditUnifiedModel(BaseModel):
    """Model identifier for unified xDiT configs."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Hugging Face repo id or absolute on-disk model path")
    remote: int = Field(
        default=0,
        ge=0,
        le=1,
        description="0 = offline/local cache only; 1 = allow runtime Hugging Face snapshot download",
    )


class PytorchXditUnifiedRuntime(BaseModel):
    """Docker runtime block for unified xDiT configs."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="docker", description="Container runtime name")
    args: Dict[str, Any] = Field(default_factory=dict, description="Runtime-specific launch arguments")


class PytorchXditUnifiedContainer(BaseModel):
    """Container block for unified xDiT configs."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lifetime: str = Field(default="per_run", description="Container lifetime policy")
    name: str = Field(description="Docker container name")
    image: str = Field(description="Docker image for the benchmark container")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables for docker run")
    runtime: PytorchXditUnifiedRuntime = Field(description="Container runtime definition")
    image_example: Optional[str] = Field(
        default=None,
        alias="_image_example",
        description="Documentation only: example container image",
    )
    image_comment: Optional[str] = Field(
        default=None,
        alias="_image_comment",
        description="Documentation only: model the container image must support",
    )


class PytorchXditUnifiedParams(BaseModel):
    """Benchmark parameters for unified xDiT configs (thresholds live in threshold_json)."""

    model_config = ConfigDict(extra="forbid")

    wan22_i2v_a14b: Optional[PytorchXditWan22Benchmarks] = Field(
        default=None, description="WAN 2.2 image-to-video A14B benchmark parameters"
    )
    flux1_dev_t2i: Optional[PytorchXditFlux1DevBenchmarks] = Field(
        default=None, description="FLUX text-to-image benchmark parameters"
    )


class PytorchXditUnifiedInference(BaseModel):
    """Optional legacy/runtime fields carried beside unified top-level keys."""

    model_config = ConfigDict(extra="allow")

    model_rev: str = Field(
        default="",
        description="Model revision (commit hash) when model.id is a Hugging Face repo id",
    )


class PytorchXditServerParams(BaseModel):
    """SGLang-style server block used by packaged xDiT configs."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="xdit", description="Inference backend identifier")
    nnodes: Any = Field(default="1", description="Participating node count")
    log_level: str = Field(default="info", description="Server log level")
    model: str = Field(description="Hugging Face repo id or absolute on-disk model path")
    benchmark_serv_node: Optional[str] = Field(
        default=None,
        description="Unused. Single jobs run on every cluster node; distributed uses the first cluster node.",
    )
    server_node_list: Optional[List[str]] = Field(
        default=None,
        description="Unused. Distributed jobs take the first nnodes hosts from the cluster.",
    )
    master_addr: str = Field(
        default="",
        description="torchrun rendezvous address. Empty probes the first address on the rank-0 node.",
    )
    master_port: Any = Field(default="29500", description="torchrun rendezvous port")
    model_rev: str = Field(default="", description="Pinned Hugging Face snapshot when model is a repo id")


class PytorchXditUnifiedConfigFile(BaseModel):
    """
    Schema for unified PyTorch xDiT inference configuration files.

    Thresholds are supplied in a sibling ``threshold_json`` file referenced at the top level.
    Packaged files use the SGLang-style ``server_params`` + flat ``benchmark_params`` layout.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    comment: Optional[str] = Field(
        default=None,
        alias="_comment",
        description="Documentation only: human-readable config summary",
    )
    schema_version: int = Field(default=1, ge=1, description="Unified config schema version")
    framework: str = Field(default="xdit", description="Inference framework identifier")
    gpu_name: Optional[str] = Field(default=None, description="Target GPU name, e.g. '<changeme> mi325'")
    gpu_arch: str = Field(default="mi3xx", description="Target GPU architecture family")
    topology: str = Field(default="single", description="Execution topology: single or distributed")
    enforce_thresholds: bool = Field(
        default=True,
        description="When true, benchmark results must satisfy the selected GPU thresholds",
    )
    threshold_json: str = Field(description="Sibling threshold JSON filename")
    paths: PytorchXditUnifiedPaths = Field(description="Host path block")
    model: Optional[PytorchXditUnifiedModel] = Field(default=None, description="Legacy model identifier block")
    container: PytorchXditUnifiedContainer = Field(description="Container definition")
    params: Optional[PytorchXditUnifiedParams] = Field(
        default=None,
        description="Legacy nested benchmark parameters without embedded thresholds",
    )
    server_params: Optional[PytorchXditServerParams] = Field(
        default=None,
        description="SGLang-style model, topology, and execution-host fields",
    )
    benchmark_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Flat xDiT workload parameters (FLUX or WAN)",
    )
    inference: Optional[PytorchXditUnifiedInference] = Field(
        default=None,
        description="Optional runtime fields such as model_rev",
    )
    benchmark_serv_node: Optional[str] = Field(
        default=None,
        description="Single execution node selected from cluster node_dict",
    )
    nnodes: Optional[int] = Field(default=None, ge=1, description="Distributed node count")
    server_node_list: Optional[List[str]] = Field(
        default=None,
        description="Ordered server nodes for distributed jobs",
    )
    master_addr: str = Field(
        default="",
        description="torchrun rendezvous address. Empty probes the first address on the rank-0 node.",
    )
    master_port: int = Field(default=29500, ge=1, le=65535, description="torchrun rendezvous port")
    nccl_ib_hca: str = Field(default="", description="NCCL_IB_HCA for multi-node jobs")
    nccl_socket_ifname: str = Field(default="", description="NCCL_SOCKET_IFNAME for multi-node jobs")
    gloo_socket_ifname: str = Field(default="", description="GLOO_SOCKET_IFNAME for multi-node jobs")
    nccl_ib_gid_index: int = Field(default=3, ge=0, description="NCCL_IB_GID_INDEX for IB/RoCE")
    nccl_debug: str = Field(default="INFO", description="NCCL_DEBUG level")
    example_nccl_ib_hca: Optional[str] = Field(
        default=None,
        alias="_example_nccl_ib_hca",
        description="Documentation only: example NCCL_IB_HCA value",
    )
    example_nccl_socket_ifname: Optional[str] = Field(
        default=None,
        alias="_example_nccl_socket_ifname",
        description="Documentation only: example nccl_socket_ifname value",
    )
    example_gloo_socket_ifname: Optional[str] = Field(
        default=None,
        alias="_example_gloo_socket_ifname",
        description="Documentation only: example gloo_socket_ifname value",
    )

    @model_validator(mode="after")
    def validate_benchmark_present(self):
        backend = self.framework
        if self.server_params is not None and self.server_params.backend:
            backend = self.server_params.backend
        if str(backend).lower() not in ("xdit", "pytorch_xdit"):
            raise ValueError("framework/backend must be 'xdit' or 'pytorch_xdit'")
        if self.topology not in ("single", "distributed"):
            raise ValueError("topology must be 'single' or 'distributed'")
        nnodes = self.nnodes
        if nnodes is None and self.server_params is not None and self.server_params.nnodes not in (None, ""):
            nnodes = int(self.server_params.nnodes)
        elif nnodes is not None:
            nnodes = int(nnodes)
        else:
            nnodes = 1
        nested = 0
        if self.params is not None:
            nested = sum(
                workload is not None
                for workload in (
                    self.params.wan22_i2v_a14b,
                    self.params.flux1_dev_t2i,
                )
            )
        flat = self.benchmark_params if isinstance(self.benchmark_params, dict) else {}
        has_flat = bool(flat) and not any(key in flat for key in ("flux1_dev_t2i", "wan22_i2v_a14b"))
        has_nested_bp = any(isinstance(flat.get(key), dict) for key in ("flux1_dev_t2i", "wan22_i2v_a14b"))
        if nested != 1 and not has_flat and not has_nested_bp:
            raise ValueError(
                "config must include params.flux1_dev_t2i, params.wan22_i2v_a14b, or flat benchmark_params"
            )
        if self.server_params is None and self.model is None:
            raise ValueError("config must include server_params.model or model.id")
        return self

    @model_validator(mode="after")
    def validate_distributed_topology(self):
        nnodes = self.nnodes
        if nnodes is None and self.server_params is not None and self.server_params.nnodes not in (None, ""):
            nnodes = int(self.server_params.nnodes)
        elif nnodes is not None:
            nnodes = int(nnodes)
        else:
            nnodes = 1
        if (self.topology == "distributed" or nnodes > 1) and nnodes < 2:
            raise ValueError("topology='distributed' requires nnodes >= 2")
        return self


class PytorchXditThresholdFile(BaseModel):
    """Schema for sibling xDiT threshold JSON files."""

    model_config = ConfigDict(extra="forbid")

    thresholds: Dict[str, PytorchXditExpectedResults] = Field(
        default_factory=dict,
        description="GPU-keyed latency thresholds",
    )

    @model_validator(mode="before")
    @classmethod
    def collect_gpu_thresholds(cls, data):
        if not isinstance(data, dict):
            return data
        thresholds = {}
        for key, value in data.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            thresholds[key] = value
        return {"thresholds": thresholds}

    @model_validator(mode="after")
    def validate_has_auto_or_specific(self):
        keys = self.thresholds
        if not keys:
            raise ValueError("threshold file must contain at least one GPU type threshold")
        if "auto" not in keys and not any(k in keys for k in ["mi300x", "mi325", "mi350", "mi355"]):
            raise ValueError("threshold file must contain either 'auto' or a specific GPU type (mi300x, mi325, etc.)")
        return self


def is_pytorch_xdit_unified_config(raw_config):
    """Return True when a loaded dict uses the unified xDiT layout."""
    if not isinstance(raw_config, dict):
        return False
    if not isinstance(raw_config.get("paths"), dict) or not isinstance(raw_config.get("container"), dict):
        return False
    framework = str(raw_config.get("framework") or "").lower()
    if framework in ("xdit", "pytorch_xdit"):
        return True
    backend = str((raw_config.get("server_params") or {}).get("backend") or "").lower()
    if backend in ("xdit", "pytorch_xdit"):
        return True
    threshold = str(raw_config.get("threshold_json") or "").lower()
    return "pytorch_xdit" in threshold


# =============================================================================
# Inference config blocks
# =============================================================================
#
# The blocks every CVS suite schema is built from, re-exported from
# cvs.lib.conf_lib.config_blocks and cvs.lib.utils.config_loader, then the
# shared inference blocks that InferenceConfig joins.
#
# Keys that start with ``_`` are comments. Each model drops them at its own
# level only, so a ``_`` key inside a free-form map (``env``, a CLI option
# payload) reaches the suite unchanged.
#
# Shipped configs are templates: ``<changeme>`` and ``<changeme-...>`` mark the
# values a user fills in for their setup. A ``ConfigStr`` field rejects them
# only when validated with ``context={"placeholders": "reject"}``, which
# ``cvs.lib.conf_lib.loader.load_config`` always passes.
#
# Nothing here imports cvs.lib at module level: cvs.lib imports this module.

_PLACEHOLDER_RE = re.compile(r"<changeme(?:-[^<>]*)?>", re.IGNORECASE)
PLACEHOLDER_MESSAGE = "still a <changeme> placeholder; replace it with the value for your setup"


def _drop_comment_keys(value):
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if not (isinstance(key, str) and key.startswith("_"))}
    return value


def has_placeholder(value):
    return isinstance(value, str) and bool(_PLACEHOLDER_RE.search(value))


def placeholders_rejected(info):
    """True when the caller validates with ``context={"placeholders": "reject"}``."""
    return bool(info.context) and info.context.get("placeholders") == "reject"


def _reject_placeholder(value, info):
    if placeholders_rejected(info) and has_placeholder(value):
        raise ValueError(PLACEHOLDER_MESSAGE)
    return value


ConfigStr = Annotated[str, AfterValidator(_reject_placeholder)]


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_comments(cls, data):
        return _drop_comment_keys(data)


class _Allow(_Forbid):
    model_config = ConfigDict(extra="allow")


class Paths(_Forbid):
    shared_fs: ConfigStr
    models_dir: ConfigStr
    log_dir: ConfigStr
    hf_token_file: ConfigStr
    # Host-user-namespaced scratch (jaxmaxtext launchers/yml). Optional so
    # inference configs that omit it still load; jaxmaxtext configs set it to
    # /tmp/{user-id}/jaxmaxtext so container-root /tmp/root is never used.
    temp_dir: ConfigStr = ""


class ModelSpec(_Forbid):
    id: str
    remote: Literal[0, 1]
    precision: str = ""


class RuntimeSpec(_Allow):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ContainerSpec(_Forbid):
    lifetime: Literal["no_launch", "per_run", "persistent"] = "per_run"
    name: ConfigStr
    image: ConfigStr
    runtime: RuntimeSpec
    # Environment variables passed to the container at `docker run` time (as
    # `-e KEY=VALUE`, merged with the orchestrator defaults). The container
    # orchestrator reads `container_config['env']`; carrying it on the spec means
    # `container.model_dump()` forwards it verbatim to the runtime.
    env: Dict[str, ConfigStr] = Field(default_factory=dict)


class RuntimeArgs(_Allow):
    """The ``runtime.args`` keys the Docker runtime reads.

    An unset key is left out of the dump, so the runtime falls back to its own
    default exactly as if the config had not named it.
    """

    volumes: Optional[List[ConfigStr]] = None
    devices: Optional[List[str]] = None
    env: Optional[Dict[str, ConfigStr]] = None
    cap_add: Optional[List[str]] = None
    security_opt: Optional[List[str]] = None
    group_add: Optional[List[Union[str, int]]] = None
    ulimit: Optional[List[str]] = None
    network: Optional[str] = None
    ipc: Optional[str] = None
    user: Optional[Union[str, int]] = None
    shm_size: Optional[Union[str, int]] = None
    privileged: Optional[bool] = None
    registry: Optional[Dict[str, str]] = None

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        return {key: value for key, value in handler(self).items() if value is not None}


FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class ValueThreshold(_Forbid):
    kind: Literal["min", "max", "max_ms", "min_tok_s", "info"]
    value: FiniteNumber
    optional: bool = False


class WithinThreshold(_Forbid):
    kind: Literal["within"]
    value: FiniteNumber
    tolerance_pct: FiniteNumber
    optional: bool = False


class MinRatioThreshold(_Forbid):
    kind: Literal["min_ratio"]
    value: FiniteNumber
    reference: str
    optional: bool = False


ThresholdSpec = Annotated[
    Union[ValueThreshold, WithinThreshold, MinRatioThreshold],
    Field(discriminator="kind"),
]
ThresholdCell = Annotated[Dict[str, ThresholdSpec], BeforeValidator(_drop_comment_keys)]


class ThresholdFile(_Allow):
    """A threshold file: one cell of metric specs per run cell, plus ``accuracy``.

    Accuracy thresholds are keyed by task and read by the accuracy harness, so
    they are not checked here.
    """

    accuracy: Dict[str, Any] = Field(default_factory=dict)
    __pydantic_extra__: Dict[str, ThresholdCell]

    def cells(self):
        return dict(self.model_extra or {})


class BaseVariantConfig(_Forbid):
    """The framework-agnostic skeleton of a variant config.

    Carries the fields every suite shares (schema/paths/model/image/container/
    thresholds + the enforce gate) and the remote-not-implemented guard.
    Per-framework subclasses add their own `framework`/`Params`/`Sweep` and the
    `cell_key`/coverage-check pair that depend on them.
    """

    schema_version: Literal[1]
    # When false, the threshold-coverage gate warns instead of raising and the
    # test records metrics without asserting pass/fail (record-only). Use for
    # un-calibrated shapes (e.g. a throughput characterization whose published
    # numbers are curves, not tabulated values). Default true keeps the gate
    # strict for calibrated configs -- no regression to the remediation work.
    enforce_thresholds: bool = True
    threshold_json: str = ""
    paths: Paths
    model: ModelSpec
    # The container image is declared once, on container.image (ContainerSpec).
    # There is no separate top-level image block.
    container: ContainerSpec
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # pydantic runs @model_validator(mode="after") hooks in definition order,
    # parent-class hooks before subclass hooks. This remote check is intentionally
    # the first to run: an unimplemented remote config fails fast
    # (NotImplementedError) before any subclass's threshold-coverage check runs,
    # which is meaningless for a config we are going to reject anyway.
    @model_validator(mode="after")
    def _check_remote_not_implemented(self):
        if self.model.remote == 1 and getattr(self, "framework", None) != "xdit":
            raise NotImplementedError(
                "model.remote=1 (remote model download) is not implemented in the PoC. "
                "Port from cvs-dtni-v1/resource_resolver.py before enabling."
            )
        return self


class AccuracyTask(_Forbid):
    """An lm-evaluation-harness 0.4.12 invocation against a live OpenAI-compatible server.

    CVS owns the live-server endpoint, model path, output path, and sample
    logging. Every other supported ``lm_eval run`` setting has an explicit field
    and its 0.4.12 default.
    """

    id: str
    # ``task`` remains accepted so existing Atom and vLLM configs need no
    # migration. ``tasks`` is the lm-eval 0.4.12 spelling and supports a
    # comma-separated string or a list of task names.
    tasks: Optional[Union[str, List[str]]] = None
    task: Optional[str] = None

    # CVS/OpenAI adapter selection. These are deliberately narrower than
    # lm-eval's generic --model field because CVS always targets a live local
    # OpenAI-compatible server.
    backend: str = ""
    lm_eval_model: Optional[Literal["local-completions", "local-chat-completions"]] = None
    extra_model_args: str = ""
    num_concurrent: int = 8

    # lm-eval 0.4.12 evaluation settings.
    num_fewshot: Optional[int] = None
    batch_size: str = "1"
    max_batch_size: Optional[int] = None
    device: Optional[str] = None
    gen_kwargs: Dict[str, Any] = Field(default_factory=dict)
    limit: Optional[float] = None
    samples: Optional[Union[str, Dict[str, Any]]] = None
    use_cache: Optional[str] = None
    cache_requests: Optional[Literal["true", "refresh", "delete"]] = None
    check_integrity: bool = False
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: Optional[bool] = None
    include_path: Optional[str] = None
    predict_only: bool = False
    seed: List[Optional[int]] = Field(default_factory=lambda: [0, 1234, 1234, 1234])
    trust_remote_code: bool = False
    confirm_run_unsafe_code: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # CVS controls execution duration, not lm-eval.
    exec_timeout_sec: int = 4 * 60 * 60

    @field_validator("apply_chat_template", mode="before")
    @classmethod
    def _coerce_boolean_template_values(cls, value):
        if isinstance(value, str) and value.lower() in {"true", "false", "0", "1"}:
            return value.lower() in {"true", "1"}
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def _normalize_seed(cls, value):
        defaults = [0, 1234, 1234, 1234]
        if isinstance(value, int):
            return [value] * 4
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, list):
            normalized = [None if item is None or item == "None" else int(item) for item in value]
            if len(normalized) == 1:
                return normalized * 4
            if len(normalized) == 3:
                return [*normalized, defaults[3]]
            return normalized
        return value

    @field_validator("tasks", mode="before")
    @classmethod
    def _reject_empty_task_lists(cls, value):
        if value == []:
            raise ValueError("tasks must not be empty")
        return value

    @field_validator("seed")
    @classmethod
    def _validate_seed_count(cls, value):
        if len(value) != 4:
            raise ValueError("seed must contain exactly four values: python,numpy,torch,fewshot")
        return value

    @model_validator(mode="after")
    def _resolve_legacy_task_alias(self):
        if self.tasks is None and self.task is None:
            raise ValueError("one of tasks or task is required")
        if self.tasks is None:
            self.tasks = self.task
        elif self.task is not None and self.task_names() != self._split_task_string(self.task):
            raise ValueError("task and tasks disagree")
        # Preserve the historical public attribute for Atom and callers that
        # consume a single comma-separated task selector.
        task_names = self.task_names()
        if not task_names:
            raise ValueError("tasks must contain at least one task name")
        self.task = ",".join(task_names)
        return self

    @staticmethod
    def _split_task_string(value: str) -> List[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def task_names(self) -> List[str]:
        if isinstance(self.tasks, list):
            return self.tasks
        return self._split_task_string(self.tasks or "")

    @property
    def resolved_lm_eval_model(self) -> str:
        if self.lm_eval_model:
            return self.lm_eval_model
        return "local-chat-completions" if self.apply_chat_template else "local-completions"


class AccuracyConfig(_Forbid):
    tasks: List[AccuracyTask] = []

    @model_validator(mode="after")
    def _check_unique_task_ids(self):
        from collections import Counter

        counts = Counter(t.id for t in self.tasks)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            rendered = ", ".join(repr(d) for d in dupes)
            raise ValueError(f"duplicate task id(s): {rendered}")
        return self


class LongContextAccCell(_Forbid):
    """Long-context (needle-in-a-haystack) accuracy cell; gating values live in the threshold file."""

    id: str
    isl: int
    osl: int = 32
    num_prompts: int = 8
    seed: int = 42


class LongContextAccuracyConfig(_Forbid):
    cells: List[LongContextAccCell] = []

    @model_validator(mode="after")
    def _check_unique_cell_ids(self):
        from collections import Counter

        counts = Counter(c.id for c in self.cells)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            rendered = ", ".join(repr(d) for d in dupes)
            raise ValueError(f"duplicate long-context cell id(s): {rendered}")
        return self


class FunctionalConfig(_Forbid):
    """Optional functional smoke gates for inference suites."""

    api_smoke: bool = False
    health_check: bool = False


class PlatformConfig(_Forbid):
    """Optional platform / post-run checks for inference suites."""

    dmesg_scan: bool = False
    gpu_metrics_poll: bool = False


class MtpQualityConfig(_Forbid):
    enabled: bool = False
    chat_template_prompt: str = "Say hello in one short sentence."
    chat_template_expected_sha256: str = ""


class QuantParityConfig(_Forbid):
    enabled: bool = False
    probe_prompt: str = "The capital of France is"
    reference_config_stem: str = ""


# ---------- shared inference blocks ----------

# The metric names every engine's threshold cells use: vLLM's metric registry
# (cvs/lib/inference/utils/vllm_metrics.py) plus ATOM's multinode scaling metric.
_LATENCY_STATS = ("mean", "median", "std", "p50", "p90", "p95", "p99")
INFERENCE_METRIC_NAMES = frozenset(
    (
        "max_concurrency",
        "max_concurrent_requests",
        "num_prompts",
        "completed",
        "failed",
        "success_rate",
        "duration",
        "request_throughput",
        "goodput",
        "output_throughput",
        "total_token_throughput",
        "per_gpu_throughput",
        "decode_throughput_p50",
        "max_output_tokens_per_s",
        "rtfx",
        "total_input_tokens",
        "total_output_tokens",
        *(f"{stat}_ttft_ms" for stat in _LATENCY_STATS),
        "normalized_ttft_ms_per_tok",
        *(f"{stat}_tpot_ms" for stat in _LATENCY_STATS),
        *(f"{stat}_itl_ms" for stat in _LATENCY_STATS),
        "decode_latency_ratio",
        *(f"{stat}_e2el_ms" for stat in _LATENCY_STATS),
        "peak_gpu_memory_mb",
        "model_load_memory_mb",
        "model_load_s",
        "gpu_bandwidth_util_pct",
        "gpu_compute_util_pct",
        "queue_time_p50_ms",
        "queue_time_p95_ms",
        "prefill_time_p50_ms",
        "prefill_time_p95_ms",
        "scaling.efficiency_pct",
    )
)


class InferenceThresholdSpec(_Forbid):
    """One metric's threshold: ``min`` and ``max`` gate, ``info`` records without gating."""

    kind: Literal["min", "max", "info"]
    value: FiniteNumber
    optional: bool = False


InferenceThresholdCell = Annotated[Dict[str, InferenceThresholdSpec], BeforeValidator(_drop_comment_keys)]


def _check_metric_names(cell):
    unknown = sorted(set(cell) - INFERENCE_METRIC_NAMES)
    if unknown:
        raise ValueError(f"unknown inference metric(s): {unknown}")
    return cell


InferenceMetricCell = Annotated[InferenceThresholdCell, AfterValidator(_check_metric_names)]


class InferenceAccuracyThresholds(_Forbid):
    """Mirrors the config's ``accuracy`` block: ``lm_eval`` and ``custom_eval`` are keyed by id."""

    lm_eval: Dict[str, InferenceThresholdCell] = Field(default_factory=dict)
    custom_eval: Dict[str, InferenceThresholdCell] = Field(default_factory=dict)
    quant_parity: InferenceThresholdCell = Field(default_factory=dict)
    mtp_quality: InferenceThresholdCell = Field(default_factory=dict)


_CELL_KEY_RE = re.compile(r"^ISL=([1-9]\d*),OSL=([1-9]\d*),TP=([1-9]\d*),PP=([1-9]\d*),CONC=([1-9]\d*)$")
_CELL_KEY_AXES = ("isl", "osl", "tp", "pp", "concurrency")


def parse_inference_cell_key(key):
    """Return a cell key's axes as ints; the key must read ``ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>``."""
    match = _CELL_KEY_RE.fullmatch(key)
    if not match:
        raise ValueError(f"{key!r} is not a cell key of the form ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>")
    return dict(zip(_CELL_KEY_AXES, (int(number) for number in match.groups())))


class InferenceThresholdFile(_Allow):
    """A threshold file: one section of metric specs per cell key, plus ``accuracy``.

    Cell metrics must be in ``INFERENCE_METRIC_NAMES``; accuracy metrics are named by the eval.
    """

    accuracy: InferenceAccuracyThresholds = Field(default_factory=InferenceAccuracyThresholds)
    __pydantic_extra__: Dict[str, InferenceMetricCell]

    @model_validator(mode="after")
    def _check_cell_keys(self):
        for key in self.model_extra or {}:
            parse_inference_cell_key(key)
        return self

    def cells(self):
        return dict(self.model_extra or {})

    def get(self, cell_key, default=None):
        """A cell's specs as plain dicts, as written, for the report code that reads threshold dicts."""
        cell = self.cells().get(cell_key)
        if cell is None:
            return default
        return {metric: spec.model_dump(exclude_unset=True) for metric, spec in cell.items()}


_HF_TOKEN_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}


class ContainerEnvTokenError(Exception):
    """A Hugging Face token name in container.env.

    Not a ValueError, so pydantic raises it as is instead of wrapping it in a
    ValidationError whose str(), errors() and json() would carry the token value.
    """


class InferenceRuntimeArgs(RuntimeArgs):
    # container.env is the one env block, so the Hugging Face token check
    # cannot be bypassed through docker run -e arguments built from here.
    @model_validator(mode="before")
    @classmethod
    def _reject_env(cls, data):
        if isinstance(data, dict) and "env" in data:
            raise ValueError("runtime.args.env is not read; set these variables in container.env")
        return data


class InferenceRuntimeSpec(RuntimeSpec):
    args: InferenceRuntimeArgs = Field(default_factory=InferenceRuntimeArgs)


class InferenceContainerSpec(ContainerSpec):
    runtime: InferenceRuntimeSpec

    @model_validator(mode="before")
    @classmethod
    def _reject_hf_token_env(cls, data):
        env = data.get("env") if isinstance(data, dict) else None
        tokens = sorted(name for name in env or {} if str(name).upper() in _HF_TOKEN_ENV)
        if tokens:
            raise ContainerEnvTokenError(
                f"container.env must not set {tokens}: container.env values become docker run -e arguments. "
                "Put the Hugging Face token in the file named by paths.hf_token_file instead."
            )
        return data


def _reject_flag_placeholder(value, info):
    if placeholders_rejected(info) and has_placeholder(value if isinstance(value, str) else repr(value)):
        raise ValueError(PLACEHOLDER_MESSAGE)
    return value


FlagValue = Annotated[Any, AfterValidator(_reject_flag_placeholder)]
_FLAG_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _check_flags(flags, section, reserved):
    for name, value in flags.items():
        if not _FLAG_NAME_RE.fullmatch(name):
            raise ValueError(f"{section}.{name}: flag names must be snake_case")
        if value is False:
            raise ValueError(f"{section}.{name}: false is ambiguous; omit the flag or use its negative form")
    harness = sorted(set(flags) & reserved)
    if harness:
        raise ValueError(f"{section} cannot set harness fields: {harness}")


# Server flags the harness sets on a multinode launch.
_HARNESS_SERVER_FLAGS = frozenset({"master_addr", "node_rank"})
# Client flags the harness sets for every cell.
_HARNESS_CLIENT_FLAGS = frozenset(
    {
        "model",
        "backend",
        "base_url",
        "max_concurrency",
        "random_input_len",
        "random_output_len",
        "result_dir",
        "result_filename",
        "percentile_metrics",
        "metric_percentiles",
    }
)


class LongContextFlags(_Forbid):
    """Server flags added only when a ``niah`` eval runs."""

    chunked_prefill_size: Optional[int] = None
    max_prefill_tokens: Optional[int] = None


class InferenceServerParams(_Forbid):
    """The server keys every engine shares; any other server flag goes in ``engine_args``."""

    model: ConfigStr
    tp: PositiveInt
    pp: PositiveInt = 1
    dp: PositiveInt = 1
    ep: PositiveInt = 1
    port: int = 8888
    master_port: int = 29501
    # null means 1 (single only); distributed needs at least 2.
    nnodes: Optional[PositiveInt] = None
    # Disaggregated only; null splits nnodes in half.
    prefill_nnodes: Optional[PositiveInt] = None
    max_model_length: Optional[PositiveInt] = None
    long_context: LongContextFlags = Field(default_factory=LongContextFlags)
    # Native server flags in snake_case; true writes a bare flag.
    engine_args: Dict[str, FlagValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_engine_args(self):
        _check_flags(self.engine_args, "engine_args", set(type(self).model_fields) | _HARNESS_SERVER_FLAGS)
        return self


class InferenceBenchmarkParams(_Forbid):
    """The client keys every engine shares; any other client flag goes in ``client_args``."""

    num_prompts: PositiveInt
    random_range_ratio: float = 0.0
    seed: int = 0
    request_rate: Union[float, ConfigStr] = "inf"
    burstiness: float = 1.0
    random_prefix_len: int = 0
    tokenizer_mode: ConfigStr = "auto"
    # Single-node output throughput; sets scaling.efficiency_pct on multinode cells.
    scaling_baseline_output_throughput: Optional[float] = None
    # Native client flags in snake_case; true writes a bare flag.
    client_args: Dict[str, FlagValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_client_args(self):
        reserved = set(type(self).model_fields) | _HARNESS_CLIENT_FLAGS
        _check_flags(self.client_args, "client_args", reserved)
        return self


class InferenceTimeouts(_Forbid):
    """Seconds to wait; null keeps the engine's default."""

    server_ready_s: Optional[PositiveInt] = None
    client_s: Optional[PositiveInt] = None


class NiahEval(LongContextAccCell):
    type: Literal["niah"]


class InferenceAccuracyConfig(_Forbid):
    lm_eval: AccuracyConfig = Field(default_factory=AccuracyConfig)
    # niah is the only custom eval type; a new type joins NiahEval in a union on ``type``.
    custom_eval: List[NiahEval] = Field(default_factory=list)
    quant_parity: QuantParityConfig = Field(default_factory=QuantParityConfig)
    mtp_quality: MtpQualityConfig = Field(default_factory=MtpQualityConfig)

    @model_validator(mode="after")
    def _check_unique_custom_eval_ids(self):
        dupes = sorted(eval_id for eval_id, n in Counter(e.id for e in self.custom_eval).items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate custom_eval id(s): {dupes}")
        return self


# One server serves every cell, so a cell can override only client flags.
_CELL_RESERVED = _HARNESS_CLIENT_FLAGS | set(InferenceServerParams.model_fields) | {"client_args"}


def _field_model(model_cls, name):
    annotation = model_cls.model_fields[name].annotation
    return annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None


class InferenceConfig(_Forbid):
    """The config every inference engine shares; an engine subclass adds only its own block.

    An engine lists in ``not_implemented`` the dotted paths of shared fields it
    does not run. A config that sets any of them to a value other than its
    default fails, naming the engine class and the field.
    """

    not_implemented: ClassVar[Tuple[str, ...]] = ()

    # false records every metric without failing.
    enforce_thresholds: bool = True
    # Threshold file next to the config or packaged; "" records without gating.
    threshold_json: ConfigStr = ""
    paths: Paths
    container: InferenceContainerSpec
    server_params: InferenceServerParams
    benchmark_params: InferenceBenchmarkParams
    timeouts: InferenceTimeouts = Field(default_factory=InferenceTimeouts)
    # Cell key -> client flags for that cell only.
    sweeps: Dict[str, Dict[str, FlagValue]]
    # The sweeps keys to run, in order.
    runs: List[str]
    accuracy: InferenceAccuracyConfig = Field(default_factory=InferenceAccuracyConfig)
    thresholds: InferenceThresholdFile = Field(default_factory=InferenceThresholdFile)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        for path in cls.not_implemented:
            model_cls = cls
            for name in path.split("."):
                if model_cls is None or name not in model_cls.model_fields:
                    raise TypeError(f"{cls.__name__}.not_implemented names {path!r}, which is not a field")
                model_cls = _field_model(model_cls, name)

    @field_validator("sweeps", mode="before")
    @classmethod
    def _drop_sweep_comments(cls, value):
        if not isinstance(value, dict):
            return value
        return {key: _drop_comment_keys(entry) for key, entry in value.items() if not key.startswith("_")}

    @model_validator(mode="after")
    def _check_not_implemented(self):
        for path in type(self).not_implemented:
            *parents, name = path.split(".")
            owner = self
            for parent in parents:
                owner = getattr(owner, parent)
            default = type(owner).model_fields[name].get_default(call_default_factory=True)
            if getattr(owner, name) != default:
                raise ValueError(f"{path} is not implemented for {type(self).__name__}; leave it at its default")
        return self

    @model_validator(mode="after")
    def _check_sweeps_and_runs(self):
        for key, overrides in self.sweeps.items():
            axes = parse_inference_cell_key(key)
            if (axes["tp"], axes["pp"]) != (self.server_params.tp, self.server_params.pp):
                raise ValueError(f"sweeps.{key}: TP and PP must equal server_params.tp and server_params.pp")
            _check_flags(overrides, f"sweeps.{key}", _CELL_RESERVED)
        if not self.runs:
            raise ValueError("runs must list at least one sweeps key")
        repeated = sorted(key for key, n in Counter(self.runs).items() if n > 1)
        if repeated:
            raise ValueError(f"runs repeats {repeated}")
        unknown = [key for key in self.runs if key not in self.sweeps]
        if unknown:
            raise ValueError(f"runs names keys missing from sweeps: {unknown}")
        return self

    @model_validator(mode="after")
    def _check_threshold_coverage(self):
        for key in self.thresholds.cells():
            axes = parse_inference_cell_key(key)
            if (axes["tp"], axes["pp"]) != (self.server_params.tp, self.server_params.pp):
                raise ValueError(f"thresholds.{key}: TP and PP must equal server_params.tp and server_params.pp")
        if self.enforce_thresholds and self.threshold_json.strip():
            missing = [key for key in self.runs if key not in self.thresholds.cells()]
            if missing:
                raise ValueError(f"runs missing from the threshold file: {missing}")
        return self


# =============================================================================
# Old Config Keys
# =============================================================================

logger = logging.getLogger(__name__)

OLD_KEY_DROPPED = object()
OLD_KEY_UNMAPPED = object()
_OLD_VALUE = object()
_MISSING = object()


def _config_path_parts(path):
    return tuple(path) if isinstance(path, tuple) else tuple(path.split("."))


class OldKey:
    """Row of an old-keys table that moves an old key to ``new_path``.

    ``value`` replaces the old value, for a rule that computes the new value
    from several old keys; such a row is written even when its own old path is
    absent, provided the input holds some old key. ``keep_old`` copies the value and leaves the old key in place.
    ``conflict`` decides what happens when ``new_path`` already holds a
    different value: ``"fail"`` raises, ``"replace"`` overwrites it.
    """

    def __init__(self, new_path, value=_OLD_VALUE, keep_old=False, conflict="fail"):
        if conflict not in ("fail", "replace"):
            raise ValueError(f"Unknown old-key conflict rule: {conflict}")
        self.new_path = _config_path_parts(new_path)
        self.value = value
        self.keep_old = keep_old
        self.conflict = conflict


def _get_config_path(config, parts):
    node = config
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _same_old_key_row(first, second):
    if isinstance(first, OldKey) and isinstance(second, OldKey):
        return (first.new_path, first.value, first.keep_old, first.conflict) == (
            second.new_path,
            second.value,
            second.keep_old,
            second.conflict,
        )
    return first is second


def _expand_old_key_rows(config, table, rules):
    rows = {}
    entries = list(table.items())
    for rule in rules:
        entries.extend(rule(config).items())
    for path, row in entries:
        parts = _config_path_parts(path)
        if isinstance(row, (str, tuple)):
            row = OldKey(row)
        if not isinstance(row, OldKey) and row is not OLD_KEY_DROPPED and row is not OLD_KEY_UNMAPPED:
            raise ValueError(f"Old-key row for {'.'.join(parts)} is not a path, OldKey or OLD_KEY_* marker")
        if parts in rows and not _same_old_key_row(rows[parts], row):
            raise ValueError(f"Old-key rows disagree for {'.'.join(parts)}")
        rows[parts] = row
    return {
        parts: row
        for parts, row in rows.items()
        if _get_config_path(config, parts) is not _MISSING or (isinstance(row, OldKey) and row.value is not _OLD_VALUE)
    }


def _write_old_key_value(converted, old_name, row, value, label, prefix):
    new_name = prefix + ".".join(row.new_path)
    node = converted
    for depth, part in enumerate(row.new_path[:-1], start=1):
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"{prefix}{'.'.join(row.new_path[:depth])} must be an object to hold {new_name}")
    key = row.new_path[-1]
    if key in node and row.conflict == "fail":
        if node[key] != value:
            raise ValueError(
                f"Conflicting {label} options: {old_name} and {new_name} must have the same value when both are provided"
            )
        return
    node[key] = value


def resolve_old_keys(
    config,
    table,
    rules=(),
    message="Deprecated config keys: {keys}. Use the new names; old keys will be removed in a future release.",
    label="config",
    prefix="",
):
    """Map old config keys to their new paths, or drop them.

    ``table`` maps an old path to a row: a new path, an ``OldKey``,
    ``OLD_KEY_DROPPED``, or ``OLD_KEY_UNMAPPED`` for a key that has no place
    in the new format. A path is a dotted string, or a tuple of keys for a key
    that holds a dot. Each of ``rules`` is a function that reads the input
    config and returns more rows, so it can decide from a value or compute one
    value from several keys. Rows for paths the input does not hold are
    ignored, and two different rows for one path fail.

    Every old value is read from the input before any key is moved, so rows
    may overlap or chain. Returns a converted deep copy and one warning message
    naming each old key, built from ``message`` by replacing ``{keys}``, or the
    input and ``None`` when no old key is present. An unmapped key raises
    ``ValueError`` naming its path, never its value. ``prefix`` is prepended to
    every path in messages. A parent left empty by a move stays in place.
    Callers that load a config pass the combined message of all their calls to
    ``warn_old_keys`` once.
    """
    if not isinstance(config, dict):
        return config, None
    rows = _expand_old_key_rows(config, table, rules)
    unmapped = [prefix + ".".join(parts) for parts, row in rows.items() if row is OLD_KEY_UNMAPPED]
    if unmapped:
        raise ValueError(
            f"Unsupported {label} options: {', '.join(unmapped)}; "
            "they have no place in the new format, so move or delete them"
        )
    if not any(_get_config_path(config, parts) is not _MISSING for parts in rows):
        return config, None

    values = {
        parts: deepcopy(_get_config_path(config, parts) if row.value is _OLD_VALUE else row.value)
        for parts, row in rows.items()
        if row is not OLD_KEY_DROPPED
    }
    converted = deepcopy(config)
    for parts, row in rows.items():
        if row is OLD_KEY_DROPPED or not row.keep_old:
            parent = _get_config_path(converted, parts[:-1])
            if isinstance(parent, dict):
                parent.pop(parts[-1], None)

    changes = []
    for parts, row in rows.items():
        old_name = prefix + ".".join(parts)
        present = _get_config_path(config, parts) is not _MISSING
        if row is OLD_KEY_DROPPED:
            changes.append(f"{old_name} (dropped)")
            continue
        _write_old_key_value(converted, old_name, row, values[parts], label, prefix)
        if present:
            changes.append(f"{old_name} -> {prefix}{'.'.join(row.new_path)}")
    return converted, message.replace("{keys}", ", ".join(changes))


def warn_old_keys(message):
    """Raise one ``FutureWarning`` with ``message`` and log the same text once.

    Under ``cvs run`` pytest captures a warning raised during test setup, so
    the log is what reaches the console and the run log. Does nothing when
    ``message`` is empty or ``None``.
    """
    if not message:
        return
    logger.warning(message)
    warnings.warn(message, FutureWarning, stacklevel=2)


# =============================================================================
# vLLM config
# =============================================================================
#
# VllmConfig is InferenceConfig plus the vllm block. An old vLLM config, and the
# old threshold file it names, load through resolve_old_keys with one
# FutureWarning. The vLLM metric registry stays in cvs.lib and is imported
# inside the validator that reads it.

_VLLM_SOCKET_ENV = frozenset({"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"})
_VLLM_HARNESS_SERVER_FLAGS = frozenset(
    {
        "headless",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "max_model_len",
        "distributed_executor_backend",
    }
)
_VLLM_HARNESS_CLIENT_FLAGS = frozenset({"dataset_name"})
_THRESHOLD_METADATA_PREFIXES = ("_comment", "_example")


def _check_vllm_flag_values(flags, section):
    for name, value in flags.items():
        if value is False:
            raise ValueError(f"{section}.{name}=false is ambiguous; omit it or use the option's negative flag")
        if isinstance(value, list) and any(isinstance(item, (list, dict)) for item in value):
            raise ValueError(f"{section}.{name} lists may contain scalar values only")
        if isinstance(value, dict):
            try:
                json.dumps(value, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{section}.{name} must be JSON serializable") from exc


def serialize_cli_options(options):
    """Serialize a snake_case option map into vLLM command-line flags; true writes a bare flag."""
    for name in options:
        if not _FLAG_NAME_RE.fullmatch(name):
            raise ValueError(f"option.{name}: flag names must be snake_case")
    _check_vllm_flag_values(options, "option")
    argv = []
    for name, value in options.items():
        if value is None:
            continue
        flag = f"--{name.replace('_', '-')}"
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            argv.extend([flag, *(str(item) for item in value)])
        elif isinstance(value, dict):
            argv.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            argv.extend([flag, str(value)])
    return argv


@dataclass(frozen=True)
class RunCell:
    key: str
    isl: int
    osl: int
    tp: int
    pp: int
    concurrency: int

    @classmethod
    def parse(cls, value):
        return cls(value, **parse_inference_cell_key(value))


@dataclass(frozen=True)
class ResolvedRun:
    cell: RunCell
    benchmark_params: Dict[str, Any]


class VllmRuntimeArgs(InferenceRuntimeArgs):
    model_config = ConfigDict(extra="forbid")

    network: str = "host"
    ipc: str = "host"
    privileged: bool = True
    volumes: List[ConfigStr] = Field(default_factory=list)
    devices: List[str] = Field(default_factory=list)


class VllmRuntime(InferenceRuntimeSpec):
    model_config = ConfigDict(extra="forbid")

    name: Literal["docker"] = "docker"
    args: VllmRuntimeArgs = Field(default_factory=VllmRuntimeArgs)


class VllmContainerSpec(InferenceContainerSpec):
    runtime: VllmRuntime = Field(default_factory=VllmRuntime)

    @model_validator(mode="after")
    def _check_network_env(self):
        configured_sockets = _VLLM_SOCKET_ENV & set(self.env)
        if configured_sockets and configured_sockets != _VLLM_SOCKET_ENV:
            missing = sorted(_VLLM_SOCKET_ENV - configured_sockets)
            raise ValueError(f"container.env must set all socket interface variables together; missing: {missing}")
        empty_sockets = sorted(name for name in configured_sockets if not self.env[name].strip())
        if empty_sockets:
            raise ValueError(f"container.env socket interface variables cannot be empty: {empty_sockets}")
        if "NCCL_IB_HCA" in self.env and not self.nccl_ib_hcas:
            raise ValueError("container.env.NCCL_IB_HCA must name at least one device")
        return self

    @property
    def nccl_ib_hcas(self):
        value = self.env.get("NCCL_IB_HCA")
        if value is None:
            return None
        return [device.strip() for device in value.split(",") if device.strip()]

    @property
    def socket_env_configured(self):
        return bool(_VLLM_SOCKET_ENV & set(self.env))


class VllmBlock(_Forbid):
    distributed_executor_backend: Literal["mp", "ray"] = "mp"


# ---------- old vLLM keys ----------

_VLLM_OLD_KEYS = {
    "server_params.tensor_parallel_size": "server_params.tp",
    "server_params.pipeline_parallel_size": "server_params.pp",
    "server_params.dist_init_port": "server_params.master_port",
    "server_params.distributed_executor_backend": "vllm.distributed_executor_backend",
    "accuracy.tasks": "accuracy.lm_eval.tasks",
}
_VLLM_OLD_SERVER_FIELDS = frozenset(
    {
        "backend",
        "model",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "port",
        "dist_init_port",
        "max_model_len",
        "server_poll_iterations",
        "server_poll_wait_s",
        "server_warmup_wait_s",
        "distributed_executor_backend",
    }
)
_VLLM_OLD_BENCHMARK_FIELDS = frozenset(
    {
        "backend",
        "dataset_name",
        "num_prompts",
        "request_rate",
        "burstiness",
        "tokenizer_mode",
        "seed",
        "random_range_ratio",
        "random_prefix_len",
        "client_poll_iterations",
        "client_poll_wait_s",
        "client_initial_wait_s",
        "trust_remote_code",
        "ignore_eos",
    }
)
# Old timeout keys and their old defaults: the first is a fixed wait, then
# iterations times the wait per poll.
_VLLM_OLD_TIMEOUTS = {
    "timeouts.server_ready_s": (
        ("server_params", "server_warmup_wait_s", 330),
        ("server_params", "server_poll_iterations", 60),
        ("server_params", "server_poll_wait_s", 60),
    ),
    "timeouts.client_s": (
        ("benchmark_params", "client_initial_wait_s", 120),
        ("benchmark_params", "client_poll_iterations", 20),
        ("benchmark_params", "client_poll_wait_s", 60),
    ),
}
# Validation context value under "old_keys" for a load that converts an old
# config without warning, such as pytest collection; the load a run uses warns.
OLD_KEYS_QUIET = "quiet"
_VLLM_OLD_BENCHMARK_DEFAULT_PROMPTS = 3200
_VLLM_OLD_CELL_TIMEOUT_KEYS = ("client_initial_wait_s", "client_poll_iterations", "client_poll_wait_s")
_VLLM_NEW_SERVER_FIELDS = frozenset(InferenceServerParams.model_fields)
_VLLM_NEW_BENCHMARK_FIELDS = frozenset(InferenceBenchmarkParams.model_fields)
_INFERENCE_ACCURACY_THRESHOLD_SECTIONS = frozenset(InferenceAccuracyThresholds.model_fields)
_OLD_THRESHOLD_KINDS = {"max_ms": "max", "min_tok_s": "min"}


def _section(config, name):
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _is_old_vllm_config(config):
    return "tensor_parallel_size" in _section(config, "server_params")


def _old_int(config, section, key, default):
    value = _section(config, section).get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{section}.{key} must be a whole number of seconds or iterations") from None


def _vllm_old_timeouts_rule(config):
    if not _is_old_vllm_config(config):
        return {}
    rows = {}
    for new_path, keys in _VLLM_OLD_TIMEOUTS.items():
        wait, count, poll = (_old_int(config, section, key, default) for section, key, default in keys)
        for section, key, _default in keys:
            rows[(section, key)] = OldKey(new_path, value=wait + count * poll)
    return rows


def _drop_at(config, section, key, expected):
    """Drop an old key the harness now always sets, or refuse it when it held another value."""
    value = _section(config, section).get(key, expected)
    return OLD_KEY_DROPPED if value == expected else OLD_KEY_UNMAPPED


def _client_trust_row(value, server_trust, new_path):
    """The client trusts remote code when the server does, so only a client-only ``true`` needs a key."""
    if value == server_trust:
        return OLD_KEY_DROPPED
    if value is True:
        return OldKey(new_path)
    return OLD_KEY_UNMAPPED


def _vllm_old_client_rule(config):
    """Drop the client keys the harness now always sets, at the values main ran with."""
    server = _section(config, "server_params")
    server_trust = (
        server.get("trust_remote_code") is True or _section(server, "engine_args").get("trust_remote_code") is True
    )
    benchmark = _section(config, "benchmark_params")
    rows = {
        "server_params.backend": _drop_at(config, "server_params", "backend", "vllm"),
        "benchmark_params.backend": _drop_at(config, "benchmark_params", "backend", "vllm"),
        "benchmark_params.dataset_name": _drop_at(config, "benchmark_params", "dataset_name", "random"),
        "benchmark_params.ignore_eos": _drop_at(config, "benchmark_params", "ignore_eos", True),
    }
    if "trust_remote_code" in benchmark:
        rows["benchmark_params.trust_remote_code"] = _client_trust_row(
            benchmark["trust_remote_code"], server_trust, "benchmark_params.client_args.trust_remote_code"
        )
    if not _is_old_vllm_config(config):
        return rows
    sweeps = _section(config, "sweeps")
    for cell_key in sweeps:
        overrides = _section(sweeps, cell_key)
        for key in overrides:
            path = ("sweeps", cell_key, key)
            if key == "ignore_eos":
                rows[path] = _drop_at(sweeps, cell_key, key, True)
            elif key == "dataset_name":
                rows[path] = _drop_at(sweeps, cell_key, key, "random")
            elif key == "trust_remote_code" and overrides[key] == server_trust:
                rows[path] = OLD_KEY_DROPPED
            elif key == "trust_remote_code" and overrides[key] is not True:
                rows[path] = OLD_KEY_UNMAPPED
            elif key in _VLLM_OLD_CELL_TIMEOUT_KEYS:
                rows[path] = OLD_KEY_UNMAPPED
    return rows


def _vllm_old_flags_rule(config):
    """Move the native flags an old config wrote beside the harness fields."""
    if not _is_old_vllm_config(config):
        return {}
    rows = {}
    server = _section(config, "server_params")
    if "max_model_len" in server:
        value = server["max_model_len"]
        try:
            usable = value is None or (not isinstance(value, bool) and int(value) > 0)
        except (TypeError, ValueError):
            usable = False
        rows["server_params.max_model_len"] = "server_params.max_model_length" if usable else OLD_KEY_UNMAPPED
    for section, old_fields, new_fields, target in (
        ("server_params", _VLLM_OLD_SERVER_FIELDS, _VLLM_NEW_SERVER_FIELDS, "engine_args"),
        ("benchmark_params", _VLLM_OLD_BENCHMARK_FIELDS, _VLLM_NEW_BENCHMARK_FIELDS, "client_args"),
    ):
        for key in _section(config, section):
            if key not in old_fields and key not in new_fields and not key.startswith("_"):
                rows[(section, key)] = OldKey((section, target, key))
    if "num_prompts" not in _section(config, "benchmark_params"):
        rows["benchmark_params.num_prompts"] = OldKey(
            "benchmark_params.num_prompts", value=_VLLM_OLD_BENCHMARK_DEFAULT_PROMPTS
        )
    return rows


def _vllm_old_network_rule(config):
    """Write the old top-level NCCL settings into container.env, where an explicit value already wins."""
    container = _section(config, "container")
    env = dict(_section(container, "env"))
    new_env = dict(env)
    hcas = config.get("ib_hca_devices")
    netdev = config.get("ib_netdev")
    hcas_used = isinstance(hcas, list) and bool(hcas) and "NCCL_IB_HCA" not in env
    netdev_used = isinstance(netdev, str) and bool(netdev) and not (_VLLM_SOCKET_ENV & set(env))
    if hcas_used:
        new_env["NCCL_IB_HCA"] = ",".join(str(device) for device in hcas)
    if netdev_used:
        new_env.update({name: netdev for name in sorted(_VLLM_SOCKET_ENV)})
    rows = {}
    for key, used in (("ib_hca_devices", hcas_used), ("ib_netdev", netdev_used)):
        if key in config:
            rows[key] = OldKey("container.env", value=new_env, conflict="replace") if used else OLD_KEY_DROPPED
    return rows


def _old_kind_rows(cell, old_prefix, new_prefix):
    rows = {}
    for metric, spec in cell.items():
        kind = spec.get("kind") if isinstance(spec, dict) else None
        if isinstance(kind, str) and kind in _OLD_THRESHOLD_KINDS:
            new_path = (*new_prefix, metric, "kind")
            rows[(*old_prefix, metric, "kind")] = OldKey(new_path, value=_OLD_THRESHOLD_KINDS[kind], conflict="replace")
        elif kind in ("within", "min_ratio"):
            rows[(*old_prefix, metric, "kind")] = OLD_KEY_UNMAPPED
    return rows


def _vllm_old_thresholds_rule(config):
    """Move old accuracy thresholds under ``lm_eval`` and rewrite the old kinds."""
    thresholds = _section(config, "thresholds")
    accuracy = _section(thresholds, "accuracy")
    rows = {}
    if not set(accuracy) & _INFERENCE_ACCURACY_THRESHOLD_SECTIONS:
        for task_id, cell in accuracy.items():
            if not task_id.startswith("_"):
                rows[("thresholds", "accuracy", task_id)] = OldKey(("thresholds", "accuracy", "lm_eval", task_id))
                if isinstance(cell, dict):
                    old = ("thresholds", "accuracy", task_id)
                    rows.update(_old_kind_rows(cell, old, ("thresholds", "accuracy", "lm_eval", task_id)))
    for cell_key, cell in thresholds.items():
        if cell_key != "accuracy" and isinstance(cell, dict):
            rows.update(_old_kind_rows(cell, ("thresholds", cell_key), ("thresholds", cell_key)))
    return rows


def resolve_old_vllm_keys(config):
    """Map an old vLLM config, with its attached ``thresholds``, to the current names.

    Returns the converted config and one warning message, or the input and
    ``None`` when it holds no old key.
    """
    return resolve_old_keys(
        config,
        _VLLM_OLD_KEYS,
        rules=(
            _vllm_old_client_rule,
            _vllm_old_timeouts_rule,
            _vllm_old_flags_rule,
            _vllm_old_network_rule,
            _vllm_old_thresholds_rule,
        ),
        message=(
            "Deprecated vLLM config keys: {keys}. Use the new names; old keys will be removed in a future release."
        ),
        label="vLLM config",
    )


def _reject_underscore_metric_keys(thresholds):
    # A misspelled metric such as _output_throughput would otherwise be dropped
    # as a comment, and enforcement only checks that the cell exists.
    for cell_key, cell in thresholds.items():
        if cell_key == "accuracy" or cell_key.startswith("_") or not isinstance(cell, dict):
            continue
        for metric, spec in cell.items():
            if metric.startswith(_THRESHOLD_METADATA_PREFIXES):
                continue
            if metric.startswith("_"):
                raise ValueError(f"unknown vLLM threshold metric {metric!r}")
            extra = sorted(key for key in spec if key.startswith("_")) if isinstance(spec, dict) else []
            if extra:
                raise ValueError(f"{cell_key}.{metric}: unknown threshold fields {extra}")


class VllmConfig(InferenceConfig):
    not_implemented: ClassVar[Tuple[str, ...]] = (
        "server_params.dp",
        "server_params.ep",
        "server_params.prefill_nnodes",
        "server_params.long_context",
        "benchmark_params.scaling_baseline_output_throughput",
        "accuracy.custom_eval",
        "accuracy.quant_parity",
        "accuracy.mtp_quality",
    )

    container: VllmContainerSpec
    vllm: VllmBlock = Field(default_factory=VllmBlock)
    _effective_topology: Any = PrivateAttr(default=None)
    _run_cells: Dict[str, RunCell] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _load_threshold_file_and_old_keys(cls, data, info):
        from cvs.lib.utils.config_loader import attach_threshold_file

        data = attach_threshold_file(data, info.context)
        if isinstance(data, dict) and isinstance(data.get("thresholds"), dict):
            _reject_underscore_metric_keys(data["thresholds"])
        data, message = resolve_old_vllm_keys(data)
        if (info.context or {}).get("old_keys") != OLD_KEYS_QUIET:
            warn_old_keys(message)
        return data

    @model_validator(mode="after")
    def _check_vllm(self):
        from cvs.lib.inference.utils.vllm_metrics import validate_threshold_spec

        _check_flags(self.server_params.engine_args, "engine_args", _VLLM_HARNESS_SERVER_FLAGS)
        _check_flags(self.benchmark_params.client_args, "client_args", _VLLM_HARNESS_CLIENT_FLAGS)
        _check_vllm_flag_values(self.server_params.engine_args, "engine_args")
        _check_vllm_flag_values(self.benchmark_params.client_args, "client_args")
        for key, overrides in self.sweeps.items():
            _check_flags(overrides, f"sweeps.{key}", _VLLM_HARNESS_CLIENT_FLAGS)
            _check_vllm_flag_values(overrides, f"sweeps.{key}")

        cells = self.thresholds.cells()
        unknown = sorted(set(cells) - set(self.sweeps))
        if unknown:
            raise ValueError(f"threshold cells have no matching sweep: {unknown}")
        for cell_key, cell in cells.items():
            for metric, spec in cell.items():
                if "optional" in spec.model_fields_set:
                    raise ValueError(f"{cell_key}.{metric}: vLLM thresholds do not support optional")
                validate_threshold_spec(metric, spec.model_dump())
        for section in ("custom_eval", "quant_parity", "mtp_quality"):
            if getattr(self.thresholds.accuracy, section):
                raise ValueError(f"thresholds accuracy.{section} is not implemented for {type(self).__name__}")
        self._run_cells = {key: RunCell.parse(key) for key in self.sweeps}
        return self

    @property
    def model_id(self):
        return self.server_params.model

    def bind_effective_topology(self, topology):
        self._effective_topology = topology

    def cell(self, key):
        return self._run_cells[key]

    def expected_cells(self):
        return list(self.runs)

    def cell_key(self, isl, osl, concurrency, **_unused):
        return f"ISL={isl},OSL={osl},TP={self.server_params.tp},PP={self.server_params.pp},CONC={concurrency}"

    def client_values(self, overrides=None):
        """The shared client fields, then ``client_args``, then ``overrides``, as one flat map."""
        values = self.benchmark_params.model_dump(exclude={"client_args", "scaling_baseline_output_throughput"})
        values.update(self.benchmark_params.client_args)
        values.update(overrides or {})
        return values

    def resolved_runs(self):
        resolved = []
        for key in self.runs:
            cell = self.cell(key)
            values = self.client_values(self.sweeps[key])
            values.update(
                {
                    "random_input_len": cell.isl,
                    "random_output_len": cell.osl,
                    "max_concurrency": cell.concurrency,
                }
            )
            resolved.append(ResolvedRun(cell=cell, benchmark_params=values))
        return resolved


# =============================================================================
# SGLang Config
# =============================================================================
#
# SglangConfig is InferenceConfig with SGLang's not-implemented list; SGLang has
# no keys of its own. resolve_old_sglang_keys maps a config from before the
# shared names, with its threshold file under ``thresholds``, onto it.
# SglangSingleVariantConfig holds the run-time views the SGLang job classes read.

# lm-eval 0.4.12 options SGLang never passes to lm-eval
# (cvs/lib/utils/model_query_lib.py LmEvalBenchmark.prepare).
_SGLANG_UNREAD_LM_EVAL_OPTIONS = (
    "backend",
    "max_batch_size",
    "device",
    "gen_kwargs",
    "samples",
    "use_cache",
    "cache_requests",
    "check_integrity",
    "system_instruction",
    "apply_chat_template",
    "fewshot_as_multiturn",
    "include_path",
    "predict_only",
    "seed",
    "trust_remote_code",
    "confirm_run_unsafe_code",
    "metadata",
)


def sglang_cell_shape(key):
    """ISL, OSL and CONC of a cell key: SGLang matches a threshold cell to a run on these alone."""
    axes = parse_inference_cell_key(key)
    return axes["isl"], axes["osl"], axes["concurrency"]


class SglangConfig(InferenceConfig):
    """SGLang's config: the shared blocks, refusing the shared fields SGLang does not run."""

    not_implemented: ClassVar[Tuple[str, ...]] = (
        "server_params.dp",
        "server_params.ep",
        "server_params.prefill_nnodes",
        "benchmark_params.seed",
        "benchmark_params.request_rate",
        "benchmark_params.burstiness",
        "benchmark_params.random_prefix_len",
        "benchmark_params.tokenizer_mode",
        "benchmark_params.scaling_baseline_output_throughput",
        "benchmark_params.client_args",
        "timeouts.server_ready_s",
        "accuracy.quant_parity",
        "accuracy.mtp_quality",
    )

    @model_validator(mode="after")
    def _check_lm_eval_options(self):
        for index, task in enumerate(self.accuracy.lm_eval.tasks):
            for name in _SGLANG_UNREAD_LM_EVAL_OPTIONS:
                if getattr(task, name) != type(task).model_fields[name].get_default(call_default_factory=True):
                    raise ValueError(
                        f"accuracy.lm_eval.tasks[{index}].{name} is not implemented for {type(self).__name__}; "
                        "leave it at its default"
                    )
        return self

    @model_validator(mode="after")
    def _check_niah_server_flags(self):
        if self.accuracy.custom_eval and (
            self.server_params.max_model_length is None or self.server_params.long_context.chunked_prefill_size is None
        ):
            raise ValueError(
                "accuracy.custom_eval needs server_params.max_model_length and "
                "server_params.long_context.chunked_prefill_size for the long-context server"
            )
        return self


_SGLANG_SERVE_PORT = 8000
_SGLANG_MASTER_PORT = 40001
# SGLang polls the client log every 90 s (30 s plus a 60 s wait) after a first
# 60 s wait, for a number of polls capped by a total in seconds
# (cvs/lib/inference/sglang/sglang_common.py poll_for_inference_completion).
SGLANG_FIRST_POLL_WAIT_S = 60
SGLANG_POLL_PERIOD_S = 90
_SGLANG_POLL_ITERATIONS = 16
SGLANG_POLL_TOTAL_S = 3600
# LongContextNiahBenchmark defaults; a long_ctx_niah option at its default has
# nothing to keep, any other value has no shared key.
_SGLANG_NIAH_OPTION_DEFAULTS = {
    "exec_timeout_sec": 21600,
    "request_timeout_sec": 7200,
    "tolerance_frac": 0.05,
    "enable_thinking": None,
    "local_files_only": False,
}
# LmEvalBenchmark options with no shared key, at their defaults.
_SGLANG_LM_EVAL_OPTION_DEFAULTS = {"tolerance_frac": 0.05, "pip_install": True}
_SGLANG_NIAH_NUM_PROMPTS = 16
_SGLANG_NIAH_SEED = 42

SGLANG_METRIC_RENAMES = {
    "request_throughput_per_sec": "request_throughput",
    "output_throughput_per_sec": "output_throughput",
    "output_throughput_per_gpu_per_sec": "per_gpu_throughput",
    "successful_requests": "completed",
    "total_requests": "num_prompts",
    "benchmark_duration": "duration",
    "total_generated_tokens": "total_output_tokens",
    **{f"{stat}_e2e_latency_ms": f"{stat}_e2el_ms" for stat in ("mean", "median", "p90", "p95", "p99")},
}
_SGLANG_DROPPED_METRICS = ("mfu",)
_SGLANG_ACC_CELL_RE = re.compile(r"^ACC_ISL=([1-9]\d*),OSL=([1-9]\d*)$")

SGLANG_OLD_KEYS = {
    "gpu_name": OLD_KEY_DROPPED,
    "schema_version": OLD_KEY_DROPPED,
    "topology": OLD_KEY_DROPPED,
    "variant_key": OLD_KEY_DROPPED,
    "gpu_arch": OLD_KEY_DROPPED,
    "server_params.backend": OLD_KEY_DROPPED,
    "server_params.max_concurrency": OLD_KEY_DROPPED,
    "server_params.tokenizer_mode": OLD_KEY_DROPPED,
    "server_params.prefill_policy": OLD_KEY_DROPPED,
    "server_params.decode_policy": OLD_KEY_DROPPED,
    "server_params.lng_ctx_activate": OLD_KEY_DROPPED,
    "server_params.tensor_parallelism": "server_params.tp",
    "server_params.pipeline_parallelism": "server_params.pp",
    "benchmark_params.data_set_name": OLD_KEY_DROPPED,
    "benchmark_params.model_num_params": OLD_KEY_DROPPED,
    "benchmark_params.peak_gpu_tflops": OLD_KEY_DROPPED,
    **{f"container.runtime.args.env.{name}": OLD_KEY_DROPPED for name in sorted(_HF_TOKEN_ENV)},
}


def _is_old_sglang_layout(config):
    server = config.get("server_params")
    return isinstance(server, dict) and "tp" not in server


def _sglang_whole_number(path, value):
    try:
        number = float(str(value).strip())
    except ValueError:
        number = math.nan
    if not number.is_integer():
        raise ValueError(f"{path} must be a whole number")
    return int(number)


def _old_tp_pp(config):
    server = config.get("server_params") if isinstance(config.get("server_params"), dict) else {}
    tp = server.get("tp", server.get("tensor_parallelism"))
    pp = server.get("pp", server.get("pipeline_parallelism", 1))
    if tp is None:
        return None, None
    return _sglang_whole_number("server_params.tensor_parallelism", tp), _sglang_whole_number(
        "server_params.pipeline_parallelism", pp
    )


def _old_run_combos(config):
    sweep = config.get("sweep") if isinstance(config.get("sweep"), dict) else {}
    return [str(run.get("combo") if isinstance(run, dict) else run).strip() for run in sweep.get("runs") or []]


def _rekey_rows(prefix, keys, own, combos):
    """Rows keying one shape's ``keys`` by ``own``.

    The key already ``own`` wins, then the one key a run names, then a key
    alone in its shape; the rest are dropped.
    """
    if own in keys:
        keep = own
    else:
        named = [key for key in keys if key in combos]
        keep = named[0] if len(named) == 1 else keys[0] if len(keys) == 1 else None
    rows = {}
    for key in keys:
        if key != own:
            rows[(prefix, key)] = OldKey((prefix, own)) if key == keep else OLD_KEY_DROPPED
    return keep, rows


def _cell_key_for(key, tp, pp):
    isl, osl, conc = sglang_cell_shape(key)
    return f"ISL={isl},OSL={osl},TP={tp},PP={pp},CONC={conc}"


def _not_token(name):
    return str(name).upper() not in _HF_TOKEN_ENV


def _old_switch_on(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return isinstance(value, int) and value == 1


def _old_sglang_value(config, path):
    return _get_config_path(config, _config_path_parts(path))


def _drop_unless(config, path, keep):
    value = _old_sglang_value(config, path)
    if value is _MISSING:
        return {}
    return {path: OLD_KEY_DROPPED if keep(value) else OLD_KEY_UNMAPPED}


def _sglang_harness_rows(config):
    """Keys whose value the harness now fixes: dropped at that value, refused at any other."""
    rows = {}
    rows.update(_drop_unless(config, "framework", lambda value: value in ("sglang", "sglang_single")))
    rows.update(_drop_unless(config, "benchmark_params.backend", lambda value: value == "sglang"))
    defaults = {
        "prefill_serv_port": "30001",
        "decode_serv_port": "30002",
        "prefill_coordinator_port": "40001",
        "decode_coordinator_port": "40002",
    }
    for name, default in defaults.items():
        rows.update(_drop_unless(config, f"server_params.{name}", lambda value, default=default: str(value) == default))
    return rows


def _sglang_port_rows(config):
    server = config.get("server_params")
    if not isinstance(server, dict):
        return {}
    old_layout = _is_old_sglang_layout(config)
    rows = {}
    sources = [name for name in ("proxy_router_serv_port", "serve_port") if name in server]
    port = _sglang_whole_number(f"server_params.{sources[0]}", server[sources[0]]) if sources else server.get("port")
    if sources:
        rows[f"server_params.{sources[0]}"] = OldKey("server_params.port", value=port)
        rows.update({f"server_params.{name}": OLD_KEY_DROPPED for name in sources[1:]})
    elif old_layout and port is None:
        port = _SGLANG_SERVE_PORT
        rows["server_params.serve_port"] = OldKey("server_params.port", value=port)
    served = port if port is not None else InferenceServerParams.model_fields["port"].default
    rows.update(_drop_unless(config, "server_params.proxy_router_port", lambda value: str(value) == str(served)))
    if "dist_init_port" in server:
        rows["server_params.dist_init_port"] = OldKey(
            "server_params.master_port",
            value=_sglang_whole_number("server_params.dist_init_port", server["dist_init_port"]),
        )
    elif old_layout and "master_port" not in server:
        rows["server_params.dist_init_port"] = OldKey("server_params.master_port", value=_SGLANG_MASTER_PORT)
    return rows


def _old_flag_value(value):
    """A number written as a string in an old config, as the number it reads as exactly."""
    if isinstance(value, str):
        for kind in (int, float):
            try:
                number = kind(value)
            except ValueError:
                continue
            if str(number) == value:
                return number
    return value


def _old_flag_args(flags):
    """``["--attention-backend aiter", "--disable-x"]`` as ``{"attention_backend": "aiter", "disable_x": True}``."""
    args = {}
    for flag in [flags] if isinstance(flags, str) else flags:
        name, _, value = str(flag).strip().replace("=", " ", 1).partition(" ")
        if name:
            args[name.lstrip("-").replace("-", "_")] = _old_flag_value(value.strip()) if value.strip() else True
    return args


def _sglang_engine_arg_rows(config):
    server = config.get("server_params")
    if not isinstance(server, dict):
        return {}
    rows = {}
    if "add_flags" in server:
        rows["server_params.add_flags"] = OldKey("server_params.engine_args", value=_old_flag_args(server["add_flags"]))
    if "log_level" in server:
        rows["server_params.log_level"] = "server_params.engine_args.log_level"
    if "memory_fraction" in server:
        rows["server_params.memory_fraction"] = OldKey(
            "server_params.engine_args.mem_fraction_static", value=_old_flag_value(server["memory_fraction"])
        )
    return rows


def _sglang_client_timeout_rows(config):
    """The client poll keys as ``timeouts.client_s``: main's effective client wait.

    SGLang waits 60 s, then polls every 90 s until the poll count or the total
    runs out, so the wait is ``60 + min(polls × 90, total)``.
    """
    server = config.get("server_params")
    names = ("inference_poll_iterations", "inference_poll_total_timeout_sec")
    present = [name for name in names if isinstance(server, dict) and name in server]
    if not present:
        return {}
    polls = _sglang_whole_number(f"server_params.{names[0]}", server.get(names[0], _SGLANG_POLL_ITERATIONS))
    cap = _sglang_whole_number(f"server_params.{names[1]}", server.get(names[1], SGLANG_POLL_TOTAL_S))
    if polls < 1 or cap < 1:
        return {f"server_params.{name}": OLD_KEY_UNMAPPED for name in present}
    client_s = SGLANG_FIRST_POLL_WAIT_S + min(polls * SGLANG_POLL_PERIOD_S, cap)
    rows = {f"server_params.{present[0]}": OldKey("timeouts.client_s", value=client_s)}
    rows.update({f"server_params.{name}": OLD_KEY_DROPPED for name in present[1:]})
    return rows


def _env_entries(lines):
    entries = {}
    for line in [lines] if isinstance(lines, str) else lines or []:
        line = str(line).strip().removeprefix("export ").strip()
        name, sep, value = line.partition("=")
        if sep and name.strip() and _not_token(name.strip()):
            entries[name.strip()] = value.strip()
    return entries


def _sglang_env_rows(config):
    runtime_env = _old_sglang_value(config, "container.runtime.args.env")
    server = config.get("server_params") if isinstance(config.get("server_params"), dict) else {}
    sources = [name for name in ("env", "add_export_env") if name in server]
    if not isinstance(runtime_env, dict) and not sources:
        return {}
    runtime_env = runtime_env if isinstance(runtime_env, dict) else {}
    existing = _old_sglang_value(config, "container.env")
    env = dict(existing) if isinstance(existing, dict) else {}
    env.update({key: value for key, value in runtime_env.items() if _not_token(key) and key != "ADD_EXPORT_ENV"})
    env.update(_env_entries(runtime_env.get("ADD_EXPORT_ENV")))
    env.update(_env_entries(server.get("add_export_env")))
    env.update({key: value for key, value in (server.get("env") or {}).items() if _not_token(key)})
    env = {key: str(value) for key, value in env.items()}
    old_paths = ["container.runtime.args.env"] + [f"server_params.{name}" for name in sources]
    rows = {old_paths[0]: OldKey("container.env", value=env, conflict="replace")}
    rows.update({path: OLD_KEY_DROPPED for path in old_paths[1:]})
    return rows


def _old_threshold_spec(metric, spec):
    if not isinstance(spec, dict):
        kind = "max" if "ms" in metric.lower() or "latency" in metric.lower() else "min"
        return {"kind": kind, "value": float(spec)}
    if spec.get("kind") not in _OLD_THRESHOLD_KINDS:
        return spec
    return {**spec, "kind": _OLD_THRESHOLD_KINDS[spec["kind"]]}


def _old_threshold_cell_rows(cell_path, new_cell_path, cell):
    rows = {}
    for metric, spec in cell.items():
        if str(metric).startswith("_"):
            continue
        old = (*cell_path, metric)
        if metric in _SGLANG_DROPPED_METRICS:
            rows[old] = OLD_KEY_DROPPED
            continue
        new_spec = _old_threshold_spec(metric, spec)
        new = (*new_cell_path, SGLANG_METRIC_RENAMES.get(metric, metric))
        if new != old or new_spec != spec:
            rows[old] = OldKey(new, value=new_spec)
    return rows


def _old_threshold_cells(config):
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        return {}
    return {
        key: cell
        for key, cell in thresholds.items()
        if isinstance(cell, dict) and (_CELL_KEY_RE.fullmatch(str(key)) or str(key).startswith("BENCH="))
    }


def _sglang_bench_threshold_rows(config):
    return {
        ("thresholds", key): OldKey(
            ("thresholds", "accuracy", "lm_eval", key.removeprefix("BENCH=")),
            value={metric: _old_threshold_spec(metric, spec) for metric, spec in cell.items()},
        )
        for key, cell in _old_threshold_cells(config).items()
        if key.startswith("BENCH=")
    }


def _sglang_threshold_spec_rows(config, rekey=True, combos=()):
    """Rows for the shared metric names and kinds, keying each cell by the config's TP and PP.

    SGLang matched a run to a threshold cell by its exact key, else by ISL, OSL
    and CONC alone, so one old file served configs of any TP and PP.
    """
    tp, pp = _old_tp_pp(config) if rekey else (None, None)
    cells = _old_threshold_cells(config)
    by_shape = {}
    for key in cells:
        if not key.startswith("BENCH="):
            by_shape.setdefault(sglang_cell_shape(key), []).append(key)
    rows = {}
    for keys in by_shape.values():
        own = keep = None
        if tp:
            own = _cell_key_for(keys[0], tp, pp)
            keep, rekeyed = _rekey_rows("thresholds", keys, own, combos)
            rows.update({path: row for path, row in rekeyed.items() if row is OLD_KEY_DROPPED})
        for key in keys:
            if tp and key != keep:
                continue
            new_key = own if tp else key
            rows.update(_old_threshold_cell_rows(("thresholds", key), ("thresholds", new_key), cells[key]))
            if new_key != key:
                rows[("thresholds", key)] = OLD_KEY_DROPPED
    return rows


def _threshold_spec_summary(config, rows):
    """One line for every threshold spec change, instead of a path per cell."""
    names, kinds, dropped, cells = {}, {}, set(), {}
    for path, row in rows.items():
        if len(path) == 2:
            cells.setdefault(path[1], "dropped")
            continue
        _, cell, metric = path
        spec = config["thresholds"][cell][metric]
        if row is OLD_KEY_DROPPED:
            dropped.add(metric)
            continue
        if row.new_path[1] != cell:
            cells[cell] = row.new_path[1]
        if row.new_path[-1] != metric:
            names[metric] = row.new_path[-1]
        old_kind = spec.get("kind") if isinstance(spec, dict) else "a bare number"
        if old_kind != row.value.get("kind"):
            kinds[old_kind] = row.value.get("kind")
    parts = [f"metric {old} -> {new}" for old, new in names.items()]
    parts += [f"kind {old} -> {new}" for old, new in kinds.items()]
    parts += [f"metric {name} (dropped)" for name in sorted(dropped)]
    parts += [f"cell {old} (dropped)" for old, new in cells.items() if new == "dropped"]
    rekeyed = sorted({new.split(",", 2)[2].rsplit(",", 1)[0] for new in cells.values() if new != "dropped"})
    parts += [f"cells keyed by {axes}" for axes in rekeyed]
    return f"Threshold specs: {', '.join(parts)}"


def _old_value_is(value, default):
    if value == default or default is OLD_KEY_UNMAPPED:
        return value == default
    if isinstance(default, bool) or default is None:
        return str(value).strip().lower() == str(default).lower()
    try:
        return float(value) == float(default)
    except (TypeError, ValueError):
        return False


def _sglang_long_context_rows(config):
    server = config.get("server_params") if isinstance(config.get("server_params"), dict) else {}
    thresholds = config.get("thresholds") if isinstance(config.get("thresholds"), dict) else {}
    acc_cells = [key for key in thresholds if _SGLANG_ACC_CELL_RE.match(str(key))]
    moves = {
        "context_length": "server_params.max_model_length",
        "chunked_prefill_size": "server_params.long_context.chunked_prefill_size",
        "max_prefill_tokens": "server_params.long_context.max_prefill_tokens",
    }
    if not _old_switch_on(server.get("lng_ctx_activate")):
        rows = {f"server_params.{name}": OLD_KEY_DROPPED for name in moves}
        rows["long_ctx_niah"] = OLD_KEY_DROPPED
        rows.update({("thresholds", key): OLD_KEY_DROPPED for key in acc_cells})
        return rows
    rows = {
        f"server_params.{name}": OldKey(new, value=_sglang_whole_number(f"server_params.{name}", server[name]))
        for name, new in moves.items()
        if name in server
    }
    niah = config.get("long_ctx_niah") if isinstance(config.get("long_ctx_niah"), dict) else {}
    for name, value in niah.items():
        if name in ("num_prompts", "seed") or str(name).startswith("_"):
            continue
        default = _SGLANG_NIAH_OPTION_DEFAULTS.get(name, OLD_KEY_UNMAPPED)
        rows[f"long_ctx_niah.{name}"] = OLD_KEY_DROPPED if _old_value_is(value, default) else OLD_KEY_UNMAPPED
    evals = []
    for key in sorted(acc_cells, key=lambda key: int(_SGLANG_ACC_CELL_RE.match(key).group(1))):
        isl, osl = (int(number) for number in _SGLANG_ACC_CELL_RE.match(key).groups())
        eval_id = f"niah_isl{isl}_osl{osl}"
        evals.append(
            {
                "type": "niah",
                "id": eval_id,
                "isl": isl,
                "osl": osl,
                "num_prompts": _sglang_whole_number(
                    "long_ctx_niah.num_prompts", niah.get("num_prompts", _SGLANG_NIAH_NUM_PROMPTS)
                ),
                "seed": _sglang_whole_number("long_ctx_niah.seed", niah.get("seed", _SGLANG_NIAH_SEED)),
            }
        )
        rows[("thresholds", key)] = OldKey(
            ("thresholds", "accuracy", "custom_eval", eval_id),
            value={metric: _old_threshold_spec(metric, spec) for metric, spec in thresholds[key].items()},
        )
    rows["long_ctx_niah"] = OldKey("accuracy.custom_eval", value=evals)
    return rows


def _sglang_accuracy_rows(config):
    tasks = _old_sglang_value(config, "accuracy.tasks")
    if tasks is _MISSING:
        return {}
    kept = []
    for index, task in enumerate(tasks or []):
        defaults = {**_SGLANG_LM_EVAL_OPTION_DEFAULTS, "label": str(task.get("id", "")).removeprefix("lm_eval_")}
        unread = [name for name, default in defaults.items() if name in task and not _old_value_is(task[name], default)]
        if unread:
            paths = ", ".join(f"accuracy.tasks[{index}].{name}" for name in unread)
            raise ValueError(
                f"Unsupported SGLang config options: {paths}; they have no place in the new format, so move or delete them"
            )
        kept.append({key: value for key, value in task.items() if key != "backend" and key not in defaults})
    return {"accuracy.tasks": OldKey("accuracy.lm_eval.tasks", value=kept)}


def _sglang_run_rows(config):
    """``sweep.runs`` to ``runs``, with ``sweeps`` and ``runs`` keyed by the config's TP and PP.

    SGLang ran each combo, and applied its override, by its exact key, else by
    ISL, OSL and CONC; an empty or missing ``sweep.runs`` ran every threshold cell.
    """
    sweep = config.get("sweep")
    if not isinstance(sweep, dict) and not _is_old_sglang_layout(config):
        return {}
    tp, pp = _old_tp_pp(config)
    if tp is None:
        return {}
    combos = _old_run_combos(config)
    if "runs" in config:
        rows = {"sweep": OLD_KEY_DROPPED} if isinstance(sweep, dict) else {}
        runs = []
    else:
        if not combos:
            thresholds = config.get("thresholds") if isinstance(config.get("thresholds"), dict) else {}
            combos = sorted((key for key in thresholds if _CELL_KEY_RE.fullmatch(str(key))), key=sglang_cell_shape)
        runs = list(dict.fromkeys(_cell_key_for(run, tp, pp) if _CELL_KEY_RE.fullmatch(run) else run for run in combos))
        rows = {"sweep": OldKey("runs", value=runs)}
    sweeps = config.get("sweeps") if isinstance(config.get("sweeps"), dict) else {}
    by_shape = {}
    for key in sweeps:
        if _CELL_KEY_RE.fullmatch(str(key)):
            by_shape.setdefault(sglang_cell_shape(key), []).append(key)
    kept = {key for key in sweeps if not _CELL_KEY_RE.fullmatch(str(key))}
    for keys in by_shape.values():
        own = _cell_key_for(keys[0], tp, pp)
        keep, rekeyed = _rekey_rows("sweeps", keys, own, combos)
        rows.update(rekeyed)
        if keep:
            kept.add(own)
    for run in runs:
        if run not in kept:
            rows[("sweeps", run)] = OldKey(("sweeps", run), value={})
    return rows


SGLANG_OLD_KEY_RULES = (
    _sglang_harness_rows,
    _sglang_port_rows,
    _sglang_engine_arg_rows,
    _sglang_client_timeout_rows,
    _sglang_env_rows,
    _sglang_long_context_rows,
    _sglang_accuracy_rows,
    _sglang_run_rows,
    _sglang_bench_threshold_rows,
)


def _sglang_threshold_cell_rows(config, spec_rows):
    """One row per threshold cell for ``spec_rows``, so the warning names cells, not every spec."""
    thresholds = config.get("thresholds") if isinstance(config.get("thresholds"), dict) else {}
    cells = dict.fromkeys(path[1] for path in spec_rows)
    rows = {}
    for cell in cells:
        new_key, new_cell, moved = cell, {}, False
        for metric, spec in thresholds[cell].items():
            row = spec_rows.get(("thresholds", cell, metric))
            if row is OLD_KEY_DROPPED:
                moved = True
            elif row is None:
                new_cell[metric] = spec
            else:
                new_key, moved = row.new_path[1], True
                new_cell[row.new_path[-1]] = row.value
        if moved:
            rows[("thresholds", cell)] = OldKey(("thresholds", new_key), value=new_cell)
        else:
            rows[("thresholds", cell)] = spec_rows[("thresholds", cell)]
    return rows


def _resolve_sglang_threshold_specs(config, table, rules, rekey, combos, message, label):
    spec_rows = _sglang_threshold_spec_rows(config, rekey=rekey, combos=combos)
    summary = f" {_threshold_spec_summary(config, spec_rows)}." if spec_rows else ""
    cell_rows = _sglang_threshold_cell_rows(config, spec_rows)
    return resolve_old_keys(
        config,
        table,
        (*rules, lambda _: cell_rows),
        message=message.replace("{summary}", summary),
        label=label,
    )


def resolve_old_sglang_keys(config):
    """Map an SGLang config from before the shared names onto ``SglangConfig``.

    ``config`` holds its threshold file under ``thresholds``. Returns the
    converted config and one ``FutureWarning`` message naming each old key, or
    the input and ``None`` when it holds no old key. The threshold specs'
    renames, the same in every cell, are summarized once.
    """
    rekey = _is_old_sglang_layout(config) or bool(_sglang_threshold_spec_rows(config, rekey=False))
    return _resolve_sglang_threshold_specs(
        config,
        SGLANG_OLD_KEYS,
        SGLANG_OLD_KEY_RULES,
        rekey,
        _old_run_combos(config),
        message="Deprecated SGLang config keys: {keys}.{summary} "
        "Use the shared inference names; old keys will be removed in a future release.",
        label="SGLang config",
    )


def resolve_old_sglang_threshold_names(thresholds):
    """Old metric names and kinds in a threshold file read on its own, keeping its cell keys."""
    converted, message = _resolve_sglang_threshold_specs(
        {"thresholds": thresholds},
        {},
        (),
        False,
        (),
        message="Deprecated SGLang threshold names: {keys}.{summary} Use the shared inference names.",
        label="SGLang threshold file",
    )
    return converted["thresholds"], message


class SglangRoleServer(_Allow):
    env: Dict[str, str] = Field(default_factory=dict)
    serve_port: str = ""


class SglangRoles(_Forbid):
    server: SglangRoleServer = Field(default_factory=SglangRoleServer)


class SglangParams(_Allow):
    """SGLang runtime parameters passed to the existing job controllers."""

    inference_tests: Dict[str, Any] = Field(default_factory=dict)
    add_export_env: List[str] = Field(default_factory=list)
    add_flags: List[str] = Field(default_factory=list)


class SglangAccuracyTask(_Allow):
    """One named lm-eval task from the unified ``accuracy.tasks`` block."""

    id: str


class SglangAccuracy(_Forbid):
    tasks: List[SglangAccuracyTask] = Field(default_factory=list)


class SglangSingleVariantConfig(BaseVariantConfig):
    """Run-time views shared by all SGLang topologies; ``config`` is the validated ``SglangConfig``."""

    framework: Literal["sglang", "sglang_single"] = "sglang"
    gpu_arch: str = ""
    topology: Literal["single", "distributed", "disaggregated"] = "single"
    variant_key: str = ""
    config_path: str = ""
    params: SglangParams = Field(default_factory=SglangParams)
    accuracy: SglangAccuracy = Field(default_factory=SglangAccuracy)
    config: Optional[SglangConfig] = None

    # Views for the SGLang job classes, built from ``config``.
    inference: Dict[str, Any] = Field(default_factory=dict)
    benchmark_params: Dict[str, Any] = Field(default_factory=dict)
    sweeps: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    sweep: Dict[str, Any] = Field(default_factory=dict)

    roles: SglangRoles = Field(default_factory=SglangRoles)

    def cell_key(self, isl, osl, concurrency):
        tp = self.benchmark_params.get("tensor_parallelism", "-")
        pp = self.benchmark_params.get("pipeline_parallelism", "-")
        return f"ISL={isl},OSL={osl},TP={tp},PP={pp},CONC={concurrency}"

    @property
    def hf_token_file(self):
        return self.paths.hf_token_file

    @model_validator(mode="after")
    def _sync_legacy_inference_container_name(self):
        """Keep legacy inference dict aligned with orchestrator container name."""
        if self.inference and self.container.name:
            self.inference["container_name"] = self.container.name
            self.inference["container_image"] = self.container.image
        return self

    @model_validator(mode="after")
    def _validate_topology_fields(self):
        required = {
            "single": (),
            "distributed": ("nnodes",),
            "disaggregated": ("nnodes",),
        }[self.topology]
        missing = [key for key in required if not self.inference.get(key)]
        if missing:
            raise ValueError(f"{self.topology} SGLang config missing required role fields: {missing}")

        if self.topology == "distributed":
            nnodes = int(self.inference.get("nnodes") or 0)
            if nnodes < 2:
                raise ValueError(f"distributed SGLang requires nnodes >= 2, got {nnodes}")
        if self.topology == "disaggregated":
            nnodes = int(self.inference.get("nnodes") or 0)
            if nnodes < 2:
                raise ValueError(f"disaggregated SGLang requires nnodes >= 2, got {nnodes}")
            if nnodes % 2:
                raise ValueError(
                    f"disaggregated SGLang requires an even nnodes so prefill and decode counts match, got {nnodes}"
                )
        return self


# =============================================================================
# Preflight Check Configuration Schema
# =============================================================================


LEGACY_PREFLIGHT_RDMA_PATHS = {
    "node_check.gid_index": "connectivity_check.rdma.gid_index",
    "node_check.rdma_interfaces": "connectivity_check.rdma.interfaces",
}

PREFLIGHT_METADATA_PREFIXES = ("_comment", "_example")
# IB verbs MTU enum values (ibv_mtu), plus 0 to disable the active MTU threshold.
VALID_RDMA_ACTIVE_MTUS = (0, 256, 512, 1024, 2048, 4096)


def strip_preflight_metadata(value):
    """Remove documentation-only pseudo-fields before schema validation.

    Preflight JSON files conventionally carry ``_comment*`` and ``_example*``
    keys so that the files remain self-documenting. They are not runtime
    options. Strip only those reserved prefixes recursively, preserving strict
    rejection of every other unknown customer-facing option.
    """
    if isinstance(value, dict):
        return {
            key: strip_preflight_metadata(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key.startswith(PREFLIGHT_METADATA_PREFIXES))
        }
    if isinstance(value, list):
        return [strip_preflight_metadata(item) for item in value]
    return value


def normalize_legacy_preflight_rdma_config(value):
    """Move the two deprecated node-check RDMA keys to their canonical block.

    Returns a deep-copied configuration and one consolidated warning message,
    or the original value and ``None`` when no legacy keys are present.
    Conflicting legacy and canonical values fail rather than silently choosing
    which RDMA inventory should be tested.
    """
    if any(_get_config_path(value, path.split(".")) is not _MISSING for path in LEGACY_PREFLIGHT_RDMA_PATHS):
        connectivity_check = value.get("connectivity_check", {})
        if not isinstance(connectivity_check, dict):
            raise ValueError(
                "preflight.connectivity_check must be an object when deprecated node_check RDMA options are used"
            )
        if not isinstance(connectivity_check.get("rdma", {}), dict):
            raise ValueError(
                "preflight.connectivity_check.rdma must be an object when deprecated node_check RDMA options are used"
            )
    return resolve_old_keys(
        value,
        LEGACY_PREFLIGHT_RDMA_PATHS,
        message=(
            "Deprecated preflight RDMA configuration detected: {keys}. "
            "Use the preflight.connectivity_check.rdma paths; legacy paths will be removed in a future release."
        ),
        label="preflight RDMA",
        prefix="preflight.",
    )


def _preflight_section_has_values(section: dict) -> bool:
    if not isinstance(section, dict):
        return False
    return any(value not in (None, "") for value in section.values())


LEGACY_PREFLIGHT_NODE_SMOKE_SECTIONS = {
    "node_smoke": "node_smoke_tier1",
    "tier3_info": "node_smoke_tier3",
}


def _legacy_node_smoke_rule(config):
    rows = {}
    for legacy_key, canonical_key in LEGACY_PREFLIGHT_NODE_SMOKE_SECTIONS.items():
        if isinstance(config.get(legacy_key), dict) and not _preflight_section_has_values(config.get(canonical_key)):
            rows[legacy_key] = OldKey(canonical_key, keep_old=True, conflict="replace")
    return rows


def normalize_legacy_preflight_node_smoke_sections(value):
    """Copy legacy Node Smoke section names to their canonical tier keys.

    Returns a deep-copied configuration and one consolidated warning message,
    or the original value and ``None`` when no legacy keys need migration.
    Canonical sections win when both legacy and canonical blocks are populated.
    """
    return resolve_old_keys(
        value,
        {},
        rules=(_legacy_node_smoke_rule,),
        message=(
            "Deprecated preflight Node Smoke section name(s) detected: {keys}. "
            "Prefer node_smoke_tier1 and node_smoke_tier3 in new configs."
        ),
        prefix="preflight.",
    )


class PreflightParallelismConfig(BaseModel):
    """Legacy parallelism settings for preflight checks."""

    model_config = ConfigDict(extra="allow")

    parallel_group_size: int = Field(
        default=128,
        ge=2,
        le=512,
        description=("Legacy alias for RDMA grouping. Prefer connectivity_check.rdma.nodes_per_full_mesh_group."),
    )


class PreflightDebugConfig(BaseModel):
    """Debug and troubleshooting settings for preflight checks."""

    model_config = ConfigDict(extra="allow")

    scriptlet: bool = Field(
        default=False,
        description=(
            "Enable ScriptLet debug: preserve generated scripts/logs on remote nodes. "
            "For RDMA connectivity, also wraps each ibv_rc_pingpong server in strace with "
            "per-test traces under /tmp/preflight/strace_server_<iface>_<port>.log (expensive at scale)."
        ),
    )


class PreflightNodeCheckConfig(BaseModel):
    """Individual node validation settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable generic GPU node health and ROCm validation")
    gpus_per_node: int = Field(default=4, ge=1, description="Expected AMD GPU count on each node")
    expected_rocm_version: str = Field(default="6.2.0", description="Expected ROCm version across all cluster nodes")
    min_up_ports_per_gpu: Optional[int] = Field(
        default=None,
        ge=0,
        description=("Optional MI4XX fabric admission override; prefer connectivity_check.ifoe.min_up_ports_per_gpu"),
    )


class PreflightRdmaConfig(BaseModel):
    """RDMA connectivity testing settings."""

    model_config = ConfigDict(extra="allow")

    connectivity_mode: str = Field(default="basic", description="RDMA connectivity testing: basic, full_mesh, or skip")
    gid_index: str = Field(default="3", description="GID index to check on all RDMA interfaces (typically 3 for RoCE)")
    gid_type: str = Field(
        default="RoCE v2",
        description="Expected sysfs GID type (gid_attrs/types/<idx>) on every RDMA interface; 'any' disables the type check. Not enforced on InfiniBand link layers.",
    )
    interfaces: List[str] = Field(
        default_factory=lambda: ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
        min_length=1,
        description="RDMA device names checked for presence, GID consistency, and connectivity",
    )
    mtu_check: bool = Field(
        default=True, description="Verify RDMA active MTU, and the netdev MTU on RoCE ports, on configured interfaces"
    )
    min_netdev_mtu: int = Field(
        default=4200,
        ge=0,
        le=65535,
        description="Minimum netdev MTU on RoCE ports; InfiniBand ports are exempt (0 disables)",
    )
    min_active_mtu: int = Field(
        default=4096, description="Minimum active_mtu from ibv_devinfo, RoCE or InfiniBand (0 disables)"
    )
    nodes_per_full_mesh_group: int = Field(
        default=128,
        ge=2,
        le=512,
        description=(
            "Number of nodes in each full-mesh partition group (2-512). "
            "Smaller groups use fewer resources per node but require more rounds."
        ),
    )
    parallel_group_size: int = Field(
        default=128,
        ge=2,
        le=512,
        description="Legacy alias for nodes_per_full_mesh_group.",
    )
    ibv_test_timeout: int = Field(
        default=90,
        ge=1,
        description="Timeout in seconds for RDMA connectivity tests using ibv_rc_pingpong",
    )
    ibv_test_port_range: str = Field(
        default="10000-50000", description="Port range for RDMA connectivity tests (format: start-end)"
    )
    inter_full_mesh_group_pairs_per_wave: str = Field(
        default="auto", description="Max ordered group-pairs per wave during inter-group testing ('auto' or integer)"
    )
    inter_group_wave_pairs: str = Field(
        default="auto",
        description="Legacy alias for inter_full_mesh_group_pairs_per_wave.",
    )
    prune_failure_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Round 1 (intra) prune before inter-group: prune nodes whose fraction of peers with ≥1 FAIL "
            "intra test is ≥ this value (default 0.5). Peers counted per distinct other node in the same partition group."
        ),
    )
    port_retry_max: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "After each ScriptLet wave (intra/inter), rerun only pairs whose logs show PORT_LISTEN_FAILED, "
            "up to this many extra batches with new TCP ports (default 3)."
        ),
    )
    port_retry_gap: int = Field(
        default=1000,
        ge=1,
        le=65535,
        description=(
            "When remapping ports for PORT_LISTEN_FAILED retries, start at (max port in batch) + this gap "
            "to reduce overlap with ephemeral ports."
        ),
    )
    exclude_failed_interface_nodes: str = Field(
        default="true",
        description=(
            "Legacy hint for reporting: preflight now prunes interface- and GID-failed nodes from the SSH "
            "host list before RDMA; interface failures are not run in the mesh regardless of this flag."
        ),
    )

    @field_validator('connectivity_mode')
    @classmethod
    def validate_connectivity_check(cls, v: str) -> str:
        """Validate RDMA connectivity check setting."""
        valid_modes = ['basic', 'full_mesh', 'skip']
        if v not in valid_modes:
            raise ValueError(f"connectivity_mode must be one of: {', '.join(valid_modes)}")
        return v

    @field_validator('min_active_mtu')
    @classmethod
    def validate_min_active_mtu(cls, v: int) -> int:
        if v not in VALID_RDMA_ACTIVE_MTUS:
            raise ValueError(f"min_active_mtu must be one of {', '.join(map(str, VALID_RDMA_ACTIVE_MTUS))}")
        return v

    @field_validator('ibv_test_port_range')
    @classmethod
    def validate_port_range(cls, v: str) -> str:
        """Validate port range format."""
        try:
            start, end = map(int, v.split('-'))
            if start >= end or start < 1024 or end > 65535:
                raise ValueError("Invalid port range")
        except (ValueError, AttributeError):
            raise ValueError("ibv_test_port_range must be in format 'start-end' with valid port numbers")
        return v


class PreflightL2PingConfig(BaseModel):
    """Small customer-facing IFoE L2 ping policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable the mandatory IFoE L2 connectivity gate")
    pings_per_port: int = Field(default=3, ge=1, description="Ping samples per selected IFoE port pair")
    loss_threshold_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Maximum tolerated packet loss percentage per traffic type",
    )
    ping_timeout: int = Field(
        default=600,
        ge=30,
        description="PSSH read timeout in seconds for each afmctl ping invocation",
    )


class PreflightTransferBenchConfig(BaseModel):
    """Small customer-facing TransferBench preflight policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable the mandatory TransferBench preflight gate")
    scope: str = Field(default="node", description="node for independent runs or cluster for one multi-rank run")
    profile: str = Field(default="smoketest", description="CVS-supported TransferBench validation profile")
    message_sizes: List[str] = Field(
        default_factory=lambda: ["1K", "16M"],
        min_length=1,
        description="Message sizes exercised by the selected profile",
    )
    iterations: int = Field(default=2, ge=1, description="Validated iterations per test and message size")
    warmup_iterations: int = Field(default=0, ge=0, description="Warmup iterations before validation")

    @field_validator('scope')
    @classmethod
    def validate_transferbench_scope(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ('node', 'cluster'):
            raise ValueError("TransferBench scope must be one of: node, cluster")
        return normalized

    @field_validator('profile')
    @classmethod
    def validate_transferbench_profile(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != 'smoketest':
            raise ValueError("TransferBench profile must be a CVS-supported profile: smoketest")
        return normalized

    @field_validator('message_sizes')
    @classmethod
    def validate_transferbench_message_sizes(cls, value: List[str]) -> List[str]:
        normalized = [str(size).strip() for size in value]
        if any(not size for size in normalized):
            raise ValueError("TransferBench message_sizes entries must not be empty")
        return normalized


class PreflightIfoeConfig(BaseModel):
    """MI4XX IFoE admission and data-path checks."""

    model_config = ConfigDict(extra="forbid")

    fabric_checks: bool = Field(
        default=False,
        description="Enable MI4XX AIFM, AFM, vPOD, station-mask, and IFoE port admission",
    )
    min_up_ports_per_gpu: int = Field(
        default=2,
        ge=0,
        description="Minimum enabled IFoE ports each GPU's station mask must report; "
        "default 2 (one station) fails an un-provisioned fabric, 0 admits it for diagnostics",
    )
    reject_partial_stations: bool = Field(
        default=True,
        description="Fail when any station mask nibble indicates a partial station",
    )
    allow_disabled_stations: bool = Field(
        default=True,
        description="Allow station masks that intentionally disable ports",
    )
    expected_station_masks: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-GPU BDF expected 18-nibble station masks",
    )
    require_uniform_station_mask: bool = Field(
        default=False,
        description="Require identical station masks across all GPUs on a node",
    )
    agent_process_name: str = Field(
        default="inb-node-agent",
        description="Process name checked for the AIFM in-band node agent",
    )
    l2ping: PreflightL2PingConfig = Field(
        default_factory=PreflightL2PingConfig,
        description="Strict IFoE L2 connectivity admission",
    )
    transferbench: PreflightTransferBenchConfig = Field(
        default_factory=PreflightTransferBenchConfig,
        description="TransferBench IFoE data-path validation",
    )


class PreflightConnectivityCheckConfig(BaseModel):
    """Connectivity check settings by protocol."""

    model_config = ConfigDict(extra="allow")

    rdma: PreflightRdmaConfig = Field(default_factory=PreflightRdmaConfig, description="RDMA connectivity settings")
    ifoe: PreflightIfoeConfig = Field(default_factory=PreflightIfoeConfig, description="IFoE connectivity settings")


class PreflightNodeSmokeConfig(BaseModel):
    """Primus node_smoke settings (primus-cli direct -- node_smoke)."""

    model_config = ConfigDict(extra="allow")

    connectivity_mode: str = Field(
        default="run",
        description="Primus node_smoke mode: 'run' (default; host/GPU/RDMA roll-call) or 'skip'",
    )
    auto_setup: bool = Field(
        default=True,
        description="Clone/update Primus and prepare venv on each node before node_smoke",
    )
    setup_timeout: int = Field(default=600, ge=60, description="SSH timeout in seconds for Primus auto_setup")
    force_reclone: bool = Field(
        default=False,
        description="Remove primus_dir and clone fresh on every run (destructive)",
    )
    shared_install: bool = Field(
        default=True,
        description=(
            "When true (default), clone and venv setup run only on the first reachable node; "
            "other nodes wait for the shared NFS home install. Set false only if each node has "
            "a local primus_dir/venv_activate path."
        ),
    )
    pip_install_mode: str = Field(
        default="minimal",
        description="Venv deps: minimal (torch only), requirements, or skip",
    )
    torch_pip_index_url: str = Field(
        default="https://download.pytorch.org/whl/rocm7.2",
        description=(
            "PyTorch ROCm wheel index URL for minimal pip_install_mode. "
            "An importable torch is reinstalled when its HIP major.minor does not match "
            "the rocm<major>.<minor> in this URL."
        ),
    )
    primus_git_url: str = Field(
        default="https://github.com/AMD-AIG-AIMA/Primus.git",
        description="Primus repository URL for auto_setup clone",
    )
    primus_git_branch: str = Field(
        default="dev/preflight-direct-test",
        description="Git branch to checkout during auto_setup",
    )
    primus_git_recurse_submodules: bool = Field(
        default=False,
        description="Clone git submodules during auto_setup (not required for node_smoke)",
    )
    primus_dir: str = Field(
        default="/home/{user-id}/INSTALL/Primus",
        description="Path to cloned Primus repo under the user's home directory (required when connectivity_mode is 'run')",
    )
    venv_activate: str = Field(
        default="/home/{user-id}/envs/preflight/.venv/bin/activate",
        description="Path to Python venv activate script on each node (required when connectivity_mode is 'run')",
    )
    gpus_per_node: int = Field(default=8, ge=1, description="GPUs per node for node_smoke")
    master_port: int = Field(default=1234, ge=1024, le=65535, description="Distributed master port for node_smoke")
    dump_path: str = Field(
        default="",
        description="Per-node dump directory for smoke JSON (default: <artifacts_root_dir>/node_smoke)",
    )
    expected_rdma_nics: Optional[int] = Field(
        default=None,
        ge=1,
        description="Hard-fail when training RDMA NIC count differs (default: len(node_check.rdma_interfaces))",
    )
    ulimit_l_min_gb: float = Field(default=32.0, ge=0, description="Minimum RLIMIT_MEMLOCK in GiB (0 disables)")
    shm_min_gb: float = Field(default=8.0, ge=0, description="Minimum /dev/shm size in GiB (0 disables)")
    skip_dmesg: bool = Field(default=False, description="Skip dmesg error scan (e.g. unprivileged containers)")
    allow_foreign_procs: bool = Field(
        default=False,
        description="Do not FAIL nodes with foreign GPU processes (still reported)",
    )
    allowed_procs: str = Field(
        default="gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter",
        description="Comma-separated process names allowed to hold GPUs",
    )
    require_tools: str = Field(
        default="",
        description="Comma-separated CLI tools that must exist in PATH (empty = warn only)",
    )
    nccl_socket_ifname: str = Field(default="", description="NCCL_SOCKET_IFNAME override for node_smoke")
    gloo_socket_ifname: str = Field(
        default="", description="GLOO_SOCKET_IFNAME override (defaults to nccl_socket_ifname)"
    )
    nccl_ib_hca: str = Field(default="", description="NCCL_IB_HCA override (defaults to node_check.rdma_interfaces)")
    nccl_ib_gid_index: Optional[int] = Field(
        default=None,
        description="NCCL_IB_GID_INDEX override (defaults to node_check.gid_index)",
    )
    rdma_nic_allowlist: str = Field(
        default="",
        description="Training NIC allowlist for node_smoke (defaults to node_check.rdma_interfaces)",
    )
    ssh_timeout: int = Field(default=300, ge=30, description="SSH timeout in seconds for each node_smoke run")
    tier2_perf: bool = Field(
        default=True,
        description=(
            "Enable Primus node_smoke Tier 2 perf sanity (--tier2-perf): "
            "8192³ GEMM TFLOPS floor, HBM D2D bandwidth, local multi-GPU RCCL all-reduce"
        ),
    )
    gemm_tflops_min: float = Field(
        default=600.0,
        ge=0,
        description="Tier 2 large GEMM TFLOPS floor (--gemm-tflops-min); used when tier2_perf is true",
    )
    hbm_gbs_min: float = Field(
        default=2000.0,
        ge=0,
        description="Tier 2 HBM device-to-device bandwidth floor in GB/s (--hbm-gbs-min)",
    )
    rccl_gbs_min: float = Field(
        default=100.0,
        ge=0,
        description="Tier 2 local multi-GPU RCCL all-reduce bandwidth floor in GB/s (--rccl-gbs-min)",
    )
    rccl_size_mb: int = Field(
        default=64,
        ge=1,
        description="Tier 2 local RCCL all-reduce message size in MB (--rccl-size-mb)",
    )
    rccl_timeout_sec: int = Field(
        default=120,
        ge=30,
        description="Tier 2 local RCCL all-reduce hard timeout in seconds (--rccl-timeout-sec)",
    )
    extra_args: List[str] = Field(
        default_factory=list,
        description="Additional node_smoke CLI flags forwarded to primus-cli",
    )

    @field_validator("connectivity_mode")
    @classmethod
    def validate_node_smoke_mode(cls, v: str) -> str:
        valid_modes = ["run", "skip"]
        if v not in valid_modes:
            raise ValueError(f"node_smoke.connectivity_mode must be one of: {', '.join(valid_modes)}")
        return v


class PreflightReportingConfig(BaseModel):
    """Report generation and output settings."""

    model_config = ConfigDict(extra="allow")

    generate_html_report: bool = Field(default=True, description="Whether to generate HTML report")
    artifacts_root_dir: str = Field(
        default="/tmp/preflight",
        description=(
            "Root directory for preflight artifacts. HTML report output and RDMA full_mesh ScriptLet logs use "
            "<artifacts_root_dir>/rdma_connectivity_workspace/<session>/<round>/ on each node (NFS-friendly)."
        ),
    )
    generate_rdma_pairs_csv: bool = Field(
        default=True,
        description="If true, write preflight_report_*_rdma_pairs.csv beside the HTML report (failed pairs only)",
    )


class PreflightConfigFile(BaseModel):
    """
    Schema for preflight check configuration file.

    Uses nested structure organized by execution phase for better organization.
    """

    model_config = ConfigDict(extra="allow")  # Allow comment fields

    parallelism: PreflightParallelismConfig = Field(
        default_factory=PreflightParallelismConfig, description="Parallel execution settings"
    )
    debug: PreflightDebugConfig = Field(
        default_factory=PreflightDebugConfig, description="Debug and troubleshooting options"
    )
    node_check: PreflightNodeCheckConfig = Field(
        default_factory=PreflightNodeCheckConfig, description="Individual node validation settings"
    )
    connectivity_check: PreflightConnectivityCheckConfig = Field(
        default_factory=PreflightConnectivityCheckConfig, description="Inter-node connectivity tests"
    )
    node_smoke: PreflightNodeSmokeConfig = Field(
        default_factory=PreflightNodeSmokeConfig, description="Primus node_smoke checks"
    )
    reporting: PreflightReportingConfig = Field(
        default_factory=PreflightReportingConfig, description="Report generation and output settings"
    )

    @model_validator(mode="before")
    @classmethod
    def reject_flat_preflight_checks(cls, value):
        if not isinstance(value, dict):
            return value
        cleaned = strip_preflight_metadata(value)
        removed = sorted(set(cleaned) & {"node_health", "l2ping", "transferbench"})
        if removed:
            raise ValueError(
                "Unsupported flat preflight block(s): "
                + ", ".join(removed)
                + "; use node_check and connectivity_check.ifoe"
            )
        normalized, warning_message = normalize_legacy_preflight_rdma_config(cleaned)
        if warning_message:
            warnings.warn(warning_message, FutureWarning, stacklevel=2)
        normalized, smoke_warning = normalize_legacy_preflight_node_smoke_sections(normalized)
        if smoke_warning:
            warnings.warn(smoke_warning, FutureWarning, stacklevel=2)
        return normalized

    @model_validator(mode="after")
    def validate_fabric_prerequisites(self):
        if self.connectivity_check.ifoe.fabric_checks and not self.node_check.enabled:
            raise ValueError("connectivity_check.ifoe.fabric_checks requires node_check.enabled=true")
        return self


def validation_problem_lines(exc):
    """One ``location: problem`` line per error in a ``ValidationError``, never the input values."""
    lines = []
    for error in exc.errors(include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "(config)"
        lines.append(f"{location}: {error['msg'].removeprefix('Value error, ')}")
    return lines


def validate_config_file(
    config_path: Union[str, Path], config_type: str = "auto"
) -> Union[
    ClusterConfigFile,
    PytorchXditWanConfigFile,
    PytorchXditFluxConfigFile,
    PytorchXditUnifiedConfigFile,
    PreflightConfigFile,
]:
    """
    Load and validate a configuration file.

    Args:
        config_path: Path to configuration file (YAML or JSON)
        config_type: Type of config - "aorta", "cluster", "pytorch_xdit_wan", "pytorch_xdit_flux", "preflight", or "auto" (detect from content)

    Returns:
        Validated Pydantic model

    Raises:
        ValueError: If config is invalid with detailed error message
        FileNotFoundError: If config file doesn't exist
    """
    import json
    import yaml

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load file
    with open(config_path) as f:
        if config_path.suffix in ('.yaml', '.yml'):
            raw_config = yaml.safe_load(f)
        else:
            raw_config = json.load(f)

    if raw_config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    # Determine config type
    if config_type == "auto":
        if "node_dict" in raw_config:
            config_type = "cluster"
        elif "preflight" in raw_config:
            config_type = "preflight"
        elif "aorta_path" in raw_config:
            config_type = "aorta"
        elif is_pytorch_xdit_unified_config(raw_config):
            config_type = "pytorch_xdit_unified"
        elif "config" in raw_config and "benchmark_params" in raw_config:
            # Check if it's a pytorch_xdit config (WAN or Flux)
            config_section = raw_config.get("config", {})
            benchmark_section = raw_config.get("benchmark_params", {})

            # Detect Flux: check for flux1_dev_t2i in benchmark_params or FLUX in model_repo
            if "flux1_dev_t2i" in benchmark_section or "FLUX" in config_section.get("model_repo", ""):
                config_type = "pytorch_xdit_flux"
            # Detect WAN: check for wan22_i2v_a14b in benchmark_params or Wan in model_repo
            elif "wan22_i2v_a14b" in benchmark_section or "Wan" in config_section.get("model_repo", ""):
                config_type = "pytorch_xdit_wan"
            else:
                # Generic pytorch_xdit - default to WAN for backward compatibility
                config_type = "pytorch_xdit_wan"
        else:
            raise ValueError(
                f"Cannot auto-detect config type for {config_path}. "
                f"Specify config_type='aorta', config_type='cluster', config_type='pytorch_xdit_unified', "
                f"config_type='pytorch_xdit_wan', config_type='pytorch_xdit_flux', or config_type='preflight'"
            )

    # Validate with appropriate schema
    try:
        if config_type == "cluster":
            return ClusterConfigFile.model_validate(raw_config)
        elif config_type == "preflight":
            # Extract preflight section for validation
            if "preflight" in raw_config:
                return PreflightConfigFile.model_validate(raw_config["preflight"])
            else:
                raise ValueError("Preflight config must contain 'preflight' section")
        elif config_type == "aorta":
            from cvs.lib.benchmark.aorta.aorta_config_loader import load_training_variant

            return load_training_variant(config_path, {})
        elif config_type == "pytorch_xdit_wan":
            return PytorchXditWanConfigFile.model_validate(raw_config)
        elif config_type == "pytorch_xdit_flux":
            return PytorchXditFluxConfigFile.model_validate(raw_config)
        elif config_type == "pytorch_xdit_unified":
            return PytorchXditUnifiedConfigFile.model_validate(raw_config)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")
    except ValidationError as e:
        problems = "\n".join(f"  {line}" for line in validation_problem_lines(e))
        raise ValueError(f"Invalid configuration in {config_path}:\n{problems}") from None
    except Exception as e:
        # Re-raise with file context
        raise ValueError(f"Invalid configuration in {config_path}:\n{e}") from e
