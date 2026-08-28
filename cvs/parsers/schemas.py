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

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
import math
import warnings

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


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


class AortaDockerConfigFile(BaseModel):
    """Schema for docker section in aorta_benchmark.yaml."""

    model_config = ConfigDict(extra="forbid")  # Catch typos

    image: str = Field(
        default="jeffdaily/pytorch:torchrec-dlrm-complete", description="Docker image for Aorta container"
    )
    container_name: str = Field(default="aorta-benchmark", description="Name for the Docker container")
    shm_size: str = Field(default="17G", description="Shared memory size")
    network_mode: str = Field(default="host", description="Docker network mode")
    privileged: bool = Field(default=True, description="Run container in privileged mode")


class AortaRcclConfigFile(BaseModel):
    """Schema for rccl section in aorta_benchmark.yaml."""

    model_config = ConfigDict(extra="forbid")

    clone_url: str = Field(
        default="https://github.com/ROCmSoftwarePlatform/rccl.git", description="RCCL git repository URL"
    )
    branch: str = Field(default="develop", description="RCCL branch to build")
    build_path: str = Field(default="/mnt/rccl", description="Path inside container for RCCL build")


class AortaEnvironmentConfigFile(BaseModel):
    """Schema for environment section in aorta_benchmark.yaml."""

    model_config = ConfigDict(extra="allow")  # Allow custom env vars

    NCCL_MAX_NCHANNELS: int = Field(default=112, ge=1, le=256, description="Maximum NCCL channels")
    NCCL_MAX_P2P_NCHANNELS: int = Field(default=112, ge=1, le=256, description="Maximum NCCL P2P channels")
    NCCL_DEBUG: str = Field(default="VERSION", description="NCCL debug level")
    TORCH_NCCL_HIGH_PRIORITY: int = Field(default=1, ge=0, le=1, description="Enable high priority NCCL streams")
    OMP_NUM_THREADS: int = Field(default=1, ge=1, description="OpenMP thread count")
    RCCL_MSCCL_ENABLE: int = Field(default=0, ge=0, le=1, description="Enable MSCCL")


class AortaExpectedResultsConfigFile(BaseModel):
    """Schema for expected_results section in aorta_benchmark.yaml."""

    model_config = ConfigDict(extra="allow")  # Allow custom thresholds

    max_avg_iteration_ms: Optional[float] = Field(
        default=None, ge=0, description="Maximum acceptable average iteration time in ms"
    )
    min_compute_ratio: Optional[float] = Field(default=None, ge=0, le=1, description="Minimum acceptable compute ratio")
    min_overlap_ratio: Optional[float] = Field(
        default=None, ge=0, le=1, description="Minimum acceptable compute-comm overlap ratio"
    )
    max_time_variance_ratio: Optional[float] = Field(
        default=None, ge=0, description="Maximum acceptable iteration time variance"
    )


class AortaAnalysisConfigFile(BaseModel):
    """Schema for analysis section in aorta_benchmark.yaml."""

    model_config = ConfigDict(extra="forbid")

    enable_tracelens: bool = Field(default=True, description="Run Aorta's TraceLens analysis after benchmark")
    enable_gemm_analysis: bool = Field(default=False, description="Run Aorta's GEMM analysis (for sweep experiments)")
    tracelens_script: str = Field(
        default="scripts/tracelens_single_config/run_tracelens_single_config.sh",
        description="Path to TraceLens analysis script relative to aorta_path",
    )
    gemm_script: str = Field(
        default="scripts/gemm_analysis/run_tracelens_analysis.sh",
        description="Path to GEMM analysis script relative to aorta_path",
    )
    skip_if_exists: bool = Field(
        default=False, description="Skip analysis if tracelens_analysis directory already exists"
    )


class AortaBenchmarkConfigFile(BaseModel):
    """
    Schema for the entire aorta_benchmark.yaml configuration file.

    Validates structure and provides sensible defaults.
    Fails fast with clear error messages if configuration is invalid.

    For ``test_aorta``, load YAML, apply ``resolve_test_config_placeholders`` with the resolved
    cluster dict (same as other CVS suites), then ``model_validate``. Standalone tools may validate
    raw YAML without placeholder resolution if paths are already absolute.
    """

    model_config = ConfigDict(extra="forbid")  # Catch typos in top-level keys

    # Path to Aorta repository on host (will be bind-mounted). If missing and aorta_auto_clone is true, it is cloned.
    aorta_path: str = Field(description="Path to Aorta repository on host (will be bind-mounted)")

    # Optional: clone Aorta repo when aorta_path does not exist
    aorta_auto_clone: bool = Field(
        default=False, description="If true and aorta_path missing, clone from aorta_clone_url"
    )
    aorta_clone_url: Optional[str] = Field(default=None, description="Git URL to clone when aorta_auto_clone is true")

    # Container settings
    container_mount_path: str = Field(default="/mnt", description="Mount point inside container for aorta_path")

    # Aorta config
    base_config: str = Field(default="config/distributed.yaml", description="Aorta config file relative to aorta_path")

    # Nested configuration sections
    docker: AortaDockerConfigFile = Field(
        default_factory=AortaDockerConfigFile, description="Docker container configuration"
    )
    rccl: AortaRcclConfigFile = Field(default_factory=AortaRcclConfigFile, description="RCCL build configuration")
    environment: AortaEnvironmentConfigFile = Field(
        default_factory=AortaEnvironmentConfigFile, description="Environment variables for RCCL/NCCL"
    )

    # Training overrides
    training_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Overrides passed to Aorta via --override flag"
    )

    # Scripts
    build_script: str = Field(
        default="scripts/build_rccl.sh", description="RCCL build script relative to container mount"
    )
    experiment_script: str = Field(
        default="scripts/rccl_exp.sh", description="Experiment script relative to container mount"
    )

    # Hardware
    gpus_per_node: int = Field(default=8, ge=1, description="Number of GPUs per node")

    # Execution settings
    timeout_seconds: int = Field(default=10800, ge=60, description="Benchmark timeout in seconds")
    skip_rccl_build: bool = Field(default=False, description="Skip RCCL build if already built")

    # Validation thresholds
    expected_results: AortaExpectedResultsConfigFile = Field(
        default_factory=AortaExpectedResultsConfigFile, description="Expected results for validation"
    )

    # Analysis configuration (use Aorta's built-in analysis scripts)
    analysis: AortaAnalysisConfigFile = Field(
        default_factory=AortaAnalysisConfigFile, description="Post-benchmark analysis configuration"
    )

    @field_validator('aorta_path')
    @classmethod
    def validate_aorta_path_not_placeholder(cls, v: str) -> str:
        """Check that aorta_path is not a placeholder."""
        if '<changeme>' in v.lower():
            raise ValueError(
                "aorta_path contains placeholder '<changeme>'. Please set the actual path to your Aorta installation."
            )
        return v

    def validate_paths_exist(self) -> List[str]:
        """
        Validate that referenced paths exist on the filesystem.

        Call this after loading config to check paths.
        Returns list of error messages (empty if all valid).
        """
        errors = []

        aorta = Path(self.aorta_path)
        if not aorta.exists():
            if self.aorta_auto_clone and self.aorta_clone_url:
                # Runner will clone in setup(); skip path checks here
                return errors
            errors.append(f"aorta_path does not exist: {self.aorta_path}")
        else:
            # Check internal paths
            base_cfg = aorta / self.base_config
            if not base_cfg.exists():
                errors.append(f"base_config does not exist: {base_cfg}")

            build_script = aorta / self.build_script
            if not build_script.exists():
                errors.append(f"build_script does not exist: {build_script}")

            exp_script = aorta / self.experiment_script
            if not exp_script.exists():
                errors.append(f"experiment_script does not exist: {exp_script}")

            # Check analysis scripts if enabled
            if self.analysis.enable_tracelens:
                tracelens_script = aorta / self.analysis.tracelens_script
                if not tracelens_script.exists():
                    errors.append(f"tracelens_script does not exist: {tracelens_script}")

            if self.analysis.enable_gemm_analysis:
                gemm_script = aorta / self.analysis.gemm_script
                if not gemm_script.exists():
                    errors.append(f"gemm_script does not exist: {gemm_script}")

        return errors


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
    compile: bool = Field(default=True, description="Whether to use torch.compile for optimization")
    torchrun_nproc: int = Field(default=8, ge=1, description="Number of processes for torchrun (usually num GPUs)")
    ulysses_size: int = Field(default=8, ge=1, description="Ulysses parallelism degree")
    ring_size: int = Field(default=1, ge=1, description="Ring parallelism degree")
    expected_results: Dict[str, PytorchXditExpectedResults] = Field(
        description="Expected results by GPU type (auto, mi300x, mi355, etc.)"
    )

    @field_validator('expected_results')
    @classmethod
    def validate_has_auto_or_specific(
        cls, v: Dict[str, PytorchXditExpectedResults]
    ) -> Dict[str, PytorchXditExpectedResults]:
        """Ensure either 'auto' or a specific GPU type is present."""
        if not v:
            raise ValueError("expected_results must contain at least one GPU type threshold")
        if 'auto' not in v and not any(k in v for k in ['mi300x', 'mi325', 'mi350', 'mi355']):
            raise ValueError("expected_results must contain either 'auto' or a specific GPU type (mi300x, mi325, etc.)")
        return v


class PytorchXditFluxExpectedResults(BaseModel):
    """Schema for expected_results in Flux benchmark params."""

    model_config = ConfigDict(extra="forbid")

    max_avg_pipe_time_s: float = Field(gt=0, description="Maximum acceptable average pipe_time in seconds")


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
            "Classifier-free guidance scale for run_usp.py. Defaults to 4.0 for FLUX.2-dev "
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
    expected_results: Dict[str, PytorchXditFluxExpectedResults] = Field(
        description="Expected results by GPU type (auto, mi300x, mi355, etc.)"
    )

    @field_validator('expected_results')
    @classmethod
    def validate_has_auto_or_specific(
        cls, v: Dict[str, PytorchXditFluxExpectedResults]
    ) -> Dict[str, PytorchXditFluxExpectedResults]:
        """Ensure either 'auto' or a specific GPU type is present."""
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


# =============================================================================
# Preflight Check Configuration Schema
# =============================================================================


LEGACY_PREFLIGHT_RDMA_PATHS = {
    "gid_index": "gid_index",
    "rdma_interfaces": "interfaces",
}

PREFLIGHT_METADATA_PREFIXES = ("_comment", "_example")


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
    if not isinstance(value, dict):
        return value, None

    node_check = value.get("node_check")
    if not isinstance(node_check, dict):
        return value, None

    legacy_keys = [key for key in LEGACY_PREFLIGHT_RDMA_PATHS if key in node_check]
    if not legacy_keys:
        return value, None

    normalized = deepcopy(value)
    normalized_node_check = normalized["node_check"]
    connectivity_check = normalized.setdefault("connectivity_check", {})
    if not isinstance(connectivity_check, dict):
        raise ValueError(
            "preflight.connectivity_check must be an object when deprecated node_check RDMA options are used"
        )
    rdma = connectivity_check.setdefault("rdma", {})
    if not isinstance(rdma, dict):
        raise ValueError(
            "preflight.connectivity_check.rdma must be an object when deprecated node_check RDMA options are used"
        )

    migrations = []
    for legacy_key in legacy_keys:
        canonical_key = LEGACY_PREFLIGHT_RDMA_PATHS[legacy_key]
        legacy_value = normalized_node_check.pop(legacy_key)
        if canonical_key in rdma and rdma[canonical_key] != legacy_value:
            raise ValueError(
                f"Conflicting preflight RDMA options: preflight.node_check.{legacy_key} and "
                f"preflight.connectivity_check.rdma.{canonical_key} must have the same value when both are provided"
            )
        rdma.setdefault(canonical_key, legacy_value)
        migrations.append(f"preflight.node_check.{legacy_key} -> preflight.connectivity_check.rdma.{canonical_key}")

    warning_message = (
        "Deprecated preflight RDMA configuration detected: "
        + ", ".join(migrations)
        + ". Use the preflight.connectivity_check.rdma paths; legacy paths will be removed in a future release."
    )
    return normalized, warning_message


LEGACY_PREFLIGHT_NODE_SMOKE_SECTIONS = {
    "node_smoke": "node_smoke_tier1",
    "tier3_info": "node_smoke_tier3",
}


def _preflight_section_has_values(section: dict) -> bool:
    if not isinstance(section, dict):
        return False
    return any(value not in (None, "") for value in section.values())


def normalize_legacy_preflight_node_smoke_sections(value):
    """Copy legacy Node Smoke section names to their canonical tier keys.

    Returns a deep-copied configuration and one consolidated warning message,
    or the original value and ``None`` when no legacy keys need migration.
    Canonical sections win when both legacy and canonical blocks are populated.
    """
    if not isinstance(value, dict):
        return value, None

    legacy_keys = [key for key in LEGACY_PREFLIGHT_NODE_SMOKE_SECTIONS if key in value]
    if not legacy_keys:
        return value, None

    normalized = deepcopy(value)
    migrations = []
    for legacy_key in legacy_keys:
        canonical_key = LEGACY_PREFLIGHT_NODE_SMOKE_SECTIONS[legacy_key]
        legacy_block = normalized.get(legacy_key)
        if not isinstance(legacy_block, dict):
            continue
        canonical_block = normalized.get(canonical_key)
        if isinstance(canonical_block, dict) and _preflight_section_has_values(canonical_block):
            continue
        normalized[canonical_key] = deepcopy(legacy_block)
        migrations.append(f"preflight.{legacy_key} -> preflight.{canonical_key}")

    if not migrations:
        return value, None

    warning_message = (
        "Deprecated preflight Node Smoke section name(s) detected: "
        + ", ".join(migrations)
        + ". Prefer node_smoke_tier1 and node_smoke_tier3 in new configs."
    )
    return normalized, warning_message


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


class PreflightRdmaConfig(BaseModel):
    """RDMA connectivity testing settings."""

    model_config = ConfigDict(extra="allow")

    connectivity_mode: str = Field(default="basic", description="RDMA connectivity testing: basic, full_mesh, or skip")
    gid_index: str = Field(default="3", description="GID index to check on all RDMA interfaces (typically 3 for RoCE)")
    interfaces: List[str] = Field(
        default_factory=lambda: ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
        min_length=1,
        description="RDMA device names checked for presence, GID consistency, and connectivity",
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
        default="skip",
        description="Primus node_smoke mode: 'run' (host/GPU/RDMA roll-call) or 'skip' (default)",
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
        default="https://download.pytorch.org/whl/rocm6.2",
        description="PyTorch ROCm wheel index URL for minimal pip_install_mode",
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
        default=False,
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


def validate_config_file(
    config_path: Union[str, Path], config_type: str = "auto"
) -> Union[
    AortaBenchmarkConfigFile,
    ClusterConfigFile,
    PytorchXditWanConfigFile,
    PytorchXditFluxConfigFile,
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
                f"Specify config_type='aorta', config_type='cluster', config_type='pytorch_xdit_wan', config_type='pytorch_xdit_flux', or config_type='preflight'"
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
            return AortaBenchmarkConfigFile.model_validate(raw_config)
        elif config_type == "pytorch_xdit_wan":
            return PytorchXditWanConfigFile.model_validate(raw_config)
        elif config_type == "pytorch_xdit_flux":
            return PytorchXditFluxConfigFile.model_validate(raw_config)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")
    except Exception as e:
        # Re-raise with file context
        raise ValueError(f"Invalid configuration in {config_path}:\n{e}") from e
