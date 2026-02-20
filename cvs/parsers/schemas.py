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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
import math

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

    model_config = ConfigDict(extra="allow")  # Allow extra fields like bmc_ip

    vpc_ip: str = Field(description="VPC IP or hostname for inter-node communication")
    bmc_ip: Optional[str] = Field(default=None, description="BMC IP for out-of-band management")


class HeadNodeConfig(BaseModel):
    """Schema for head_node_dict in cluster.json."""

    model_config = ConfigDict(extra="allow")

    mgmt_ip: str = Field(description="Management IP of head node")


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

    Usage:
        with open("aorta_benchmark.yaml") as f:
            raw = yaml.safe_load(f)
        config = AortaBenchmarkConfigFile.model_validate(raw)
    """

    model_config = ConfigDict(extra="forbid")  # Catch typos in top-level keys

    # Path to Aorta repository on host (will be bind-mounted). If missing and aorta_auto_clone is true, it is cloned.
    aorta_path: str = Field(description="Path to Aorta repository on host (will be bind-mounted)")

    # Optional: clone Aorta repo when aorta_path does not exist
    aorta_auto_clone: bool = Field(default=False, description="If true and aorta_path missing, clone from aorta_clone_url")
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


class PytorchXditContainerConfig(BaseModel):
    """Schema for container_config section in pytorch-xdit configs."""

    model_config = ConfigDict(extra="allow")

    device_list: List[str] = Field(
        default=["/dev/dri", "/dev/kfd"], description="List of device paths to mount in container"
    )
    volume_dict: Dict[str, str] = Field(default_factory=dict, description="Host:container volume mount mappings")
    env_dict: Dict[str, str] = Field(default_factory=dict, description="Environment variables for container")


class PytorchXditExpectedResults(BaseModel):
    """Schema for expected_results in pytorch-xdit benchmark params."""

    model_config = ConfigDict(extra="forbid")

    max_avg_total_time_s: float = Field(gt=0, description="Maximum acceptable average total_time in seconds")


class PytorchXditWan22Benchmarks(BaseModel):
    """Schema for wan22_i2v_a14b benchmark parameters."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(description="Text prompt for image-to-video generation")
    size: str = Field(default="720*1280", pattern=r"^\d+\*\d+$", description="Video resolution (format: height*width)")
    frame_num: int = Field(default=81, ge=1, description="Number of frames to generate")
    num_benchmark_steps: int = Field(default=5, ge=1, description="Number of benchmark iterations to run")
    compile: bool = Field(default=True, description="Whether to use torch.compile for optimization")
    torchrun_nproc: int = Field(default=8, ge=1, description="Number of processes for torchrun (usually num GPUs)")
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
    no_use_resolution_binning: bool = Field(default=True, description="Disable resolution binning")
    warmup_steps: int = Field(default=1, ge=0, description="Number of warmup steps before benchmarking")
    warmup_calls: int = Field(default=5, ge=0, description="Number of warmup calls")
    num_repetitions: int = Field(default=25, ge=1, description="Number of benchmark repetitions")
    height: int = Field(default=1024, ge=1, description="Output image height in pixels")
    width: int = Field(default=1024, ge=1, description="Output image width in pixels")
    ulysses_degree: int = Field(default=8, ge=1, description="Ulysses parallelism degree")
    ring_degree: int = Field(default=1, ge=1, description="Ring parallelism degree")
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


class PytorchXditWanConfig(BaseModel):
    """Schema for config section in pytorch-xdit WAN configs."""

    model_config = ConfigDict(extra="forbid")

    container_image: str = Field(
        default="amdsiloai/pytorch-xdit:v25.11.2", description="Docker image for pytorch-xdit container"
    )
    container_name: str = Field(default="wan22-benchmark", description="Name for the Docker container")
    hf_token_file: str = Field(description="Path to Hugging Face token file")
    hf_home: str = Field(description="Host directory for Hugging Face cache (mounted to /hf_home)")
    output_base_dir: str = Field(description="Host base directory for benchmark outputs")
    model_repo: str = Field(default="Wan-AI/Wan2.2-I2V-A14B", description="Hugging Face model repository")
    model_rev: str = Field(
        default="206a9ee1b7bfaaf8f7e4d81335650533490646a3", description="Model revision (commit hash)"
    )
    container_config: PytorchXditContainerConfig = Field(
        default_factory=PytorchXditContainerConfig, description="Container device/volume/env configuration"
    )

    @field_validator('hf_token_file', 'hf_home', 'output_base_dir')
    @classmethod
    def validate_path_not_placeholder(cls, v: str, info) -> str:
        """Check that paths are not still placeholders."""
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


class PytorchXditFluxConfig(BaseModel):
    """Schema for config section in pytorch-xdit Flux configs."""

    model_config = ConfigDict(extra="forbid")

    container_image: str = Field(
        default="amdsiloai/pytorch-xdit:v25.11.2", description="Docker image for pytorch-xdit container"
    )
    container_name: str = Field(default="flux-benchmark", description="Name for the Docker container")
    hf_token_file: str = Field(description="Path to Hugging Face token file")
    hf_home: str = Field(description="Host directory for Hugging Face cache (mounted to /hf_home)")
    output_base_dir: str = Field(description="Host base directory for benchmark outputs")
    model_repo: str = Field(default="black-forest-labs/FLUX.1-dev", description="Hugging Face model repository")
    model_rev: str = Field(
        default="", description="Model revision (commit hash) - empty means use any available snapshot"
    )
    container_config: PytorchXditContainerConfig = Field(
        default_factory=PytorchXditContainerConfig, description="Container device/volume/env configuration"
    )

    @field_validator('hf_token_file', 'hf_home', 'output_base_dir')
    @classmethod
    def validate_path_not_placeholder(cls, v: str, info) -> str:
        """Check that paths are not still placeholders."""
        if '<changeme>' in v.lower():
            raise ValueError(f"{info.field_name} contains placeholder '<changeme>'. Please set a valid path in config.")
        return v


def validate_config_file(
    config_path: Union[str, Path], config_type: str = "auto"
) -> Union[AortaBenchmarkConfigFile, ClusterConfigFile, PytorchXditWanConfigFile, PytorchXditFluxConfigFile]:
    """
    Load and validate a configuration file.

    Args:
        config_path: Path to configuration file (YAML or JSON)
        config_type: Type of config - "aorta", "cluster", "pytorch_xdit_wan", "pytorch_xdit_flux", or "auto" (detect from content)

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
                f"Specify config_type='aorta', config_type='cluster', config_type='pytorch_xdit_wan', or config_type='pytorch_xdit_flux'"
            )

    # Validate with appropriate schema
    try:
        if config_type == "cluster":
            return ClusterConfigFile.model_validate(raw_config)
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

