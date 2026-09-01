"""
Aorta benchmark configuration file schema.

Mirrors ``cvs/input/config_file/aorta/`` (``aorta_benchmark.yaml``).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
