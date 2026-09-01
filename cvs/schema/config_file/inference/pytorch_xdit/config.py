"""
PyTorch xDiT (WAN / Flux) inference configuration file schemas.

Mirrors ``cvs/input/config_file/inference/pytorch_xdit/``.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    model_config = ConfigDict(extra="allow")  # Allow comment fields

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

    model_config = ConfigDict(extra="allow")  # Allow comment fields

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


class PytorchXditFluxConfig(BaseModel):
    """Schema for config section in pytorch-xdit Flux configs."""

    model_config = ConfigDict(extra="forbid")

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


PytorchXditWanConfigFile.model_rebuild()
PytorchXditFluxConfigFile.model_rebuild()
