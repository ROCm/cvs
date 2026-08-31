"""
Pydantic schemas for benchmark result parsing.

Configuration file schemas live under ``cvs/schema/`` (mirroring ``cvs/input/``); use
``cvs.schema.validate.validate_config_file`` to load and validate configs.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
