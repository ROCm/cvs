"""
Pydantic data models for RCCL monitoring.
RCCLJobState is the canonical definition — import from here, not from collectors.
"""

import time
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class RCCLJobState(str, Enum):
    NO_JOB = "no_job"          # Connection refused — no RCCL job running
    UNREACHABLE = "unreachable"  # SSH/TCP timeout — node down
    HEALTHY = "healthy"          # Job running, all communicators healthy
    DEGRADED = "degraded"        # Job running, some ranks missing or async errors
    ERROR = "error"              # Unexpected protocol error


class NCCLFunction(str, Enum):
    """NCCL collective function names as they appear in VERBOSE STATUS text output.

    Used as typed dict keys after the text parser extracts named op counts from
    lines like 'AllReduce=N AllGather=N ...'. Enum order does NOT affect correctness —
    parsing is by string name, not by index.
    """
    BROADCAST = "Broadcast"
    REDUCE = "Reduce"
    ALL_GATHER = "AllGather"
    REDUCE_SCATTER = "ReduceScatter"
    ALL_REDUCE = "AllReduce"
    GATHER = "Gather"
    SCATTER = "Scatter"
    ALL_TO_ALL = "AllToAll"
    ALL_TO_ALL_V = "AllToAllv"
    SEND = "Send"
    RECV = "Recv"
    SEND_RECV = "SendRecv"


class RCCLRankStatus(BaseModel):
    init_state: int        # ncclResult_t value (0 = ncclSuccess)
    async_error: int       # ncclResult_t value
    finalize_called: bool
    destroy_flag: bool
    abort_flag: bool


class RCCLRank(BaseModel):
    comm_rank: int
    node_addr: str         # IP address of the node
    pid: int
    cuda_dev: int          # CUDA device index (CUDA_VISIBLE_DEVICES-aware)
    nvml_dev: int          # NVML device index (raw hardware index)
    coll_op_counts: dict[NCCLFunction, int]  # Keyed by NCCLFunction string enum
    status: RCCLRankStatus


class RCCLCommunicator(BaseModel):
    comm_hash: str         # Hex string of the 3-component commId hash
    total_ranks: int       # commNRanks from RAS collective
    responding_ranks: int  # nRanks — ranks we received data from
    missing_ranks: int     # nMissingRanks — declared missing by other ranks
    ranks: list[RCCLRank]
    health: RCCLJobState   # Derived: HEALTHY/DEGRADED/ERROR


class RCCLPeer(BaseModel):
    addr: str
    pid: int
    cuda_devs: int   # Bitmask
    nvml_devs: int   # Bitmask
    is_dead: bool


class RCCLJobSummary(BaseModel):
    total_nodes: int
    total_processes: int
    total_gpus: int
    rccl_version: str
    hip_runtime_version: int
    amdgpu_driver_version: int
    inconsistent_topology: bool  # True when nodes have different process/GPU counts


class RCCLSnapshot(BaseModel):
    timestamp: float
    state: RCCLJobState
    job_summary: Optional[RCCLJobSummary] = None
    communicators: list[RCCLCommunicator] = []
    peers: list[RCCLPeer] = []
    dead_peers: list[str] = []  # IP:port strings of declared-dead peers
    errors: list[str] = []     # Raw error lines from the Errors section

    @classmethod
    def empty(cls, state: RCCLJobState = RCCLJobState.NO_JOB) -> "RCCLSnapshot":
        return cls(timestamp=time.time(), state=state)


class RCCLEvent(BaseModel):
    timestamp: float
    event_type: str    # "lifecycle" or "trace" (Phase 3+)
    source_node: str
    details: str
    peer_addr: Optional[str] = None


class RCCLMarker(BaseModel):
    """Posted by the PyTorch callback via POST /api/rccl/markers."""
    type: str              # e.g., "training_step"
    step: int
    loss: Optional[float] = None
    rank: int
    timestamp: str         # ISO 8601


# ---------------------------------------------------------------------------
# Inspector plugin models
# ---------------------------------------------------------------------------

class InspectorKernelChannel(BaseModel):
    """Per-channel kernel timing from NCCL_INSPECTOR_DUMP_VERBOSE=1."""
    channel_id: int
    # Sequence numbers — monotonic counters across all profiler events
    kernel_start_sn: Optional[int] = None
    kernel_stop_sn: Optional[int] = None
    kernel_record_sn: Optional[int] = None
    # Raw timestamps — units depend on timing_source:
    #   kernel_gpu:       GPU clock ticks (not directly convertible without clock freq)
    #   kernel_cpu / collective_cpu: microseconds
    kernel_start_ts: Optional[int] = None
    kernel_stop_ts: Optional[int] = None
    kernel_record_ts: Optional[int] = None


class InspectorEventTrace(BaseModel):
    """Verbose event trace from coll_perf.event_trace_sn / event_trace_ts."""
    coll_start_sn: Optional[int] = None
    coll_stop_sn: Optional[int] = None
    coll_start_ts: Optional[int] = None
    coll_stop_ts: Optional[int] = None
    channels: list[InspectorKernelChannel] = []


class InspectorCollPerf(BaseModel):
    """Single completed collective record from one Inspector JSONL line."""
    timestamp: float            # dump_timestamp_us / 1e6 (Unix seconds)
    comm_hash: str              # header.id — communicator identity
    rank: int                   # header.rank
    nranks: int                 # header.n_ranks
    nnodes: int                 # header.nnodes
    hostname: str               # metadata.hostname
    pid: int                    # metadata.pid
    collective: str             # coll_perf.coll (e.g. "AllReduce")
    sequence_num: int           # coll_perf.coll_sn — monotonic counter per comm
    msg_size_bytes: int         # coll_perf.coll_msg_size_bytes
    exec_time_us: int           # coll_perf.coll_exec_time_us
    timing_source: str          # coll_perf.coll_timing_source
    algo_bw_gbps: float         # coll_perf.coll_algobw_gbs
    bus_bw_gbps: float          # coll_perf.coll_busbw_gbs
    event_trace: Optional[InspectorEventTrace] = None  # present only with NCCL_INSPECTOR_DUMP_VERBOSE=1


class InspectorSnapshot(BaseModel):
    """Aggregated Inspector data across all ranks collected in one poll cycle."""
    timestamp: float
    records: list[InspectorCollPerf]
    avg_bus_bw_gbps: Optional[float] = None
    min_bus_bw_gbps: Optional[float] = None
    max_bus_bw_gbps: Optional[float] = None
    slowest_rank: Optional[int] = None
    collective_breakdown: dict[str, int] = {}  # collective name → count
