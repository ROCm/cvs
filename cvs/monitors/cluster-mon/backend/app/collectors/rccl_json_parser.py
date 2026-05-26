"""
Parser for rcclras JSON output (RCCL v2.28.7+ with -f json / SET FORMAT json).

The RAS server emits a single JSON document (from jsonWriteHeader etc. in
client_support.cc). This parser populates per-rank status flags that the text
parser cannot extract, including init_state, async_error, abort_flag, etc.
"""

import json
import time
import logging
from typing import Any

from app.models.rccl_models import (
    RCCLSnapshot,
    RCCLJobState,
    RCCLJobSummary,
    RCCLCommunicator,
    RCCLRank,
    RCCLRankStatus,
    NCCLFunction,
)

logger = logging.getLogger(__name__)


class RCCLJsonParser:
    """
    Parses rcclras JSON output into RCCLSnapshot.

    Schema (from client_support.cc jsonWrite* functions, RCCL v2.28.7+):
      {
        "nccl_version": "2.28.9",
        "cuda_runtime_version": 12040,
        "cuda_driver_version": 12040,
        "timestamp": "...",
        "communicators_count": N,
        "communicators": [{
          "hash": "0x...",
          "size": N,
          "ranks_count": N,
          "missing_ranks_count": N,
          "ranks": [{
            "rank": 0, "host": "...", "pid": ..., "cuda_dev": ..., "nvml_dev": ...,
            "status": {"init_state": 0, "async_error": 0, "finalize_called": false,
                       "destroy_flag": false, "abort_flag": false},
            "collective_counts": {"Broadcast": 0, "Reduce": 0, ...}
          }]
        }]
      }
    """

    def parse(self, raw_text: str) -> RCCLSnapshot:
        if not raw_text or not raw_text.strip():
            return RCCLSnapshot.empty(state=RCCLJobState.NO_JOB)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error(f"RCCLJsonParser: invalid JSON: {e}")
            return RCCLSnapshot.empty(state=RCCLJobState.ERROR)

        if not isinstance(data, dict):
            logger.error("RCCLJsonParser: top-level JSON is not an object")
            return RCCLSnapshot.empty(state=RCCLJobState.ERROR)

        try:
            job_summary = self._parse_job_summary(data)
            communicators = self._parse_communicators(data)
            state = self._determine_state(communicators, job_summary)

            return RCCLSnapshot(
                timestamp=time.time(),
                state=state,
                job_summary=job_summary,
                communicators=communicators,
                peers=[],
                dead_peers=[],
                errors=[],
            )
        except Exception as e:
            logger.error(f"RCCLJsonParser: failed to parse JSON output: {e}", exc_info=True)
            return RCCLSnapshot.empty(state=RCCLJobState.ERROR)

    def _parse_job_summary(self, data: dict[str, Any]) -> RCCLJobSummary:
        rccl_version = data.get("nccl_version", "unknown")
        hip_version = int(data.get("cuda_runtime_version", 0))
        driver_version = int(data.get("cuda_driver_version", 0))

        comms = data.get("communicators", [])
        total_gpus = sum(c.get("size", 0) for c in comms)

        # Derive node count from unique hostnames across all ranks
        hosts: set[str] = set()
        for comm in comms:
            for rank in comm.get("ranks", []):
                host = rank.get("host", "")
                if host:
                    hosts.add(host)
        total_nodes = len(hosts) if hosts else 1

        # Process count: distinct (host, pid) pairs
        procs: set[tuple[str, int]] = set()
        for comm in comms:
            for rank in comm.get("ranks", []):
                host = rank.get("host", "")
                pid = rank.get("pid", 0)
                if host:
                    procs.add((host, pid))
        total_processes = len(procs) if procs else total_gpus

        return RCCLJobSummary(
            total_nodes=total_nodes,
            total_processes=total_processes,
            total_gpus=total_gpus,
            rccl_version=rccl_version,
            hip_runtime_version=hip_version,
            amdgpu_driver_version=driver_version,
            inconsistent_topology=False,
        )

    def _parse_communicators(self, data: dict[str, Any]) -> list[RCCLCommunicator]:
        comms = []
        for comm_data in data.get("communicators", []):
            comm_hash = comm_data.get("hash", "unknown")
            total_ranks = int(comm_data.get("size", 0))
            missing = int(comm_data.get("missing_ranks_count", 0))
            ranks_count = int(comm_data.get("ranks_count", total_ranks))
            responding = ranks_count - missing

            ranks = [self._parse_rank(r) for r in comm_data.get("ranks", [])]

            # Degraded if any rank has a non-zero error flag
            any_error = any(
                r.status.async_error != 0
                or r.status.abort_flag
                or r.status.init_state != 0
                for r in ranks
            )
            if missing > 0 or any_error:
                health = RCCLJobState.DEGRADED
            else:
                health = RCCLJobState.HEALTHY

            comms.append(RCCLCommunicator(
                comm_hash=comm_hash,
                total_ranks=total_ranks,
                responding_ranks=responding,
                missing_ranks=missing,
                ranks=ranks,
                health=health,
            ))
        return comms

    def _parse_rank(self, rank_data: dict[str, Any]) -> RCCLRank:
        status_data = rank_data.get("status", {})
        status = RCCLRankStatus(
            init_state=int(status_data.get("init_state", 0)),
            async_error=int(status_data.get("async_error", 0)),
            finalize_called=bool(status_data.get("finalize_called", False)),
            destroy_flag=bool(status_data.get("destroy_flag", False)),
            abort_flag=bool(status_data.get("abort_flag", False)),
        )

        raw_counts: dict[str, int] = rank_data.get("collective_counts", {})
        coll_counts: dict[NCCLFunction, int] = {}
        for fn in NCCLFunction:
            coll_counts[fn] = int(raw_counts.get(fn.value, 0))

        return RCCLRank(
            comm_rank=int(rank_data.get("rank", 0)),
            node_addr=rank_data.get("host", ""),
            pid=int(rank_data.get("pid", 0)),
            cuda_dev=int(rank_data.get("cuda_dev", 0)),
            nvml_dev=int(rank_data.get("nvml_dev", 0)),
            coll_op_counts=coll_counts,
            status=status,
        )

    def _determine_state(
        self,
        communicators: list[RCCLCommunicator],
        job_summary: RCCLJobSummary,
    ) -> RCCLJobState:
        if not communicators:
            return RCCLJobState.HEALTHY  # Job starting up, no comms yet
        for comm in communicators:
            if comm.health == RCCLJobState.DEGRADED or comm.missing_ranks > 0:
                return RCCLJobState.DEGRADED
        return RCCLJobState.HEALTHY
