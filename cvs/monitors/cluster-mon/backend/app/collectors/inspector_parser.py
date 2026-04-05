"""
Parser for RCCL Inspector plugin JSONL output (format version v4.0).

Each line in an Inspector log file is one JSON object representing the
most recently completed collective for a communicator during a dump interval.
This is a "latest snapshot" model — not a complete event log.

Reference: ext-profiler/inspector/inspector.cc (RCCL v2.28.3)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from app.models.rccl_models import (
    InspectorCollPerf,
    InspectorEventTrace,
    InspectorKernelChannel,
    InspectorSnapshot,
)

logger = logging.getLogger(__name__)


class InspectorParser:
    """
    Parse Inspector JSONL log files into InspectorCollPerf records.

    Usage:
        parser = InspectorParser()
        records = parser.parse_file(Path("/nfs/inspector-logs/gpu-node-01-pid12345.log"))
        records = parser.parse_lines("line1\\nline2\\n...")
    """

    def parse_file(self, path: Path, tail: int = 100) -> list[InspectorCollPerf]:
        """
        Read the last `tail` lines from an Inspector log file and parse them.
        Only the tail is read to bound memory usage on long-running jobs.
        """
        try:
            text = path.read_text(errors="replace")
        except OSError as e:
            logger.warning(f"Inspector: cannot read {path}: {e}")
            return []
        lines = text.splitlines()
        return self.parse_lines("\n".join(lines[-tail:]))

    def parse_lines(self, text: str) -> list[InspectorCollPerf]:
        """
        Parse a block of text (one JSON object per line). Malformed lines
        are silently skipped with a debug-level log.
        """
        records: list[InspectorCollPerf] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            record = self._parse_line(line, lineno)
            if record is not None:
                records.append(record)
        return records

    def _parse_line(self, line: str, lineno: int) -> Optional[InspectorCollPerf]:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"Inspector: skipping malformed JSON at line {lineno}")
            return None

        try:
            header = obj["header"]
            meta = obj["metadata"]
            perf = obj["coll_perf"]
            return InspectorCollPerf(
                timestamp=meta["dump_timestamp_us"] / 1_000_000.0,
                comm_hash=header["id"],
                rank=header["rank"],
                nranks=header["n_ranks"],
                nnodes=header["nnodes"],
                hostname=meta["hostname"],
                pid=meta["pid"],
                collective=perf["coll"],
                sequence_num=perf["coll_sn"],
                msg_size_bytes=perf["coll_msg_size_bytes"],
                exec_time_us=perf["coll_exec_time_us"],
                timing_source=perf["coll_timing_source"],
                algo_bw_gbps=float(perf["coll_algobw_gbs"]),
                bus_bw_gbps=float(perf["coll_busbw_gbs"]),
                event_trace=self._parse_event_trace(perf),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Inspector: skipping line {lineno} — missing field: {e}")
            return None

    def _parse_event_trace(self, perf: dict) -> Optional[InspectorEventTrace]:
        """
        Parse verbose event trace from coll_perf when NCCL_INSPECTOR_DUMP_VERBOSE=1.
        event_trace_sn and event_trace_ts are both nested inside the coll_perf object.
        Returns None if neither key is present (non-verbose mode).
        """
        sn = perf.get("event_trace_sn")
        ts = perf.get("event_trace_ts")
        if sn is None and ts is None:
            return None
        # Guard against non-dict values (malformed verbose data)
        if not isinstance(sn, dict):
            sn = None
        if not isinstance(ts, dict):
            ts = None
        if sn is None and ts is None:
            return None

        try:
            # Merge per-channel sn and ts by channel_id
            sn_channels = {c["channel_id"]: c for c in (sn or {}).get("kernel_events", [])}
            ts_channels = {c["channel_id"]: c for c in (ts or {}).get("kernel_events", [])}
            all_ids = sorted(set(sn_channels) | set(ts_channels))

            channels = [
                InspectorKernelChannel(
                    channel_id=ch_id,
                    kernel_start_sn=sn_channels.get(ch_id, {}).get("kernel_start_sn"),
                    kernel_stop_sn=sn_channels.get(ch_id, {}).get("kernel_stop_sn"),
                    kernel_record_sn=sn_channels.get(ch_id, {}).get("kernel_record_sn"),
                    kernel_start_ts=ts_channels.get(ch_id, {}).get("kernel_start_ts"),
                    kernel_stop_ts=ts_channels.get(ch_id, {}).get("kernel_stop_ts"),
                    kernel_record_ts=ts_channels.get(ch_id, {}).get("kernel_record_ts"),
                )
                for ch_id in all_ids
            ]

            return InspectorEventTrace(
                coll_start_sn=(sn or {}).get("coll_start_sn"),
                coll_stop_sn=(sn or {}).get("coll_stop_sn"),
                coll_start_ts=(ts or {}).get("coll_start_ts"),
                coll_stop_ts=(ts or {}).get("coll_stop_ts"),
                channels=channels,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Inspector: skipping malformed event_trace: {e}")
            return None


def aggregate_snapshot(records: list[InspectorCollPerf]) -> InspectorSnapshot:
    """
    Aggregate a list of InspectorCollPerf records into an InspectorSnapshot.

    Stats (avg/min/max busBw, collective breakdown) are computed from ALL
    records in the tail window, giving a richer sample for accuracy.

    The `records` field stored in the snapshot is deduplicated to the LATEST
    entry per (rank, comm_hash) — one row per rank per communicator. This
    prevents the frontend table and WebSocket payload from growing with the
    tail window size (e.g. 500 lines × 8 files = 4000 rows → 8 rows).

    Zero-bandwidth records (exec_time_us == 0) are excluded from bandwidth
    statistics since they indicate a timing fallback or tiny collective.
    """
    import time

    # Stats over all records in the tail window
    bw_records = [r for r in records if r.bus_bw_gbps > 0.0]

    avg_bw: Optional[float] = None
    min_bw: Optional[float] = None
    max_bw: Optional[float] = None
    slowest_rank: Optional[int] = None

    if bw_records:
        bws = [r.bus_bw_gbps for r in bw_records]
        avg_bw = sum(bws) / len(bws)
        min_bw = min(bws)
        max_bw = max(bws)
        slowest_rank = bw_records[bws.index(min_bw)].rank

    collective_breakdown: dict[str, int] = {}
    for r in records:
        collective_breakdown[r.collective] = collective_breakdown.get(r.collective, 0) + 1

    ts = records[0].timestamp if records else time.time()

    # Deduplicate to latest record per (rank, comm_hash) by sequence_num.
    # sequence_num is a monotonically increasing counter per communicator,
    # so the highest value is the most recent collective.
    latest: dict[tuple, InspectorCollPerf] = {}
    for r in records:
        key = (r.rank, r.comm_hash)
        existing = latest.get(key)
        if existing is None or r.sequence_num > existing.sequence_num:
            latest[key] = r
    display_records = sorted(latest.values(), key=lambda r: (r.rank, r.comm_hash))

    return InspectorSnapshot(
        timestamp=ts,
        records=display_records,
        avg_bus_bw_gbps=avg_bw,
        min_bus_bw_gbps=min_bw,
        max_bus_bw_gbps=max_bw,
        slowest_rank=slowest_rank,
        collective_breakdown=collective_breakdown,
    )
