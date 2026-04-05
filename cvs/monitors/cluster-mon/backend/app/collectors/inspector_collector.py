"""
InspectorCollector — reads RCCL Inspector JSONL log files and pushes snapshots.

Two collection modes:
  - "file": reads directly from an NFS path visible to the CVS backend (primary)
  - "ssh":  reads via SSH exec_async, using the existing JumpHostPssh/Pssh manager

The Inspector plugin writes one file per process:
    <dump_dir>/<hostname>-pid<PID>.log

This collector glob-matches all *.log files in dump_dir (file mode) or
runs `tail -n <max_records>` on each compute node (ssh mode).

critical = False — Inspector failure never affects overall cluster health status.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.collectors.inspector_parser import InspectorParser, aggregate_snapshot
from app.models.rccl_models import InspectorCollPerf, InspectorSnapshot

logger = logging.getLogger(__name__)


class InspectorCollector(BaseCollector):
    """
    Polls RCCL Inspector log files every poll_interval seconds.

    File mode (default):
        Reads all *.log files from rccl.inspector.dump_dir via the local
        filesystem. Use this when dump_dir is on a shared NFS mount visible
        from the CVS backend host.

    SSH mode:
        For each healthy node, runs `tail -n <max_records> <dump_dir>/<hostname>-pid*.log`
        via exec_async. Use this when NFS is not available.
    """

    name = "inspector"
    poll_interval: int = 30
    collect_timeout: float = 20.0
    critical = False

    def __init__(self):
        self._parser = InspectorParser()

    async def collect(self, ssh_manager) -> CollectorResult:
        from app.core.config import settings

        cfg = settings.rccl.inspector
        if not cfg.enabled:
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.NO_SERVICE,
                data={},
                error="Inspector collector is disabled (rccl.inspector.enabled=false)",
            )

        try:
            if cfg.mode == "ssh":
                records = await self._collect_ssh(ssh_manager, cfg)
            else:
                records = self._collect_file(cfg)
        except Exception as e:
            logger.error(f"InspectorCollector unexpected error: {e}", exc_info=True)
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.ERROR,
                data={},
                error=str(e),
            )

        snapshot = aggregate_snapshot(records)
        snapshot_dict = snapshot.model_dump()

        # Push to data store if available
        from app.main import app_state
        data_store = getattr(app_state, 'rccl_data_store', None)
        if data_store is not None:
            await data_store.push_inspector_snapshot(snapshot_dict)

        return CollectorResult(
            collector_name=self.name,
            timestamp=CollectorResult.now_iso(),
            state=CollectorState.OK,
            data=snapshot_dict,
        )

    # ------------------------------------------------------------------
    # File mode
    # ------------------------------------------------------------------

    def _collect_file(self, cfg) -> list[InspectorCollPerf]:
        """Read all *.log files in dump_dir from the local/NFS filesystem."""
        if not cfg.dump_dir:
            logger.warning("Inspector file mode: rccl.inspector.dump_dir is not set")
            return []

        dump_path = Path(cfg.dump_dir)
        if not dump_path.exists():
            logger.warning(f"Inspector dump_dir does not exist: {dump_path}")
            return []

        records: list[InspectorCollPerf] = []
        log_files = list(dump_path.glob("*.log"))
        if not log_files:
            logger.debug(f"Inspector: no *.log files found in {dump_path}")
            return []

        for log_file in log_files:
            file_records = self._parser.parse_file(log_file, tail=cfg.max_records_per_file)
            records.extend(file_records)
            logger.debug(f"Inspector: parsed {len(file_records)} records from {log_file.name}")

        logger.info(f"Inspector file mode: {len(records)} records from {len(log_files)} files")
        return records

    # ------------------------------------------------------------------
    # SSH mode
    # ------------------------------------------------------------------

    def _active_pids(self) -> set[int]:
        """
        Extract PIDs of currently active RCCL ranks from the latest rcclras snapshot.
        Returns an empty set when no job is running or rcclras hasn't connected yet.
        """
        from app.main import app_state
        snapshot = getattr(app_state, 'latest_rccl_snapshot', None)
        if not snapshot:
            return set()
        pids: set[int] = set()
        for comm in snapshot.get('communicators', []):
            for rank in comm.get('ranks', []):
                pid = rank.get('pid')
                if pid:
                    pids.add(pid)
        return pids

    async def _collect_ssh(self, ssh_manager, cfg) -> list[InspectorCollPerf]:
        """Collect from each compute node via SSH tail.

        When active PIDs are known from the rcclras snapshot, only log files
        belonging to those PIDs are read — stale files from previous runs in
        the same dump_dir are ignored.
        Falls back to reading all *-pid*.log files when no snapshot is available.
        """
        if not cfg.dump_dir:
            logger.warning("Inspector SSH mode: rccl.inspector.dump_dir is not set")
            return []

        active_pids = self._active_pids()

        if active_pids:
            # Build a grep pattern so only files for the current job are read.
            # Example: grep -E 'pid(3404720|3404721|...)\.log$'
            pid_pattern = "|".join(f"pid{p}" for p in sorted(active_pids))
            cmd = (
                f"ls {cfg.dump_dir}/ 2>/dev/null "
                f"| grep -E '({pid_pattern})\\.log$' "
                f"| xargs -I{{}} tail -n {cfg.max_records_per_file} {cfg.dump_dir}/{{}} "
                f"2>/dev/null || true"
            )
            logger.debug(f"Inspector SSH: filtering to {len(active_pids)} active PIDs")
        else:
            # No rcclras snapshot yet — read all pid log files as fallback
            cmd = (
                f"tail -n {cfg.max_records_per_file} "
                f"{cfg.dump_dir}/*-pid*.log 2>/dev/null || true"
            )
            logger.debug("Inspector SSH: no active PIDs known, reading all log files")

        try:
            outputs = await ssh_manager.exec_async(cmd)
        except Exception as e:
            logger.warning(f"Inspector SSH exec failed: {e}")
            return []

        records: list[InspectorCollPerf] = []
        for node, output in outputs.items():
            if not output:
                continue
            node_records = self._parser.parse_lines(output)
            records.extend(node_records)
            logger.debug(f"Inspector SSH: {len(node_records)} records from {node}")

        logger.info(
            f"Inspector SSH mode: {len(records)} records from {len(outputs)} nodes"
            + (f" (PIDs: {sorted(active_pids)})" if active_pids else " (all files)")
        )
        return records


# Set poll_interval from config at import time (consistent with other collectors)
from app.core.config import settings as _settings
InspectorCollector.poll_interval = _settings.rccl.inspector.poll_interval
