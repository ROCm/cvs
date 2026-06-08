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

import logging
from pathlib import Path

from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.collectors.inspector_parser import InspectorParser, aggregate_snapshot
from app.core.config import settings as _settings
from app.models.rccl_models import InspectorCollPerf

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
    collect_timeout: float = 25.0  # overridden at module level from settings
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

        # Secondary oracle: if any record carries v5 fields, mark per-node capabilities.
        # This lets RCCLCollector cross-check: json_ras=False + inspector_v5=True → mismatch.
        self._apply_inspector_v5_oracle(records, app_state)

        return CollectorResult(
            collector_name=self.name,
            timestamp=CollectorResult.now_iso(),
            state=CollectorState.OK,
            data=snapshot_dict,
        )

    # ------------------------------------------------------------------
    # Inspector v5 secondary oracle
    # ------------------------------------------------------------------

    def _apply_inspector_v5_oracle(self, records: list[InspectorCollPerf], app_state) -> None:
        """
        Mark node_capabilities.inspector_v5=True for any hostname whose Inspector
        records carry v5 fields (graph_captured is not None).

        If a node is marked inspector_v5=True but its NodeRCCLCapability has
        json_ras=False, that is a warning-level inconsistency: Inspector reports
        v5 fields but the RAS probe returned text-only output.
        Likely cause: node upgrade in progress or RAS port bound before upgrade.
        """
        node_capabilities = getattr(app_state, 'node_capabilities', {})
        v5_hosts: set[str] = {r.hostname for r in records if r.graph_captured is not None}

        for hostname in v5_hosts:
            cap = node_capabilities.get(hostname)
            if cap is None:
                continue
            if not cap.inspector_v5:
                cap.inspector_v5 = True
                logger.info(f"Inspector oracle: {hostname} has Inspector v5 fields (graphCaptured present)")
            # Cross-check: Inspector says v5 but RAS is still text-mode
            if not cap.json_ras:
                logger.warning(
                    f"Inspector oracle: {hostname} shows v5 Inspector but json_ras=False — "
                    "node may be mid-upgrade or RAS probe needs refresh"
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

        # __INSP_EOF__ sentinel: appended to every SSH command so pssh always gets
        # at least one line of output. Without it, libssh2 waits the full read_timeout
        # when tail produces no output (no matching files), because it polls for channel
        # EOF rather than detecting an empty stdout stream. The sentinel guarantees at
        # least one byte of output so the channel drains immediately.
        _SENTINEL = "__INSP_EOF__"

        # Inspector log filenames encode the writing host: <hostname>-pid<PID>.log
        # When dump_dir is on shared NFS all 16 files are visible from every node.
        # Scope each node's tail to files whose name starts with that node's own
        # hostname so we read each file exactly once and avoid cross-node duplication.
        # The ${HOSTNAME} shell variable is expanded on the remote node at runtime.

        if active_pids:
            # Filter to current-job PIDs AND current-node hostname.
            pid_pattern = "|".join(f"pid{p}" for p in sorted(active_pids))
            cmd = (
                f"ls {cfg.dump_dir}/ 2>/dev/null "
                f"| grep -E '^${{HOSTNAME}}-({pid_pattern})\\.log$' "
                f"| xargs -I{{}} tail -n {cfg.max_records_per_file} {cfg.dump_dir}/{{}} "
                f"2>/dev/null; echo {_SENTINEL}"
            )
            logger.debug(f"Inspector SSH: filtering to {len(active_pids)} active PIDs on ${{HOSTNAME}}")
        else:
            # No rcclras snapshot — read all files belonging to this node.
            cmd = (
                f"tail -n {cfg.max_records_per_file} "
                f"{cfg.dump_dir}/${{HOSTNAME}}-pid*.log 2>/dev/null; echo {_SENTINEL}"
            )
            logger.debug("Inspector SSH: no active PIDs known, reading all local-node log files")

        # Pass collect_timeout as read_timeout so pssh doesn't block beyond the window.
        ssh_timeout = max(5.0, InspectorCollector.collect_timeout - 2.0)
        try:
            outputs = await ssh_manager.exec_async(cmd, timeout=ssh_timeout)
        except Exception as e:
            logger.warning(f"Inspector SSH exec failed: {e}")
            return []

        records: list[InspectorCollPerf] = []
        for node, output in outputs.items():
            if not output:
                continue
            # Strip sentinel and tail's multi-file headers (==> filename <==)
            # before parsing — neither is a JSONL record.
            clean = "\n".join(
                line for line in output.splitlines() if line.strip() != _SENTINEL and not line.startswith("==>")
            )
            if not clean.strip():
                continue
            node_records = self._parser.parse_lines(clean)
            records.extend(node_records)
            logger.debug(f"Inspector SSH: {len(node_records)} records from {node}")

        logger.info(
            f"Inspector SSH mode: {len(records)} records from {len(outputs)} nodes"
            + (f" (PIDs: {sorted(active_pids)})" if active_pids else " (all files)")
        )
        return records


# Set poll_interval and collect_timeout from config at import time.
# collect_timeout = max(15s, poll_interval * 0.8): allows most of the poll window
# for SSH exec to complete. Minimum 15s — SSH tail across 2+ nodes needs headroom.
# NOTE: poll_interval < 20s is not recommended for SSH mode; file mode can go lower.
InspectorCollector.poll_interval = _settings.rccl.inspector.poll_interval
InspectorCollector.collect_timeout = max(15.0, _settings.rccl.inspector.poll_interval * 0.8)
