"""
RCCL Collector -- CVS cluster-mon Phase 1.

Implements BaseCollector. Polls ncclras via SSH port-forward on each cycle.
Lifecycle managed by the unified REGISTERED_COLLECTORS loop in main.py.

Critical=False: RCCLJobState.NO_JOB is expected when no RCCL job is running
and does NOT count as a collector failure for overall_status purposes.
"""

import asyncio
import logging
from typing import Optional, Any

from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.collectors.rccl_ras_client import RCCLRasClient, ProtocolError
from app.models.rccl_models import RCCLJobState, RCCLSnapshot

logger = logging.getLogger(__name__)


class RCCLCollector(BaseCollector):
    """
    Polls ncclras (the RCCL RAS TCP service on port 28028) via SSH port-forward.

    - name = "rccl"
    - poll_interval: read from settings.rccl.poll_interval (default 30s)
    - collect_timeout: collective_timeout_secs + 10s for SSH + protocol overhead
    - critical = False (NO_JOB is expected; not a system failure)
    """

    name = "rccl"
    poll_interval: int = 30         # overridden at module level from settings
    collect_timeout: float = 20.0   # overridden at module level from settings
    critical = False

    def __init__(self):
        self.job_state: RCCLJobState = RCCLJobState.NO_JOB
        self._app_state: Optional[Any] = None  # set in run() before collect()

    def _pick_leader(self, app_state: Any) -> Optional[str]:
        """
        Return the first node with healthy status from app_state.node_health_status.
        The existing host probe already maintains this dict -- no extra reachability
        check needed. Returns None if all nodes are unhealthy/unreachable.
        """
        for node, status in app_state.node_health_status.items():
            if status == "healthy":
                return node
        return None

    def _health_from_snapshot(self, snapshot: RCCLSnapshot) -> RCCLJobState:
        """Derive job state from snapshot: HEALTHY if no missing ranks, DEGRADED otherwise."""
        if not snapshot.communicators:
            return RCCLJobState.HEALTHY
        for comm in snapshot.communicators:
            if comm.missing_ranks > 0:
                return RCCLJobState.DEGRADED
        return RCCLJobState.HEALTHY

    async def run(self, ssh_manager, app_state: Any) -> None:
        """
        Override BaseCollector.run() to pass app_state to collect().
        Stores app_state reference so _pick_leader() and data_store are accessible.
        """
        self._app_state = app_state
        self._ssh_manager = ssh_manager
        await super().run(ssh_manager, app_state)

    async def collect(self, ssh_manager) -> CollectorResult:
        """
        One RCCL poll cycle:
        1. Pick one healthy node as 'leader' (VERBOSE STATUS -> triggers RAS_COLL_COMMS).
        2. Open SSH port-forward to leader:ras_port.
        3. Run ncclras protocol: handshake -> set_timeout -> get_status(verbose=True).
        4. Parse response -> RCCLSnapshot -> store in Redis and app_state.
        5. Return CollectorResult.

        Connection refused -> NO_JOB (expected state, not an error).
        Timeout -> UNREACHABLE.
        Protocol error -> ERROR.
        """
        from app.core.config import settings

        app_state = self._app_state
        if app_state is None:
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.ERROR,
                data={},
                error="collect() called before run() -- app_state not set",
            )

        leader = self._pick_leader(app_state)
        if leader is None:
            self.job_state = RCCLJobState.UNREACHABLE
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.UNREACHABLE,
                data={},
                error="No healthy nodes available for RCCL leader",
            )

        ras_port = settings.rccl.ras_port
        collective_timeout = settings.rccl.collective_timeout_secs

        try:
            async with ssh_manager.open_port_forward(leader, ras_port) as (reader, writer):
                client = RCCLRasClient(reader, writer)
                await client.handshake()
                await client.set_timeout(collective_timeout)
                raw_text = await client.get_status(verbose=True)
                # Connection closed automatically when context exits

            # Phase 1: text parser (placeholder -- returns minimal snapshot)
            snapshot = self._parse_text_response(raw_text, leader)
            self.job_state = self._health_from_snapshot(snapshot)
            snapshot_dict = snapshot.model_dump()

            # Store in Redis and app_state (non-blocking, degrades gracefully)
            if hasattr(app_state, 'rccl_data_store') and app_state.rccl_data_store:
                await app_state.rccl_data_store.push_snapshot(snapshot_dict)
            if hasattr(app_state, 'latest_rccl_snapshot'):
                app_state.latest_rccl_snapshot = snapshot_dict

            # Broadcast RCCL snapshot to WebSocket clients
            try:
                from app.main import broadcast_rccl
                await broadcast_rccl(snapshot_dict)
            except Exception:
                pass

            collector_state = (
                CollectorState.OK
                if self.job_state in (RCCLJobState.HEALTHY, RCCLJobState.DEGRADED)
                else CollectorState.NO_SERVICE
            )
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=collector_state,
                data=snapshot_dict,
            )

        except ConnectionRefusedError:
            self.job_state = RCCLJobState.NO_JOB
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.NO_SERVICE,
                data={},
                error=f"Port {ras_port} not listening on {leader} -- no RCCL job running",
            )
        except asyncio.TimeoutError:
            self.job_state = RCCLJobState.UNREACHABLE
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.UNREACHABLE,
                data={},
                error=f"RAS collective timed out on {leader}",
            )
        except ProtocolError as e:
            self.job_state = RCCLJobState.ERROR
            logger.error(f"RAS protocol error on {leader}: {e}")
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.ERROR,
                data={},
                error=str(e),
            )
        except Exception as e:
            self.job_state = RCCLJobState.ERROR
            logger.error(f"RCCL collect() unexpected error on {leader}: {e}", exc_info=True)
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.ERROR,
                data={},
                error=str(e),
            )

    def _parse_text_response(self, raw_text: str, leader: str) -> RCCLSnapshot:
        """
        Phase 1 text parser stub.

        The full parser must be written test-first against captured ncclras fixture
        files (see plan Section 4a). This stub returns a minimal snapshot indicating
        the job is running so the HEALTHY/DEGRADED state machine works end-to-end.

        TODO: Replace with regex-based parser against real ncclras VERBOSE STATUS output.
        Run: ncclras -v > tests/fixtures/rccl_verbose_status_healthy.txt
        """
        import time
        from app.models.rccl_models import RCCLSnapshot, RCCLJobState

        # Detect obvious failure patterns from raw text
        if not raw_text.strip():
            return RCCLSnapshot.empty(state=RCCLJobState.NO_JOB)

        # Return minimal healthy snapshot -- communicators empty until full parser is built
        return RCCLSnapshot(
            timestamp=time.time(),
            state=RCCLJobState.HEALTHY,
            communicators=[],
            peers=[],
            dead_peers=[],
        )


try:
    from app.core.config import settings as _settings
    RCCLCollector.poll_interval = _settings.rccl.poll_interval
    RCCLCollector.collect_timeout = _settings.rccl.collective_timeout_secs + 10
except Exception:
    pass  # use class defaults
