"""
RCCL Collector -- CVS cluster-mon Phase 1.

Implements BaseCollector. Polls rcclras via SSH port-forward on each cycle.
Lifecycle managed by the unified REGISTERED_COLLECTORS loop in main.py.

Critical=False: RCCLJobState.NO_JOB is expected when no RCCL job is running
and does NOT count as a collector failure for overall_status purposes.
"""

import asyncio
import logging
import time
from typing import Optional, Any

from paramiko.ssh_exception import ChannelException

from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.collectors.rccl_ras_client import RCCLRasClient, ProtocolError
from app.models.rccl_models import RCCLJobState, RCCLSnapshot

logger = logging.getLogger(__name__)


class RCCLCollector(BaseCollector):
    """
    Polls rcclras (the RCCL RAS TCP service on port 28028) via SSH port-forward.

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

    def _healthy_nodes(self, app_state: Any) -> list[str]:
        """
        Return all nodes with healthy status from app_state.node_health_status,
        in the order they appear in the config. The rcclras listener (port 28028)
        only runs on nodes that are part of an active RCCL job, which may be any
        subset of the configured nodes — so we must try each one.
        """
        return [
            node
            for node, status in app_state.node_health_status.items()
            if status == "healthy"
        ]

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
        3. Run rcclras protocol: handshake -> set_timeout -> get_status(verbose=True).
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

        candidates = self._healthy_nodes(app_state)
        if not candidates:
            self.job_state = RCCLJobState.UNREACHABLE
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.UNREACHABLE,
                data={},
                error="No healthy nodes available for RCCL polling",
            )

        ras_port = settings.rccl.ras_port
        collective_timeout = settings.rccl.collective_timeout_secs

        # Try each healthy node — rcclras only listens on nodes that are part of
        # an active RCCL job, which may be a subset of the configured nodes.
        for leader in candidates:
            try:
                async with ssh_manager.open_port_forward(leader, ras_port) as (reader, writer):
                    client = RCCLRasClient(reader, writer)
                    await client.handshake()
                    await client.set_timeout(collective_timeout)
                    raw_text = await client.get_status(verbose=True)

                snapshot = self._parse_text_response(raw_text, leader)
                self.job_state = self._health_from_snapshot(snapshot)
                snapshot_dict = snapshot.model_dump()

                if hasattr(app_state, 'rccl_data_store') and app_state.rccl_data_store:
                    await app_state.rccl_data_store.push_snapshot(snapshot_dict)
                if hasattr(app_state, 'latest_rccl_snapshot'):
                    app_state.latest_rccl_snapshot = snapshot_dict

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

            except (ConnectionRefusedError, ChannelException):
                # Port 28028 closed on this node — no RCCL job here, try next.
                logger.debug(f"No rcclras listener on {leader}:{ras_port}, trying next node")
                continue
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

        # All healthy nodes tried — no rcclras listener found anywhere.
        self.job_state = RCCLJobState.NO_JOB
        return CollectorResult(
            collector_name=self.name,
            timestamp=CollectorResult.now_iso(),
            state=CollectorState.NO_SERVICE,
            data={},
            error=f"Port {ras_port} not listening on any healthy node -- no RCCL job running",
        )

    def _parse_text_response(self, raw_text: str, leader: str) -> RCCLSnapshot:
        """Parse rcclras VERBOSE STATUS text output using RCCLTextParser."""
        from app.collectors.rccl_text_parser import RCCLTextParser
        return RCCLTextParser().parse(raw_text)

    async def _push_state_event(
        self,
        prev: RCCLJobState,
        curr: RCCLJobState,
        app_state: Any,
        leader: Optional[str] = None,
    ) -> None:
        """Push a state_change event when job_state transitions between polls."""
        if prev == curr:
            return
        data_store = getattr(app_state, 'rccl_data_store', None)
        if not data_store:
            return

        _TYPE_MAP = {
            (RCCLJobState.NO_JOB,      RCCLJobState.HEALTHY):    "job_start",
            (RCCLJobState.NO_JOB,      RCCLJobState.DEGRADED):   "job_start_degraded",
            (RCCLJobState.HEALTHY,     RCCLJobState.DEGRADED):   "job_degraded",
            (RCCLJobState.DEGRADED,    RCCLJobState.HEALTHY):    "job_recovered",
            (RCCLJobState.HEALTHY,     RCCLJobState.NO_JOB):     "job_end",
            (RCCLJobState.DEGRADED,    RCCLJobState.NO_JOB):     "job_end",
            (RCCLJobState.HEALTHY,     RCCLJobState.UNREACHABLE):"node_unreachable",
            (RCCLJobState.DEGRADED,    RCCLJobState.UNREACHABLE):"node_unreachable",
            (RCCLJobState.HEALTHY,     RCCLJobState.ERROR):      "collector_error",
            (RCCLJobState.DEGRADED,    RCCLJobState.ERROR):      "collector_error",
        }
        event_type = _TYPE_MAP.get((prev, curr), "state_change")

        await data_store.push_event({
            "event_type": event_type,
            "timestamp": time.time(),
            "from_state": prev,
            "to_state": curr,
            "leader_node": leader,
        })
        logger.info(f"RCCL state transition: {prev} → {curr} (event: {event_type})")


try:
    from app.core.config import settings as _settings
    RCCLCollector.poll_interval = _settings.rccl.poll_interval
    RCCLCollector.collect_timeout = _settings.rccl.collective_timeout_secs + 10
except Exception:
    pass  # use class defaults
