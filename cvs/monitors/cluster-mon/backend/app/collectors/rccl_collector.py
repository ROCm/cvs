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
        self._bootstrapped: bool = False  # True after first poll seeds job_state from store

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
        """Return the job state already computed by the parser (covers missing ranks,
        async errors, dead peers, and inconsistent topology)."""
        return snapshot.state

    async def _bootstrap_job_state(self, app_state: Any) -> None:
        """
        On the first poll after startup, seed job_state from the data store's last
        known snapshot state. This prevents a spurious job_start event when the
        backend restarts mid-job.
        """
        if self._bootstrapped:
            return
        self._bootstrapped = True
        data_store = getattr(app_state, 'rccl_data_store', None)
        if not data_store:
            return
        try:
            last = await data_store.get_current_snapshot()
            if last and 'state' in last:
                self.job_state = RCCLJobState(last['state'])
                logger.info(f"RCCL collector bootstrapped from stored state: {self.job_state}")
        except (ValueError, Exception):
            pass  # Unknown state value or store error — start from NO_JOB

    async def on_collect_timeout(self, app_state: Any) -> None:
        """
        Called by BaseCollector.run() when collect() is cancelled by collect_timeout.
        Updates the state machine so the timeout is visible as an UNREACHABLE transition.
        """
        prev = self.job_state
        self.job_state = RCCLJobState.UNREACHABLE
        await self._push_state_event(prev, self.job_state, app_state)
        if hasattr(app_state, 'latest_rccl_snapshot'):
            app_state.latest_rccl_snapshot = {"state": "unreachable"}

    async def run(self, ssh_manager, app_state: Any) -> None:
        """
        Override BaseCollector.run() to pass app_state to collect().
        Stores app_state reference so _healthy_nodes() and data_store are accessible.
        """
        self._app_state = app_state
        self._ssh_manager = ssh_manager
        await super().run(ssh_manager, app_state)

    async def collect(self, ssh_manager) -> CollectorResult:
        """
        One RCCL poll cycle:
        1. Bootstrap job_state from data store on first poll (prevents spurious job_start).
        2. Try each healthy node for an active rcclras listener on ras_port.
        3. On ConnectionRefused/ChannelException: continue to next node (not our job).
        4. On TimeoutError: continue to next node (may be transient SSH delay).
        5. On ProtocolError/Exception: abort cycle with ERROR.
        6. If no node has a listener: NO_JOB.

        Connection refused -> try next node -> NO_JOB if none respond.
        Timeout -> try next node -> UNREACHABLE only if ALL candidates time out.
        Protocol error -> ERROR (abort immediately; protocol errors are not transient).
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

        await self._bootstrap_job_state(app_state)

        prev_state = self.job_state

        candidates = self._healthy_nodes(app_state)
        if not candidates:
            self.job_state = RCCLJobState.UNREACHABLE
            await self._push_state_event(prev_state, self.job_state, app_state)
            if hasattr(app_state, 'latest_rccl_snapshot'):
                app_state.latest_rccl_snapshot = {"state": "unreachable"}
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.UNREACHABLE,
                data={},
                error="No healthy nodes available for RCCL polling",
            )

        ras_port = settings.rccl.ras_port
        collective_timeout = settings.rccl.collective_timeout_secs

        timed_out_nodes: list[str] = []

        # Try each healthy node — rcclras only listens on nodes that are part of
        # an active RCCL job, which may be a subset of the configured nodes.
        for leader in candidates:
            try:
                async with ssh_manager.open_port_forward(leader, ras_port) as (reader, writer):
                    client = RCCLRasClient(reader, writer)
                    await client.handshake()
                    await client.set_timeout(collective_timeout)
                    raw_text = await client.get_status(verbose=True)
                    logger.debug(f"rcclras raw output from {leader}:\n{raw_text}")

                snapshot = self._parse_text_response(raw_text, leader)
                self.job_state = self._health_from_snapshot(snapshot)
                await self._push_state_event(prev_state, self.job_state, app_state, leader)
                snapshot_dict = snapshot.model_dump()

                if hasattr(app_state, 'rccl_data_store') and app_state.rccl_data_store:
                    await app_state.rccl_data_store.push_snapshot(snapshot_dict)
                if hasattr(app_state, 'latest_rccl_snapshot'):
                    app_state.latest_rccl_snapshot = snapshot_dict

                try:
                    from app.main import broadcast_rccl
                    await broadcast_rccl(snapshot_dict)
                except Exception as e:
                    logger.warning(f"broadcast_rccl failed (snapshot not sent to WebSocket clients): {e}")

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
                # SSH/protocol timeout on this node — could be transient, try remaining nodes.
                logger.debug(f"Timeout on {leader}:{ras_port}, trying next node")
                timed_out_nodes.append(leader)
                continue

            except ProtocolError as e:
                # Protocol errors are not transient — abort immediately.
                prev = self.job_state
                self.job_state = RCCLJobState.ERROR
                await self._push_state_event(prev, self.job_state, app_state, leader)
                logger.error(f"RAS protocol error on {leader}: {e}")
                return CollectorResult(
                    collector_name=self.name,
                    timestamp=CollectorResult.now_iso(),
                    state=CollectorState.ERROR,
                    data={},
                    error=str(e),
                )

            except Exception as e:
                prev = self.job_state
                self.job_state = RCCLJobState.ERROR
                await self._push_state_event(prev, self.job_state, app_state, leader)
                logger.error(f"RCCL collect() unexpected error on {leader}: {e}", exc_info=True)
                return CollectorResult(
                    collector_name=self.name,
                    timestamp=CollectorResult.now_iso(),
                    state=CollectorState.ERROR,
                    data={},
                    error=str(e),
                )

        # All candidates tried. If some timed out and none responded, mark UNREACHABLE.
        if timed_out_nodes:
            prev = self.job_state
            self.job_state = RCCLJobState.UNREACHABLE
            await self._push_state_event(prev, self.job_state, app_state)
            if hasattr(app_state, 'latest_rccl_snapshot'):
                app_state.latest_rccl_snapshot = {"state": "unreachable"}
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.UNREACHABLE,
                data={},
                error=f"RAS collective timed out on all nodes: {timed_out_nodes}",
            )

        # All healthy nodes tried — no rcclras listener found anywhere.
        self.job_state = RCCLJobState.NO_JOB
        await self._push_state_event(prev_state, self.job_state, app_state)
        if hasattr(app_state, 'latest_rccl_snapshot'):
            app_state.latest_rccl_snapshot = {"state": "no_job"}
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
            (RCCLJobState.NO_JOB,         RCCLJobState.HEALTHY):      "job_start",
            (RCCLJobState.NO_JOB,         RCCLJobState.DEGRADED):     "job_start_degraded",
            (RCCLJobState.NO_JOB,         RCCLJobState.UNREACHABLE):  "nodes_unreachable",
            (RCCLJobState.NO_JOB,         RCCLJobState.ERROR):        "collector_error",
            (RCCLJobState.HEALTHY,        RCCLJobState.DEGRADED):     "job_degraded",
            (RCCLJobState.HEALTHY,        RCCLJobState.NO_JOB):       "job_end",
            (RCCLJobState.HEALTHY,        RCCLJobState.UNREACHABLE):  "node_unreachable",
            (RCCLJobState.HEALTHY,        RCCLJobState.ERROR):        "collector_error",
            (RCCLJobState.DEGRADED,       RCCLJobState.HEALTHY):      "job_recovered",
            (RCCLJobState.DEGRADED,       RCCLJobState.NO_JOB):       "job_end",
            (RCCLJobState.DEGRADED,       RCCLJobState.UNREACHABLE):  "node_unreachable",
            (RCCLJobState.DEGRADED,       RCCLJobState.ERROR):        "collector_error",
            (RCCLJobState.UNREACHABLE,    RCCLJobState.HEALTHY):      "node_recovered",
            (RCCLJobState.UNREACHABLE,    RCCLJobState.DEGRADED):     "node_recovered_degraded",
            (RCCLJobState.UNREACHABLE,    RCCLJobState.NO_JOB):       "job_end",
            (RCCLJobState.UNREACHABLE,    RCCLJobState.ERROR):        "collector_error",
            (RCCLJobState.ERROR,          RCCLJobState.HEALTHY):      "job_start",
            (RCCLJobState.ERROR,          RCCLJobState.DEGRADED):     "job_start_degraded",
            (RCCLJobState.ERROR,          RCCLJobState.NO_JOB):       "job_end",
            (RCCLJobState.ERROR,          RCCLJobState.UNREACHABLE):  "node_unreachable",
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


# Fail loudly if settings cannot be loaded — consistent with GPU/NIC collectors.
from app.core.config import settings as _settings
RCCLCollector.poll_interval = _settings.rccl.poll_interval
RCCLCollector.collect_timeout = _settings.rccl.collective_timeout_secs + 10
