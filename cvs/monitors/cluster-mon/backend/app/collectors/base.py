"""
BaseCollector ABC and supporting types for CVS cluster-mon collectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class CollectorState(str, Enum):
    OK = "ok"
    NO_SERVICE = "no_service"  # Service not running (e.g. no RCCL job)
    UNREACHABLE = "unreachable"  # SSH/TCP timeout — node down
    ERROR = "error"  # Parse or protocol failure


@dataclass
class CollectorResult:
    collector_name: str
    timestamp: str  # ISO-8601 UTC string
    state: CollectorState
    data: dict[str, Any]
    error: Optional[str] = None
    node_errors: dict[str, bool] = field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


class BaseCollector(ABC):
    name: str  # class-level attribute — set on each subclass
    poll_interval: int  # seconds between collection cycles
    collect_timeout: float  # max seconds per collect() call
    critical: bool = False  # if True, failures affect overall_status as "critical"

    @abstractmethod
    async def collect(self, ssh_manager) -> CollectorResult:
        """
        One collection cycle. Must NOT raise — all errors go into CollectorResult.
        ssh_manager is Union[Pssh, JumpHostPssh].
        Must call ssh_manager.exec_async() (not exec()) to avoid blocking the event loop.
        """
        ...

    async def run(self, ssh_manager, app_state: Any) -> None:
        """
        Default task body. Loops until app_state.is_collecting is False.
        Wraps collect() in asyncio.wait_for to enforce collect_timeout.
        Subclasses with non-poll lifecycles (e.g. RCCL monitor mode) override this.
        """
        while app_state.is_collecting:
            try:
                result = await asyncio.wait_for(
                    self.collect(ssh_manager),
                    timeout=self.collect_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"{self.name} collect() timed out after {self.collect_timeout}s"
                )
                result = CollectorResult(
                    collector_name=self.name,
                    timestamp=CollectorResult.now_iso(),
                    state=CollectorState.ERROR,
                    data={},
                    error=f"collect() timed out after {self.collect_timeout}s",
                )
            except asyncio.CancelledError:
                raise
            except ConnectionError as e:
                logger.error(f"{self.name} collector ConnectionError: {e}")
                if hasattr(app_state, 'probe_requested') and app_state.probe_requested:
                    app_state.probe_requested.set()
                result = CollectorResult(
                    collector_name=self.name,
                    timestamp=CollectorResult.now_iso(),
                    state=CollectorState.UNREACHABLE,
                    data={},
                    error=str(e),
                )
            except Exception as e:
                logger.error(
                    f"{self.name} collector unexpected error: {e}", exc_info=True
                )
                result = CollectorResult(
                    collector_name=self.name,
                    timestamp=CollectorResult.now_iso(),
                    state=CollectorState.ERROR,
                    data={},
                    error=str(e),
                )

            # Store per-collector result
            if hasattr(app_state, 'collector_results'):
                app_state.collector_results[self.name] = result

            # Node health update — only GPU collector populates node_errors
            if hasattr(app_state, 'node_health_status'):
                for node, has_error in result.node_errors.items():
                    _update_node_status_via_app_state(app_state, node, has_error)

            # Update latest_metrics for WebSocket broadcast
            if hasattr(app_state, 'latest_metrics'):
                app_state.latest_metrics[self.name] = result.data
                # Shared timestamp key: last-writer-wins across collectors.
                # This preserves the existing WebSocket contract
                # {"gpu": ..., "nic": ..., "timestamp": "..."}.
                # Clients needing per-collector timestamps should use
                # GET /api/collectors/status instead.
                app_state.latest_metrics["timestamp"] = result.timestamp

            # Broadcast (imported lazily to avoid circular imports)
            try:
                from app.main import broadcast_metrics
                await broadcast_metrics(app_state.latest_metrics)
            except Exception as e:
                logger.debug(f"broadcast_metrics not available: {e}")

            await asyncio.sleep(self.poll_interval)


def _update_node_status_via_app_state(app_state: Any, node: str, has_error: bool) -> None:
    """Update node health status via app_state. Avoids circular import from main."""
    try:
        from app.main import update_node_status
        update_node_status(node, has_error, "unreachable")
    except Exception as e:
        logger.debug(f"update_node_status not available: {e}")
