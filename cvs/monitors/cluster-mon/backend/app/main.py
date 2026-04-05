"""
Main FastAPI application for CVS Cluster Monitor.
"""

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Union, Optional
import os
import time
from pathlib import Path

from app.core.config import settings
from app.core.cvs_parallel_ssh_reliable import Pssh
from app.core.jump_host_pssh import JumpHostPssh
from app.collectors.gpu_collector import GPUMetricsCollector
from app.collectors.nic_collector import NICMetricsCollector
from app.collectors.rccl_collector import RCCLCollector
from app.collectors.inspector_collector import InspectorCollector
from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.api import router as api_router

import redis.asyncio as aioredis

# Configure logging based on DEBUG environment variable
# Using RotatingFileHandler for circular log files with 1MB max size
DEBUG_MODE = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend.log")
rotating_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=1024 * 1024,  # 1MB
    backupCount=3,  # Keep 3 backup files (backend.log.1, backend.log.2, backend.log.3)
)
rotating_handler.setLevel(LOG_LEVEL)
rotating_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Also keep console output
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL,
    handlers=[rotating_handler, console_handler],
)

# Suppress verbose logging from parallel-ssh library unless in DEBUG mode
if not DEBUG_MODE:
    logging.getLogger("pssh").setLevel(logging.WARNING)
    logging.getLogger("pssh.host_logger").setLevel(logging.WARNING)
    logging.getLogger("pssh.clients.base.parallel").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized - DEBUG_MODE: {DEBUG_MODE}, LOG_LEVEL: {logging.getLevelName(LOG_LEVEL)}")


# Global state
class AppState:
    """Global application state."""

    def __init__(self):
        # SSH manager
        self.ssh_manager: Optional[Union[Pssh, JumpHostPssh]] = None

        # Unified collector registry (BaseCollector pattern)
        self.collectors: dict[str, BaseCollector] = {}
        self.collector_tasks: dict[str, asyncio.Task] = {}
        self.collector_results: dict[str, CollectorResult] = {}

        # Legacy fields kept for backward compat during transition
        self.gpu_collector: GPUMetricsCollector = None
        self.nic_collector: NICMetricsCollector = None
        self.collection_task: asyncio.Task = None  # deprecated

        self.latest_metrics: dict = {}
        self.websocket_clients: List[WebSocket] = []
        self.is_collecting: bool = False

        # Node health tracking
        self.node_failure_count: dict = {}
        self.node_health_status: dict = {}

        # Software info cache
        self.cached_gpu_software: dict = {}
        self.cached_nic_software: dict = {}
        self.cached_nic_advanced: dict = {}
        self.gpu_software_cache_time: float = 0
        self.nic_software_cache_time: float = 0
        self.nic_advanced_cache_time: float = 0
        self.software_cache_ttl: int = 180

        # SECURITY: Passwords stored in memory only
        self.ssh_password: str = None
        self.jump_host_password: str = None

        # Periodic host probe
        self.probe_task: Optional[asyncio.Task] = None
        self.last_probe_time: Optional[float] = None
        self.probe_count: int = 0
        self.probe_requested: asyncio.Event = None  # set by collectors on ConnectionError

        # Redis client
        self.redis: Optional[object] = None

        # RCCL state
        self.rccl_data_store = None   # RCCLDataStore, set in lifespan
        self.latest_rccl_snapshot: Optional[dict] = None
        self.rccl_websocket_clients: List[WebSocket] = []


app_state = AppState()

_reload_lock = asyncio.Lock()


# SSH Transport Scaling Note:
# The SSH-based collection transport has a practical limit of ~500-800 nodes at
# 60-second poll intervals. Known constraints at 600 nodes: 3-5GB RSS, pool_size
# reduced to 50, global threading lock serializes SSH batches. For clusters
# significantly larger, consider deploying lightweight push agents (Telegraf
# amd_rocm_smi plugin or rocm-smi-exporter) on compute nodes.

REGISTERED_COLLECTORS: list[type[BaseCollector]] = [
    GPUMetricsCollector,
    NICMetricsCollector,
    RCCLCollector,
    InspectorCollector,
]


def _start_collector_task(c: BaseCollector) -> asyncio.Task:
    """Create a supervised collector task that restarts on crash with exponential backoff."""
    _backoff = [1.0]  # mutable cell for closure

    def _on_done(task: asyncio.Task) -> None:
        if task.cancelled() or not app_state.is_collecting:
            return
        exc = task.exception()
        if exc is None:
            logger.warning(f"Collector {c.name} task exited unexpectedly — restarting")
        else:
            delay = _backoff[0]
            logger.error(
                f"Collector {c.name} crashed: {exc!r} — restarting in {delay:.0f}s",
                exc_info=exc,
            )

        async def _restart():
            await asyncio.sleep(_backoff[0])
            _backoff[0] = min(_backoff[0] * 2, 120)
            new_task = _start_collector_task(c)
            app_state.collector_tasks[c.name] = new_task

        def _schedule_restart():
            restart_task = asyncio.create_task(_restart(), name=f"restart-{c.name}")
            app_state.collector_tasks[f"_restart_{c.name}"] = restart_task

        asyncio.get_running_loop().call_soon(_schedule_restart)

    task = asyncio.create_task(
        c.run(app_state.ssh_manager, app_state),
        name=f"collector-{c.name}",
    )
    task.add_done_callback(_on_done)
    return task


async def reload_configuration():
    """
    Reload configuration without restarting the entire process.
    Uses topology-diff to restart only collectors whose config actually changed.

    Returns:
        dict: Status of reload operation with success/error details
    """
    async with _reload_lock:
        return await _reload_configuration_inner()


async def _reload_configuration_inner():
    try:
        logger.info("Starting configuration reload (topology-diff)...")

        # 1. Snapshot old settings and load new settings
        from app.core.config import Settings
        import app.core.config as config_module

        old_settings = config_module.settings
        new_config = Settings()  # re-reads YAML and env vars

        # 2. Determine which config sections changed
        ssh_changed = (
            old_settings.ssh.model_dump() != new_config.ssh.model_dump()
        )
        rccl_changed = (
            old_settings.rccl.model_dump() != new_config.rccl.model_dump()
        )
        polling_changed = (
            old_settings.polling.model_dump() != new_config.polling.model_dump()
        )

        logger.info(
            f"Config diff: ssh_changed={ssh_changed}, rccl_changed={rccl_changed}, "
            f"polling_changed={polling_changed}"
        )

        # 3. Update the global settings reference
        config_module.settings = new_config

        # 4. Determine which collectors need restart
        collectors_to_restart: set[str] = set()
        if polling_changed:
            # Interval changed — restart all polling collectors
            collectors_to_restart = {cls.name for cls in REGISTERED_COLLECTORS}
        else:
            if ssh_changed:
                collectors_to_restart.update({"gpu", "nic"})  # SSH-dependent
            if rccl_changed:
                collectors_to_restart.add("rccl")

        # 5. Load new nodes
        nodes = new_config.load_nodes_from_file()
        if not nodes:
            logger.warning("No nodes found in configuration after reload")
            return {"success": False, "error": "No nodes configured in nodes.txt", "nodes_count": 0}

        logger.info(f"Loaded {len(nodes)} nodes from configuration")

        # 6. Check if SSH keys exist (only if using key-based auth, not password)
        using_jump_password = new_config.ssh.jump_host.enabled and new_config.ssh.jump_host.password
        using_direct_password = not new_config.ssh.jump_host.enabled and app_state.ssh_password

        if not using_jump_password and not using_direct_password:
            # Using key-based auth - verify key exists
            key_file_path = (
                new_config.ssh.jump_host.key_file
                if (new_config.ssh.jump_host.enabled and new_config.ssh.jump_host.host)
                else new_config.ssh.key_file
            )
            key_file_expanded = os.path.expanduser(key_file_path) if key_file_path.startswith("~") else key_file_path

            logger.info(f"Checking for SSH key (key-based auth): {key_file_expanded}")
            if not os.path.exists(key_file_expanded):
                logger.warning(f"SSH key file not found: {key_file_expanded}")
                logger.warning("Please upload SSH keys via Configuration UI or run refresh-ssh-keys.sh")
                return {
                    "success": False,
                    "error": f"SSH key file not found: {key_file_path}. Please upload SSH keys via the Configuration UI.",
                    "nodes_count": len(nodes),
                    "requires_key_upload": True,
                }
            else:
                logger.info(f"SSH key file found: {key_file_expanded}")
                # List the key file to verify
                import subprocess

                try:
                    result = subprocess.run(['ls', '-l', key_file_expanded], capture_output=True, text=True)
                    logger.info(f"Key file details: {result.stdout.strip()}")
                except:
                    pass
        else:
            logger.info("Using password authentication - no key file check needed")

        # 7. Cancel only affected collector tasks
        for name in collectors_to_restart:
            task = app_state.collector_tasks.get(name)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            # Also cancel any pending restart tasks
            restart_key = f"_restart_{name}"
            restart_task = app_state.collector_tasks.get(restart_key)
            if restart_task:
                restart_task.cancel()
                try:
                    await restart_task
                except asyncio.CancelledError:
                    pass

        # If nothing changed, we can skip SSH and collector restart
        if not collectors_to_restart and not ssh_changed:
            logger.info("No config sections changed — nothing to restart")
            return {
                "success": True,
                "message": "Configuration reloaded (no changes detected)",
                "nodes_count": len(nodes),
                "jump_host_enabled": new_config.ssh.jump_host.enabled,
            }

        # 8. Recreate SSH manager only if SSH config changed
        if ssh_changed:
            # Stop probe task — it depends on the SSH manager
            if app_state.probe_task:
                app_state.probe_task.cancel()
                try:
                    await app_state.probe_task
                except asyncio.CancelledError:
                    pass

            if app_state.ssh_manager:
                logger.info("Closing existing SSH connections (ssh config changed)...")
                app_state.ssh_manager.destroy_clients()
                app_state.ssh_manager = None

            # Clear cached data (node topology may have changed)
            app_state.latest_metrics = {}
            app_state.node_failure_count = {}
            app_state.node_health_status = {}
            app_state.cached_gpu_software = {}
            app_state.cached_nic_software = {}
            app_state.cached_nic_advanced = {}
            app_state.gpu_software_cache_time = 0
            app_state.nic_software_cache_time = 0
            app_state.nic_advanced_cache_time = 0

            try:
                if new_config.ssh.jump_host.enabled and new_config.ssh.jump_host.host:
                    num_nodes = len(nodes)
                    min(num_nodes, 5)

                    logger.info(f"Reinitializing with jump host: {new_config.ssh.jump_host.host}")
                    logger.info(f"Jump Host Username: {new_config.ssh.jump_host.username}")
                    logger.info(f"Cluster Nodes: {len(nodes)} nodes")
                    logger.info(f"Cluster Username: {new_config.ssh.jump_host.node_username}")

                    # Use JumpHostPssh - working approach from test_auth_script.py
                    app_state.ssh_manager = JumpHostPssh(
                        jump_host=new_config.ssh.jump_host.host,
                        jump_user=new_config.ssh.jump_host.username,
                        jump_password=new_config.ssh.jump_host.password,
                        jump_pkey=new_config.ssh.jump_host.key_file if not new_config.ssh.jump_host.password else None,
                        target_hosts=nodes,
                        target_user=new_config.ssh.jump_host.node_username,
                        target_pkey=new_config.ssh.jump_host.node_key_file,
                        max_parallel=min(len(nodes), 5),  # Limit to 5 to avoid exhausting paramiko channels (conservative)
                        timeout=new_config.ssh.timeout,
                    )
                    logger.info("JumpHostPssh initialized successfully")
                else:
                    logger.info("Reinitializing with direct SSH (no jump host)")
                    logger.info(f"Username: {new_config.ssh.username}")
                    logger.info(f"Nodes: {len(nodes)} nodes")

                    app_state.ssh_manager = Pssh(
                        log=logger,
                        host_list=nodes,
                        user=new_config.ssh.username,
                        password=app_state.ssh_password,  # Use in-memory password
                        pkey=new_config.ssh.key_file,
                        timeout=new_config.ssh.timeout,
                        stop_on_errors=False,
                    )
                    logger.info("Direct SSH manager reinitialized")
            except Exception as e:
                logger.error(f"Failed to reinitialize SSH manager: {e}")
                return {"success": False, "error": f"Failed to initialize SSH manager: {str(e)}", "nodes_count": len(nodes)}

            # Restart probe task with new SSH manager
            app_state.probe_requested = asyncio.Event()
            app_state.probe_task = asyncio.create_task(periodic_host_probe())

        # 9. Restart only the affected collectors
        if app_state.ssh_manager and nodes:
            app_state.is_collecting = True
            for cls in REGISTERED_COLLECTORS:
                if cls.name in collectors_to_restart:
                    old_collector = app_state.collectors.get(cls.name)
                    c = cls()
                    # Transfer stateful fields so a config reload doesn't emit a
                    # spurious job_start event (new instance initialises to NO_JOB).
                    if old_collector is not None:
                        if hasattr(old_collector, 'job_state') and hasattr(c, 'job_state'):
                            c.job_state = old_collector.job_state
                        if hasattr(c, '_bootstrapped'):
                            c._bootstrapped = True  # skip bootstrap — state already known
                    app_state.collectors[c.name] = c
                    app_state.collector_tasks[c.name] = _start_collector_task(c)
                    logger.info(f"Restarted collector: {c.name}")
                else:
                    logger.info(f"Collector unchanged, kept running: {cls.name}")

        logger.info("Configuration reload completed successfully!")
        return {
            "success": True,
            "message": "Configuration reloaded successfully",
            "nodes_count": len(nodes),
            "jump_host_enabled": new_config.ssh.jump_host.enabled,
            "collectors_restarted": list(collectors_to_restart),
        }

    except Exception as e:
        logger.error(f"Error during configuration reload: {e}", exc_info=True)
        return {"success": False, "error": str(e), "nodes_count": 0}


def update_node_status(node: str, is_error: bool, error_type: str = 'unreachable'):
    """
    Update node status with stability check.
    Only marks node as unhealthy/unreachable after failure_threshold consecutive failures.

    Args:
        node: Node hostname
        is_error: True if current poll had an error
        error_type: 'unhealthy' or 'unreachable'
    """
    failure_threshold = settings.polling.failure_threshold

    # Initialize if not exists
    if node not in app_state.node_failure_count:
        app_state.node_failure_count[node] = 0
        app_state.node_health_status[node] = 'healthy'

    if is_error:
        # Increment failure counter
        app_state.node_failure_count[node] += 1

        # Only change status after consecutive failures exceed threshold
        if app_state.node_failure_count[node] >= failure_threshold:
            app_state.node_health_status[node] = error_type
            logger.warning(
                f"Node {node} marked as {error_type} after {app_state.node_failure_count[node]} consecutive failures"
            )
    else:
        # Success - reset counter and mark healthy
        if app_state.node_failure_count[node] > 0:
            logger.info(f"Node {node} recovered (was {app_state.node_failure_count[node]} failures)")
        app_state.node_failure_count[node] = 0
        app_state.node_health_status[node] = 'healthy'

    return app_state.node_health_status[node]


class ConnectionManager:
    """
    WebSocket connection manager with per-client bounded queues.
    Slow clients are disconnected instead of blocking the broadcast loop.
    """
    def __init__(self, max_queue_size: int = 64):
        self._clients: dict[int, WebSocket] = {}
        self._queues: dict[int, asyncio.Queue] = {}
        self._send_tasks: dict[int, asyncio.Task] = {}
        self._max_queue_size = max_queue_size
        self._closing: set[int] = set()  # guard against concurrent double-close

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        client_id = id(websocket)
        self._clients[client_id] = websocket
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._queues[client_id] = q
        self._send_tasks[client_id] = asyncio.create_task(
            self._sender(client_id, websocket, q)
        )

    async def _sender(self, client_id: int, ws: WebSocket, queue: asyncio.Queue):
        try:
            while True:
                message = await queue.get()
                await ws.send_json(message)
        except Exception as e:
            logger.debug(f"WebSocket sender error for client {client_id}: {e}")
        finally:
            await self._remove(client_id)

    async def disconnect(self, websocket: WebSocket):
        await self._remove(id(websocket))

    async def _remove(self, client_id: int):
        if client_id in self._closing:
            return
        self._closing.add(client_id)
        try:
            task = self._send_tasks.pop(client_id, None)
            if task and not task.done():
                task.cancel()
            self._queues.pop(client_id, None)
            ws = self._clients.pop(client_id, None)
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass
        finally:
            self._closing.discard(client_id)

    def broadcast(self, message: dict):
        """Non-blocking broadcast: enqueues to each client's queue."""
        to_remove = []
        for client_id, q in self._queues.items():
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(f"WebSocket client {client_id} queue full — disconnecting")
                to_remove.append(client_id)
        for client_id in to_remove:
            asyncio.create_task(self._remove(client_id))

    @property
    def client_count(self) -> int:
        return len(self._clients)


metrics_ws_manager = ConnectionManager()
rccl_ws_manager = ConnectionManager()


async def broadcast_metrics(metrics: dict):
    """Broadcast metrics to all connected WebSocket clients (non-blocking)."""
    metrics_ws_manager.broadcast({"type": "metrics", "data": metrics})


async def broadcast_rccl(snapshot: dict):
    """Broadcast RCCL snapshot to /ws/rccl WebSocket clients (non-blocking)."""
    rccl_ws_manager.broadcast({"type": "rccl_snapshot", "data": snapshot})


async def periodic_host_probe():
    """
    Periodically re-probe hosts every 5 minutes to detect changes.

    This background task runs continuously while metrics collection is active.
    It detects nodes that have come online or gone offline and updates the
    SSH client accordingly.
    """
    PROBE_INTERVAL = 300  # 5 minutes

    logger.info("Periodic host probe task started (every 5 minutes)")

    while app_state.is_collecting:
        try:
            try:
                await asyncio.wait_for(app_state.probe_requested.wait(), timeout=300)
                app_state.probe_requested.clear()
            except asyncio.TimeoutError:
                pass  # normal 5-minute periodic probe

            if not app_state.ssh_manager:
                logger.debug("Skipping periodic probe - no SSH manager")
                continue

            logger.info("Running periodic host probe...")

            # Get current lists before probe
            old_reachable = set(app_state.ssh_manager.reachable_hosts)
            old_unreachable = set(app_state.ssh_manager.unreachable_hosts)

            # Re-probe (run in executor to avoid blocking event loop)
            changed = await asyncio.to_thread(app_state.ssh_manager.refresh_host_reachability)

            new_reachable = set(app_state.ssh_manager.reachable_hosts)
            new_unreachable = set(app_state.ssh_manager.unreachable_hosts)
            logger.info(new_unreachable)

            # Check for changes
            newly_reachable = new_reachable - old_reachable
            newly_unreachable = old_unreachable - new_reachable

            # Increment probe counter
            app_state.probe_count += 1

            # Determine if client recreation is needed
            should_recreate = False
            recreation_reason = ""

            if newly_reachable or newly_unreachable:
                logger.info("Host reachability changed during periodic probe:")
                if newly_reachable:
                    logger.info(f"  Newly reachable: {newly_reachable}")
                if newly_unreachable:
                    logger.info(f"  Newly unreachable: {newly_unreachable}")
                should_recreate = changed
                recreation_reason = "reachability changed"
            elif app_state.probe_count % 12 == 0:
                # Force recreation every 12 probes (1 hour) to refresh stale connections
                should_recreate = True
                recreation_reason = f"periodic refresh (probe #{app_state.probe_count})"
                logger.info(f"Forcing SSH client recreation after {app_state.probe_count} probes (1 hour)")

            # Recreate client if needed
            if should_recreate:
                await asyncio.to_thread(app_state.ssh_manager.recreate_client)
                logger.info(f"✅ SSH client recreated - reason: {recreation_reason}")

            app_state.last_probe_time = time.time()
            logger.info(f"Periodic probe completed - next probe in {PROBE_INTERVAL} seconds")

        except asyncio.CancelledError:
            logger.info("Periodic probe task cancelled")
            raise
        except Exception as e:
            logger.error(f"Periodic probe failed: {e}", exc_info=True)
            # Continue running despite errors

    logger.info("Periodic host probe task stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting CVS Cluster Monitor")

    # Initialize probe_requested event (before collectors start)
    app_state.probe_requested = asyncio.Event()

    # Load nodes from file
    nodes = settings.load_nodes_from_file()

    # Also set legacy fields for backward-compat with existing API endpoints
    app_state.gpu_collector = GPUMetricsCollector()
    app_state.nic_collector = NICMetricsCollector()
    logger.info("Collectors initialized")

    if not nodes:
        logger.warning("No nodes configured! Please add nodes to config/nodes.txt")
        logger.info("Waiting for user to configure nodes via web UI...")
    else:
        logger.info(f"Configuration found: {len(nodes)} nodes")
        logger.info("Auto-initializing SSH manager on startup...")

        # Auto-initialize SSH manager if configuration exists
        try:
            if settings.ssh.jump_host.enabled and settings.ssh.jump_host.host:
                logger.info(f"Initializing with jump host: {settings.ssh.jump_host.host}")
                logger.info(f"Jump Host Username: {settings.ssh.jump_host.username}")
                logger.info(f"Cluster Nodes: {len(nodes)} nodes")
                logger.info(f"Cluster Username: {settings.ssh.jump_host.node_username}")

                app_state.ssh_manager = JumpHostPssh(
                    jump_host=settings.ssh.jump_host.host,
                    jump_user=settings.ssh.jump_host.username,
                    jump_password=settings.ssh.jump_host.password,
                    jump_pkey=settings.ssh.jump_host.key_file if not settings.ssh.jump_host.password else None,
                    target_hosts=nodes,
                    target_user=settings.ssh.jump_host.node_username,
                    target_pkey=settings.ssh.jump_host.node_key_file,
                    max_parallel=min(len(nodes), 5),
                    timeout=settings.ssh.timeout,
                )
                logger.info("✅ JumpHostPssh initialized successfully")
            else:
                logger.info("Initializing with direct SSH (no jump host)")
                logger.info(f"Username: {settings.ssh.username}")
                logger.info(f"Nodes: {len(nodes)} nodes")

                app_state.ssh_manager = Pssh(
                    log=logger,
                    host_list=nodes,
                    user=settings.ssh.username,
                    password=settings.ssh.password,
                    pkey=settings.ssh.key_file,
                    timeout=settings.ssh.timeout,
                    stop_on_errors=False,
                )
                logger.info("✅ Direct SSH manager initialized")

        except Exception as e:
            logger.error(f"Failed to auto-initialize SSH manager: {e}", exc_info=True)
            logger.warning("Will wait for manual configuration via web UI")

    # Initialize Redis (optional — app continues without it)
    try:
        redis_kwargs = {
            "db": settings.storage.redis.db,
            "decode_responses": True,
        }
        if settings.storage.redis.password:
            redis_kwargs["password"] = settings.storage.redis.password
        app_state.redis = aioredis.from_url(
            settings.storage.redis.url,
            **redis_kwargs,
        )
        await app_state.redis.ping()
        logger.info(f"Redis connected: {settings.storage.redis.url}")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}. History features disabled.")
        app_state.redis = None

    # Initialize RCCL data store (uses app_state.redis, degrades if None)
    from app.collectors.rccl_data_store import RCCLDataStore
    app_state.rccl_data_store = RCCLDataStore(
        app_state.redis,
        snapshot_max=settings.storage.redis.snapshot_max_entries,
        event_max=settings.storage.redis.event_max_entries,
    )

    # Start metrics collection using unified collector registry
    if app_state.ssh_manager:
        logger.info("Starting metrics collection (BaseCollector pattern)...")

        # Pre-seed node_health_status so RCCL collector can pick a leader on its
        # first poll cycle, before any GPU/NIC poll has completed.
        startup_nodes = settings.load_nodes_from_file()
        for node in startup_nodes:
            if node not in app_state.node_health_status:
                app_state.node_health_status[node] = "healthy"
                app_state.node_failure_count[node] = 0

        app_state.is_collecting = True

        for cls in REGISTERED_COLLECTORS:
            c = cls()
            app_state.collectors[c.name] = c
            app_state.collector_tasks[c.name] = _start_collector_task(c)

        app_state.probe_task = asyncio.create_task(periodic_host_probe())
        logger.info("✅ Metrics collection started")

    yield

    # Shutdown
    logger.info("Shutting down CVS Cluster Monitor")

    # 1. Signal all background loops to stop accepting new work
    app_state.is_collecting = False

    # 2. Cancel collector tasks.  Collectors call SSH inside asyncio.to_thread()
    #    which cannot be interrupted once the thread has started.  Set client to
    #    None first so any in-flight thread that finishes and tries to issue the
    #    next SSH command gets the "no client" early-return rather than an
    #    AttributeError after destroy_clients() deletes the attribute.
    if app_state.ssh_manager:
        app_state.ssh_manager.client = None  # type: ignore[assignment]

    for task in app_state.collector_tasks.values():
        task.cancel()
    if app_state.collector_tasks:
        await asyncio.gather(*app_state.collector_tasks.values(), return_exceptions=True)

    if app_state.probe_task:
        app_state.probe_task.cancel()
        try:
            await app_state.probe_task
        except asyncio.CancelledError:
            pass

    # 3. Close Redis
    if app_state.redis:
        await app_state.redis.aclose()

    # 4. Destroy SSH connections (client already None, this cleans up port-forward state)
    if app_state.ssh_manager:
        app_state.ssh_manager.destroy_clients()

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Real-time GPU cluster monitoring dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket endpoint
@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await metrics_ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await metrics_ws_manager.disconnect(websocket)


@app.websocket("/ws/rccl")
async def websocket_rccl(websocket: WebSocket):
    await rccl_ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await rccl_ws_manager.disconnect(websocket)


# Include API router FIRST (highest priority)
app.include_router(api_router, prefix=settings.api_prefix)


# Health check (specific route - defined before static files)
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ssh_manager": app_state.ssh_manager is not None,
        "collecting": app_state.is_collecting,
        "clients": metrics_ws_manager.client_count,
    }


# Mount static files LAST (after all API and WebSocket routes)
# This serves the built React frontend at the root path
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    logger.info(f"Mounting static files from: {static_dir}")
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")

    # Fallback root endpoint if static files don't exist
    @app.get("/")
    async def root():
        """Root endpoint (fallback when static files not available)."""
        return {
            "name": settings.app_name,
            "version": "0.1.0",
            "status": "running",
            "nodes": len(settings.load_nodes_from_file()),
            "collecting": app_state.is_collecting,
            "note": "Frontend not built. Run 'cd frontend && npm run build' to build the UI.",
        }
