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
from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.api import router as api_router

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

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


app_state = AppState()


REGISTERED_COLLECTORS: list[type[BaseCollector]] = [
    GPUMetricsCollector,
    NICMetricsCollector,
    # RCCLCollector added in Batch 6
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

        asyncio.get_event_loop().call_soon(lambda: asyncio.create_task(_restart()))

    task = asyncio.create_task(
        c.run(app_state.ssh_manager, app_state),
        name=f"collector-{c.name}",
    )
    task.add_done_callback(_on_done)
    return task


async def reload_configuration():
    """
    Reload configuration without restarting the entire process.
    Stops metrics collection, closes SSH connections, reloads config, reinitializes, and restarts.

    Returns:
        dict: Status of reload operation with success/error details
    """
    try:
        logger.info("Starting configuration reload...")

        # 1. Stop metrics collection and periodic probe
        if app_state.is_collecting:
            logger.info("Stopping metrics collection and periodic probe...")
            app_state.is_collecting = False
            for task in app_state.collector_tasks.values():
                task.cancel()
            if app_state.collector_tasks:
                await asyncio.gather(*app_state.collector_tasks.values(), return_exceptions=True)
            app_state.collector_tasks.clear()
            app_state.collectors.clear()
            if app_state.collection_task:
                app_state.collection_task.cancel()
                try:
                    await app_state.collection_task
                except asyncio.CancelledError:
                    pass
            if app_state.probe_task:
                app_state.probe_task.cancel()
                try:
                    await app_state.probe_task
                except asyncio.CancelledError:
                    pass

        # 2. Close existing SSH connections
        if app_state.ssh_manager:
            logger.info("Closing existing SSH connections...")
            app_state.ssh_manager.destroy_clients()
            app_state.ssh_manager = None

        # 3. Clear cached data
        app_state.latest_metrics = {}
        app_state.node_failure_count = {}
        app_state.node_health_status = {}
        app_state.cached_gpu_software = {}
        app_state.cached_nic_software = {}
        app_state.cached_nic_advanced = {}
        app_state.gpu_software_cache_time = 0
        app_state.nic_software_cache_time = 0
        app_state.nic_advanced_cache_time = 0

        # 4. Reload configuration from files
        logger.info("Reloading configuration from cluster.yaml and nodes.txt...")
        from app.core.config import Settings

        new_config = Settings()  # re-reads YAML and env vars

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
                logger.warning(f"❌ SSH key file not found: {key_file_expanded}")
                logger.warning("Please upload SSH keys via Configuration UI or run refresh-ssh-keys.sh")
                return {
                    "success": False,
                    "error": f"SSH key file not found: {key_file_path}. Please upload SSH keys via the Configuration UI.",
                    "nodes_count": len(nodes),
                    "requires_key_upload": True,
                }
            else:
                logger.info(f"✅ SSH key file found: {key_file_expanded}")
                # List the key file to verify
                import subprocess

                try:
                    result = subprocess.run(['ls', '-l', key_file_expanded], capture_output=True, text=True)
                    logger.info(f"Key file details: {result.stdout.strip()}")
                except:
                    pass
        else:
            logger.info("✅ Using password authentication - no key file check needed")

        # 7. Reinitialize SSH manager with new configuration
        try:
            if new_config.ssh.jump_host.enabled and new_config.ssh.jump_host.host:
                num_nodes = len(nodes)
                min(num_nodes, 5)

                logger.info(f"Reinitializing with jump host: {new_config.ssh.jump_host.host}")
                logger.info(f"Jump Host Username: {new_config.ssh.jump_host.username}")
                logger.info(f"Cluster Nodes: {len(nodes)} nodes")
                logger.info(f"Cluster Username: {new_config.node_username_via_jumphost}")

                # Use JumpHostPssh - working approach from test_auth_script.py
                app_state.ssh_manager = JumpHostPssh(
                    jump_host=new_config.ssh.jump_host.host,
                    jump_user=new_config.ssh.jump_host.username,
                    jump_password=new_config.ssh.jump_host.password,
                    jump_pkey=new_config.ssh.jump_host.key_file if not new_config.ssh.jump_host.password else None,
                    target_hosts=nodes,
                    target_user=new_config.node_username_via_jumphost,
                    target_pkey=new_config.node_key_file_on_jumphost,
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

        # 7. Restart metrics collection and periodic probe
        if app_state.ssh_manager and nodes:
            logger.info("Restarting metrics collection and periodic probe...")
            app_state.is_collecting = True
            app_state.probe_requested = asyncio.Event()
            for cls in REGISTERED_COLLECTORS:
                c = cls()
                app_state.collectors[c.name] = c
                app_state.collector_tasks[c.name] = _start_collector_task(c)
            app_state.probe_task = asyncio.create_task(periodic_host_probe())
            logger.info("Metrics collection and periodic probe restarted")

        logger.info("Configuration reload completed successfully!")
        return {
            "success": True,
            "message": "Configuration reloaded successfully",
            "nodes_count": len(nodes),
            "jump_host_enabled": new_config.ssh.jump_host.enabled,
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


async def collect_metrics_loop():
    """Background task to collect metrics periodically."""
    logger.info("Starting metrics collection loop")

    while app_state.is_collecting:
        try:
            logger.info("Collecting metrics...")

            # Collect GPU and NIC metrics with connection error handling
            try:
                gpu_metrics = await app_state.gpu_collector.collect_all_metrics(app_state.ssh_manager)
                nic_metrics = await app_state.nic_collector.collect_all_metrics(app_state.ssh_manager)
            except ConnectionError as e:
                # Connection error during metrics collection - trigger immediate re-probe
                logger.error(f"ConnectionError during metrics collection: {e}")
                logger.info("Triggering immediate host re-probe...")

                # Trigger immediate re-probe
                if app_state.ssh_manager:
                    changed = await asyncio.to_thread(app_state.ssh_manager.refresh_host_reachability)
                    if changed:
                        await asyncio.to_thread(app_state.ssh_manager.recreate_client)
                        logger.info("SSH client recreated with updated reachable hosts")

                # Continue to next iteration (skip this round)
                logger.info("Skipping this metrics collection round, will retry next interval")
                await asyncio.sleep(settings.polling.interval)
                continue

            # Package metrics
            metrics_payload = {
                "timestamp": gpu_metrics.get("timestamp") if isinstance(gpu_metrics, dict) else None,
                "gpu": gpu_metrics if not isinstance(gpu_metrics, Exception) else {"error": str(gpu_metrics)},
                "nic": nic_metrics if not isinstance(nic_metrics, Exception) else {"error": str(nic_metrics)},
            }

            # Update node status based on metrics collection success/failure
            # Check each node and update failure counters
            if isinstance(gpu_metrics, dict):
                util_data = gpu_metrics.get("utilization", {})
                for node in app_state.ssh_manager.host_list:
                    has_error = False

                    if node in util_data:
                        node_data = util_data[node]
                        if isinstance(node_data, dict) and 'error' in node_data:
                            has_error = True

                    # Update status with stability check (5 consecutive failures required)
                    update_node_status(node, has_error, 'unreachable')

            # Store in app state
            app_state.latest_metrics = metrics_payload

            # Broadcast to WebSocket clients
            await broadcast_metrics(metrics_payload)

            logger.info(f"Metrics collected successfully. {len(app_state.websocket_clients)} clients notified")

        except asyncio.CancelledError:
            logger.info("Metrics collection task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in metrics collection loop: {e}", exc_info=True)

        # Wait for next interval
        await asyncio.sleep(settings.polling.interval)

    logger.info("Metrics collection loop stopped")


async def broadcast_metrics(metrics: dict):
    """Broadcast metrics to all connected WebSocket clients (non-blocking per client)."""
    if not app_state.websocket_clients:
        return

    disconnected_clients = []
    message = {"type": "metrics", "data": metrics}

    for client in list(app_state.websocket_clients):
        try:
            # Use asyncio.wait_for to prevent one slow client from blocking others
            await asyncio.wait_for(
                client.send_json(message),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning("WebSocket client timed out on send — disconnecting")
            disconnected_clients.append(client)
        except Exception as e:
            logger.warning(f"Failed to send metrics to client: {e}")
            disconnected_clients.append(client)

    for client in disconnected_clients:
        if client in app_state.websocket_clients:
            app_state.websocket_clients.remove(client)


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
                logger.info(f"Cluster Username: {settings.node_username_via_jumphost}")

                app_state.ssh_manager = JumpHostPssh(
                    jump_host=settings.ssh.jump_host.host,
                    jump_user=settings.ssh.jump_host.username,
                    jump_password=settings.ssh.jump_host.password,
                    jump_pkey=settings.ssh.jump_host.key_file if not settings.ssh.jump_host.password else None,
                    target_hosts=nodes,
                    target_user=settings.node_username_via_jumphost,
                    target_pkey=settings.node_key_file_on_jumphost,
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

            # Start metrics collection using unified collector registry
            if app_state.ssh_manager:
                logger.info("Starting metrics collection (BaseCollector pattern)...")
                app_state.is_collecting = True

                for cls in REGISTERED_COLLECTORS:
                    c = cls()
                    app_state.collectors[c.name] = c
                    app_state.collector_tasks[c.name] = _start_collector_task(c)

                app_state.probe_task = asyncio.create_task(periodic_host_probe())
                logger.info("✅ Metrics collection started")

        except Exception as e:
            logger.error(f"Failed to auto-initialize SSH manager: {e}", exc_info=True)
            logger.warning("Will wait for manual configuration via web UI")

    yield

    # Shutdown
    logger.info("Shutting down CVS Cluster Monitor")

    # Stop collection
    app_state.is_collecting = False
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
    # Close Redis
    if app_state.redis:
        await app_state.redis.aclose()
    # Close SSH
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
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming."""
    await websocket.accept()
    app_state.websocket_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(app_state.websocket_clients)}")

    try:
        # Send initial metrics
        if app_state.latest_metrics:
            await websocket.send_json({"type": "metrics", "data": app_state.latest_metrics})

        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in app_state.websocket_clients:
            app_state.websocket_clients.remove(websocket)


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
        "clients": len(app_state.websocket_clients),
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
