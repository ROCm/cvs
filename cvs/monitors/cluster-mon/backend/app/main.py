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
from pathlib import Path

from app.core.simple_config import config as settings
from app.core.cvs_parallel_ssh_reliable import Pssh
from app.core.jump_host_pssh import JumpHostPssh
from app.collectors.gpu_collector import GPUMetricsCollector
from app.collectors.nic_collector import NICMetricsCollector
from app.api import router as api_router

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
        self.ssh_manager: Optional[Union[Pssh, JumpHostPssh]] = None
        self.gpu_collector: GPUMetricsCollector = None
        self.nic_collector: NICMetricsCollector = None
        self.latest_metrics: dict = {}
        self.websocket_clients: List[WebSocket] = []
        self.collection_task: asyncio.Task = None
        self.is_collecting: bool = False
        # Node health tracking (for stability - require 5 consecutive failures)
        self.node_failure_count: dict = {}  # {node: consecutive_failure_count}
        self.node_health_status: dict = {}  # {node: 'healthy'|'unhealthy'|'unreachable'}
        # Software info cache (updated every 180 seconds since it rarely changes)
        self.cached_gpu_software: dict = {}
        self.cached_nic_software: dict = {}
        self.cached_nic_advanced: dict = {}
        self.gpu_software_cache_time: float = 0
        self.nic_software_cache_time: float = 0
        self.nic_advanced_cache_time: float = 0
        self.software_cache_ttl: int = 180  # 3 minutes
        # SECURITY: Passwords stored in memory only (never persisted to disk)
        self.ssh_password: str = None  # Direct SSH password
        self.jump_host_password: str = None  # Jump host password


app_state = AppState()


async def reload_configuration():
    """
    Reload configuration without restarting the entire process.
    Stops metrics collection, closes SSH connections, reloads config, reinitializes, and restarts.

    Returns:
        dict: Status of reload operation with success/error details
    """
    try:
        logger.info("Starting configuration reload...")

        # 1. Stop metrics collection
        if app_state.is_collecting:
            logger.info("Stopping metrics collection...")
            app_state.is_collecting = False
            if app_state.collection_task:
                app_state.collection_task.cancel()
                try:
                    await app_state.collection_task
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
        from app.core.simple_config import SimpleConfig

        new_config = SimpleConfig()

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

        # 7. Restart metrics collection
        if app_state.ssh_manager and nodes:
            logger.info("Restarting metrics collection...")
            app_state.is_collecting = True
            app_state.collection_task = asyncio.create_task(collect_metrics_loop())
            logger.info("Metrics collection restarted")

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

            # Collect GPU and NIC metrics
            gpu_metrics = await app_state.gpu_collector.collect_all_metrics(app_state.ssh_manager)
            nic_metrics = await app_state.nic_collector.collect_all_metrics(app_state.ssh_manager)

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

        except Exception as e:
            logger.error(f"Error in metrics collection loop: {e}", exc_info=True)

        # Wait for next interval
        await asyncio.sleep(settings.polling.interval)

    logger.info("Metrics collection loop stopped")


async def broadcast_metrics(metrics: dict):
    """Broadcast metrics to all connected WebSocket clients."""
    if not app_state.websocket_clients:
        return

    disconnected_clients = []

    for client in app_state.websocket_clients:
        try:
            await client.send_json({"type": "metrics", "data": metrics})
        except Exception as e:
            logger.warning(f"Failed to send metrics to client: {e}")
            disconnected_clients.append(client)

    # Remove disconnected clients
    for client in disconnected_clients:
        app_state.websocket_clients.remove(client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting CVS Cluster Monitor")

    # Load nodes from file
    nodes = settings.load_nodes_from_file()
    if not nodes:
        logger.warning("No nodes configured! Please add nodes to config/nodes.txt")
        logger.info("Waiting for user to configure nodes via web UI...")
    else:
        logger.info(f"Configuration found: {len(nodes)} nodes")
        logger.info("SSH manager will be initialized when user saves configuration")

    # Don't initialize SSH or start metrics collection on startup
    # Wait for user to configure and click "Save Configuration and Start Monitoring"
    logger.info("Application ready - waiting for configuration via web UI")

    # Initialize collectors (lightweight, no SSH needed)
    app_state.gpu_collector = GPUMetricsCollector()
    app_state.nic_collector = NICMetricsCollector()
    logger.info("Collectors initialized")

    # SSH initialization disabled on startup - will be triggered via reload endpoint
    # (Keeping old code commented for reference)
    """
    # Old code - SSH initialized on startup (caused blocking issues)
    try:
        if settings.ssh.jump_host.enabled and settings.ssh.jump_host.host:
            logger.info(f"Using jump host with pssh ParallelSSHClient: {settings.ssh.jump_host.host}")
            logger.info(f"SSH Configuration Details:")
            logger.info(f"  Jump Host: {settings.ssh.jump_host.host}")
            logger.info(f"  Jump Host Username: {settings.ssh.jump_host.username}")
            logger.info(f"  Jump Host Key File: {settings.ssh.jump_host.key_file}")
            logger.info(f"  Jump Host Password: {'***SET***' if settings.ssh.jump_host.password else 'NOT SET'}")
            logger.info(f"  Cluster Nodes: {len(nodes)} nodes")
            logger.info(f"  Cluster Username: {settings.node_username_via_jumphost}")
            logger.info(f"  Cluster Key File (on jump host): {settings.node_key_file_on_jumphost}")

            # Use ParallelSSHManager with proxy parameters (uses pssh native jump host support)
            app_state.ssh_manager = ParallelSSHManager(
                host_list=nodes,
                user=settings.node_username_via_jumphost,
                password=None,  # Nodes use key-based auth from jump host
                pkey=settings.node_key_file_on_jumphost,
                timeout=settings.ssh.timeout,
                stop_on_errors=False,
                proxy_host=settings.ssh.jump_host.host,
                proxy_user=settings.ssh.jump_host.username,
                proxy_password=app_state.jump_host_password,
                proxy_pkey=settings.ssh.jump_host.key_file,
            )
            logger.info(f"ParallelSSHClient initialized with jump host proxy (native pssh)")
        else:
            # Direct SSH to nodes (no jump host)
            logger.info("Using direct SSH (no jump host)")
            logger.info(f"SSH Configuration Details:")
            logger.info(f"  Nodes: {len(nodes)} nodes")
            logger.info(f"  Username: {settings.ssh.username}")
            logger.info(f"  Key File: {settings.ssh.key_file}")
            logger.info(f"  Password: {'***SET***' if settings.ssh.password else 'NOT SET'}")
            logger.info(f"  Timeout: {settings.ssh.timeout}s")

            app_state.ssh_manager = ParallelSSHManager(
                host_list=nodes,
                user=settings.ssh.username,
                password=settings.ssh.password,
                pkey=settings.ssh.key_file,
                timeout=settings.ssh.timeout,
                stop_on_errors=False,
            )
            logger.info("Direct SSH manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize SSH manager: {e}")
    """

    # Metrics collection will start after user saves configuration
    logger.info("Ready to accept configuration via web UI")

    yield

    # Shutdown
    logger.info("Shutting down CVS Cluster Monitor")

    # Stop metrics collection
    app_state.is_collecting = False
    if app_state.collection_task:
        app_state.collection_task.cancel()
        try:
            await app_state.collection_task
        except asyncio.CancelledError:
            pass

    # Close SSH connections
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
            "nodes": len(settings.nodes) if settings.nodes else 0,
            "collecting": app_state.is_collecting,
            "note": "Frontend not built. Run 'cd frontend && npm run build' to build the UI.",
        }
