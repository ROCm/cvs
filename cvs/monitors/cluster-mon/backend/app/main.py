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
from typing import List, Optional
import os
import time
from pathlib import Path

from app.core.config import settings
from app.core.ssh_manager import SshManager
import app.core.go_collector as go_collector
from app.collectors.gpu_collector import GPUMetricsCollector
from app.collectors.nic_collector import NICMetricsCollector
from app.collectors.rccl_collector import RCCLCollector
from app.collectors.inspector_collector import InspectorCollector
from app.collectors.base import BaseCollector, CollectorResult
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

# Suppress paramiko's ERROR-level "Secsh channel N open FAILED: Connection refused"
# messages. These fire when rcclras (port 28028) is not listening (i.e. no active
# RCCL job), which is normal/expected.
if not DEBUG_MODE:
    logging.getLogger("paramiko.transport").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized - DEBUG_MODE: {DEBUG_MODE}, LOG_LEVEL: {logging.getLevelName(LOG_LEVEL)}")


# Global state
class AppState:
    """Global application state."""

    def __init__(self):
        # SSH manager
        self.ssh_manager: Optional[SshManager] = None
        # Go daemon lifecycle task
        self.lifecycle_task: Optional[asyncio.Task] = None

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

        # SECURITY: Passwords and keys stored in memory only — never persisted to disk.
        self.ssh_password: str = None
        self.jump_host_password: str = None
        # node_key_bytes: PEM bytes of the node SSH private key fetched from the
        # jump host via SFTP. Delivered to the Go daemon in-memory via the UDS
        # refresh_nodes message (key_bytes field) — never written to the container
        # filesystem. Retained across daemon crash-restarts so each new process
        # receives the key immediately after its socket opens.
        self.node_key_bytes: Optional[bytes] = None

        # Periodic host probe
        self.probe_task: Optional[asyncio.Task] = None
        self.last_probe_time: Optional[float] = None
        self.probe_count: int = 0
        self.probe_requested: asyncio.Event = None  # set by collectors on ConnectionError

        # Redis client
        self.redis: Optional[object] = None

        # RCCL state
        self.rccl_data_store = None  # RCCLDataStore, set in lifespan
        self.latest_rccl_snapshot: Optional[dict] = None
        self.rccl_websocket_clients: List[WebSocket] = []
        # Per-node capability map: populated on first successful RAS connection.
        # Key = node hostname. Value = NodeRCCLCapability (probed, with TTL).
        self.node_capabilities: dict = {}  # dict[str, NodeRCCLCapability]


app_state = AppState()

_reload_lock = asyncio.Lock()

# ─── Go daemon lifecycle ───────────────────────────────────────────────────────

_GO_BINARY  = os.environ.get("GPU_COLLECTOR_BIN", "/usr/local/bin/gpu-collector")
_HOSTS_FILE = "/tmp/go-collector-hosts.txt"
_daemon_stopping = False
_MAX_DAEMON_RESTARTS = 10

# Set by _run_daemon_lifecycle() the moment the socket file appears.
# lifespan() awaits this instead of polling the socket itself — removes the
# duplicate _wait_socket_ready race where both coroutines watched simultaneously.
_daemon_ready_event: Optional[asyncio.Event] = None


def _write_hosts_file(hosts: list) -> None:
    with open(_HOSTS_FILE, "w") as f:
        f.write("\n".join(hosts))


def _build_daemon_args(ssh_manager: SshManager) -> list:
    args = [
        _GO_BINARY,
        "--socket", go_collector._SOCKET_PATH,
        "--ssh-user", ssh_manager.user,
        "--hosts-file", _HOSTS_FILE,
    ]
    if ssh_manager.pkey:
        args += ["--ssh-key", ssh_manager.pkey]
    if ssh_manager.jump_host:
        args += ["--jump-host", ssh_manager.jump_host,
                 "--jump-user", ssh_manager.jump_user]
        if ssh_manager.jump_pkey:
            args += ["--jump-key", ssh_manager.jump_pkey]
        if ssh_manager.jump_password:
            args += ["--jump-password", ssh_manager.jump_password]
    return args


async def _fetch_node_key_from_jump_host(
    host: str,
    jump_user: str,
    jump_key_path: Optional[str],
    jump_password: Optional[str],
    remote_key_path: str,
) -> bytes:
    """
    Fetch the node SSH private key from the jump host via SFTP.

    The key is returned as raw PEM bytes and never written to the container's
    filesystem. The caller stores the result in app_state.node_key_bytes and
    delivers it to the running daemon via a refresh_nodes UDS message.
    """
    import paramiko

    def _do_fetch() -> bytes:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kw: dict = {"timeout": 30, "banner_timeout": 60}
        if jump_key_path:
            connect_kw["key_filename"] = jump_key_path
        if jump_password:
            connect_kw["password"] = jump_password
        client.connect(host, username=jump_user, **connect_kw)
        # Expand leading ~ relative to the jump user's home directory.
        path = remote_key_path
        if path == "~" or path.startswith("~/"):
            path = f"/home/{jump_user}" + path[1:]
        sftp = client.open_sftp()
        try:
            with sftp.open(path, "rb") as fh:
                return fh.read()
        finally:
            sftp.close()
            client.close()

    return await asyncio.to_thread(_do_fetch)


async def _wait_socket_ready(timeout: float = 30.0, poll: float = 0.1) -> bool:
    """Poll until the daemon's socket file appears or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(go_collector._SOCKET_PATH):
            return True
        await asyncio.sleep(poll)
    return False


async def _run_daemon_lifecycle(ssh_manager: SshManager) -> None:
    """
    Spawn the Go daemon and respawn it immediately on unexpected exit.

    Uses asyncio.create_subprocess_exec + await proc.wait() so crash detection
    is sub-100 ms (OS SIGCHLD, not polling).  Exponential backoff (2 → 4 → …
    → 120 s) prevents restart storms on repeated failures.
    """
    global _daemon_stopping
    restart_count = 0
    backoff = 0.0

    while True:
        if backoff > 0:
            logger.warning("Daemon restart #%d: waiting %.0fs backoff", restart_count, backoff)
            await asyncio.sleep(backoff)

        try:
            _write_hosts_file(ssh_manager.host_list)
            args = _build_daemon_args(ssh_manager)

            proc = await asyncio.create_subprocess_exec(
                *args,
                # Inherit stdout/stderr so Go daemon logs appear in `docker logs`
                # without needing a separate reader coroutine.
                stdin=None,
                stdout=None,
                stderr=None,
            )
            go_collector._daemon_proc = proc
            logger.info("Daemon process started (pid=%d)", proc.pid)

            ready = await _wait_socket_ready(timeout=30.0)
            if not ready:
                logger.error("Daemon did not open socket within 30 s — killing pid=%d", proc.pid)
                try:
                    proc.kill()
                except Exception:
                    pass
            else:
                # Socket is up — deliver node key in-memory if we have it.
                # This covers both the initial start and crash-recovery restarts.
                if app_state.node_key_bytes:
                    try:
                        await asyncio.to_thread(
                            go_collector._refresh_nodes_in_daemon,
                            ssh_manager.host_list,
                            key_bytes=app_state.node_key_bytes,
                        )
                        logger.info(
                            "Node SSH key delivered to daemon via UDS (%d bytes)",
                            len(app_state.node_key_bytes),
                        )
                    except Exception as exc:
                        logger.warning("Failed to deliver node key via UDS: %s", exc)

                if _daemon_ready_event is not None and not _daemon_ready_event.is_set():
                    # Signal lifespan (and reload) that the daemon is up.
                    _daemon_ready_event.set()

            if restart_count > 0:
                logger.info("Daemon process running (restart attempt #%d)", restart_count)

            # Wait for the child to exit. Use a poll-based fallback in case
            # asyncio's child watcher fails to deliver the exit event (seen on
            # some Linux container runtimes with inherited stdio).
            try:
                exit_code = await asyncio.wait_for(proc.wait(), timeout=None)
            except Exception:
                # If wait() itself raises (shouldn't happen), poll manually.
                exit_code = proc.returncode
                if exit_code is None:
                    while True:
                        await asyncio.sleep(1.0)
                        exit_code = proc.returncode
                        if exit_code is not None:
                            break
            go_collector._daemon_proc = None

        except asyncio.CancelledError:
            logger.info("Daemon lifecycle task cancelled")
            raise
        except Exception as exc:
            logger.exception("Unexpected exception in daemon lifecycle loop: %s", exc)
            go_collector._daemon_proc = None
            exit_code = -1

        if _daemon_stopping:
            logger.info("Daemon stopped intentionally, not restarting")
            break

        restart_count += 1
        logger.error(
            "Daemon exited unexpectedly (code=%s), restart #%d/%d",
            exit_code, restart_count, _MAX_DAEMON_RESTARTS,
        )

        if restart_count > _MAX_DAEMON_RESTARTS:
            logger.critical(
                "Go daemon restart limit exceeded — crashing container for Docker restart"
            )
            await asyncio.sleep(5)  # flush RotatingFileHandler before os._exit bypasses shutdown
            os._exit(1)

        backoff = min(2.0 ** restart_count, 120.0)

    logger.info("Daemon lifecycle task exiting")


async def _stop_daemon() -> None:
    """
    Signal the daemon to stop and wait for it.

    Sets _daemon_stopping so _run_daemon_lifecycle does not respawn.
    """
    global _daemon_stopping
    _daemon_stopping = True
    proc = go_collector._daemon_proc
    if proc is not None and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Daemon did not stop in 5 s, sending SIGKILL")
            proc.kill()
            await proc.wait()
    # Remove stale socket so next start doesn't find a ghost file.
    try:
        os.remove(go_collector._SOCKET_PATH)
    except OSError:
        pass


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
        ssh_changed = old_settings.ssh.model_dump() != new_config.ssh.model_dump()
        rccl_changed = old_settings.rccl.model_dump() != new_config.rccl.model_dump()
        polling_changed = old_settings.polling.model_dump() != new_config.polling.model_dump()

        # In-memory passwords are not stored in YAML, so the diff above won't
        # catch them. Compare against the password the active SshManager has.
        old_jump_pw = (app_state.ssh_manager.jump_password if app_state.ssh_manager else None)
        old_ssh_pw  = (app_state.ssh_manager.password     if app_state.ssh_manager else None)
        if (app_state.jump_host_password != old_jump_pw or
                app_state.ssh_password != old_ssh_pw):
            ssh_changed = True
            logger.info("In-memory SSH password changed — treating as ssh_changed")

        # Node list diff (computed after loading new nodes below)
        old_nodes = set(old_settings.load_nodes_from_file())

        logger.info(
            f"Config diff: ssh_changed={ssh_changed}, rccl_changed={rccl_changed}, polling_changed={polling_changed}"
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
                collectors_to_restart.update({"gpu", "nic", "rccl"})  # All use ssh_manager
            if rccl_changed:
                collectors_to_restart.add("rccl")

        # 5. Load new nodes
        nodes = new_config.load_nodes_from_file()
        if not nodes:
            logger.warning("No nodes found in configuration after reload")
            return {"success": False, "error": "No nodes configured in nodes.txt", "nodes_count": 0}

        logger.info(f"Loaded {len(nodes)} nodes from configuration")
        nodes_changed = set(nodes) != old_nodes

        # 6. Check if SSH keys exist (only if using key-based auth, not password)
        using_jump_password = new_config.ssh.jump_host.enabled and app_state.jump_host_password
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

        if not collectors_to_restart and not ssh_changed and not nodes_changed:
            # No structural changes. Always send a reprobe nudge so the daemon
            # retries any currently unreachable nodes.
            #
            # Special case: if jump host is active but we don't yet have the node
            # key in memory (e.g. SFTP fetch failed at startup because the jump
            # key wasn't uploaded yet), try to fetch it now and deliver via UDS —
            # no daemon restart required.
            node_key_missing = (
                new_config.ssh.jump_host.enabled
                and new_config.ssh.jump_host.host
                and new_config.ssh.jump_host.node_key_file
                and app_state.node_key_bytes is None
            )
            if node_key_missing:
                logger.info(
                    "Node key not in memory — fetching from jump host and delivering via UDS"
                )
                try:
                    app_state.node_key_bytes = await _fetch_node_key_from_jump_host(
                        host=new_config.ssh.jump_host.host,
                        jump_user=new_config.ssh.jump_host.username,
                        jump_key_path=new_config.ssh.jump_host.key_file if not app_state.jump_host_password else None,
                        jump_password=app_state.jump_host_password,
                        remote_key_path=new_config.ssh.jump_host.node_key_file,
                    )
                    logger.info("Node SSH key fetched (%d bytes) — delivering to daemon via UDS", len(app_state.node_key_bytes))
                    await asyncio.to_thread(
                        go_collector._refresh_nodes_in_daemon,
                        nodes,
                        key_bytes=app_state.node_key_bytes,
                    )
                except Exception as exc:
                    logger.warning("Could not fetch/deliver node key: %s", exc)
                    app_state.node_key_bytes = None
            else:
                # Still send a reprobe nudge: if a key file was just uploaded to
                # the already-configured path, the Go pool reads keys lazily and
                # will succeed on the next conn() attempt — but only if reprobe is
                # triggered.
                logger.info("No config sections changed — sending reprobe nudge to daemon")
                await asyncio.to_thread(go_collector._refresh_nodes_in_daemon, nodes)
            return {
                "success": True,
                "message": "Configuration reloaded (no changes detected)",
                "nodes_count": len(nodes),
                "jump_host_enabled": new_config.ssh.jump_host.enabled,
            }

        # 8. Handle SSH config changes.
        #
        # Jump-host change: the Go daemon's dialFunc is baked in at startup and
        # cannot be updated in-place — full daemon restart required.
        #
        # Direct-SSH change (username or key file path): update the running pool
        # in-place via refresh_nodes with the new credentials. The pool drops all
        # cached connections and re-dials with the new credentials immediately.
        # No daemon restart, no downtime.
        if ssh_changed:
            if new_config.ssh.jump_host.enabled and new_config.ssh.jump_host.host:
                # Jump-host change: full daemon restart.
                if app_state.probe_task:
                    app_state.probe_task.cancel()
                    try:
                        await app_state.probe_task
                    except asyncio.CancelledError:
                        pass

                if app_state.lifecycle_task:
                    await _stop_daemon()
                    app_state.lifecycle_task.cancel()
                    try:
                        await app_state.lifecycle_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    app_state.lifecycle_task = None

                if app_state.ssh_manager:
                    app_state.ssh_manager.destroy_clients()
                    app_state.ssh_manager = None

                app_state.latest_metrics = {}
                app_state.node_failure_count = {}
                app_state.node_health_status = {}
                app_state.cached_gpu_software = {}
                app_state.cached_nic_software = {}
                app_state.cached_nic_advanced = {}
                app_state.gpu_software_cache_time = 0
                app_state.nic_software_cache_time = 0
                app_state.nic_advanced_cache_time = 0

                logger.info("Reinitializing with jump host: %s", new_config.ssh.jump_host.host)

                # Fetch node key from jump host via SFTP (in-memory, never written to disk).
                if new_config.ssh.jump_host.node_key_file:
                    try:
                        app_state.node_key_bytes = await _fetch_node_key_from_jump_host(
                            host=new_config.ssh.jump_host.host,
                            jump_user=new_config.ssh.jump_host.username,
                            jump_key_path=new_config.ssh.jump_host.key_file if not app_state.jump_host_password else None,
                            jump_password=app_state.jump_host_password,
                            remote_key_path=new_config.ssh.jump_host.node_key_file,
                        )
                        logger.info("Node SSH key fetched from jump host (%d bytes)", len(app_state.node_key_bytes))
                    except Exception as exc:
                        logger.warning("Could not fetch node key from jump host: %s", exc)
                        app_state.node_key_bytes = None
                else:
                    app_state.node_key_bytes = None

                app_state.ssh_manager = SshManager(
                    host_list=nodes,
                    user=new_config.ssh.jump_host.node_username,
                    pkey=None,  # Key delivered in-memory via UDS refresh_nodes; not a container path
                    timeout=new_config.ssh.timeout,
                    jump_host=new_config.ssh.jump_host.host,
                    jump_user=new_config.ssh.jump_host.username,
                    jump_pkey=new_config.ssh.jump_host.key_file if not app_state.jump_host_password else None,
                    jump_password=app_state.jump_host_password,
                )
                logger.info("SshManager (jump host) initialized")

                global _daemon_stopping, _daemon_ready_event
                _daemon_stopping = False
                _daemon_ready_event = asyncio.Event()
                app_state.lifecycle_task = asyncio.create_task(
                    _run_daemon_lifecycle(app_state.ssh_manager),
                    name="daemon-lifecycle",
                )
                try:
                    await asyncio.wait_for(_daemon_ready_event.wait(), timeout=35.0)
                except asyncio.TimeoutError:
                    logger.warning("Daemon socket did not appear within 35 s after reload")

                app_state.probe_requested = asyncio.Event()
                app_state.probe_task = asyncio.create_task(periodic_host_probe())

            else:
                # Direct-SSH change: update credentials in-place — no restart.
                # The Go pool drops all connections and re-dials with new creds.
                logger.info("Direct-SSH credentials changed — updating daemon in-place")
                diff = await asyncio.to_thread(
                    go_collector._refresh_nodes_in_daemon,
                    nodes,
                    user=new_config.ssh.username,
                    key_path=new_config.ssh.key_file,
                )
                logger.info(
                    "Daemon credential update: +%d -%d (total %d)",
                    len(diff.get("added", [])),
                    len(diff.get("removed", [])),
                    diff.get("total", len(nodes)),
                )

                # Switching to direct SSH — discard any in-memory node key.
                app_state.node_key_bytes = None
                # Recreate the paramiko SshManager for Python collectors.
                if app_state.ssh_manager:
                    app_state.ssh_manager.destroy_clients()
                app_state.ssh_manager = SshManager(
                    host_list=nodes,
                    user=new_config.ssh.username,
                    pkey=new_config.ssh.key_file,
                    password=app_state.ssh_password,
                    timeout=new_config.ssh.timeout,
                )
                logger.info("SshManager (direct) reinitialized")

        elif nodes_changed:
            # Node list changed but SSH credentials unchanged — update in-place.
            app_state.ssh_manager._host_list = nodes
            diff = await asyncio.to_thread(
                go_collector._refresh_nodes_in_daemon, nodes
            )
            logger.info(
                "Daemon node list refreshed: +%d -%d (total %d)",
                len(diff.get("added", [])),
                len(diff.get("removed", [])),
                diff.get("total", len(nodes)),
            )

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
        self._send_tasks[client_id] = asyncio.create_task(self._sender(client_id, websocket, q))

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

        if settings.ssh.jump_host.enabled and settings.ssh.jump_host.host:
            logger.info(f"Initializing with jump host: {settings.ssh.jump_host.host}")
            # Seed in-memory password from YAML on cold start (YAML is the only
            # time it can come from config; reloads use app_state.jump_host_password).
            if settings.ssh.jump_host.password and not app_state.jump_host_password:
                app_state.jump_host_password = settings.ssh.jump_host.password

            # Fetch node key from jump host via SFTP so it is never stored in the container.
            if settings.ssh.jump_host.node_key_file:
                try:
                    app_state.node_key_bytes = await _fetch_node_key_from_jump_host(
                        host=settings.ssh.jump_host.host,
                        jump_user=settings.ssh.jump_host.username,
                        jump_key_path=settings.ssh.jump_host.key_file if not app_state.jump_host_password else None,
                        jump_password=app_state.jump_host_password,
                        remote_key_path=settings.ssh.jump_host.node_key_file,
                    )
                    logger.info("✅ Node SSH key fetched from jump host (%d bytes)", len(app_state.node_key_bytes))
                except Exception as exc:
                    logger.warning("Could not fetch node key from jump host: %s — nodes may be unreachable", exc)

            app_state.ssh_manager = SshManager(
                host_list=nodes,
                user=settings.ssh.jump_host.node_username,
                pkey=None,  # Key delivered in-memory via UDS refresh_nodes; not a container path
                timeout=settings.ssh.timeout,
                jump_host=settings.ssh.jump_host.host,
                jump_user=settings.ssh.jump_host.username,
                jump_pkey=settings.ssh.jump_host.key_file if not app_state.jump_host_password else None,
                jump_password=app_state.jump_host_password,
            )
            logger.info("✅ SshManager (jump host) initialized")
        else:
            logger.info(f"Initializing with direct SSH (no jump host), user={settings.ssh.username}")
            app_state.ssh_manager = SshManager(
                host_list=nodes,
                user=settings.ssh.username,
                pkey=settings.ssh.key_file,
                password=settings.ssh.password,
                timeout=settings.ssh.timeout,
            )
            logger.info("✅ SshManager (direct) initialized")

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

    # Start Go daemon + metrics collection
    if app_state.ssh_manager:
        logger.info("Starting Go SSH daemon and metrics collection...")

        # Pre-seed node_health_status so RCCL collector can pick a leader on its
        # first poll cycle, before any GPU/NIC poll has completed.
        startup_nodes = settings.load_nodes_from_file()
        for node in startup_nodes:
            if node not in app_state.node_health_status:
                app_state.node_health_status[node] = "healthy"
                app_state.node_failure_count[node] = 0

        # Launch daemon lifecycle task — spawns daemon and respawns on crash.
        global _daemon_stopping, _daemon_ready_event
        _daemon_stopping = False
        _daemon_ready_event = asyncio.Event()
        app_state.lifecycle_task = asyncio.create_task(
            _run_daemon_lifecycle(app_state.ssh_manager),
            name="daemon-lifecycle",
        )
        # Lifecycle task is the sole socket watcher and sets _daemon_ready_event
        # when the socket appears.  Awaiting the event here (not _wait_socket_ready)
        # eliminates the dual-poll race where both coroutines woke simultaneously.
        try:
            await asyncio.wait_for(_daemon_ready_event.wait(), timeout=35.0)
            logger.info("Go daemon ready")
        except asyncio.TimeoutError:
            logger.warning("Go daemon socket did not appear within 35 s — collectors starting anyway")

        app_state.is_collecting = True

        for cls in REGISTERED_COLLECTORS:
            c = cls()
            app_state.collectors[c.name] = c
            app_state.collector_tasks[c.name] = _start_collector_task(c)

        app_state.probe_task = asyncio.create_task(periodic_host_probe())
        logger.info("✅ Daemon and metrics collection started")

    yield

    # Shutdown
    logger.info("Shutting down CVS Cluster Monitor")

    # 1. Signal all background loops to stop accepting new work.
    app_state.is_collecting = False

    # 2. Cancel collector tasks. _exec_one calls will fail immediately
    #    once the daemon stops; 5 s deadline catches any in-flight thread.
    for task in app_state.collector_tasks.values():
        task.cancel()
    if app_state.collector_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*app_state.collector_tasks.values(), return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Collector tasks did not finish within 5 s — forcing shutdown")

    if app_state.probe_task:
        app_state.probe_task.cancel()
        try:
            await asyncio.wait_for(app_state.probe_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    # 3. Stop Go daemon (SIGTERM → wait 5 s → SIGKILL).
    await _stop_daemon()
    if app_state.lifecycle_task:
        app_state.lifecycle_task.cancel()
        try:
            await asyncio.wait_for(app_state.lifecycle_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    # 4. Clean up paramiko port-forward clients.
    if app_state.ssh_manager:
        app_state.ssh_manager.destroy_clients()

    # 5. Close Redis
    if app_state.redis:
        await app_state.redis.aclose()

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
