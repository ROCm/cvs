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
from app.core import go_collector
from app.collectors.gpu_collector import GPUMetricsCollector
from app.collectors.nic_collector import NICMetricsCollector
from app.collectors.rccl_collector import RCCLCollector
from app.collectors.inspector_collector import InspectorCollector
from app.collectors.rack_collector import RackCollector
from app.collectors.cpu_collector import CPUCollector
from app.collectors.storage_collector import StorageCollector
from app.collectors.base import BaseCollector
from app.api import router as api_router

import redis.asyncio as aioredis

DEBUG_MODE = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend.log")
rotating_handler = RotatingFileHandler(log_file_path, maxBytes=1024 * 1024, backupCount=3)
rotating_handler.setLevel(LOG_LEVEL)
rotating_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logging.basicConfig(level=LOG_LEVEL, handlers=[rotating_handler, console_handler])

if not DEBUG_MODE:
    # Suppress paramiko's internal SSH transport errors — these are noisy when
    # RCCL port-forwarding fails (expected when no RCCL job is running).
    logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized - DEBUG_MODE: {DEBUG_MODE}")


class AppState:
    def __init__(self):
        self.ssh_manager: Optional[SshManager] = None
        self.collectors: dict = {}
        self.collector_tasks: dict = {}
        self.collector_results: dict = {}
        self.gpu_collector: Optional[GPUMetricsCollector] = None
        self.nic_collector: Optional[NICMetricsCollector] = None
        self.collection_task: Optional[asyncio.Task] = None
        self.latest_metrics: dict = {}
        self.websocket_clients: List[WebSocket] = []
        self.is_collecting: bool = False
        self.node_failure_count: dict = {}
        self.node_health_status: dict = {}
        self.cached_gpu_software: dict = {}
        self.cached_nic_software: dict = {}
        self.cached_nic_advanced: dict = {}
        self.gpu_software_cache_time: float = 0
        self.nic_software_cache_time: float = 0
        self.nic_advanced_cache_time: float = 0
        self.software_cache_ttl: int = 180
        self.ssh_password: Optional[str] = None
        self.jump_host_password: Optional[str] = None
        self.probe_task: Optional[asyncio.Task] = None
        self.last_probe_time: Optional[float] = None
        self.probe_count: int = 0
        self.probe_requested: Optional[asyncio.Event] = None
        self.redis: Optional[object] = None
        self.rccl_data_store = None
        self.latest_rccl_snapshot: Optional[dict] = None
        self.rccl_websocket_clients: List[WebSocket] = []
        self.node_capabilities: dict = {}
        # Rack / IFoE
        self.rack_settings: Optional[object] = None
        self.rack_passwords: dict = {}
        self.latest_rack_data: Optional[dict] = None
        # CPU/memory collector cache
        self.latest_cpu_data: Optional[dict] = None
        self.latest_storage_data: Optional[dict] = None
        # Node groups
        self.node_groups: Optional[object] = None
        self.node_groups_passwords: dict = {}
        self.node_groups_jump_passwords: dict = {}
        # Go daemon lifecycle tasks
        self.daemon_task: Optional[asyncio.Task] = None  # cluster (GPU nodes)
        self.switch_daemon_task: Optional[asyncio.Task] = None  # switch trays


app_state = AppState()
_reload_lock = asyncio.Lock()


REGISTERED_COLLECTORS: list[type[BaseCollector]] = [
    GPUMetricsCollector,
    NICMetricsCollector,
    RCCLCollector,
    InspectorCollector,
    RackCollector,
    CPUCollector,
    StorageCollector,
]


def _start_collector_task(c: BaseCollector) -> asyncio.Task:
    _backoff = [1.0]

    def _on_done(task: asyncio.Task) -> None:
        if task.cancelled() or not app_state.is_collecting:
            return
        exc = task.exception()
        delay = _backoff[0]
        if exc is not None:
            logger.error(f"Collector {c.name} crashed: {exc!r} — restarting in {delay:.0f}s", exc_info=exc)
        else:
            logger.warning(f"Collector {c.name} task exited — restarting")

        async def _restart():
            await asyncio.sleep(_backoff[0])
            _backoff[0] = min(_backoff[0] * 2, 120)
            new_task = _start_collector_task(c)
            app_state.collector_tasks[c.name] = new_task

        def _schedule():
            asyncio.create_task(_restart(), name=f"restart-{c.name}")

        asyncio.get_running_loop().call_soon(_schedule)

    task = asyncio.create_task(c.run(app_state.ssh_manager, app_state), name=f"collector-{c.name}")
    task.add_done_callback(_on_done)
    return task


async def _run_daemon_lifecycle() -> None:
    daemon_binary = os.environ.get("GO_COLLECTOR_BIN", "/usr/local/bin/gpu-collector")

    if not os.path.exists(daemon_binary):
        logger.warning(f"Go daemon binary not found at {daemon_binary} — SSH collection unavailable")
        return

    backoff = 1.0

    while app_state.is_collecting:
        # ── Resolve hosts file ──────────────────────────────────────────────
        hosts_file = None
        for candidate in ["/app/config/nodes.txt", "config/nodes.txt"]:
            if os.path.exists(candidate):
                hosts_file = candidate
                break
        if hosts_file is None:
            # No nodes.txt yet — create an empty one so the daemon can start.
            # Hosts will be populated via refresh_nodes socket messages later.
            hosts_file = "/tmp/go_hosts.txt"
            with open(hosts_file, "w") as _f:
                pass

        # ── Resolve SSH username ────────────────────────────────────────────
        # Prefer node_groups gpu_nodes username if configured, else cluster SSH
        ng = getattr(app_state, "node_groups", None)
        if ng and ng.gpu_nodes.ssh.username:
            username = ng.gpu_nodes.ssh.username
        else:
            username = settings.ssh.username or "root"

        # ── Build command-line arguments ───────────────────────────────────
        args = [
            daemon_binary,
            "--hosts-file",
            hosts_file,
            "--ssh-user",
            username,
            "--socket",
            go_collector.CLUSTER_SOCKET,
        ]

        # SSH key — only include if the file actually exists on disk
        def _resolve_key(key_path: Optional[str]) -> Optional[str]:
            if not key_path:
                return None
            expanded = os.path.expanduser(key_path)
            return expanded if os.path.exists(expanded) else None

        ssh_key: Optional[str] = None
        if ng and ng.gpu_nodes.ssh.auth_method == "key":
            ssh_key = _resolve_key(ng.gpu_nodes.ssh.key_file)
        if ssh_key is None and not getattr(app_state, "ssh_password", None):
            ssh_key = _resolve_key(settings.ssh.key_file)
        if ssh_key:
            args.extend(["--ssh-key", ssh_key])
        else:
            # Password auth — pass at startup for immediate use; also re-sent
            # dynamically via refresh_nodes after socket opens.
            startup_pw = (
                getattr(app_state, "node_groups_passwords", {}).get("gpu_nodes")
                or getattr(app_state, "node_groups_passwords", {}).get("cluster")
                or getattr(app_state, "ssh_password", None)
            )
            if startup_pw:
                args.extend(["--ssh-password", startup_pw])

        # Jump host (cluster-level jump host)
        jh = settings.ssh.jump_host
        ng_jh = ng.gpu_nodes.jump_host if ng else None
        active_jh = ng_jh if (ng_jh and ng_jh.enabled and ng_jh.host) else (jh if (jh.enabled and jh.host) else None)
        if active_jh:
            _jh_host = getattr(active_jh, "host", "")
            _jh_user = getattr(active_jh, "username", getattr(active_jh, "username", "root"))
            _jh_key = _resolve_key(getattr(active_jh, "key_file", None))
            _jh_pw = getattr(app_state, "node_groups_jump_passwords", {}).get("gpu_nodes") or getattr(
                app_state, "jump_host_password", None
            )
            if _jh_host:
                args.extend(["--jump-host", _jh_host, "--jump-user", _jh_user])
                if _jh_key:
                    args.extend(["--jump-key", _jh_key])
                elif _jh_pw:
                    args.extend(["--jump-password", _jh_pw])

        logger.info(
            f"Starting Go SSH daemon: user={username} "
            f"hosts_file={hosts_file} key={'yes' if ssh_key else 'no (password/socket)'}"
        )

        # ── Remove stale socket ────────────────────────────────────────────
        try:
            os.unlink(go_collector.CLUSTER_SOCKET)
        except FileNotFoundError:
            pass

        # ── Launch ─────────────────────────────────────────────────────────
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            go_collector._daemon_proc = proc
            logger.info(f"Go daemon started (pid={proc.pid})")
            backoff = 1.0

            async def _drain_stderr():
                async for line in proc.stderr:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        logger.warning(f"[go-daemon stderr] {text}")

            async def _drain_stdout():
                async for line in proc.stdout:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        logger.info(f"[go-daemon] {text}")

            asyncio.create_task(_drain_stderr())
            asyncio.create_task(_drain_stdout())
            await proc.wait()
            go_collector._daemon_proc = None
            logger.warning(f"Go daemon exited (rc={proc.returncode}) — restarting in {backoff:.0f}s")
        except asyncio.CancelledError:
            if go_collector._daemon_proc:
                go_collector._daemon_proc.terminate()
                go_collector._daemon_proc = None
            raise
        except Exception as e:
            go_collector._daemon_proc = None
            logger.error(f"Go daemon error: {e}")

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)

    logger.info("Go daemon lifecycle task stopped")


async def _run_switch_daemon_lifecycle() -> None:
    """
    Lifecycle for the SWITCH tray Go daemon (separate from the cluster daemon).
    Uses SWITCH_SOCKET so switch tray SSH connections are fully isolated from
    GPU node connections, each with their own credential set.
    """
    daemon_binary = os.environ.get("GO_COLLECTOR_BIN", "/usr/local/bin/gpu-collector")
    if not os.path.exists(daemon_binary):
        logger.warning("Go daemon binary not found — switch daemon not started")
        return

    backoff = 1.0

    while app_state.is_collecting:
        ng = getattr(app_state, "node_groups", None)
        switch_hosts = []
        if ng:
            switch_hosts = list(ng.scale_up_switches.hosts) + list(ng.scale_out_switches.hosts)

        if not switch_hosts:
            # No switch trays configured yet — wait and retry
            await asyncio.sleep(30)
            continue

        # Write combined switch hosts to a temp file
        hosts_file = "/tmp/go_switch_hosts.txt"
        try:
            with open(hosts_file, "w") as fh:
                for h in switch_hosts:
                    fh.write(f"{h}\n")
        except Exception as e:
            logger.error(f"Switch daemon: cannot write hosts file: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        # Resolve credentials — try scale_up first (same creds assumed for all switches)
        switch_user = "admin"
        switch_key = None
        switch_pw = None
        if ng and ng.scale_up_switches.ssh.username:
            switch_user = ng.scale_up_switches.ssh.username
        if ng and ng.scale_up_switches.ssh.auth_method == "key" and ng.scale_up_switches.ssh.key_file:
            expanded = os.path.expanduser(ng.scale_up_switches.ssh.key_file)
            if os.path.exists(expanded):
                switch_key = expanded
        if not switch_key:
            switch_pw = getattr(app_state, "node_groups_passwords", {}).get("scale_up_switches")

        # If password auth is configured but no password is available yet (e.g. after
        # container restart), wait instead of starting the daemon with no credentials.
        # The user must save Configuration → Scale-up Switches to provide the password.
        auth_method = ng.scale_up_switches.ssh.auth_method if ng else "password"
        if auth_method == "password" and not switch_key and not switch_pw:
            logger.warning(
                "Switch daemon: password auth required but no password in memory. "
                "Go to Configuration → Scale-up Switches and click Save & Apply."
            )
            await asyncio.sleep(30)
            continue

        args = [
            daemon_binary,
            "--hosts-file",
            hosts_file,
            "--ssh-user",
            switch_user,
            "--socket",
            go_collector.SWITCH_SOCKET,
        ]
        if switch_key:
            args.extend(["--ssh-key", switch_key])
        elif switch_pw:
            args.extend(["--ssh-password", switch_pw])

        logger.info(
            f"Starting switch Go daemon: {len(switch_hosts)} hosts, "
            f"user={switch_user}, key={'yes' if switch_key else 'no'}, "
            f"password={'yes' if switch_pw else 'no (re-save config)'}"
        )

        try:
            os.unlink(go_collector.SWITCH_SOCKET)
        except FileNotFoundError:
            pass

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            go_collector._switch_daemon_proc = proc
            logger.info(f"Switch Go daemon started (pid={proc.pid})")
            backoff = 1.0

            async def _drain_sw_stderr():
                async for line in proc.stderr:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        logger.warning(f"[go-switch stderr] {text}")

            async def _drain_sw_stdout():
                async for line in proc.stdout:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        logger.info(f"[go-switch] {text}")

            asyncio.create_task(_drain_sw_stderr())
            asyncio.create_task(_drain_sw_stdout())
            await proc.wait()
            go_collector._switch_daemon_proc = None
            logger.warning(f"Switch Go daemon exited (rc={proc.returncode}) — restarting in {backoff:.0f}s")
        except asyncio.CancelledError:
            if go_collector._switch_daemon_proc:
                go_collector._switch_daemon_proc.terminate()
                go_collector._switch_daemon_proc = None
            raise
        except Exception as e:
            go_collector._switch_daemon_proc = None
            logger.error(f"Switch Go daemon error: {e}")

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)

    logger.info("Switch Go daemon lifecycle task stopped")


async def reload_configuration():
    async with _reload_lock:
        return await _reload_configuration_inner()


async def _reload_configuration_inner():
    try:
        logger.info("Starting configuration reload...")
        from app.core.config import Settings
        import app.core.config as config_module

        old_settings = config_module.settings
        new_config = Settings()
        ssh_changed = old_settings.ssh.model_dump() != new_config.ssh.model_dump()
        rccl_changed = old_settings.rccl.model_dump() != new_config.rccl.model_dump()
        polling_changed = old_settings.polling.model_dump() != new_config.polling.model_dump()
        config_module.settings = new_config

        collectors_to_restart: set[str] = set()
        if polling_changed:
            collectors_to_restart = {cls.name for cls in REGISTERED_COLLECTORS}
        else:
            if ssh_changed:
                collectors_to_restart.update({"gpu", "nic"})
            if rccl_changed:
                collectors_to_restart.add("rccl")

        nodes = new_config.load_nodes_from_file()
        if not nodes:
            logger.info("No cluster nodes configured — rack-only deployment mode")
            return {
                "success": True,
                "message": "Configuration reloaded (no cluster nodes)",
                "nodes_count": 0,
                "jump_host_enabled": new_config.ssh.jump_host.enabled,
            }

        logger.info(f"Loaded {len(nodes)} nodes")

        for name in collectors_to_restart:
            task = app_state.collector_tasks.get(name)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if not collectors_to_restart and not ssh_changed:
            return {
                "success": True,
                "message": "Configuration reloaded (no changes)",
                "nodes_count": len(nodes),
                "jump_host_enabled": new_config.ssh.jump_host.enabled,
            }

        if ssh_changed:
            if app_state.probe_task:
                app_state.probe_task.cancel()
                try:
                    await app_state.probe_task
                except asyncio.CancelledError:
                    pass

            app_state.latest_metrics = {}
            app_state.node_failure_count = {}
            app_state.node_health_status = {}

            try:
                jump_cfg = new_config.ssh.jump_host
                app_state.ssh_manager = SshManager(
                    host_list=nodes,
                    user=new_config.ssh.username,
                    pkey=new_config.ssh.key_file if not app_state.ssh_password else None,
                    password=app_state.ssh_password,
                    timeout=new_config.ssh.timeout,
                    jump_host=jump_cfg.host if jump_cfg.enabled and jump_cfg.host else None,
                    jump_user=jump_cfg.username if jump_cfg.enabled else None,
                    jump_pkey=jump_cfg.key_file if jump_cfg.enabled and not app_state.jump_host_password else None,
                    jump_password=app_state.jump_host_password if jump_cfg.enabled else None,
                )
                key_path = new_config.ssh.key_file if not app_state.ssh_password else ""
                await asyncio.to_thread(
                    go_collector._refresh_nodes_in_daemon,
                    nodes,
                    new_config.ssh.username,
                    key_path,
                    None,
                    app_state.ssh_password,
                    "cluster",
                )
                logger.info(f"SSH manager reinitialized with {len(nodes)} nodes")
            except Exception as e:
                logger.error(f"Failed to reinitialize SSH manager: {e}")
                return {"success": False, "error": str(e), "nodes_count": len(nodes)}

            app_state.probe_requested = asyncio.Event()
            app_state.probe_task = asyncio.create_task(periodic_host_probe())

        if app_state.ssh_manager and nodes:
            app_state.is_collecting = True
            for cls in REGISTERED_COLLECTORS:
                if cls.name in collectors_to_restart:
                    c = cls()
                    app_state.collectors[c.name] = c
                    app_state.collector_tasks[c.name] = _start_collector_task(c)

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
    failure_threshold = settings.polling.failure_threshold
    if node not in app_state.node_failure_count:
        app_state.node_failure_count[node] = 0
        app_state.node_health_status[node] = 'healthy'
    if is_error:
        app_state.node_failure_count[node] += 1
        if app_state.node_failure_count[node] >= failure_threshold:
            app_state.node_health_status[node] = error_type
    else:
        if app_state.node_failure_count[node] > 0:
            logger.info(f"Node {node} recovered")
        app_state.node_failure_count[node] = 0
        app_state.node_health_status[node] = 'healthy'
    return app_state.node_health_status[node]


class ConnectionManager:
    def __init__(self, max_queue_size: int = 64):
        self._clients: dict[int, WebSocket] = {}
        self._queues: dict[int, asyncio.Queue] = {}
        self._send_tasks: dict[int, asyncio.Task] = {}
        self._max_queue_size = max_queue_size
        self._closing: set[int] = set()

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
        to_remove = []
        for client_id, q in self._queues.items():
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                to_remove.append(client_id)
        for client_id in to_remove:
            asyncio.create_task(self._remove(client_id))

    @property
    def client_count(self) -> int:
        return len(self._clients)


metrics_ws_manager = ConnectionManager()
rccl_ws_manager = ConnectionManager()


async def broadcast_metrics(metrics: dict):
    metrics_ws_manager.broadcast({"type": "metrics", "data": metrics})


async def broadcast_rccl(snapshot: dict):
    rccl_ws_manager.broadcast({"type": "rccl_snapshot", "data": snapshot})


async def periodic_host_probe():
    logger.info("Periodic host probe task started")
    while app_state.is_collecting:
        try:
            try:
                await asyncio.wait_for(app_state.probe_requested.wait(), timeout=300)
                app_state.probe_requested.clear()
            except asyncio.TimeoutError:
                pass
            if not app_state.ssh_manager:
                continue
            # changed = await asyncio.to_thread(app_state.ssh_manager.refresh_host_reachability)
            app_state.probe_count += 1
            app_state.last_probe_time = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Periodic probe failed: {e}", exc_info=True)
    logger.info("Periodic host probe task stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CVS Cluster Monitor")
    app_state.probe_requested = asyncio.Event()
    app_state.is_collecting = True

    # Start Go daemons (cluster for GPU nodes, switch for switch trays)
    app_state.daemon_task = asyncio.create_task(_run_daemon_lifecycle(), name="go-daemon")
    app_state.switch_daemon_task = asyncio.create_task(_run_switch_daemon_lifecycle(), name="go-switch-daemon")

    # After the daemon socket appears, push any persisted node-group credentials.
    # Key file paths survive restarts (stored in node_groups.yaml).
    # Passwords are in-memory only — user must save Configuration once after restart.
    async def _push_initial_credentials():
        # Wait up to 30 s for cluster socket
        for _ in range(60):
            await asyncio.sleep(0.5)
            if go_collector.is_daemon_ready():
                break

        ng = getattr(app_state, "node_groups", None)
        if ng is None:
            logger.info("No node groups configured — skipping initial credential push")
            return

        # ── GPU nodes → CLUSTER_SOCKET ────────────────────────────────────
        if ng.gpu_nodes.hosts and go_collector.is_daemon_ready(go_collector.CLUSTER_SOCKET):
            grp = ng.gpu_nodes
            key_path = ""
            if grp.ssh.auth_method == "key" and grp.ssh.key_file:
                expanded = os.path.expanduser(grp.ssh.key_file)
                if os.path.exists(expanded):
                    key_path = expanded
            gpu_pw = app_state.node_groups_passwords.get("gpu_nodes")
            try:
                await asyncio.to_thread(
                    go_collector._refresh_nodes_in_daemon,
                    grp.hosts,
                    grp.ssh.username,
                    key_path,
                    None,
                    gpu_pw,
                    "cluster",
                    socket_path=go_collector.CLUSTER_SOCKET,
                )
                logger.info(
                    f"Pushed initial credentials for {len(grp.hosts)} GPU nodes: "
                    f"auth={grp.ssh.auth_method}, key={'yes' if key_path else 'no'}, "
                    f"password={'yes' if gpu_pw else 'no (re-save Configuration after restart)'}"
                )
            except Exception as exc:
                logger.warning(f"Could not push initial GPU credentials: {exc}")

        # ── Switch trays → SWITCH_SOCKET ──────────────────────────────────
        # Wait up to 30 s for switch socket
        for _ in range(60):
            await asyncio.sleep(0.5)
            if go_collector.is_daemon_ready(go_collector.SWITCH_SOCKET):
                break

        switch_hosts = list(ng.scale_up_switches.hosts) + list(ng.scale_out_switches.hosts)
        if switch_hosts and go_collector.is_daemon_ready(go_collector.SWITCH_SOCKET):
            sw_grp = ng.scale_up_switches
            sw_key = ""
            if sw_grp.ssh.auth_method == "key" and sw_grp.ssh.key_file:
                expanded = os.path.expanduser(sw_grp.ssh.key_file)
                if os.path.exists(expanded):
                    sw_key = expanded
            sw_pw = app_state.node_groups_passwords.get("scale_up_switches")
            try:
                await asyncio.to_thread(
                    go_collector._refresh_nodes_in_daemon,
                    switch_hosts,
                    sw_grp.ssh.username,
                    sw_key,
                    None,
                    sw_pw,
                    "switches",
                    socket_path=go_collector.SWITCH_SOCKET,
                )
                logger.info(
                    f"Pushed initial credentials for {len(switch_hosts)} switch trays: "
                    f"auth={sw_grp.ssh.auth_method}, key={'yes' if sw_key else 'no'}, "
                    f"password={'yes' if sw_pw else 'no (re-save Configuration after restart)'}"
                )
            except Exception as exc:
                logger.warning(f"Could not push initial switch credentials: {exc}")

    asyncio.create_task(_push_initial_credentials(), name="push-initial-creds")

    # Load node groups config
    try:
        from app.core.node_groups import load_node_groups

        app_state.node_groups = load_node_groups()
        ng = app_state.node_groups
        logger.info(
            f"Node groups loaded: gpu={len(ng.gpu_nodes.hosts)}, "
            f"scale_up={len(ng.scale_up_switches.hosts)}, "
            f"scale_out={len(ng.scale_out_switches.hosts)}"
        )
    except Exception as e:
        logger.warning(f"Could not load node_groups.yaml: {e}")

    # Load rack config (legacy)
    try:
        from app.core.rack_config import load_rack_config

        app_state.rack_settings = load_rack_config()
    except Exception as e:
        logger.warning(f"Could not load rack config: {e}")

    # Legacy collector references
    app_state.gpu_collector = GPUMetricsCollector()
    app_state.nic_collector = NICMetricsCollector()

    # Redis
    try:
        redis_kwargs = {"db": settings.storage.redis.db, "decode_responses": True}
        if settings.storage.redis.password:
            redis_kwargs["password"] = settings.storage.redis.password
        app_state.redis = aioredis.from_url(settings.storage.redis.url, **redis_kwargs)
        await app_state.redis.ping()
        logger.info(f"Redis connected: {settings.storage.redis.url}")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        app_state.redis = None

    # RCCL data store
    from app.collectors.rccl_data_store import RCCLDataStore

    app_state.rccl_data_store = RCCLDataStore(
        app_state.redis,
        snapshot_max=settings.storage.redis.snapshot_max_entries,
        event_max=settings.storage.redis.event_max_entries,
    )

    # Initialize SSH manager from cluster nodes (used for RCCL port-forward)
    nodes = settings.load_nodes_from_file()
    # Prefer node_groups GPU nodes config if available (more up-to-date)
    ng = getattr(app_state, "node_groups", None)
    if ng and ng.gpu_nodes.hosts:
        nodes = ng.gpu_nodes.hosts
        ssh_user = ng.gpu_nodes.ssh.username
        ssh_key = ng.gpu_nodes.ssh.key_file if ng.gpu_nodes.ssh.auth_method == "key" else None
    else:
        ssh_user = settings.ssh.username
        ssh_key = settings.ssh.key_file
    # Password: in-memory only, cleared on restart — will be set after user saves config
    node_pw = getattr(app_state, "node_groups_passwords", {}).get("gpu_nodes") or app_state.ssh_password

    if not nodes:
        logger.warning("No cluster nodes configured. Configure via web UI.")
    else:
        logger.info(
            f"Initializing SSH manager with {len(nodes)} nodes, user={ssh_user}, "
            f"key={'yes' if ssh_key else 'no'}, password={'yes' if node_pw else 'no (re-save config after restart)'}"
        )
        try:
            jump_cfg = settings.ssh.jump_host
            app_state.ssh_manager = SshManager(
                host_list=nodes,
                user=ssh_user,
                pkey=ssh_key,
                password=node_pw,
                timeout=settings.ssh.timeout,
                jump_host=jump_cfg.host if jump_cfg.enabled and jump_cfg.host else None,
                jump_user=jump_cfg.username if jump_cfg.enabled else None,
                jump_pkey=jump_cfg.key_file if jump_cfg.enabled else None,
            )
            logger.info("SSH manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SSH manager: {e}", exc_info=True)

    for node in nodes:
        if node not in app_state.node_health_status:
            app_state.node_health_status[node] = "healthy"
            app_state.node_failure_count[node] = 0

    # Start collectors
    logger.info("Starting collectors...")
    for cls in REGISTERED_COLLECTORS:
        c = cls()
        app_state.collectors[c.name] = c
        app_state.collector_tasks[c.name] = _start_collector_task(c)

    app_state.probe_task = asyncio.create_task(periodic_host_probe())
    logger.info("CVS Cluster Monitor started")

    yield

    logger.info("Shutting down CVS Cluster Monitor")
    app_state.is_collecting = False

    if app_state.ssh_manager:
        app_state.ssh_manager.destroy_clients()

    for task_attr, proc_attr, name in [
        ("daemon_task", "_daemon_proc", "cluster"),
        ("switch_daemon_task", "_switch_daemon_proc", "switch"),
    ]:
        task = getattr(app_state, task_attr, None)
        if task:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        proc = getattr(go_collector, proc_attr, None)
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass
            setattr(go_collector, proc_attr, None)
        logger.info(f"Go {name} daemon stopped")

    for task in app_state.collector_tasks.values():
        task.cancel()
    if app_state.collector_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*app_state.collector_tasks.values(), return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Collector tasks did not finish within 5s")

    if app_state.probe_task:
        app_state.probe_task.cancel()
        try:
            await asyncio.wait_for(app_state.probe_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    if app_state.redis:
        await app_state.redis.aclose()

    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="Real-time GPU cluster monitoring dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ssh_manager": app_state.ssh_manager is not None,
        "collecting": app_state.is_collecting,
        "clients": metrics_ws_manager.client_count,
        "daemon_ready": go_collector.is_daemon_ready(),
    }


static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    logger.info(f"Mounting static files from: {static_dir}")
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")

    @app.get("/")
    async def root():
        return {"name": settings.app_name, "status": "running", "collecting": app_state.is_collecting}
