"""
go_collector: socket-client module for the Go SSH daemon(s).

Two daemon instances are supported:
  CLUSTER_SOCKET — GPU/compute nodes (main daemon)
  SWITCH_SOCKET  — Switch trays (scale-up + scale-out, separate credentials)

Each daemon has its own Unix socket so they maintain independent connection
pools with independent credentials. Python directs commands to the right
daemon by passing the appropriate socket_path to each call.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CLUSTER_SOCKET: str = os.environ.get("GO_COLLECTOR_SOCKET", "/tmp/go-collector.sock")
SWITCH_SOCKET: str = os.environ.get("GO_SWITCH_COLLECTOR_SOCKET", "/tmp/go-switch.sock")

# Process references set by main.py lifecycle tasks.
_daemon_proc = None  # asyncio.subprocess.Process — cluster daemon
_switch_daemon_proc = None  # asyncio.subprocess.Process — switch daemon


# ─── readiness ────────────────────────────────────────────────────────────────


def _proc_ready(proc, socket_path: str) -> bool:
    if proc is None:
        return False
    if getattr(proc, "returncode", -1) is not None:
        return False
    return os.path.exists(socket_path)


def is_daemon_ready(socket_path: str = CLUSTER_SOCKET) -> bool:
    if socket_path == SWITCH_SOCKET:
        return _proc_ready(_switch_daemon_proc, socket_path)
    return _proc_ready(_daemon_proc, socket_path)


# ─── low-level socket I/O ─────────────────────────────────────────────────────


def _send_recv(msg: dict, timeout: int = 120, socket_path: str = CLUSTER_SOCKET) -> Optional[dict]:
    if not is_daemon_ready(socket_path):
        return None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(socket_path)
        try:
            sock.sendall(json.dumps(msg).encode() + b"\n")
            buf = bytearray()
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
            return json.loads(buf.decode())
        finally:
            sock.close()
    except Exception as exc:
        logger.warning("go_collector socket error (%s): %s", socket_path, exc)
        return None


# ─── public API ───────────────────────────────────────────────────────────────


def _exec_one(cmd: str, timeout: int = 60, socket_path: str = CLUSTER_SOCKET) -> Tuple[Dict[str, str], List[str]]:
    """Run cmd on ALL reachable hosts managed by the daemon at socket_path."""
    resp = _send_recv(
        {"id": str(uuid.uuid4()), "type": "exec", "command": cmd, "timeout_s": timeout},
        timeout=timeout + 30,
        socket_path=socket_path,
    )
    if resp is None:
        return {}, []
    return resp.get("results", {}), resp.get("unreachable", [])


def _exec_on_hosts(
    hosts: List[str],
    cmd: str,
    timeout: int = 60,
    socket_path: str = CLUSTER_SOCKET,
) -> Dict[str, str]:
    """
    Run cmd on a specific subset of hosts via the daemon at socket_path.
    Returns {host: output}; unreachable hosts get "ABORT: Host Unreachable Error".
    """
    resp = _send_recv(
        {
            "id": str(uuid.uuid4()),
            "type": "exec",
            "command": cmd,
            "hosts": hosts,
            "timeout_s": timeout,
        },
        timeout=timeout + 30,
        socket_path=socket_path,
    )
    if resp is None:
        return {h: "ABORT: Host Unreachable Error" for h in hosts}
    results: Dict[str, str] = resp.get("results", {})
    unreachable: List[str] = resp.get("unreachable", [])
    for h in hosts:
        if h not in results or h in unreachable:
            results[h] = "ABORT: Host Unreachable Error"
    return {h: results.get(h, "ABORT: Host Unreachable Error") for h in hosts}


def query_daemon_health(socket_path: str = CLUSTER_SOCKET) -> Optional[dict]:
    resp = _send_recv({"id": str(uuid.uuid4()), "type": "health"}, timeout=30, socket_path=socket_path)
    if resp is None:
        return None
    if resp.get("probe_status") == "in-progress":
        return None
    return resp


def _refresh_nodes_in_daemon(
    hosts: List[str],
    user: str = "",
    key_path: str = "",
    key_bytes: Optional[bytes] = None,
    password: Optional[str] = None,
    group: str = "",
    jump_host: str = "",
    jump_user: str = "",
    jump_key: str = "",
    jump_password: Optional[str] = None,
    socket_path: str = CLUSTER_SOCKET,
) -> dict:
    """
    Register hosts and credentials with the daemon at socket_path.

    Credential priority: key_bytes > key_path > password.
    group is informational metadata logged by the daemon.
    """
    import base64

    msg: dict = {"id": str(uuid.uuid4()), "type": "refresh_nodes", "hosts": hosts}
    if user:
        msg["user"] = user
    if group:
        msg["group"] = group
    if password:
        msg["password"] = password
    elif key_bytes is not None:
        msg["key_bytes"] = base64.b64encode(key_bytes).decode()
    elif key_path:
        msg["key_path"] = key_path
    if jump_host:
        msg["jump_host"] = jump_host
        if jump_user:
            msg["jump_user"] = jump_user
        if jump_password:
            msg["jump_password"] = jump_password
        elif jump_key:
            msg["jump_key"] = jump_key
    return _send_recv(msg, timeout=60, socket_path=socket_path) or {}
