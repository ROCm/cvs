"""
go_collector: pure socket-client module for the Go SSH daemon.

No process management here — daemon lifecycle (spawn, watch, respawn) lives in
main.py._run_daemon_lifecycle(). This module only holds the socket path, the
daemon process reference (set by main.py), and the three blocking socket calls
that Python threads use to communicate with the daemon.

Thread-safety: each call opens an independent Unix socket connection so responses
are always routed back to the caller that sent the request.  Multiple concurrent
callers are safe.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Env-overridable paths so integration tests can redirect without changing config.
_SOCKET_PATH: str = os.environ.get("GO_COLLECTOR_SOCKET", "/tmp/go-collector.sock")

# asyncio.subprocess.Process set by main.py._run_daemon_lifecycle().
# Accessed read-only here to guard against dead-daemon scenarios.
_daemon_proc = None  # type: Optional[object]  # asyncio.subprocess.Process


# ─── readiness ────────────────────────────────────────────────────────────────

def is_daemon_ready() -> bool:
    """Return True if the daemon process is alive and its socket is visible."""
    proc = _daemon_proc
    if proc is None:
        return False
    # asyncio.subprocess.Process.returncode is None while the process is running.
    if getattr(proc, "returncode", -1) is not None:
        return False
    return os.path.exists(_SOCKET_PATH)


# ─── low-level socket I/O ─────────────────────────────────────────────────────

def _send_recv(msg: dict, timeout: int = 120) -> Optional[dict]:
    """
    Open a fresh UDS connection, send one JSON line, read one JSON response line.

    Called from thread-pool workers (asyncio.to_thread).  Each call gets its own
    file descriptor so concurrent requests never mix their responses.

    Returns None on any I/O or decode error.
    """
    if not is_daemon_ready():
        return None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(_SOCKET_PATH)
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
        logger.warning("go_collector socket error: %s", exc)
        return None


# ─── public API ───────────────────────────────────────────────────────────────

def _exec_one(cmd: str, timeout: int = 60) -> Tuple[Dict[str, str], List[str]]:
    """
    Run *cmd* on all reachable hosts.

    Returns (results, unreachable) where:
      results     – {host: output_str} for every host that was attempted.
                    Pruned / connection-failed hosts have "ABORT: Host Unreachable Error".
      unreachable – list of hosts currently known to be unreachable (not attempted).

    Returns ({}, []) if the daemon is not ready.
    Called from thread-pool workers.
    """
    resp = _send_recv(
        {"id": str(uuid.uuid4()), "type": "exec", "command": cmd, "timeout_s": timeout},
        timeout=timeout + 30,
    )
    if resp is None:
        return {}, []
    return resp.get("results", {}), resp.get("unreachable", [])


def query_daemon_health() -> Optional[dict]:
    """
    Fetch fleet SSH health from the daemon.

    Returns the parsed response dict, or None when:
      - the daemon is not ready, or
      - the initial probe is still in progress (probe_status == "in-progress").

    Callers should treat None as "no data yet, skip sync".
    Called from thread-pool workers.
    """
    resp = _send_recv(
        {"id": str(uuid.uuid4()), "type": "health"},
        timeout=30,
    )
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
) -> dict:
    """
    Send a refresh_nodes message with the full current host list.

    The daemon computes the diff (added / removed) internally, closes connections
    for removed hosts, and starts a background ProbeSubset for added hosts.
    A reprobe nudge is always sent, so unreachable nodes are retried immediately.

    Credential update options (all optional, can combine):
      user       – SSH username change; daemon drops all connections and re-dials.
      key_path   – New key file path on the container (file-based, lazy read).
      key_bytes  – Raw PEM bytes of the node SSH private key (in-memory delivery).
                   Encoded as standard base64 in JSON, decoded by Go automatically.
                   key_bytes takes priority over key_path when both are provided.
                   Use this to deliver a key fetched from the jump host via SFTP —
                   the key is never written to the container filesystem.

    Returns {"added": [...], "removed": [...], "total": N} or {} on error.
    Called from thread-pool workers.
    """
    import base64
    msg: dict = {"id": str(uuid.uuid4()), "type": "refresh_nodes", "hosts": hosts}
    if user:
        msg["user"] = user
    if key_bytes is not None:
        # Go's json.Unmarshal decodes []byte fields from standard base64.
        msg["key_bytes"] = base64.b64encode(key_bytes).decode()
    elif key_path:
        msg["key_path"] = key_path
    return _send_recv(msg, timeout=60) or {}
