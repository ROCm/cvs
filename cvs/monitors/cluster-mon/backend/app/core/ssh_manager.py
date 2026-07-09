"""
SshManager: SSH credential store, exec shims, and port-forward provider.

All SSH command execution is delegated to the Go daemon via go_collector._exec_one().
SshManager holds:
  - the host list and SSH credentials (used to build daemon startup args)
  - reachable / unreachable state (synced from daemon responses)
  - open_port_forward() — paramiko direct-tcpip tunnel for RCCL
  - backward-compatible exec/exec_async interface so collectors are unchanged

The class is intentionally thin.  It will be renamed (e.g. SshContext) in a
follow-up refactor once this changeset stabilises.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional

import paramiko

from app.core.go_collector import _exec_one, query_daemon_health
from app.core.ssh_port_forward import _run_bridge

logger = logging.getLogger(__name__)


class SshManager:
    """
    Credential store + port-forward provider.

    All command execution goes through the Go SSH daemon via go_collector.
    The Go daemon owns persistent SSH connections; SshManager owns paramiko
    connections only for open_port_forward() (RCCL port tunnelling).
    """

    def __init__(
        self,
        host_list: List[str],
        user: str,
        pkey: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
        # Jump host (optional)
        jump_host: Optional[str] = None,
        jump_user: Optional[str] = None,
        jump_pkey: Optional[str] = None,
        jump_password: Optional[str] = None,
    ) -> None:
        self._host_list: List[str] = list(host_list)
        self.user: str = user
        self.pkey: Optional[str] = pkey
        self.password: Optional[str] = password
        self.timeout: int = timeout

        self.jump_host: Optional[str] = jump_host
        self.jump_user: Optional[str] = jump_user
        self.jump_pkey: Optional[str] = jump_pkey
        self.jump_password: Optional[str] = jump_password

        # Reachability state — updated by exec/health responses.
        self.reachable_hosts: List[str] = list(host_list)
        self.unreachable_hosts: List[str] = []
        self._unreachable_reasons: Dict[str, str] = {}

        # Paramiko clients for port-forwarding (RCCL only).
        self._pf_clients: Dict[str, paramiko.SSHClient] = {}
        self._pf_lock = threading.Lock()

        # Jump-host paramiko connection for two-hop port-forwarding.
        self._jump_client: Optional[paramiko.SSHClient] = None
        self._jump_transport: Optional[paramiko.Transport] = None
        self._jump_lock = threading.Lock()

    # ─── properties for backward-compat ──────────────────────────────────────

    @property
    def host_list(self) -> List[str]:
        return self._host_list

    @host_list.setter
    def host_list(self, value: List[str]) -> None:
        self._host_list = list(value)

    @property
    def client(self):
        """Backward-compat stub — Go daemon owns connections, not Python."""
        return None

    @client.setter
    def client(self, _value) -> None:  # noqa: D401
        pass

    # ─── execution ────────────────────────────────────────────────────────────

    async def exec_async(
        self,
        cmd: str,
        timeout: int = 60,
        print_console: bool = True,
    ) -> Dict[str, str]:
        """
        Run *cmd* on all reachable hosts.  Non-blocking (runs _exec_one in a
        thread-pool worker).

        Returns {host: output} for ALL hosts — unreachable hosts get
        "ABORT: Host Unreachable Error" so collectors see a complete map.
        """
        results, unreachable = await asyncio.to_thread(_exec_one, cmd, timeout)
        self._sync_reachability(unreachable)
        return self._fill_unreachable(results)

    def exec(
        self,
        cmd: str,
        timeout: int = 60,
        print_console: bool = True,
    ) -> Dict[str, str]:
        """Synchronous exec — calls _exec_one directly (use from threads only)."""
        results, unreachable = _exec_one(cmd, timeout)
        self._sync_reachability(unreachable)
        return self._fill_unreachable(results)

    def exec_cmd_list(
        self,
        cmd_list: Dict[str, str],
        timeout: int = 60,
        print_console: bool = True,
    ) -> Dict[str, str]:
        """
        Run different commands on different hosts.

        Groups hosts by their command, calls _exec_one per unique command, and
        merges results.  Only hosts present in cmd_list appear in the output.
        """
        cmd_to_hosts: Dict[str, List[str]] = {}
        for host, cmd in cmd_list.items():
            cmd_to_hosts.setdefault(cmd, []).append(host)

        combined: Dict[str, str] = {}
        for cmd, hosts in cmd_to_hosts.items():
            results, unreachable = _exec_one(cmd, timeout)
            unreachable_set = set(unreachable)
            for host in hosts:
                if host in results:
                    combined[host] = results[host]
                elif host in unreachable_set:
                    combined[host] = "ABORT: Host Unreachable Error"
                else:
                    combined[host] = "ABORT: Host Unreachable Error"
        return combined

    # ─── reachability / health ────────────────────────────────────────────────

    def refresh_host_reachability(self) -> bool:
        """
        Query daemon's SSH-level fleet health and sync reachable/unreachable sets.

        Returns True if the reachable set changed.
        Returns False (no sync) when:
          - daemon is not ready, or
          - initial probe is still in progress (probe_status == "in-progress").

        Replaces TCP probe (host_probe.py).  Called from periodic_host_probe()
        via asyncio.to_thread().
        """
        health = query_daemon_health()
        if not health:
            return False

        old_set = set(self.reachable_hosts)
        unreachable_map: Dict[str, str] = health.get("unreachable", {})
        self._unreachable_reasons = dict(unreachable_map)
        self._sync_reachability(list(unreachable_map.keys()))
        return set(self.reachable_hosts) != old_set

    def recreate_client(self) -> None:
        """No-op — the Go daemon owns persistent connections."""

    def destroy_clients(self) -> None:
        """No-op — lifecycle task in main.py handles daemon shutdown."""
        self._close_pf_clients()

    def get_reachable_hosts(self) -> List[str]:
        return list(self.reachable_hosts)

    def get_unreachable_hosts(self) -> List[str]:
        return list(self.unreachable_hosts)

    # ─── port forwarding (paramiko, RCCL only) ────────────────────────────────

    @asynccontextmanager
    async def open_port_forward(
        self,
        node: str,
        remote_port: int,
    ) -> AsyncIterator[tuple]:
        """
        Open an SSH tunnel to node:remote_port.

        Direct (no jump host):
            paramiko → node → direct-tcpip → ::1:remote_port on node

        Jump host:
            uses the persistent jump_transport → direct-tcpip → ::1:remote_port
            (rcclras binds to IPv6 loopback only)

        Yields (asyncio.StreamReader, asyncio.StreamWriter).
        Uses a Unix socketpair() — no ephemeral TCP port, no TOCTOU race.
        """
        asyncio_end, thread_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            if self.jump_host:
                channel = await asyncio.to_thread(self._open_jump_channel, remote_port)
            else:
                channel = await asyncio.to_thread(self._open_direct_channel, node, remote_port)
        except Exception:
            asyncio_end.close()
            thread_end.close()
            raise

        _run_bridge(channel, thread_end)

        try:
            reader, writer = await asyncio.open_connection(sock=asyncio_end)
        except Exception:
            asyncio_end.close()
            channel.close()
            raise

        try:
            yield reader, writer
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            channel.close()
            thread_end.close()

    # ─── internal helpers ─────────────────────────────────────────────────────

    def _sync_reachability(self, unreachable: List[str]) -> None:
        unreachable_set = set(unreachable)
        self.reachable_hosts = [h for h in self._host_list if h not in unreachable_set]
        self.unreachable_hosts = [h for h in self._host_list if h in unreachable_set]

    def _fill_unreachable(self, results: Dict[str, str]) -> Dict[str, str]:
        """
        Ensure every host in _host_list appears in results.

        Hosts not attempted by the daemon (pre-existing unreachable) get an
        ABORT entry so collectors that iterate over the full host list don't
        silently skip them.
        """
        for host in self._host_list:
            if host not in results:
                results[host] = "ABORT: Host Unreachable Error"
        return results

    # ─── paramiko port-forward helpers ────────────────────────────────────────

    def _get_pf_transport(self, node: str) -> paramiko.Transport:
        """Get or create a dedicated paramiko SSH client for port-forwarding to node."""
        with self._pf_lock:
            client = self._pf_clients.get(node)
            transport = client.get_transport() if client else None
            if transport is None or not transport.is_active():
                new_client = paramiko.SSHClient()
                new_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                import os as _os

                # Only pass key_filename if the file actually exists
                _pkey = self.pkey
                if _pkey and not _os.path.exists(_os.path.expanduser(_pkey)):
                    _pkey = None
                connect_kwargs: dict = {
                    "username": self.user,
                    "timeout": self.timeout,
                    "look_for_keys": _pkey is None and self.password is None,
                }
                if self.password:
                    connect_kwargs["password"] = self.password
                elif _pkey:
                    connect_kwargs["key_filename"] = _pkey
                new_client.connect(node, **connect_kwargs)
                if client:
                    try:
                        client.close()
                    except Exception:
                        pass
                self._pf_clients[node] = new_client
            return self._pf_clients[node].get_transport()

    def _open_direct_channel(self, node: str, remote_port: int) -> paramiko.Channel:
        transport = self._get_pf_transport(node)
        return transport.open_channel(
            "direct-tcpip",
            ("::1", remote_port),
            ("127.0.0.1", 0),
        )

    def _ensure_jump_connection(self) -> None:
        """Establish (or re-establish) the jump-host paramiko connection."""
        with self._jump_lock:
            transport = self._jump_client.get_transport() if self._jump_client else None
            if transport is not None and transport.is_active():
                return
            if self._jump_client:
                try:
                    self._jump_client.close()
                except Exception:
                    pass
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            if self.jump_password:
                client.connect(
                    self.jump_host,
                    username=self.jump_user,
                    password=self.jump_password,
                    timeout=self.timeout,
                    banner_timeout=60,
                )
            else:
                client.connect(
                    self.jump_host,
                    username=self.jump_user,
                    key_filename=self.jump_pkey,
                    timeout=self.timeout,
                    banner_timeout=60,
                )
            self._jump_client = client
            self._jump_transport = client.get_transport()
            logger.info("Jump host paramiko connection established: %s", self.jump_host)

    def _open_jump_channel(self, remote_port: int) -> paramiko.Channel:
        self._ensure_jump_connection()
        with self._jump_lock:
            transport = self._jump_transport
        return transport.open_channel(
            "direct-tcpip",
            ("::1", remote_port),
            ("127.0.0.1", 0),
        )

    def _close_pf_clients(self) -> None:
        with self._pf_lock:
            for c in self._pf_clients.values():
                try:
                    c.close()
                except Exception:
                    pass
            self._pf_clients.clear()
        with self._jump_lock:
            if self._jump_client:
                try:
                    self._jump_client.close()
                except Exception:
                    pass
                self._jump_client = None
                self._jump_transport = None
