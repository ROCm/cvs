"""
Tests for SSH port-forwarding bridge (_run_bridge).
Tests use real socketpairs and mock paramiko channels.
"""
import asyncio
import socket
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from app.core.ssh_port_forward import _run_bridge


class MockChannel:
    """Minimal mock paramiko channel backed by a real socket."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self.closed = False

    def recv(self, nbytes: int) -> bytes:
        try:
            return self._sock.recv(nbytes)
        except Exception:
            return b""

    def sendall(self, data: bytes) -> None:
        self._sock.sendall(data)

    def close(self) -> None:
        self.closed = True
        try:
            self._sock.close()
        except Exception:
            pass


def test_run_bridge_forwards_data_ch_to_sock():
    """Data written to channel side arrives on the socket side."""
    # Create two connected socket pairs to simulate channel <-> bridge <-> user
    ch_side, bridge_ch_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    bridge_sock_end, user_sock_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    channel = MockChannel(ch_side)
    _run_bridge(channel, bridge_sock_end)

    # Write to channel side -> should arrive at user_sock_end
    bridge_ch_end.sendall(b"hello from channel")
    user_sock_end.settimeout(2.0)
    data = user_sock_end.recv(100)
    assert data == b"hello from channel"

    bridge_ch_end.close()
    user_sock_end.close()


def test_run_bridge_forwards_data_sock_to_ch():
    """Data written to the socket side arrives on the channel side."""
    ch_side, bridge_ch_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    bridge_sock_end, user_sock_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    channel = MockChannel(ch_side)
    _run_bridge(channel, bridge_sock_end)

    # Write to user_sock_end -> should arrive at bridge_ch_end (channel side)
    user_sock_end.sendall(b"hello from socket")
    bridge_ch_end.settimeout(2.0)
    data = bridge_ch_end.recv(100)
    assert data == b"hello from socket"

    bridge_ch_end.close()
    user_sock_end.close()


def test_run_bridge_closes_both_on_channel_close():
    """When the channel closes, the socket side also closes (no thread leak)."""
    ch_side, bridge_ch_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    bridge_sock_end, user_sock_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    channel = MockChannel(ch_side)
    _run_bridge(channel, bridge_sock_end)

    # Close the channel side -- this sends EOF
    bridge_ch_end.close()
    ch_side.close()

    # The user_sock_end should eventually get EOF too
    user_sock_end.settimeout(2.0)
    data = user_sock_end.recv(100)
    assert data == b""  # EOF propagated

    user_sock_end.close()


def test_run_bridge_daemon_threads():
    """Bridge threads must be daemon threads (don't block process exit)."""
    ch_side, bridge_ch_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    bridge_sock_end, user_sock_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    before = {t.name for t in threading.enumerate()}
    channel = MockChannel(ch_side)
    _run_bridge(channel, bridge_sock_end)

    # Find the new threads
    after = {t for t in threading.enumerate() if t.name not in before}
    bridge_threads = [t for t in after if "bridge" in t.name]
    assert len(bridge_threads) == 2
    assert all(t.daemon for t in bridge_threads)

    bridge_ch_end.close()
    user_sock_end.close()
