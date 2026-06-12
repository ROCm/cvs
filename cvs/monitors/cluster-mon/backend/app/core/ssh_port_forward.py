"""
Shared SSH port-forwarding bridge for CVS cluster-mon.

_run_bridge() creates a bidirectional byte-copy between a paramiko Channel
and a Unix socketpair. Used by both Pssh and JumpHostPssh to implement
open_port_forward() without ephemeral TCP port allocation.
"""

import socket
import threading
import paramiko
import logging

logger = logging.getLogger(__name__)


def _run_bridge(channel: paramiko.Channel, sock: socket.socket) -> None:
    """
    Start two daemon threads that copy bytes bidirectionally between
    a paramiko channel and a Unix socket.

    When either direction closes (clean EOF or exception), close_all()
    is called, which closes both the channel and the socket. This causes
    the other direction's recv() to return empty bytes or raise, causing
    the other thread to also exit cleanly. No thread leaks.

    Args:
        channel: Open paramiko.Channel (e.g., from transport.open_channel)
        sock: One end of a socketpair (the thread end, not the asyncio end)
    """

    def copy(src_recv, dst_send, cleanup):
        try:
            while True:
                data = src_recv(4096)
                if not data:
                    break
                dst_send(data)
        except Exception:
            pass
        finally:
            cleanup()

    def close_all():
        try:
            channel.close()
        except Exception:
            pass
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

    threading.Thread(
        target=copy,
        args=(channel.recv, sock.sendall, close_all),
        daemon=True,
        name=f"bridge-ch→sock-{id(channel)}",
    ).start()
    threading.Thread(
        target=copy,
        args=(sock.recv, channel.sendall, close_all),
        daemon=True,
        name=f"bridge-sock→ch-{id(channel)}",
    ).start()
