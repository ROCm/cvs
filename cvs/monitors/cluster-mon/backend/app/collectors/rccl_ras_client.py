"""
Async rcclras TCP client for the CVS RCCL monitoring extension.

Speaks the rcclras wire protocol (newline-terminated ASCII over TCP).
Connection is owned by the caller's context manager (ssh_manager.open_port_forward).

Warning: Protocol version caveat: Values 3 and 4 for JSON_FORMAT and MONITOR_MODE
are ASSUMPTIONS -- not verified against actual rcclras server responses for
v2.28.9 and v2.29.2. Verify by running a v2.28.9 rcclras server and checking
its handshake response before implementing version-gated features. If the
server always responds SERVER PROTOCOL 2, the version guards will never
activate and an alternative feature-detection mechanism will be required.
"""

import asyncio
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class ProtocolError(Exception):
    """Raised when the rcclras server responds unexpectedly."""


class ProtocolVersionError(ProtocolError):
    """Raised when a feature requires a higher protocol version than the server supports."""


class ProtocolVersion:
    TEXT_ONLY = 2    # v2.28.3: STATUS, VERBOSE STATUS, TIMEOUT only
    JSON_FORMAT = 3  # v2.28.9+: adds SET FORMAT json (ASSUMPTION — verify)
    MONITOR_MODE = 4  # v2.29.2+: adds MONITOR [groups] (ASSUMPTION — verify)


class RCCLRasClient:
    """
    Async rcclras TCP client.

    Takes pre-connected reader/writer (from ssh_manager.open_port_forward()).
    Connection lifetime is managed by the caller's context manager -- do not
    close writer here.

    Version guards: methods requiring features not in the server's protocol
    version raise ProtocolVersionError rather than sending unknown commands
    (which would return ERROR: Unknown command and stall the reader).
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self._reader = reader
        self._writer = writer
        self.server_protocol: int = 0  # Set during handshake

    async def handshake(self) -> int:
        """
        Send CLIENT PROTOCOL 2, read SERVER PROTOCOL N.
        Returns server protocol version.
        Protocol mismatch is logged but not fatal (matches rcclras behavior).
        """
        self._writer.write(b"CLIENT PROTOCOL 2\n")
        await self._writer.drain()
        line = await asyncio.wait_for(self._reader.readline(), timeout=5.0)
        version_str = line.decode().strip().removeprefix("SERVER PROTOCOL ")
        try:
            self.server_protocol = int(version_str)
        except ValueError:
            raise ProtocolError(f"Unexpected handshake response: {line!r}")
        return self.server_protocol

    async def set_timeout(self, secs: int) -> None:
        """Set collective timeout. Available in all versions (v2.28.3+)."""
        self._writer.write(f"TIMEOUT {secs}\n".encode())
        await self._writer.drain()
        line = await asyncio.wait_for(self._reader.readline(), timeout=5.0)
        if line.decode().strip() != "OK":
            raise ProtocolError(f"Expected OK after TIMEOUT, got: {line!r}")

    async def set_format(self, fmt: str = "json") -> None:
        """
        Set output format. Available only in v2.28.9+ (protocol 3+).
        Raises ProtocolVersionError if server does not support it.
        """
        if self.server_protocol < ProtocolVersion.JSON_FORMAT:
            raise ProtocolVersionError(
                f"SET FORMAT requires protocol {ProtocolVersion.JSON_FORMAT}+, "
                f"server is {self.server_protocol}"
            )
        self._writer.write(f"SET FORMAT {fmt}\n".encode())
        await self._writer.drain()
        line = await asyncio.wait_for(self._reader.readline(), timeout=5.0)
        if line.decode().strip() != "OK":
            raise ProtocolError(f"Expected OK after SET FORMAT, got: {line!r}")

    async def get_status(self, verbose: bool = True) -> str:
        """
        Send STATUS or VERBOSE STATUS. Reads until EOF (server closes after dump).
        Returns raw text. The caller is responsible for parsing.
        Available in all versions.
        """
        cmd = b"VERBOSE STATUS\n" if verbose else b"STATUS\n"
        self._writer.write(cmd)
        await self._writer.drain()
        data = await asyncio.wait_for(self._reader.read(-1), timeout=30.0)
        return data.decode()

    async def start_monitor(self, groups: str = "all") -> AsyncIterator[str]:
        """
        Send MONITOR [groups] and yield lines until connection closes.
        Available only in v2.29.2+ (protocol 4+).
        Raises ProtocolVersionError if server does not support it.
        """
        if self.server_protocol < ProtocolVersion.MONITOR_MODE:
            raise ProtocolVersionError(
                f"MONITOR requires protocol {ProtocolVersion.MONITOR_MODE}+, "
                f"server is {self.server_protocol}"
            )
        self._writer.write(f"MONITOR {groups}\n".encode())
        await self._writer.drain()
        ok = await asyncio.wait_for(self._reader.readline(), timeout=5.0)
        if ok.decode().strip() != "OK":
            raise ProtocolError(f"Expected OK after MONITOR, got: {ok!r}")
        async for line in self._reader:
            yield line.decode()
