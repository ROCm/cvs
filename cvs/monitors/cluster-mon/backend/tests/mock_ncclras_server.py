"""
Mock ncclras TCP server for unit testing.
Replays a fixture response over TCP so collector tests run without a real RCCL job.
"""

import asyncio
from typing import Optional


class MockNcclRasServer:
    """Minimal ncclras server for unit testing. Replays a fixed fixture response."""

    def __init__(self, fixture_data: bytes, protocol_version: int = 2):
        self.fixture_data = fixture_data
        self.protocol_version = protocol_version
        self._server: Optional[asyncio.Server] = None

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            assert line.lower().startswith(b"client protocol"), \
                f"Expected CLIENT PROTOCOL, got: {line!r}"

            writer.write(f"SERVER PROTOCOL {self.protocol_version}\n".encode())
            await writer.drain()

            # Handle optional TIMEOUT command
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if line.lower().startswith(b"timeout"):
                writer.write(b"OK\n")
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=2.0)

            # Expect STATUS or VERBOSE STATUS
            assert b"status" in line.lower(), \
                f"Expected STATUS command, got: {line!r}"

            writer.write(self.fixture_data)
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Start the server. Returns the bound port (use port=0 for ephemeral)."""
        self._server = await asyncio.start_server(self.handle, host, port)
        return self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
