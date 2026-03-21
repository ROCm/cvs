"""Tests for ConnectionManager WebSocket pattern."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import ConnectionManager


@pytest.fixture
def manager():
    return ConnectionManager(max_queue_size=4)


@pytest.mark.asyncio
async def test_broadcast_is_nonblocking(manager):
    """broadcast() should return immediately even with no clients."""
    manager.broadcast({"type": "test"})  # should not raise or block


@pytest.mark.asyncio
async def test_broadcast_to_connected_client(manager):
    """broadcast() should put messages in client queues."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    await manager.connect(ws)

    manager.broadcast({"type": "test", "data": "hello"})

    # Give the sender task a moment to process
    await asyncio.sleep(0.05)

    # The sender task should have called send_json
    ws.send_json.assert_called()
    assert manager.client_count == 1


@pytest.mark.asyncio
async def test_slow_client_disconnected_on_full_queue(manager):
    """Client with full queue should be disconnected."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    # Make send_json block forever to fill the queue
    ws.send_json = AsyncMock(side_effect=lambda msg: asyncio.sleep(100))

    await manager.connect(ws)
    assert manager.client_count == 1

    # Fill the queue (maxsize=4)
    for i in range(10):
        manager.broadcast({"msg": i})

    await asyncio.sleep(0.1)  # let cleanup tasks run
    # Client should have been removed due to full queue
    assert manager.client_count == 0


@pytest.mark.asyncio
async def test_disconnect_cleans_up(manager):
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    await manager.connect(ws)
    assert manager.client_count == 1

    await manager.disconnect(ws)
    assert manager.client_count == 0
