"""
Tests for RCCLRasClient against MockNcclRasServer.
"""
import asyncio
import pytest

from app.collectors.rccl_ras_client import (
    RCCLRasClient,
    ProtocolError,
    ProtocolVersionError,
    ProtocolVersion,
)
from tests.mock_ncclras_server import MockNcclRasServer


SAMPLE_STATUS = b"""RCCL version 2.28.3 compiled with ROCm 6.0

Job summary
===========

  Nodes  Processes
(total)   per node
      2          8

Communicator abc123 HEALTHY
  Ranks: 16 total, 16 responding, 0 missing
"""


@pytest.fixture
async def mock_server():
    server = MockNcclRasServer(fixture_data=SAMPLE_STATUS, protocol_version=2)
    port = await server.start()
    yield port
    await server.stop()


async def _connect(port: int):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    return RCCLRasClient(reader, writer)


@pytest.mark.asyncio
async def test_handshake_returns_server_version(mock_server):
    client = await _connect(mock_server)
    version = await client.handshake()
    assert version == 2
    assert client.server_protocol == 2
    client._writer.close()


@pytest.mark.asyncio
async def test_set_timeout_ok(mock_server):
    client = await _connect(mock_server)
    await client.handshake()
    await client.set_timeout(10)
    client._writer.close()


@pytest.mark.asyncio
async def test_get_status_returns_text(mock_server):
    client = await _connect(mock_server)
    await client.handshake()
    await client.set_timeout(10)
    text = await client.get_status(verbose=True)
    assert "RCCL version" in text
    assert "Communicator" in text


@pytest.mark.asyncio
async def test_set_format_raises_on_protocol_2(mock_server):
    """set_format requires protocol 3+ — should raise on a protocol 2 server."""
    client = await _connect(mock_server)
    await client.handshake()
    assert client.server_protocol == 2
    with pytest.raises(ProtocolVersionError):
        await client.set_format("json")
    client._writer.close()


@pytest.mark.asyncio
async def test_start_monitor_raises_on_protocol_2(mock_server):
    """start_monitor requires protocol 4+ — should raise on a protocol 2 server."""
    client = await _connect(mock_server)
    await client.handshake()
    with pytest.raises(ProtocolVersionError):
        async for _ in client.start_monitor():
            pass
    client._writer.close()


def test_rccl_models_import():
    from app.models.rccl_models import (
        RCCLSnapshot,
        RCCLJobState,
        NCCLFunction,
        RCCLMarker,
    )
    snapshot = RCCLSnapshot.empty()
    assert snapshot.state == RCCLJobState.NO_JOB
    assert snapshot.communicators == []


@pytest.mark.asyncio
async def test_rccl_data_store_degrades_without_redis():
    from app.collectors.rccl_data_store import RCCLDataStore
    store = RCCLDataStore(redis_client=None)
    await store.push_snapshot({"timestamp": 1.0})
    await store.push_event({"timestamp": 1.0})
    result = await store.get_recent_snapshots()
    assert result == []
    result = await store.get_current_snapshot()
    assert result is None


def test_ncclfunction_enum_str_values():
    from app.models.rccl_models import NCCLFunction
    assert NCCLFunction.ALL_REDUCE == "AllReduce"
    assert NCCLFunction.ALL_GATHER == "AllGather"
    assert NCCLFunction.SEND == "Send"


def test_rccl_job_state_values():
    from app.models.rccl_models import RCCLJobState
    assert RCCLJobState.NO_JOB == "no_job"
    assert RCCLJobState.HEALTHY == "healthy"
    assert RCCLJobState.DEGRADED == "degraded"
