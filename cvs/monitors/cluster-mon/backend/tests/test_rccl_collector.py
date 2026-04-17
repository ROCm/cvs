"""Tests for RCCLCollector."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.collectors.base import CollectorState
from app.collectors.rccl_collector import RCCLCollector
from app.models.rccl_models import RCCLJobState


def _make_app_state(healthy_nodes=None):
    app_state = MagicMock()
    app_state.node_health_status = {n: "healthy" for n in (healthy_nodes or ["node1"])}
    app_state.rccl_data_store = None
    app_state.latest_rccl_snapshot = None
    app_state.collector_results = {}
    app_state.latest_metrics = {}
    app_state.is_collecting = False
    app_state.probe_requested = None
    return app_state


def test_rccl_collector_attrs():
    assert RCCLCollector.name == "rccl"
    assert RCCLCollector.critical is False
    assert hasattr(RCCLCollector, 'poll_interval')
    assert hasattr(RCCLCollector, 'collect_timeout')


def test_healthy_nodes_returns_all_healthy():
    collector = RCCLCollector()
    app_state = _make_app_state(["node1", "node2"])
    nodes = collector._healthy_nodes(app_state)
    assert set(nodes) == {"node1", "node2"}


def test_healthy_nodes_returns_empty_when_all_unhealthy():
    collector = RCCLCollector()
    app_state = MagicMock()
    app_state.node_health_status = {"node1": "unhealthy", "node2": "unreachable"}
    assert collector._healthy_nodes(app_state) == []


@pytest.mark.asyncio
async def test_collect_returns_unreachable_when_no_healthy_nodes():
    collector = RCCLCollector()
    collector._app_state = _make_app_state()
    collector._app_state.node_health_status = {}  # no nodes

    ssh_manager = MagicMock()
    result = await collector.collect(ssh_manager)
    assert result.state == CollectorState.UNREACHABLE
    assert collector.job_state == RCCLJobState.UNREACHABLE


@pytest.mark.asyncio
async def test_collect_returns_no_service_on_connection_refused():
    collector = RCCLCollector()
    app_state = _make_app_state(["node1"])
    collector._app_state = app_state

    ssh_manager = MagicMock()
    ssh_mock_ctx = AsyncMock()
    ssh_mock_ctx.__aenter__ = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    ssh_mock_ctx.__aexit__ = AsyncMock(return_value=False)
    ssh_manager.open_port_forward = MagicMock(return_value=ssh_mock_ctx)

    result = await collector.collect(ssh_manager)
    assert result.state == CollectorState.NO_SERVICE
    assert collector.job_state == RCCLJobState.NO_JOB


@pytest.mark.asyncio
async def test_collect_returns_error_when_app_state_not_set():
    collector = RCCLCollector()
    # _app_state is None (run() not called)
    result = await collector.collect(MagicMock())
    assert result.state == CollectorState.ERROR


def test_health_from_snapshot_healthy():
    from app.models.rccl_models import RCCLSnapshot, RCCLCommunicator, RCCLJobState
    collector = RCCLCollector()
    snapshot = RCCLSnapshot(
        timestamp=1.0,
        state=RCCLJobState.HEALTHY,
        communicators=[],
    )
    assert collector._health_from_snapshot(snapshot) == RCCLJobState.HEALTHY


@pytest.mark.asyncio
async def test_state_change_event_emitted_on_job_start():
    """Transition NO_JOB → HEALTHY should push a job_start event."""
    collector = RCCLCollector()
    collector.job_state = RCCLJobState.NO_JOB

    data_store = MagicMock()
    data_store.push_event = AsyncMock()

    app_state = MagicMock()
    app_state.rccl_data_store = data_store

    await collector._push_state_event(
        RCCLJobState.NO_JOB, RCCLJobState.HEALTHY, app_state, leader="node1"
    )

    data_store.push_event.assert_called_once()
    event = data_store.push_event.call_args[0][0]
    assert event["event_type"] == "job_start"
    assert event["from_state"] == RCCLJobState.NO_JOB
    assert event["to_state"] == RCCLJobState.HEALTHY
    assert event["leader_node"] == "node1"


@pytest.mark.asyncio
async def test_no_event_when_state_unchanged():
    """No event should be emitted when state does not change."""
    collector = RCCLCollector()
    data_store = MagicMock()
    data_store.push_event = AsyncMock()
    app_state = MagicMock()
    app_state.rccl_data_store = data_store

    await collector._push_state_event(
        RCCLJobState.HEALTHY, RCCLJobState.HEALTHY, app_state
    )
    data_store.push_event.assert_not_called()


def test_rccl_endpoints_importable():
    from app.api.rccl_endpoints import router
    routes = [r.path for r in router.routes]
    assert any("status" in r for r in routes)
    assert any("markers" in r for r in routes)
