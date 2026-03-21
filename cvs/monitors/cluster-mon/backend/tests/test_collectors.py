"""
Tests for GPUMetricsCollector and NICMetricsCollector as BaseCollector subclasses.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.collectors.base import CollectorState, CollectorResult
from app.collectors.gpu_collector import GPUMetricsCollector
from app.collectors.nic_collector import NICMetricsCollector


def test_gpu_collector_has_base_collector_attrs():
    assert hasattr(GPUMetricsCollector, 'name')
    assert GPUMetricsCollector.name == "gpu"
    assert hasattr(GPUMetricsCollector, 'poll_interval')
    assert hasattr(GPUMetricsCollector, 'collect_timeout')
    assert GPUMetricsCollector.critical is True


def test_nic_collector_has_base_collector_attrs():
    assert hasattr(NICMetricsCollector, 'name')
    assert NICMetricsCollector.name == "nic"
    assert hasattr(NICMetricsCollector, 'poll_interval')
    assert hasattr(NICMetricsCollector, 'collect_timeout')
    assert NICMetricsCollector.critical is True


@pytest.mark.asyncio
async def test_gpu_collector_collect_returns_collector_result():
    collector = GPUMetricsCollector()
    ssh_manager = MagicMock()

    # Mock collect_all_metrics to return a simple metrics dict
    fake_metrics = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "utilization": {"node1": {"gpu0": 80}},
    }
    collector.collect_all_metrics = AsyncMock(return_value=fake_metrics)

    result = await collector.collect(ssh_manager)

    assert isinstance(result, CollectorResult)
    assert result.collector_name == "gpu"
    assert result.state == CollectorState.OK
    assert result.data == fake_metrics


@pytest.mark.asyncio
async def test_gpu_collector_collect_handles_exception():
    collector = GPUMetricsCollector()
    ssh_manager = MagicMock()
    collector.collect_all_metrics = AsyncMock(side_effect=RuntimeError("SSH failed"))

    result = await collector.collect(ssh_manager)

    assert result.state == CollectorState.ERROR
    assert "SSH failed" in result.error


@pytest.mark.asyncio
async def test_nic_collector_collect_returns_collector_result():
    collector = NICMetricsCollector()
    ssh_manager = MagicMock()
    fake_metrics = {"rdma_links": {"node1": {}}}
    collector.collect_all_metrics = AsyncMock(return_value=fake_metrics)

    result = await collector.collect(ssh_manager)

    assert isinstance(result, CollectorResult)
    assert result.collector_name == "nic"
    assert result.state == CollectorState.OK


def test_registered_collectors_list():
    pytest.importorskip("fastapi")
    from app.main import REGISTERED_COLLECTORS
    from app.collectors.base import BaseCollector
    assert len(REGISTERED_COLLECTORS) >= 2
    for cls in REGISTERED_COLLECTORS:
        assert issubclass(cls, BaseCollector)
