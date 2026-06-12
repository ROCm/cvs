"""
Tests for the per-node RAS capability map (probe + parser routing).

Covers:
- _ensure_capability: probe result stored in app_state.node_capabilities
- Cache hit (TTL not expired) skips re-probe
- Cache miss (TTL expired) re-probes
- _parse_response: routes to JSON parser when json_ras=True
- _parse_response: routes to text parser when json_ras=False
- NodeRCCLCapability fields: detected_rccl_version back-filled after first JSON parse
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.collectors.rccl_collector import RCCLCollector
from app.collectors.rccl_ras_client import ProtocolVersionError
from app.models.rccl_models import NodeRCCLCapability, RCCLJobState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_state(node_capabilities=None):
    app_state = MagicMock()
    app_state.node_health_status = {"node1": "healthy"}
    app_state.node_capabilities = node_capabilities if node_capabilities is not None else {}
    app_state.rccl_data_store = None
    app_state.latest_rccl_snapshot = None
    return app_state


def _make_client(set_format_ok=True):
    client = MagicMock()
    if set_format_ok:
        client.set_format = AsyncMock(return_value=None)
    else:
        client.set_format = AsyncMock(side_effect=ProtocolVersionError("protocol too old"))
    return client


# ---------------------------------------------------------------------------
# _ensure_capability: probe path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_json_supported_stores_capability():
    collector = RCCLCollector()
    app_state = _make_app_state()
    client = _make_client(set_format_ok=True)

    cap = await collector._ensure_capability(client, "node1", app_state)

    assert cap.json_ras is True
    assert cap.detection_method == "probe"
    assert "node1" in app_state.node_capabilities
    assert app_state.node_capabilities["node1"].json_ras is True


@pytest.mark.asyncio
async def test_probe_json_not_supported_stores_capability():
    collector = RCCLCollector()
    app_state = _make_app_state()
    client = _make_client(set_format_ok=False)

    cap = await collector._ensure_capability(client, "node1", app_state)

    assert cap.json_ras is False
    assert app_state.node_capabilities["node1"].json_ras is False


@pytest.mark.asyncio
async def test_probe_ttl_longer_when_json_supported():
    collector = RCCLCollector()
    app_state = _make_app_state()
    client = _make_client(set_format_ok=True)

    cap = await collector._ensure_capability(client, "node1", app_state)
    # JSON-capable nodes get 1-hour TTL; text-only gets 5-minute TTL
    assert cap.ttl == 3600.0


@pytest.mark.asyncio
async def test_probe_ttl_shorter_for_text_only():
    collector = RCCLCollector()
    app_state = _make_app_state()
    client = _make_client(set_format_ok=False)

    cap = await collector._ensure_capability(client, "node1", app_state)
    assert cap.ttl == 300.0


# ---------------------------------------------------------------------------
# _ensure_capability: cache hit / miss
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_skips_probe():
    collector = RCCLCollector()
    fresh_cap = NodeRCCLCapability(
        json_ras=True,
        detected_rccl_version="2.28.9",
        detection_method="probe",
        probed_at=time.time(),
        ttl=3600.0,
    )
    app_state = _make_app_state(node_capabilities={"node1": fresh_cap})
    client = _make_client(set_format_ok=True)

    cap = await collector._ensure_capability(client, "node1", app_state)

    # Should return cached record without probing
    client.set_format.assert_not_awaited()
    assert cap is fresh_cap


@pytest.mark.asyncio
async def test_cache_miss_on_expired_ttl_reprobes():
    collector = RCCLCollector()
    stale_cap = NodeRCCLCapability(
        json_ras=False,
        detected_rccl_version=None,
        detection_method="probe",
        probed_at=time.time() - 400.0,  # 400s ago, TTL=300 → expired
        ttl=300.0,
    )
    app_state = _make_app_state(node_capabilities={"node1": stale_cap})
    client = _make_client(set_format_ok=True)  # server upgraded since last probe

    cap = await collector._ensure_capability(client, "node1", app_state)

    client.set_format.assert_awaited_once()
    assert cap.json_ras is True  # fresh probe result


@pytest.mark.asyncio
async def test_probe_error_falls_back_to_text_only():
    """Unexpected error during probe (not ProtocolVersionError) → text-only safe default."""
    collector = RCCLCollector()
    app_state = _make_app_state()
    client = MagicMock()
    client.set_format = AsyncMock(side_effect=asyncio.TimeoutError())

    cap = await collector._ensure_capability(client, "node1", app_state)

    assert cap.json_ras is False


# ---------------------------------------------------------------------------
# _parse_response: routing
# ---------------------------------------------------------------------------


def test_parse_response_uses_json_parser_when_json_ras():
    collector = RCCLCollector()
    cap = NodeRCCLCapability(
        json_ras=True,
        detected_rccl_version=None,
        detection_method="probe",
    )
    json_input = (
        '{"nccl_version": "2.28.9", "cuda_runtime_version": 70226015, '
        '"cuda_driver_version": 70226015, "communicators_count": 0, "communicators": []}'
    )
    snapshot = collector._parse_response(json_input, "node1", cap)
    assert snapshot.state == RCCLJobState.HEALTHY
    assert snapshot.job_summary.rccl_version == "2.28.9"


def test_parse_response_uses_text_parser_when_not_json_ras():
    collector = RCCLCollector()
    cap = NodeRCCLCapability(
        json_ras=False,
        detected_rccl_version=None,
        detection_method="probe",
    )
    # Connection-refused text triggers NO_JOB from text parser
    text_input = "Connection refused\nFailed to connect to the NCCL RAS service!"
    snapshot = collector._parse_response(text_input, "node1", cap)
    assert snapshot.state == RCCLJobState.NO_JOB


def test_parse_response_backfills_version_into_capability():
    collector = RCCLCollector()
    cap = NodeRCCLCapability(
        json_ras=True,
        detected_rccl_version=None,
        detection_method="probe",
    )
    json_input = (
        '{"nccl_version": "2.30.4", "cuda_runtime_version": 0, '
        '"cuda_driver_version": 0, "communicators_count": 0, "communicators": []}'
    )
    collector._parse_response(json_input, "node1", cap)
    assert cap.detected_rccl_version == "2.30.4"


def test_parse_response_does_not_overwrite_existing_version():
    """Once detected_rccl_version is set, it must not be overwritten by later parses."""
    collector = RCCLCollector()
    cap = NodeRCCLCapability(
        json_ras=True,
        detected_rccl_version="2.28.9",  # already set
        detection_method="probe",
    )
    json_input = (
        '{"nccl_version": "2.30.4", "cuda_runtime_version": 0, '
        '"cuda_driver_version": 0, "communicators_count": 0, "communicators": []}'
    )
    collector._parse_response(json_input, "node1", cap)
    # Should NOT overwrite
    assert cap.detected_rccl_version == "2.28.9"


# ---------------------------------------------------------------------------
# NodeRCCLCapability dataclass
# ---------------------------------------------------------------------------


def test_node_rccl_capability_defaults():
    cap = NodeRCCLCapability(
        json_ras=False,
        detected_rccl_version=None,
        detection_method="probe",
    )
    assert cap.ttl == 300.0
    assert cap.probed_at <= time.time()
    assert cap.detection_method == "probe"
