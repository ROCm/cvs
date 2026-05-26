"""
Tests for P2 features:
- hip_version_str() decode utility
- Cross-node version skew detection (_check_and_emit_skew)
- Text-path version back-fill in _parse_response
- /api/rccl/status augmentation (version strings + mismatch flag)
"""
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models.rccl_models import NodeRCCLCapability, hip_version_str
from app.collectors.rccl_collector import RCCLCollector
from app.collectors.rccl_data_store import RCCLDataStore


# ---------------------------------------------------------------------------
# hip_version_str
# ---------------------------------------------------------------------------

def test_hip_version_str_known_value():
    # ROCm 6.5.0 → 60500000
    assert hip_version_str(60500000) == "6.5.0"


def test_hip_version_str_from_fixture():
    # Value seen in rccl_v2283_text_healthy.txt
    assert hip_version_str(70226015) == "7.2.26015"


def test_hip_version_str_zero_returns_unknown():
    assert hip_version_str(0) == "unknown"


def test_hip_version_str_negative_returns_unknown():
    assert hip_version_str(-1) == "unknown"


def test_hip_version_str_round_trip():
    major, minor, patch = 8, 1, 500
    v = major * 10_000_000 + minor * 100_000 + patch
    assert hip_version_str(v) == f"{major}.{minor}.{patch}"


# ---------------------------------------------------------------------------
# Text-path version back-fill in _parse_response
# ---------------------------------------------------------------------------

def _text_cap() -> NodeRCCLCapability:
    return NodeRCCLCapability(
        json_ras=False,
        detected_rccl_version=None,
        detection_method="probe",
    )


def test_text_path_backfills_version(tmp_path):
    """After parsing text output, detected_rccl_version must be filled in."""
    from pathlib import Path
    fixtures = Path(__file__).parent / "fixtures"
    healthy_text = (fixtures / "rccl_v2283_text_healthy.txt").read_text()

    collector = RCCLCollector()
    cap = _text_cap()
    collector._parse_response(healthy_text, "node1", cap)

    assert cap.detected_rccl_version == "2.28.3"


def test_text_path_does_not_overwrite_existing_version():
    from pathlib import Path
    fixtures = Path(__file__).parent / "fixtures"
    healthy_text = (fixtures / "rccl_v2283_text_healthy.txt").read_text()

    collector = RCCLCollector()
    cap = _text_cap()
    cap.detected_rccl_version = "2.28.0"  # already set
    collector._parse_response(healthy_text, "node1", cap)

    assert cap.detected_rccl_version == "2.28.0"  # must not overwrite


def test_connection_refused_text_does_not_backfill():
    """NO_JOB snapshot has no job_summary — version must stay None."""
    collector = RCCLCollector()
    cap = _text_cap()
    collector._parse_response(
        "Connection refused\nFailed to connect to the NCCL RAS service!",
        "node1",
        cap,
    )
    assert cap.detected_rccl_version is None


# ---------------------------------------------------------------------------
# _check_and_emit_skew
# ---------------------------------------------------------------------------

def _cap_with_version(v: str) -> NodeRCCLCapability:
    return NodeRCCLCapability(
        json_ras=False,
        detected_rccl_version=v,
        detection_method="probe",
    )


def _make_snapshot(inconsistent: bool = False):
    from app.models.rccl_models import RCCLSnapshot, RCCLJobState, RCCLJobSummary
    return RCCLSnapshot(
        timestamp=time.time(),
        state=RCCLJobState.HEALTHY,
        job_summary=RCCLJobSummary(
            total_nodes=2,
            total_processes=16,
            total_gpus=16,
            rccl_version="2.28.3",
            hip_runtime_version=70226015,
            amdgpu_driver_version=70226015,
            inconsistent_topology=inconsistent,
        ),
    )


@pytest.mark.asyncio
async def test_no_skew_when_single_node():
    collector = RCCLCollector()
    store = RCCLDataStore(redis_client=None)
    app_state = MagicMock()
    app_state.node_capabilities = {"node1": _cap_with_version("2.28.3")}
    app_state.rccl_data_store = store

    snap = _make_snapshot()
    await collector._check_and_emit_skew(snap, app_state)

    assert snap.job_summary.inconsistent_topology is False
    events = await store.get_events_in_range(0, time.time() + 1)
    assert not any(e["event_type"] == "version_skew" for e in events)


@pytest.mark.asyncio
async def test_no_skew_when_same_version_two_nodes():
    collector = RCCLCollector()
    store = RCCLDataStore(redis_client=None)
    app_state = MagicMock()
    app_state.node_capabilities = {
        "node1": _cap_with_version("2.28.3"),
        "node2": _cap_with_version("2.28.3"),
    }
    app_state.rccl_data_store = store

    snap = _make_snapshot()
    await collector._check_and_emit_skew(snap, app_state)

    assert snap.job_summary.inconsistent_topology is False
    events = await store.get_events_in_range(0, time.time() + 1)
    assert not any(e["event_type"] == "version_skew" for e in events)


@pytest.mark.asyncio
async def test_skew_detected_emits_event():
    collector = RCCLCollector()
    store = RCCLDataStore(redis_client=None)
    app_state = MagicMock()
    app_state.node_capabilities = {
        "node1": _cap_with_version("2.28.3"),
        "node2": _cap_with_version("2.28.9"),
    }
    app_state.rccl_data_store = store

    snap = _make_snapshot()
    await collector._check_and_emit_skew(snap, app_state)

    events = await store.get_events_in_range(0, time.time() + 1)
    skew_events = [e for e in events if e["event_type"] == "version_skew"]
    assert len(skew_events) == 1
    evt = skew_events[0]
    assert set(evt["unique_versions"]) == {"2.28.3", "2.28.9"}
    assert evt["versions_by_node"]["node1"] == "2.28.3"
    assert evt["versions_by_node"]["node2"] == "2.28.9"


@pytest.mark.asyncio
async def test_skew_sets_inconsistent_topology():
    collector = RCCLCollector()
    store = RCCLDataStore(redis_client=None)
    app_state = MagicMock()
    app_state.node_capabilities = {
        "node1": _cap_with_version("2.28.3"),
        "node2": _cap_with_version("2.30.4"),
    }
    app_state.rccl_data_store = store

    snap = _make_snapshot()
    assert snap.job_summary.inconsistent_topology is False
    await collector._check_and_emit_skew(snap, app_state)
    assert snap.job_summary.inconsistent_topology is True


@pytest.mark.asyncio
async def test_skew_skips_nodes_without_version():
    """Nodes with detected_rccl_version=None must not count toward skew."""
    collector = RCCLCollector()
    store = RCCLDataStore(redis_client=None)
    app_state = MagicMock()
    app_state.node_capabilities = {
        "node1": _cap_with_version("2.28.3"),
        "node2": NodeRCCLCapability(
            json_ras=False, detected_rccl_version=None, detection_method="probe"
        ),
    }
    app_state.rccl_data_store = store

    snap = _make_snapshot()
    await collector._check_and_emit_skew(snap, app_state)

    assert snap.job_summary.inconsistent_topology is False
    events = await store.get_events_in_range(0, time.time() + 1)
    assert not any(e["event_type"] == "version_skew" for e in events)


@pytest.mark.asyncio
async def test_skew_no_crash_without_data_store():
    """Skew detection must not raise even when rccl_data_store is None."""
    collector = RCCLCollector()
    app_state = MagicMock()
    app_state.node_capabilities = {
        "node1": _cap_with_version("2.28.3"),
        "node2": _cap_with_version("2.28.9"),
    }
    app_state.rccl_data_store = None

    snap = _make_snapshot()
    await collector._check_and_emit_skew(snap, app_state)
    assert snap.job_summary.inconsistent_topology is True  # still set even without store


# ---------------------------------------------------------------------------
# /api/rccl/status augmentation
# ---------------------------------------------------------------------------

def _make_raw_snapshot(hip: int = 70226015, drv: int = 70226015) -> dict:
    return {
        "timestamp": time.time(),
        "state": "healthy",
        "job_summary": {
            "total_nodes": 1,
            "total_gpus": 8,
            "rccl_version": "2.28.3",
            "hip_runtime_version": hip,
            "amdgpu_driver_version": drv,
            "inconsistent_topology": False,
        },
        "communicators": [],
    }


@pytest.mark.asyncio
async def test_status_endpoint_adds_version_strings():
    import httpx
    from app.main import app, app_state

    app_state.latest_rccl_snapshot = _make_raw_snapshot()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/rccl/status")
    assert resp.status_code == 200
    summary = resp.json()["job_summary"]
    assert summary["hip_runtime_version_str"] == "7.2.26015"
    assert summary["amdgpu_driver_version_str"] == "7.2.26015"


@pytest.mark.asyncio
async def test_status_endpoint_mismatch_false_when_equal():
    import httpx
    from app.main import app, app_state

    app_state.latest_rccl_snapshot = _make_raw_snapshot(hip=70226015, drv=70226015)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/rccl/status")
    assert resp.json()["job_summary"]["driver_runtime_mismatch"] is False


@pytest.mark.asyncio
async def test_status_endpoint_mismatch_true_when_differ():
    import httpx
    from app.main import app, app_state

    app_state.latest_rccl_snapshot = _make_raw_snapshot(hip=70226015, drv=60500000)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/rccl/status")
    assert resp.json()["job_summary"]["driver_runtime_mismatch"] is True


@pytest.mark.asyncio
async def test_status_endpoint_no_job_no_summary():
    import httpx
    from app.main import app, app_state

    app_state.latest_rccl_snapshot = {"state": "no_job"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/rccl/status")
    data = resp.json()
    assert "job_summary" not in data or data.get("job_summary") is None
