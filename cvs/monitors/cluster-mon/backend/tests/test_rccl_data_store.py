"""
Tests for RCCLDataStore — in-memory path (redis_client=None).

Covers:
- push_snapshot / get_current_snapshot / get_recent_snapshots
- push_event / get_events_in_range
- Version-as-timeline-event: software_upgrade emitted on rccl_version change
"""

import time
import pytest

from app.collectors.rccl_data_store import RCCLDataStore


def _store() -> RCCLDataStore:
    return RCCLDataStore(redis_client=None)


def _snapshot(version: str = "2.28.3", ts: float | None = None) -> dict:
    return {
        "timestamp": ts or time.time(),
        "state": "healthy",
        "job_summary": {"rccl_version": version, "total_nodes": 1, "total_gpus": 8},
        "communicators": [],
    }


# ---------------------------------------------------------------------------
# Basic push / get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_and_get_current_snapshot():
    store = _store()
    snap = _snapshot()
    await store.push_snapshot(snap)
    current = await store.get_current_snapshot()
    assert current == snap


@pytest.mark.asyncio
async def test_get_recent_snapshots_newest_first():
    store = _store()
    now = time.time()
    snaps = [_snapshot(ts=now + i) for i in range(3)]
    for s in snaps:
        await store.push_snapshot(s)
    recent = await store.get_recent_snapshots(count=3)
    assert recent[0]["timestamp"] == snaps[2]["timestamp"]
    assert recent[-1]["timestamp"] == snaps[0]["timestamp"]


@pytest.mark.asyncio
async def test_get_recent_snapshots_count_limit():
    store = _store()
    for i in range(5):
        await store.push_snapshot(_snapshot(ts=float(i)))
    assert len(await store.get_recent_snapshots(count=2)) == 2


@pytest.mark.asyncio
async def test_push_event_and_get_in_range():
    store = _store()
    now = time.time()
    event = {"event_type": "test_event", "timestamp": now}
    await store.push_event(event)
    results = await store.get_events_in_range(now - 1, now + 1)
    assert len(results) == 1
    assert results[0]["event_type"] == "test_event"


@pytest.mark.asyncio
async def test_get_events_out_of_range_returns_empty():
    store = _store()
    now = time.time()
    await store.push_event({"event_type": "old", "timestamp": now - 100})
    results = await store.get_events_in_range(now - 5, now)
    assert results == []


# ---------------------------------------------------------------------------
# Version-as-timeline-event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_upgrade_event_on_first_push():
    store = _store()
    await store.push_snapshot(_snapshot("2.28.3"))
    events = await store.get_events_in_range(0, time.time() + 1)
    upgrade_events = [e for e in events if e.get("event_type") == "software_upgrade"]
    assert upgrade_events == []


@pytest.mark.asyncio
async def test_no_upgrade_event_when_version_unchanged():
    store = _store()
    await store.push_snapshot(_snapshot("2.28.3"))
    await store.push_snapshot(_snapshot("2.28.3"))
    events = await store.get_events_in_range(0, time.time() + 1)
    upgrade_events = [e for e in events if e.get("event_type") == "software_upgrade"]
    assert upgrade_events == []


@pytest.mark.asyncio
async def test_upgrade_event_emitted_on_version_change():
    store = _store()
    await store.push_snapshot(_snapshot("2.28.3"))
    await store.push_snapshot(_snapshot("2.28.9"))
    events = await store.get_events_in_range(0, time.time() + 1)
    upgrade_events = [e for e in events if e.get("event_type") == "software_upgrade"]
    assert len(upgrade_events) == 1
    assert upgrade_events[0]["from_version"] == "2.28.3"
    assert upgrade_events[0]["to_version"] == "2.28.9"


@pytest.mark.asyncio
async def test_upgrade_event_has_timestamp():
    store = _store()
    before = time.time()
    await store.push_snapshot(_snapshot("2.28.3"))
    await store.push_snapshot(_snapshot("2.28.9"))
    after = time.time()
    events = await store.get_events_in_range(0, after + 1)
    evt = next(e for e in events if e.get("event_type") == "software_upgrade")
    assert before <= evt["timestamp"] <= after


@pytest.mark.asyncio
async def test_no_upgrade_event_when_version_is_unknown():
    """'unknown' version strings must not trigger software_upgrade events."""
    store = _store()
    await store.push_snapshot(_snapshot("2.28.3"))
    await store.push_snapshot(_snapshot("unknown"))
    events = await store.get_events_in_range(0, time.time() + 1)
    upgrade_events = [e for e in events if e.get("event_type") == "software_upgrade"]
    assert upgrade_events == []


@pytest.mark.asyncio
async def test_no_upgrade_event_when_no_job_summary():
    """Snapshots with no job_summary (e.g. NO_JOB) must not trigger events."""
    store = _store()
    await store.push_snapshot(_snapshot("2.28.3"))
    await store.push_snapshot({"timestamp": time.time(), "state": "no_job"})
    events = await store.get_events_in_range(0, time.time() + 1)
    upgrade_events = [e for e in events if e.get("event_type") == "software_upgrade"]
    assert upgrade_events == []


@pytest.mark.asyncio
async def test_multiple_upgrades_each_emit_event():
    """Each distinct version transition emits its own event."""
    store = _store()
    for version in ("2.28.3", "2.28.9", "2.30.4"):
        await store.push_snapshot(_snapshot(version))
    events = await store.get_events_in_range(0, time.time() + 1)
    upgrade_events = [e for e in events if e.get("event_type") == "software_upgrade"]
    assert len(upgrade_events) == 2
    assert upgrade_events[0]["from_version"] == "2.28.3"
    assert upgrade_events[0]["to_version"] == "2.28.9"
    assert upgrade_events[1]["from_version"] == "2.28.9"
    assert upgrade_events[1]["to_version"] == "2.30.4"


@pytest.mark.asyncio
async def test_version_tracked_across_pushes():
    """_last_rccl_version persists in the store instance."""
    store = _store()
    assert store._last_rccl_version is None
    await store.push_snapshot(_snapshot("2.28.3"))
    assert store._last_rccl_version == "2.28.3"
    await store.push_snapshot(_snapshot("2.28.9"))
    assert store._last_rccl_version == "2.28.9"
