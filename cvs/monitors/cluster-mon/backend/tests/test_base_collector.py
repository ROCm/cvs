"""
Tests for BaseCollector ABC, CollectorResult, CollectorState.
Uses TDD - written before/alongside implementation.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.collectors.base import (
    BaseCollector,
    CollectorResult,
    CollectorState,
)


# ── Concrete test subclass ──────────────────────────────────────────────────


class FakeCollector(BaseCollector):
    name = "fake"
    poll_interval = 1
    collect_timeout = 5.0
    critical = False

    def __init__(self, result: CollectorResult = None, raise_exc=None):
        self._result = result or CollectorResult(
            collector_name="fake",
            timestamp=CollectorResult.now_iso(),
            state=CollectorState.OK,
            data={"value": 42},
        )
        self._raise_exc = raise_exc
        self.collect_call_count = 0

    async def collect(self, ssh_manager) -> CollectorResult:
        self.collect_call_count += 1
        if self._raise_exc:
            raise self._raise_exc
        return self._result


class HangingCollector(BaseCollector):
    name = "hanging"
    poll_interval = 1
    collect_timeout = 0.1  # very short — will timeout
    critical = False

    async def collect(self, ssh_manager) -> CollectorResult:
        # Use an Event that is never set so collect() genuinely blocks,
        # even if asyncio.sleep is patched in the test.
        await asyncio.Event().wait()
        return CollectorResult(
            collector_name=self.name,
            timestamp=CollectorResult.now_iso(),
            state=CollectorState.OK,
            data={},
        )


# ── CollectorResult tests ───────────────────────────────────────────────────


def test_collector_result_now_iso_is_utc():
    ts = CollectorResult.now_iso()
    assert "T" in ts
    assert ts.endswith("+00:00") or ts.endswith("Z") or "UTC" in ts or "+00" in ts


def test_collector_result_defaults():
    result = CollectorResult(
        collector_name="gpu",
        timestamp="2026-01-01T00:00:00+00:00",
        state=CollectorState.OK,
        data={"x": 1},
    )
    assert result.error is None
    assert result.node_errors == {}


# ── CollectorState tests ────────────────────────────────────────────────────


def test_collector_state_values():
    assert CollectorState.OK == "ok"
    assert CollectorState.NO_SERVICE == "no_service"
    assert CollectorState.UNREACHABLE == "unreachable"
    assert CollectorState.ERROR == "error"


# ── BaseCollector.collect() is abstract ─────────────────────────────────────


def test_base_collector_is_abstract():
    with pytest.raises(TypeError):
        BaseCollector()  # cannot instantiate abstract class


# ── BaseCollector.run() — timeout enforcement ────────────────────────────────


@pytest.mark.asyncio
async def test_run_times_out_and_produces_error_result():
    """If collect() hangs beyond collect_timeout, run() produces an ERROR result."""
    collector = HangingCollector()
    ssh_manager = MagicMock()

    app_state = MagicMock()
    app_state.is_collecting = True
    app_state.collector_results = {}
    app_state.latest_metrics = {}
    app_state.probe_requested = None  # no probe event

    # Run one iteration then stop
    call_count = 0
    original_sleep = asyncio.sleep

    async def stop_after_one(seconds):
        nonlocal call_count
        call_count += 1
        app_state.is_collecting = False
        await original_sleep(0)

    with patch("asyncio.sleep", side_effect=stop_after_one):
        with patch("app.collectors.base._update_node_status_via_app_state"):
            # broadcast_metrics is imported lazily inside run(); the import
            # will fail in the test environment, but that's caught by the
            # try/except inside run(). No need to patch it.
            task = asyncio.create_task(collector.run(ssh_manager, app_state))
            await original_sleep(0.5)  # let it run
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # The collector_results should have an ERROR entry from timeout
    assert "hanging" in app_state.collector_results, \
        "Expected collector_results to have 'hanging' entry after timeout"
    result = app_state.collector_results["hanging"]
    assert result.state == CollectorState.ERROR
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_run_cancelled_error_propagates():
    """CancelledError must propagate out of run() without being swallowed."""
    collector = FakeCollector()
    ssh_manager = MagicMock()
    app_state = MagicMock()
    app_state.is_collecting = True
    app_state.collector_results = {}
    app_state.latest_metrics = {}
    app_state.probe_requested = None

    task = asyncio.create_task(collector.run(ssh_manager, app_state))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_run_connection_error_sets_probe_requested():
    """ConnectionError in collect() should call probe_requested.set()."""
    collector = FakeCollector(raise_exc=ConnectionError("SSH timeout"))
    ssh_manager = MagicMock()

    probe_event = asyncio.Event()
    app_state = MagicMock()
    app_state.is_collecting = True
    app_state.collector_results = {}
    app_state.latest_metrics = {}
    app_state.probe_requested = probe_event

    task = asyncio.create_task(collector.run(ssh_manager, app_state))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert probe_event.is_set()


# ── Config tests ─────────────────────────────────────────────────────────────


def test_settings_defaults():
    from app.core.config import Settings
    s = Settings()
    assert s.polling.interval == 60
    assert s.polling.failure_threshold == 5
    assert s.rccl.ras_port == 28028
    assert s.storage.redis.url == "redis://localhost:6379"


def test_settings_backward_compat_properties():
    from app.core.config import Settings
    s = Settings()
    # These properties must exist for the existing main.py to keep working
    assert hasattr(s, 'node_username_via_jumphost')
    assert hasattr(s, 'node_key_file_on_jumphost')
    assert hasattr(s, 'ssh_username')


def test_collector_state_str_enum():
    # CollectorState must be usable as a string (for JSON serialization)
    # In Python 3.10, str() on (str, Enum) returns "ClassName.MEMBER",
    # but the value and f-string formatting yield the raw string.
    assert CollectorState.OK.value == "ok"
    assert f"{CollectorState.OK}" == "ok"
    # Enum equality with str works because CollectorState inherits from str
    assert CollectorState.OK == "ok"
