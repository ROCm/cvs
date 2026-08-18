"""
Tests for cluster metrics snapshot store, persist, capture, and diff.
"""

import asyncio
import json
from pathlib import Path

from app.core.snapshot import (
    SnapshotEmptyError,
    SnapshotFullError,
    SnapshotStore,
    capture_snapshot,
    compare_cluster_metrics_snapshots,
    diff_warnings,
    get_metrics_snapshot_diff_dict,
    snapshot_metadata,
    stored_snapshot_list_item,
)


def _sample_snapshot(rx_errors: int = 0, retry_count: int = 0, extra_node: bool = False) -> dict:
    data = {
        "eth_stats": {
            "node1": {
                "ens1": {
                    "rx_errors": rx_errors,
                    "rx_packets": 1000,
                }
            }
        },
        "rdma_stats": {
            "node1": {
                "rdma0/1": {
                    "port_rcv_data": 500,
                    "rx_rdma_ecn_pkts": 0,
                }
            }
        },
        "gpu_ras_stats": {
            "node1": {
                "0": {
                    "total_uncorrectable": 0,
                    "total_correctable": 5,
                }
            }
        },
        "gpu_pcie_stats": {
            "node1": {
                "0": {
                    "retry_count": retry_count,
                    "nak_sent_count": 0,
                }
            }
        },
    }
    if extra_node:
        data["eth_stats"]["node2"] = {"ens1": {"rx_errors": 0, "rx_packets": 1}}
    return data


def test_get_metrics_snapshot_diff_dict_computes_deltas():
    before = _sample_snapshot(rx_errors=0, retry_count=10)
    after = _sample_snapshot(rx_errors=3, retry_count=15)

    diff = get_metrics_snapshot_diff_dict(before, after)

    assert diff["eth_stats"]["node1"]["ens1"]["rx_errors"] == 3
    assert diff["eth_stats"]["node1"]["ens1"]["rx_packets"] == 0
    assert diff["gpu_pcie_stats"]["node1"]["0"]["retry_count"] == 5


def test_float_counters_are_diffed():
    before = {"eth_stats": {"n": {"d": {"rx_errors": 1.5}}}}
    after = {"eth_stats": {"n": {"d": {"rx_errors": 4.0}}}}
    diff = get_metrics_snapshot_diff_dict(before, after)
    assert diff["eth_stats"]["n"]["d"]["rx_errors"] == 2.5


def test_compare_classifies_error_counters():
    before = _sample_snapshot(rx_errors=0)
    after = _sample_snapshot(rx_errors=2)

    result = compare_cluster_metrics_snapshots(before, after)

    assert result["summary"]["errors"] == 1
    assert result["summary"]["total_increments"] == 1
    assert result["rows"][0]["severity"] == "error"
    assert result["rows"][0]["stat"] == "rx_errors"
    assert result["rows"][0]["diff"] == 2


def test_compare_classifies_warning_counters():
    before = _sample_snapshot(retry_count=0)
    after = _sample_snapshot(retry_count=5)

    result = compare_cluster_metrics_snapshots(before, after)

    assert result["summary"]["warnings"] == 1
    assert result["rows"][0]["severity"] == "warning"
    assert result["rows"][0]["stat"] == "retry_count"


def test_compare_ignores_zero_or_negative_deltas():
    before = _sample_snapshot(rx_errors=10)
    after = _sample_snapshot(rx_errors=10)

    result = compare_cluster_metrics_snapshots(before, after)

    assert result["summary"]["total_increments"] == 0
    assert result["rows"] == []


def test_snapshot_metadata_summarizes_categories():
    snapshot = _sample_snapshot()
    meta = snapshot_metadata(snapshot, "2026-01-01T00:00:00+00:00")

    assert meta["captured_at"] == "2026-01-01T00:00:00+00:00"
    assert meta["categories"]["eth_stats"]["nodes"] == 1
    assert meta["categories"]["eth_stats"]["devices"] == 1
    assert meta["categories"]["eth_stats"]["stats"] == 2


def _store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(tmp_path / "snapshots.json")


def test_store_cap_and_persist(tmp_path):
    store = _store(tmp_path)

    async def collect_factory(i):
        async def collect():
            return _sample_snapshot(rx_errors=i)

        return collect

    async def run():
        for i in range(5):
            await capture_snapshot(store, await collect_factory(i), label=f"s{i}")
        assert store.count() == 5
        try:
            await capture_snapshot(store, await collect_factory(99), label="overflow")
            raise AssertionError("expected SnapshotFullError")
        except SnapshotFullError:
            pass
        assert store.count() == 5

    asyncio.run(run())

    raw = json.loads((tmp_path / "snapshots.json").read_text())
    assert len(raw["snapshots"]) == 5
    assert all(item["label"] != "overflow" for item in raw["snapshots"])


def test_diff_order_is_after_minus_before(tmp_path):
    store = _store(tmp_path)

    async def before_collect():
        return _sample_snapshot(rx_errors=1)

    async def after_collect():
        return _sample_snapshot(rx_errors=4)

    async def run():
        before, _ = await capture_snapshot(store, before_collect)
        after, _ = await capture_snapshot(store, after_collect)
        result = compare_cluster_metrics_snapshots(before.data, after.data)
        assert result["rows"][0]["diff"] == 3
        reversed_result = compare_cluster_metrics_snapshots(after.data, before.data)
        assert reversed_result["rows"] == []

    asyncio.run(run())


def test_delete_one_and_clear_all(tmp_path):
    store = _store(tmp_path)

    async def collect():
        return _sample_snapshot()

    async def run():
        first, _ = await capture_snapshot(store, collect)
        await capture_snapshot(store, collect)
        assert store.count() == 2
        assert store.delete(first.id) is True
        assert store.count() == 1
        assert store.get(first.id) is None
        store.clear()
        assert store.count() == 0
        assert json.loads((tmp_path / "snapshots.json").read_text())["snapshots"] == []

    asyncio.run(run())


def test_corrupt_file_loads_empty(tmp_path):
    path = tmp_path / "snapshots.json"
    path.write_text("{not-json")
    store = SnapshotStore(path)
    store.load()
    assert store.count() == 0


def test_empty_collect_does_not_consume_slot(tmp_path):
    store = _store(tmp_path)

    async def empty():
        return {"eth_stats": {}, "rdma_stats": {}, "gpu_ras_stats": {}, "gpu_pcie_stats": {}}

    async def run():
        try:
            await capture_snapshot(store, empty)
            raise AssertionError("expected SnapshotEmptyError")
        except SnapshotEmptyError:
            pass
        assert store.count() == 0
        assert not (tmp_path / "snapshots.json").exists()

    asyncio.run(run())


def test_timeout_does_not_consume_slot(tmp_path, monkeypatch):
    import app.core.snapshot as snapshot_mod

    monkeypatch.setattr(snapshot_mod, "CAPTURE_TIMEOUT_S", 0.01)
    store = _store(tmp_path)

    async def slow():
        await asyncio.sleep(1)
        return _sample_snapshot()

    async def run():
        try:
            await snapshot_mod.capture_snapshot(store, slow)
            raise AssertionError("expected TimeoutError")
        except asyncio.TimeoutError:
            pass
        assert store.count() == 0

    asyncio.run(run())


def test_degraded_and_node_set_warnings(tmp_path):
    store = _store(tmp_path)

    async def degraded():
        data = _sample_snapshot()
        data["eth_stats"]["node1"] = {"error": "ssh fail"}
        return data

    async def extra():
        return _sample_snapshot(extra_node=True)

    async def run():
        before, _ = await capture_snapshot(store, degraded, label="bad")
        after, _ = await capture_snapshot(store, extra, label="more")
        assert before.degraded is True
        assert "node1" in before.failed_nodes
        warnings = diff_warnings(before, after)
        assert any("degraded" in w.lower() for w in warnings)
        assert any("node set differs" in w.lower() for w in warnings)
        item = stored_snapshot_list_item(before)
        assert "data" not in item
        assert item["degraded"] is True

    asyncio.run(run())


def test_same_snapshot_compare_has_no_increments():
    before = _sample_snapshot()
    result = compare_cluster_metrics_snapshots(before, before)
    assert result["rows"] == []
