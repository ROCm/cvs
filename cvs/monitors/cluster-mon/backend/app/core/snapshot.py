"""
Cluster metrics snapshot capture, store, and diffing.

Mirrors the counter snapshot workflow in ``cvs.monitors.check_cluster_health``
so operators can capture up to five snapshots from the cluster-mon UI and
diff any pair on demand.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

ERR_STATS_PATTERN = "err|drop|discard|overflow|fcs|nak|uncorrect|loss"
WARN_STATS_PATTERN = "retry|timeout|exceeded|ooo|retransmit"
THRESHOLD_STATS_PATTERN = "cnp|ecn"
THRESHOLD_COUNTER_VAL = 1000

SNAPSHOT_CATEGORIES = ("eth_stats", "rdma_stats", "gpu_ras_stats", "gpu_pcie_stats")

MAX_SNAPSHOTS = 5
CAPTURE_TIMEOUT_S = 180
LABEL_MAX_LEN = 80


class SnapshotFullError(Exception):
    """Gallery already holds MAX_SNAPSHOTS entries."""


class SnapshotEmptyError(Exception):
    """Collector returned no per-node data in any category."""


@dataclass
class StoredSnapshot:
    id: str
    captured_at: str
    label: Optional[str]
    degraded: bool
    failed_nodes: list[str]
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StoredSnapshot":
        return cls(
            id=str(raw["id"]),
            captured_at=str(raw["captured_at"]),
            label=raw.get("label"),
            degraded=bool(raw.get("degraded", False)),
            failed_nodes=list(raw.get("failed_nodes") or []),
            data=raw.get("data") or {},
        )


def snapshots_file_path() -> Path:
    """Resolve snapshots.json the same way cluster.yaml is resolved."""
    docker_dir = Path("/app/config")
    if docker_dir.is_dir():
        return docker_dir / "snapshots.json"
    cluster_mon_root = Path(__file__).resolve().parents[3]
    return cluster_mon_root / "config" / "snapshots.json"


class SnapshotStore:
    """In-memory gallery persisted atomically to snapshots.json."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or snapshots_file_path()
        self._items: dict[str, StoredSnapshot] = {}
        self._order: list[str] = []  # oldest first

    def count(self) -> int:
        return len(self._order)

    def list_newest_first(self) -> list[StoredSnapshot]:
        return [self._items[sid] for sid in reversed(self._order)]

    def get(self, snapshot_id: str) -> Optional[StoredSnapshot]:
        return self._items.get(snapshot_id)

    def load(self) -> None:
        self._items.clear()
        self._order.clear()
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict):
                snaps = raw.get("snapshots") or []
            elif isinstance(raw, list):
                snaps = raw
            else:
                raise ValueError("unexpected snapshots.json shape")
            for item in snaps:
                if not isinstance(item, dict):
                    raise ValueError("snapshot entry is not an object")
                snap = StoredSnapshot.from_dict(item)
                self._items[snap.id] = snap
                self._order.append(snap.id)
            while len(self._order) > MAX_SNAPSHOTS:
                oldest = self._order.pop(0)
                self._items.pop(oldest, None)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error("Corrupt snapshots file %s: %s — starting empty", self.path, exc)
            self._items.clear()
            self._order.clear()

    def add(self, snap: StoredSnapshot) -> Optional[str]:
        if self.count() >= MAX_SNAPSHOTS:
            raise SnapshotFullError()
        self._items[snap.id] = snap
        self._order.append(snap.id)
        return self._persist()

    def delete(self, snapshot_id: str) -> bool:
        if snapshot_id not in self._items:
            return False
        del self._items[snapshot_id]
        self._order.remove(snapshot_id)
        self._persist()
        return True

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()
        self._persist()

    def _persist(self) -> Optional[str]:
        payload = {"snapshots": [self._items[sid].to_dict() for sid in self._order]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_name(self.path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload))
            os.replace(tmp_path, self.path)
            return None
        except OSError as exc:
            logger.exception("Failed to persist snapshots to %s", self.path)
            return str(exc)


def _flatten_numeric_stats(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dicts, keeping only numeric counter values."""
    flat: dict[str, Any] = {}
    if not isinstance(obj, dict):
        return flat

    for key, value in obj.items():
        stat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            if "value" in value:
                raw = value["value"]
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    flat[stat_key] = raw
                elif isinstance(raw, str) and raw.lstrip("-").isdigit():
                    flat[stat_key] = int(raw)
            else:
                flat.update(_flatten_numeric_stats(value, stat_key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[stat_key] = value
        elif isinstance(value, str) and value.lstrip("-").isdigit():
            flat[stat_key] = int(value)

    return flat


def _normalize_gpu_metric_snapshot(raw_data: dict[str, Any], metric_key: str) -> dict[str, Any]:
    """Convert amd-smi collector output to {node: {device: {stat: value}}}."""
    result: dict[str, Any] = {}
    for node, data in raw_data.items():
        if not isinstance(data, dict) or "error" in data:
            continue

        result[node] = {}
        gpu_list = None
        if isinstance(data, list):
            gpu_list = data
        elif "gpu_data" in data:
            gpu_list = data["gpu_data"]

        if not gpu_list:
            continue

        for gpu in gpu_list:
            if not isinstance(gpu, dict):
                continue
            device = str(gpu.get("gpu", 0))
            metric_data = gpu.get(metric_key)
            if metric_data is None and metric_key == "ecc":
                metric_data = gpu.get("ras", gpu.get("ras_errors", gpu.get("ecc_blocks", {})))
            if isinstance(metric_data, dict):
                flattened = _flatten_numeric_stats(metric_data)
                if flattened:
                    result[node][device] = flattened

    return result


def _coerce_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if re.search(r"[a-z]", value, re.I):
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_delta(delta: float) -> int | float:
    if delta == int(delta):
        return int(delta)
    return delta


def get_metrics_snapshot_diff_dict(
    snapshot_before: dict[str, Any],
    snapshot_after: dict[str, Any],
) -> dict[str, Any]:
    """Compute numeric deltas (after - before) for matching snapshot keys."""
    diff_dict: dict[str, Any] = {}

    for category in snapshot_before.keys():
        diff_dict[category] = {}
        before_nodes = snapshot_before.get(category) or {}
        after_nodes = snapshot_after.get(category) or {}

        for node in before_nodes.keys():
            diff_dict[category][node] = {}
            before_devices = before_nodes.get(node) or {}
            after_devices = after_nodes.get(node) or {}

            for device in before_devices.keys():
                diff_dict[category][node][device] = {}
                before_stats = before_devices.get(device) or {}
                after_stats = after_devices.get(device) or {}

                for stat_name in before_stats.keys():
                    before_val = before_stats.get(stat_name)
                    after_val = after_stats.get(stat_name)
                    if isinstance(before_val, list):
                        continue
                    before_num = _coerce_number(before_val)
                    after_num = _coerce_number(after_val)
                    if before_num is None or after_num is None:
                        continue
                    diff_dict[category][node][device][stat_name] = _as_delta(after_num - before_num)

    return diff_dict


def _classify_stat(stat_name: str, diff: int | float) -> Optional[str]:
    if diff <= 0:
        return None
    if re.search(ERR_STATS_PATTERN, stat_name, re.I):
        return "error"
    if re.search(WARN_STATS_PATTERN, stat_name, re.I):
        return "warning"
    if re.search(THRESHOLD_STATS_PATTERN, stat_name, re.I) and diff > THRESHOLD_COUNTER_VAL:
        return "threshold_warning"
    return None


def compare_cluster_metrics_snapshots(
    snapshot_before: dict[str, Any],
    snapshot_after: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare two snapshots and return structured diff rows for the UI.

    Returns:
        {
            "summary": {"errors": int, "warnings": int, "threshold_warnings": int, "total_increments": int},
            "rows": [{"severity", "category", "node", "device", "stat", "before", "after", "diff"}, ...],
        }
    """
    diff_dict = get_metrics_snapshot_diff_dict(snapshot_before, snapshot_after)
    rows: list[dict[str, Any]] = []
    summary = {"errors": 0, "warnings": 0, "threshold_warnings": 0, "total_increments": 0}

    for category in diff_dict.keys():
        for node in diff_dict[category].keys():
            for device in diff_dict[category][node].keys():
                for stat_name, diff in diff_dict[category][node][device].items():
                    try:
                        diff_val = (
                            diff if isinstance(diff, (int, float)) and not isinstance(diff, bool) else float(diff)
                        )
                    except (TypeError, ValueError):
                        continue

                    if diff_val <= 0:
                        continue

                    before_val = snapshot_before[category][node][device][stat_name]
                    after_val = snapshot_after[category][node][device].get(stat_name, before_val)
                    severity = _classify_stat(stat_name, diff_val) or "info"

                    if severity == "error":
                        summary["errors"] += 1
                    elif severity == "warning":
                        summary["warnings"] += 1
                    elif severity == "threshold_warning":
                        summary["threshold_warnings"] += 1
                    summary["total_increments"] += 1

                    rows.append(
                        {
                            "severity": severity,
                            "category": category,
                            "node": node,
                            "device": device,
                            "stat": stat_name,
                            "before": before_val,
                            "after": after_val,
                            "diff": _as_delta(float(diff_val)),
                        }
                    )

    rows.sort(
        key=lambda row: (
            {"error": 0, "warning": 1, "threshold_warning": 2, "info": 3}.get(row["severity"], 4),
            row["category"],
            row["node"],
            row["device"],
            row["stat"],
        )
    )

    return {"summary": summary, "rows": rows}


async def create_cluster_metrics_snapshot(ssh_manager) -> dict[str, Any]:
    """
    Collect a point-in-time snapshot across all cluster nodes.

    Categories mirror ``verify_lib.create_cluster_metrics_snapshot``:
    eth_stats, rdma_stats, gpu_ras_stats, gpu_pcie_stats.
    """
    from app.collectors.gpu_collector import GPUMetricsCollector
    from app.collectors.nic_collector import NICMetricsCollector

    gpu_collector = GPUMetricsCollector()
    nic_collector = NICMetricsCollector()

    eth_stats, rdma_stats, ras_raw, pcie_raw = await asyncio.gather(
        nic_collector.collect_ethtool_stats(ssh_manager),
        nic_collector.collect_rdma_stats(ssh_manager),
        gpu_collector.collect_ras_errors(ssh_manager),
        gpu_collector.collect_pcie_metrics(ssh_manager),
    )

    return {
        "eth_stats": eth_stats,
        "rdma_stats": rdma_stats,
        "gpu_ras_stats": _normalize_gpu_metric_snapshot(ras_raw, "ecc"),
        "gpu_pcie_stats": _normalize_gpu_metric_snapshot(pcie_raw, "pcie"),
    }


def snapshot_metadata(snapshot: dict[str, Any], captured_at: str) -> dict[str, Any]:
    """Summarize a snapshot for API responses without returning full counter payloads."""
    categories: dict[str, Any] = {}
    for category in SNAPSHOT_CATEGORIES:
        category_data = snapshot.get(category) or {}
        node_count = len(category_data) if isinstance(category_data, dict) else 0
        device_count = 0
        stat_count = 0
        if isinstance(category_data, dict):
            device_count = sum(len(devices or {}) for devices in category_data.values() if isinstance(devices, dict))
            for devices in category_data.values():
                if not isinstance(devices, dict):
                    continue
                for stats in devices.values():
                    if isinstance(stats, dict):
                        stat_count += len(stats)
        categories[category] = {
            "nodes": node_count,
            "devices": device_count,
            "stats": stat_count,
        }

    return {
        "captured_at": captured_at,
        "categories": categories,
    }


def stored_snapshot_list_item(snap: StoredSnapshot) -> dict[str, Any]:
    meta = snapshot_metadata(snap.data, snap.captured_at)
    return {
        "id": snap.id,
        "captured_at": snap.captured_at,
        "label": snap.label,
        "degraded": snap.degraded,
        "failed_nodes": snap.failed_nodes,
        "categories": meta["categories"],
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    stripped = label.strip()[:LABEL_MAX_LEN]
    return stripped or None


def snapshot_has_nodes(data: dict[str, Any]) -> bool:
    for category in SNAPSHOT_CATEGORIES:
        nodes = data.get(category) or {}
        if isinstance(nodes, dict) and nodes:
            return True
    return False


def failed_nodes_from_data(data: dict[str, Any]) -> list[str]:
    failed: set[str] = set()
    for category in SNAPSHOT_CATEGORIES:
        nodes = data.get(category) or {}
        if not isinstance(nodes, dict):
            continue
        for node, payload in nodes.items():
            if isinstance(payload, dict) and "error" in payload:
                failed.add(node)
    return sorted(failed)


def nodes_in_snapshot(data: dict[str, Any]) -> set[str]:
    nodes: set[str] = set()
    for category in SNAPSHOT_CATEGORIES:
        cat_data = data.get(category) or {}
        if isinstance(cat_data, dict):
            nodes.update(cat_data.keys())
    return nodes


def diff_warnings(before: StoredSnapshot, after: StoredSnapshot) -> list[str]:
    warnings: list[str] = []
    before_nodes = nodes_in_snapshot(before.data)
    after_nodes = nodes_in_snapshot(after.data)
    if before_nodes != after_nodes:
        only_before = sorted(before_nodes - after_nodes)
        only_after = sorted(after_nodes - before_nodes)
        parts = []
        if only_before:
            parts.append("only in before: " + ", ".join(only_before))
        if only_after:
            parts.append("only in after: " + ", ".join(only_after))
        warnings.append("Node set differs (" + "; ".join(parts) + ")")
    if before.degraded:
        who = ", ".join(before.failed_nodes) if before.failed_nodes else "unknown nodes"
        warnings.append(f"Before snapshot is degraded ({who})")
    if after.degraded:
        who = ", ".join(after.failed_nodes) if after.failed_nodes else "unknown nodes"
        warnings.append(f"After snapshot is degraded ({who})")
    return warnings


async def capture_snapshot(
    store: SnapshotStore,
    collector: Callable[[], Awaitable[dict[str, Any]]],
    label: Optional[str] = None,
) -> tuple[StoredSnapshot, Optional[str]]:
    """
    Run collector, store the result, persist.

    Raises SnapshotFullError, SnapshotEmptyError, asyncio.TimeoutError,
    or whatever the collector raises.
    """
    if store.count() >= MAX_SNAPSHOTS:
        raise SnapshotFullError()

    data = await asyncio.wait_for(collector(), timeout=CAPTURE_TIMEOUT_S)
    if not snapshot_has_nodes(data):
        raise SnapshotEmptyError()

    failed = failed_nodes_from_data(data)
    snap = StoredSnapshot(
        id=str(uuid.uuid4()),
        captured_at=now_iso(),
        label=normalize_label(label),
        degraded=bool(failed),
        failed_nodes=failed,
        data=data,
    )
    persist_warning = store.add(snap)
    return snap, persist_warning
