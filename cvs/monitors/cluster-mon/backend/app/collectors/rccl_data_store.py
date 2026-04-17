"""
Redis-backed RCCL data store using Redis Streams.

XADD+MAXLEN is atomic (single command), fixing the LPUSH+LTRIM race condition.
Stream IDs embed millisecond timestamps, enabling time-range queries without
a separate sorted set.

When redis_client is None, falls back to a bounded in-memory buffer so that
events and snapshots are available for the current session without Redis.
"""

import collections
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory fallback caps (no Redis)
_MEMORY_EVENT_MAX = 500
_MEMORY_SNAPSHOT_MAX = 100
_MEMORY_INSPECTOR_MAX = 100


class RCCLDataStore:
    # Redis Streams — atomic append+cap in one command
    SNAPSHOT_STREAM = "rccl:snapshots"            # Stream, capped at 1000 entries
    EVENT_STREAM = "rccl:events"                  # Stream, capped at 10000 entries
    CURRENT_KEY = "rccl:current"                  # Hash, latest snapshot only
    INSPECTOR_STREAM = "rccl:inspector:snapshots"  # Stream, capped at 1000 entries
    INSPECTOR_CURRENT_KEY = "rccl:inspector:current"  # Hash, latest Inspector snapshot

    def __init__(self, redis_client, snapshot_max: int = 1000, event_max: int = 10000):
        """
        Args:
            redis_client: redis.asyncio.Redis instance from app_state.redis, or None.
                          When None, an in-memory deque is used as a fallback so that
                          events are available for the current session.
            snapshot_max: Maximum number of entries in the snapshot stream.
            event_max: Maximum number of entries in the event stream.
        """
        self._r = redis_client
        self._snapshot_max = snapshot_max
        self._event_max = event_max
        # In-memory fallback buffers (used only when Redis is unavailable)
        self._mem_events: collections.deque[dict] = collections.deque(maxlen=_MEMORY_EVENT_MAX)
        self._mem_snapshots: collections.deque[dict] = collections.deque(maxlen=_MEMORY_SNAPSHOT_MAX)
        self._mem_current: Optional[dict] = None
        self._mem_inspector_snapshots: collections.deque[dict] = collections.deque(maxlen=_MEMORY_INSPECTOR_MAX)
        self._mem_inspector_current: Optional[dict] = None

    async def push_snapshot(self, snapshot: dict) -> None:
        """Atomically append snapshot to ring buffer and update current."""
        if self._r is None:
            self._mem_snapshots.append(snapshot)
            self._mem_current = snapshot
            return
        try:
            payload = json.dumps(snapshot)
            await self._r.xadd(
                self.SNAPSHOT_STREAM,
                {"data": payload},
                maxlen=self._snapshot_max,
            )
            await self._r.hset(
                self.CURRENT_KEY,
                mapping={"data": payload, "ts": str(snapshot.get("timestamp", ""))},
            )
        except Exception as e:
            logger.warning(f"RCCLDataStore.push_snapshot failed (falling back to memory): {e}")
            self._mem_snapshots.append(snapshot)
            self._mem_current = snapshot

    async def push_event(self, event: dict) -> None:
        """Atomically append event to event stream."""
        if self._r is None:
            self._mem_events.append(event)
            return
        try:
            # approximate=True trims in whole radix tree nodes — efficient for high-volume
            await self._r.xadd(
                self.EVENT_STREAM,
                {"data": json.dumps(event)},
                maxlen=self._event_max,
                approximate=True,
            )
        except Exception as e:
            logger.warning(f"RCCLDataStore.push_event failed (falling back to memory): {e}")
            self._mem_events.append(event)

    async def get_recent_snapshots(self, count: int = 50) -> list[dict]:
        """Return the most recent N snapshots, newest first."""
        if self._r is None:
            return list(reversed(list(self._mem_snapshots)))[:count]
        try:
            entries = await self._r.xrevrange(self.SNAPSHOT_STREAM, count=count)
            return [json.loads(e[1]["data"]) for e in entries]
        except Exception as e:
            logger.warning(f"RCCLDataStore.get_recent_snapshots failed: {e}")
            return []

    async def get_current_snapshot(self) -> Optional[dict]:
        """Return the latest snapshot from the CURRENT_KEY hash."""
        if self._r is None:
            return self._mem_current
        try:
            result = await self._r.hget(self.CURRENT_KEY, "data")
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.warning(f"RCCLDataStore.get_current_snapshot failed: {e}")
            return None

    @property
    def is_memory_capped(self) -> bool:
        """True when the in-memory event buffer has reached its maximum capacity."""
        return self._r is None and len(self._mem_events) >= _MEMORY_EVENT_MAX

    async def push_inspector_snapshot(self, snapshot: dict) -> None:
        """Append an Inspector performance snapshot and update the current-key."""
        if self._r is None:
            self._mem_inspector_snapshots.append(snapshot)
            self._mem_inspector_current = snapshot
            return
        try:
            payload = json.dumps(snapshot)
            await self._r.xadd(
                self.INSPECTOR_STREAM,
                {"data": payload},
                maxlen=self._snapshot_max,
            )
            await self._r.hset(
                self.INSPECTOR_CURRENT_KEY,
                mapping={"data": payload, "ts": str(snapshot.get("timestamp", ""))},
            )
        except Exception as e:
            logger.warning(f"RCCLDataStore.push_inspector_snapshot failed (falling back to memory): {e}")
            self._mem_inspector_snapshots.append(snapshot)
            self._mem_inspector_current = snapshot

    async def get_inspector_current(self) -> Optional[dict]:
        """Return the latest Inspector snapshot."""
        if self._r is None:
            return self._mem_inspector_current
        try:
            result = await self._r.hget(self.INSPECTOR_CURRENT_KEY, "data")
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.warning(f"RCCLDataStore.get_inspector_current failed: {e}")
            return None

    async def get_inspector_snapshots(self, count: int = 50) -> list[dict]:
        """Return the most recent N Inspector snapshots, newest first."""
        if self._r is None:
            return list(reversed(list(self._mem_inspector_snapshots)))[:count]
        try:
            entries = await self._r.xrevrange(self.INSPECTOR_STREAM, count=count)
            return [json.loads(e[1]["data"]) for e in entries]
        except Exception as e:
            logger.warning(f"RCCLDataStore.get_inspector_snapshots failed: {e}")
            return []

    async def get_events_in_range(self, start_ts: float, end_ts: float) -> list[dict]:
        """Return events within a UTC timestamp range using stream entry IDs.
        In-memory results are sorted by timestamp to handle NTP clock adjustments."""
        if self._r is None:
            results = [
                e for e in self._mem_events
                if start_ts <= e.get("timestamp", 0) <= end_ts
            ]
            return sorted(results, key=lambda e: e.get("timestamp", 0))
        try:
            start_id = f"{int(start_ts * 1000)}-0"
            end_id = f"{int(end_ts * 1000)}-0"
            entries = await self._r.xrange(self.EVENT_STREAM, min=start_id, max=end_id)
            return [json.loads(e[1]["data"]) for e in entries]
        except Exception as e:
            logger.warning(f"RCCLDataStore.get_events_in_range failed: {e}")
            return []
