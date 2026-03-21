"""
Redis-backed RCCL data store using Redis Streams.

XADD+MAXLEN is atomic (single command), fixing the LPUSH+LTRIM race condition.
Stream IDs embed millisecond timestamps, enabling time-range queries without
a separate sorted set.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RCCLDataStore:
    # Redis Streams — atomic append+cap in one command
    SNAPSHOT_STREAM = "rccl:snapshots"   # Stream, capped at 1000 entries
    EVENT_STREAM = "rccl:events"         # Stream, capped at 10000 entries
    CURRENT_KEY = "rccl:current"         # Hash, latest snapshot only

    def __init__(self, redis_client, snapshot_max: int = 1000, event_max: int = 10000):
        """
        Args:
            redis_client: redis.asyncio.Redis instance from app_state.redis, or None.
                          All methods degrade silently when redis_client is None.
            snapshot_max: Maximum number of entries in the snapshot stream.
            event_max: Maximum number of entries in the event stream.
        """
        self._r = redis_client
        self._snapshot_max = snapshot_max
        self._event_max = event_max

    async def push_snapshot(self, snapshot: dict) -> None:
        """Atomically append snapshot to ring buffer and update current."""
        if self._r is None:
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
            logger.warning(f"RCCLDataStore.push_snapshot failed: {e}")

    async def push_event(self, event: dict) -> None:
        """Atomically append event to event stream."""
        if self._r is None:
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
            logger.warning(f"RCCLDataStore.push_event failed: {e}")

    async def get_recent_snapshots(self, count: int = 50) -> list[dict]:
        """Return the most recent N snapshots, newest first."""
        if self._r is None:
            return []
        try:
            entries = await self._r.xrevrange(self.SNAPSHOT_STREAM, count=count)
            return [json.loads(e[1]["data"]) for e in entries]
        except Exception as e:
            logger.warning(f"RCCLDataStore.get_recent_snapshots failed: {e}")
            return []

    async def get_current_snapshot(self) -> Optional[dict]:
        """Return the latest snapshot from the CURRENT_KEY hash."""
        if self._r is None:
            return None
        try:
            result = await self._r.hget(self.CURRENT_KEY, "data")
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.warning(f"RCCLDataStore.get_current_snapshot failed: {e}")
            return None

    async def get_events_in_range(self, start_ts: float, end_ts: float) -> list[dict]:
        """Return events within a UTC timestamp range using stream entry IDs."""
        if self._r is None:
            return []
        try:
            start_id = f"{int(start_ts * 1000)}-0"
            end_id = f"{int(end_ts * 1000)}-0"
            entries = await self._r.xrange(self.EVENT_STREAM, min=start_id, max=end_id)
            return [json.loads(e[1]["data"]) for e in entries]
        except Exception as e:
            logger.warning(f"RCCLDataStore.get_events_in_range failed: {e}")
            return []
