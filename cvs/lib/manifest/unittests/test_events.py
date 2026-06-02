"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/manifest/events.py: the closed event vocabulary and the
# append-only EventWriter.
#
# Pinned invariants:
#   - EVENT_VOCAB is a frozen set; emitting an unknown name raises.
#   - events.jsonl is append-only: one parseable JSON object per line, each with
#     a timestamp, preserved across separate writer sessions.
#   - A file-backed writer streams (does not also retain records in RAM) and
#     raises on emit-after-close; an in-memory writer still buffers .records and
#     tolerates emit-after-close.

import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.manifest import EventWriter
from cvs.lib.manifest.events import EVENT_VOCAB, UnknownEventError


class TestEvents(unittest.TestCase):
    def test_closed_vocabulary(self):
        """Flow 6: valid name emits, unknown raises, vocab is frozen."""
        ew = EventWriter(None)
        ew.emit("prepare.start")
        with self.assertRaises(UnknownEventError):
            ew.emit("not.a.real.event")
        self.assertIn("verify.passed", EVENT_VOCAB)
        self.assertIsInstance(EVENT_VOCAB, frozenset)

    def test_jsonl_file_is_append_only(self):
        """Flow 7: events.jsonl written one parseable JSON object per line, appended."""
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "events.jsonl"
        with EventWriter(path) as ew:
            ew.emit("prepare.start", run_id="r1")
            ew.emit("step", step=1)
        with EventWriter(path) as ew:
            ew.emit("teardown.done", run_id="r1")
        lines = path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        records = [json.loads(line) for line in lines]
        self.assertEqual([r["event"] for r in records], ["prepare.start", "step", "teardown.done"])
        self.assertTrue(all("ts" in r for r in records))
        self.assertEqual(records[1]["step"], 1)

    def test_streaming_mode_does_not_buffer_in_memory(self):
        """Flow 7: when streaming to a file, records are not also retained in RAM."""
        tmp = Path(tempfile.mkdtemp())
        with EventWriter(tmp / "events.jsonl") as ew:
            for i in range(100):
                ew.emit("step", step=i)
            self.assertEqual(ew.records, [])
        # In-memory mode (no path) still buffers for callers that read .records.
        mem = EventWriter(None)
        mem.emit("step", step=0)
        self.assertEqual(len(mem.records), 1)

    def test_emit_after_close_raises(self):
        """A file-backed writer raises on emit-after-close; in-memory is unaffected."""
        tmp = Path(tempfile.mkdtemp())
        ew = EventWriter(tmp / "events.jsonl")
        ew.emit("prepare.start")
        ew.close()
        with self.assertRaisesRegex(ValueError, "after close"):
            ew.emit("prepare.done")
        mem = EventWriter(None)
        mem.close()
        self.assertEqual(mem.emit("step", step=1)["event"], "step")


if __name__ == "__main__":
    unittest.main()
