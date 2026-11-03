'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/training/jaxmaxtext/utils/tb_events.py.

A tiny in-test encoder emits real TFRecord-framed Event/Summary protobufs so the
reader is exercised against the actual on-disk format (no TensorFlow dependency).
'''

import os
import struct
import tempfile
import unittest

from cvs.lib.training.jaxmaxtext.utils.tb_events import read_scalars, read_scalars_from_bytes


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field, wire):
    return _varint((field << 3) | wire)


def _ld(field, payload):
    return _key(field, 2) + _varint(len(payload)) + payload


def _value_simple(tag, val):
    return _ld(1, tag.encode()) + _key(2, 5) + struct.pack("<f", val)


def _value_tensor(tag, val):
    # TensorProto: dtype=DT_FLOAT(1) at field 1, float_val at field 5 (32-bit).
    tensor = _key(1, 0) + _varint(1) + _key(5, 5) + struct.pack("<f", val)
    return _ld(1, tag.encode()) + _ld(8, tensor)


def _event(step, scalars, *, tensor=False):
    enc = _value_tensor if tensor else _value_simple
    summary = b"".join(_ld(1, enc(t, v)) for t, v in scalars.items())
    return _key(2, 0) + _varint(step) + _ld(5, summary)


def _tfrecord(payload):
    # length + (unverified) length-crc + payload + (unverified) data-crc
    return struct.pack("<Q", len(payload)) + b"\x00\x00\x00\x00" + payload + b"\x00\x00\x00\x00"


def _blob(events, *, tensor=False):
    return b"".join(_tfrecord(_event(s, sc, tensor=tensor)) for s, sc in events)


class ReadScalarsFromBytesTests(unittest.TestCase):
    def test_simple_value_round_trip(self):
        blob = _blob([(0, {"learning/loss": 12.0}), (1, {"learning/loss": 11.0}), (2, {"learning/loss": 10.5})])
        out = read_scalars_from_bytes([blob])
        self.assertEqual([s for s, _v in out["learning/loss"]], [0, 1, 2])
        self.assertAlmostEqual(out["learning/loss"][-1][1], 10.5, places=5)

    def test_multiple_tags_per_event(self):
        blob = _blob([(0, {"learning/loss": 12.0, "learning/grad_norm": 1.5, "learning/current_learning_rate": 0.001})])
        out = read_scalars_from_bytes([blob])
        self.assertIn("learning/grad_norm", out)
        self.assertIn("learning/current_learning_rate", out)
        self.assertAlmostEqual(out["learning/grad_norm"][0][1], 1.5, places=5)

    def test_tensor_scalar_value(self):
        blob = _blob([(3, {"perf/step_time_seconds": 9.8})], tensor=True)
        out = read_scalars_from_bytes([blob])
        self.assertAlmostEqual(out["perf/step_time_seconds"][0][1], 9.8, places=4)

    def test_merge_overlapping_steps_last_wins(self):
        first = _blob([(0, {"learning/loss": 12.0}), (1, {"learning/loss": 11.0})])
        resume = _blob([(1, {"learning/loss": 99.0}), (2, {"learning/loss": 9.0})])  # step 1 rewritten
        out = read_scalars_from_bytes([first, resume])
        series = dict(out["learning/loss"])
        self.assertEqual(series[1], 99.0)  # resume wins on the overlapping step
        self.assertEqual(sorted(series), [0, 1, 2])

    def test_truncated_tail_record_is_ignored(self):
        blob = _blob([(0, {"learning/loss": 12.0})])
        out = read_scalars_from_bytes([blob + b"\x40\x00\x00\x00\x00\x00\x00\x00\xff\xff"])  # bogus long length
        self.assertEqual(out["learning/loss"], [(0, 12.0)])

    def test_empty_and_none_inputs(self):
        self.assertEqual(read_scalars_from_bytes([]), {})
        self.assertEqual(read_scalars_from_bytes([b"", None]), {})


class ReadScalarsFromPathsTests(unittest.TestCase):
    def test_directory_glob_merges_files(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "events.out.tfevents.1.host"), "wb") as fh:
                fh.write(_blob([(0, {"learning/loss": 12.0})]))
            with open(os.path.join(d, "events.out.tfevents.2.host"), "wb") as fh:
                fh.write(_blob([(1, {"learning/loss": 11.0})]))
            out = read_scalars(d)
            self.assertEqual([s for s, _v in out["learning/loss"]], [0, 1])

    def test_missing_dir_returns_empty(self):
        self.assertEqual(read_scalars("/no/such/dir/xyz"), {})


if __name__ == "__main__":
    unittest.main()
