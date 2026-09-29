'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Dependency-free TensorBoard ``tfevents`` scalar reader.

MaxText writes per-step scalars (learning/loss, learning/grad_norm,
perf/step_time_seconds, ...) to TensorBoard event files. TensorFlow / tensorboard
are NOT CVS dependencies, so this module reads the on-disk format directly:

  * TFRecord framing: ``<uint64 length LE><uint32 crc><payload><uint32 crc>``
    (the CRCs are not verified -- a corrupt tail record is simply dropped).
  * The ``payload`` is a serialized ``Event`` protobuf. Only the fields needed for
    scalars are decoded by a minimal wire-format walker: Event.step (2),
    Event.summary (5) -> Summary.value (1) -> Value.tag (1) + Value.simple_value (2)
    or Value.tensor (8) for TF2 rank-0 float scalars.

Public API returns ``{tag: [(step, value), ...]}`` sorted by step, merged across
multiple event files (restarts/resumes) with last-writer-wins on a repeated step.
'''

from __future__ import annotations

import glob
import os
import struct

from cvs.lib import globals

log = globals.log


def _read_varint(buf, i):
    shift = 0
    result = 0
    n = len(buf)
    while i < n:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
    raise ValueError("truncated varint")


def _iter_fields(buf):
    """Yield ``(field_number, wire_type, value)`` for a protobuf message.

    value is an int (varint), or a ``bytes`` slice (64-bit / length-delimited /
    32-bit). Unsupported wire types raise, which the callers treat as a decode miss.
    """
    i = 0
    n = len(buf)
    while i < n:
        key, i = _read_varint(buf, i)
        field = key >> 3
        wire = key & 0x7
        if wire == 0:
            val, i = _read_varint(buf, i)
            yield field, wire, val
        elif wire == 1:
            yield field, wire, buf[i : i + 8]
            i += 8
        elif wire == 2:
            length, i = _read_varint(buf, i)
            yield field, wire, buf[i : i + length]
            i += length
        elif wire == 5:
            yield field, wire, buf[i : i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")


def _iter_tfrecords(data):
    """Yield each record payload from TFRecord-framed ``data`` (CRCs skipped)."""
    i = 0
    n = len(data)
    while i + 12 <= n:
        (length,) = struct.unpack_from("<Q", data, i)
        i += 8 + 4  # length + its masked-crc
        if i + length + 4 > n:
            break  # truncated tail record
        yield data[i : i + length]
        i += length + 4  # payload + its masked-crc


def _tensor_scalar(buf):
    """Extract a rank-0 float from a TensorProto (TF2 scalar summaries)."""
    float_vals = []
    content = None
    for field, wire, val in _iter_fields(buf):
        if field == 5 and wire == 5:
            float_vals.append(struct.unpack("<f", val)[0])
        elif field == 5 and wire == 2:  # packed float_val
            for j in range(0, len(val) - 3, 4):
                float_vals.append(struct.unpack_from("<f", val, j)[0])
        elif field == 4 and wire == 2:
            content = val
    if float_vals:
        return float_vals[0]
    if content is not None and len(content) >= 4:
        return struct.unpack_from("<f", content, 0)[0]
    return None


def _parse_event(payload):
    """Return ``(step, [(tag, value), ...])`` for one Event payload."""
    step = None
    summary = None
    for field, wire, val in _iter_fields(payload):
        if field == 2 and wire == 0:
            step = val
        elif field == 5 and wire == 2:
            summary = val
    if summary is None:
        return step, []

    pairs = []
    for field, wire, val in _iter_fields(summary):
        if field != 1 or wire != 2:
            continue  # Summary.value is field 1, length-delimited
        tag = None
        value = None
        for vfield, vwire, vval in _iter_fields(val):
            if vfield == 1 and vwire == 2:
                tag = vval.decode("utf-8", "replace")
            elif vfield == 2 and vwire == 5:
                value = struct.unpack("<f", vval)[0]
            elif vfield == 8 and vwire == 2:
                value = _tensor_scalar(vval)
        if tag is not None and value is not None:
            pairs.append((tag, value))
    return step, pairs


def read_scalars_from_bytes(blobs):
    """Merge scalar series from an iterable of raw event-file ``bytes``.

    Returns ``{tag: [(step, value), ...]}`` sorted by step. Overlapping steps
    (from restarts) keep the last value seen across the given blobs.
    """
    per_tag = {}
    for blob in blobs or []:
        if not blob:
            continue
        for payload in _iter_tfrecords(blob):
            try:
                step, pairs = _parse_event(payload)
            except (ValueError, IndexError, struct.error):
                continue  # skip an undecodable record, keep the rest
            if step is None:
                continue
            for tag, value in pairs:
                per_tag.setdefault(tag, {})[step] = value
    return {tag: sorted(steps.items()) for tag, steps in per_tag.items()}


def read_scalars(source):
    """Read merged scalar series from a directory or a list of event-file paths.

    A directory is globbed for ``events.out.tfevents.*``. Returns the same
    ``{tag: [(step, value), ...]}`` mapping as :func:`read_scalars_from_bytes`;
    an empty mapping when nothing is readable (never raises for missing paths).
    """
    if isinstance(source, str):
        paths = sorted(glob.glob(os.path.join(source, "events.out.tfevents.*"))) if os.path.isdir(source) else [source]
    else:
        paths = list(source or [])

    blobs = []
    for path in paths:
        try:
            with open(path, "rb") as fh:
                blobs.append(fh.read())
        except OSError as e:
            log.warning("tb_events: cannot read %s (%s)", path, e)
    return read_scalars_from_bytes(blobs)
