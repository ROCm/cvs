'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Checkpoint I/O timing utilities for TorchTitan training logs.

Parses save and load elapsed times directly from training log text so
test_checkpoint can report checkpoint I/O performance without any external
instrumentation.

TorchTitan uses torchrun output format, different from Primus/Megatron:
  - Save timing: extract timestamps around checkpoint save operations
  - Load timing: extract timestamps around checkpoint load operations

Both single-node and distributed runs emit these lines on the master node (rank-0).
The caller is responsible for passing the node-0 log for distributed runs.
'''

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple

from cvs.lib import globals

log = globals.log

# ---------------------------------------------------------------------------
# Regex patterns for TorchTitan log format
# ---------------------------------------------------------------------------

# TorchTitan log format examples:
# [2025:08:21-14:30:45] [rank0]: Saving checkpoint to /path/to/checkpoint...
# [2025:08:21-14:30:48] [rank0]: Successfully saved checkpoint to /path/to/checkpoint

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# TorchTitan checkpoint save patterns
_SAVE_START_RE = re.compile(
    r'\[(\d{4}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2})\].*saving\s+checkpoint',
    re.I
)
_SAVE_END_RE = re.compile(
    r'\[(\d{4}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2})\].*successfully\s+saved\s+checkpoint',
    re.I
)

# TorchTitan checkpoint load patterns
_LOAD_START_RE = re.compile(
    r'\[(\d{4}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2})\].*loading\s+checkpoint',
    re.I
)
_LOAD_END_RE = re.compile(
    r'\[(\d{4}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2})\].*successfully\s+loaded\s+checkpoint',
    re.I
)

_TS_FMT = '%Y:%m:%d-%H:%M:%S'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_checkpoint_io_seconds(
    log_text: str,
) -> Tuple[List[Tuple[int, float]], Optional[float]]:
    """Parse checkpoint save and load I/O times from a TorchTitan training log.

    Returns:
        save_times  : list of (iteration, elapsed_seconds) for each checkpoint saved.
        load_seconds: seconds to load the checkpoint, or None if not found.

    Iterates line-by-line (via splitlines) so mixed line endings from SSH output
    never interfere with the patterns.
    """
    save_starts = []
    save_ends = []
    load_start_ts = None
    load_end_ts = None

    for line in (_ANSI_RE.sub('', ln) for ln in log_text.splitlines()):
        # Save start
        m = _SAVE_START_RE.search(line)
        if m:
            save_starts.append(datetime.strptime(m.group(1), _TS_FMT))
            continue

        # Save end
        m = _SAVE_END_RE.search(line)
        if m:
            save_ends.append(datetime.strptime(m.group(1), _TS_FMT))
            continue

        # Load start
        m = _LOAD_START_RE.search(line)
        if m and not load_start_ts:
            load_start_ts = datetime.strptime(m.group(1), _TS_FMT)
            continue

        # Load end
        m = _LOAD_END_RE.search(line)
        if m and not load_end_ts:
            load_end_ts = datetime.strptime(m.group(1), _TS_FMT)

    # Pair up save start/end times
    save_times = []
    for i, (start, end) in enumerate(zip(save_starts, save_ends)):
        elapsed = (end - start).total_seconds()
        # Use index+1 as iteration placeholder (TorchTitan may not log iteration number in checkpoint messages)
        save_times.append((i + 1, elapsed))

    load_seconds = None
    if load_start_ts and load_end_ts:
        load_seconds = (load_end_ts - load_start_ts).total_seconds()

    return save_times, load_seconds


def log_checkpoint_io_times(
    save_times: List[Tuple[int, float]],
    load_seconds: Optional[float],
) -> None:
    """Log checkpoint I/O timing results."""
    if save_times:
        for itr, elapsed in save_times:
            log.info("checkpoint save I/O: iteration=%d time=%.2fs", itr, elapsed)
        avg_save = sum(e for _, e in save_times) / len(save_times)
        log.info(
            "checkpoint save I/O avg (across %d checkpoints): %.2fs",
            len(save_times),
            avg_save,
        )
    else:
        log.warning("checkpoint save I/O times could not be parsed from save log")

    if load_seconds is not None:
        log.info("checkpoint load I/O time: %.2fs", load_seconds)
    else:
        log.warning("checkpoint load I/O time could not be parsed from resume log")
