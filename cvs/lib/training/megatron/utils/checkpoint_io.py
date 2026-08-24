'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Checkpoint I/O timing utilities for Primus-Megatron training logs.

Parses save and load elapsed times directly from training log text so
test_checkpoint can report checkpoint I/O performance without any external
instrumentation.

Save timing uses the embedded microsecond timestamps on save bracket lines:
  [YYYY-MM-DD HH:MM:SS.ffffff] saving checkpoint at iteration N to ...
  [YYYY-MM-DD HH:MM:SS.ffffff] successfully saved checkpoint from iteration N to ...

Load timing uses the outer second-precision header timestamps:
  [YYYYMMDD HH:MM:SS]... loading checkpoint from ...
  [YYYYMMDD HH:MM:SS]... successfully loaded checkpoint from ...

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
# Regex patterns (single-line, applied per line after splitlines())
# ---------------------------------------------------------------------------

_OUTER_TS_RE = re.compile(r'\[(\d{8} \d{2}:\d{2}:\d{2})\]')
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_SAVE_START_RE = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]\s+saving checkpoint at iteration\s+(\d+)',
)
_SAVE_END_RE = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]\s+successfully saved checkpoint from iteration\s+(\d+)',
)

_TS_FMT_US = '%Y-%m-%d %H:%M:%S.%f'
_TS_FMT_S = '%Y%m%d %H:%M:%S'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_checkpoint_io_seconds(
    log_text: str,
) -> Tuple[List[Tuple[int, float]], Optional[float]]:
    """Parse checkpoint save and load I/O times from a Primus training log.

    Returns:
        save_times  : list of (iteration, elapsed_seconds) for each checkpoint saved.
        load_seconds: seconds to load the checkpoint, or None if not found.

    Iterates line-by-line (via splitlines) so mixed line endings from SSH output
    never interfere with the patterns.
    """
    save_starts: dict = {}
    save_ends: dict = {}
    load_start_ts = None
    load_end_ts = None

    for line in (_ANSI_RE.sub('', ln) for ln in log_text.splitlines()):
        # Save start — embedded microsecond timestamp
        m = _SAVE_START_RE.search(line)
        if m:
            save_starts[int(m.group(2))] = datetime.strptime(m.group(1), _TS_FMT_US)
            continue

        # Save end — embedded microsecond timestamp
        m = _SAVE_END_RE.search(line)
        if m:
            save_ends[int(m.group(2))] = datetime.strptime(m.group(1), _TS_FMT_US)
            continue

        # Load start / end — keyword check first, then extract outer timestamp via
        # search() (not match()) so any invisible prefix bytes don't block the match.
        if 'loading checkpoint from' in line and 'successfully' not in line:
            outer_m = _OUTER_TS_RE.search(line)
            if outer_m:
                load_start_ts = datetime.strptime(outer_m.group(1), _TS_FMT_S)
        elif 'successfully loaded checkpoint from' in line:
            outer_m = _OUTER_TS_RE.search(line)
            if outer_m:
                load_end_ts = datetime.strptime(outer_m.group(1), _TS_FMT_S)

    save_times = [
        (itr, (save_ends[itr] - save_starts[itr]).total_seconds()) for itr in sorted(save_starts) if itr in save_ends
    ]

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
