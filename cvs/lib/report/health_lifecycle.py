'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Stage timings for health Run Decks.

Health suites do not have an inference-style lifecycle. They record the wall
clock of each command that actually ran, in the same ``report`` shape the
timeline card already aggregates.
'''

import time
from contextlib import contextmanager


class HealthLifecycle:
    """Module-scoped stage timings bound as the ``lifecycle`` fixture."""

    def __init__(self):
        self.report = {}

    def record(self, label, seconds):
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return
        if value < 0:
            return
        self.report.setdefault("health", []).append((str(label), value, "s"))


def _record_stage(lifecycle, label, seconds):
    """Best-effort: a reporting failure must not change the suite verdict."""
    if lifecycle is None:
        return
    try:
        lifecycle.record(label, seconds)
    except (AttributeError, TypeError, ValueError):
        return


@contextmanager
def timed_stage(lifecycle, label):
    """Time one command. Exceptions from the block always propagate; recording is best-effort."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _record_stage(lifecycle, label, time.perf_counter() - t0)
