'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Deadline-bounded polling shared by CVS wait loops.
'''

import time


def poll_until(probe, done, timeout, interval):
    """Call ``probe()`` until ``done(result)`` is true or ``timeout`` seconds pass.

    Probes at least once and never sleeps past the deadline. Returns ``(result, satisfied)``
    with the last probe result, so each caller decides what a timeout means.
    """
    deadline = time.monotonic() + timeout
    while True:
        result = probe()
        if done(result):
            return result, True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return result, False
        time.sleep(min(interval, remaining))
