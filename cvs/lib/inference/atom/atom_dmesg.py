'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Time-bounded dmesg verification helpers for the atom suite (INF-6).
'''

from __future__ import annotations

from cvs.lib import globals

log = globals.log

_DMESG_TS_CMD = 'date +"%a %b %e %H:%M:%S"'


def capture_dmesg_timestamp(orch) -> dict:
    """Return per-node timestamp strings suitable for verify_dmesg_for_errors."""
    return orch.exec(_DMESG_TS_CMD) or {}


def verify_dmesg_window(orch, start_time_dict: dict, end_time_dict: dict) -> dict:
    """Scan kernel logs between start/end on all nodes; fail on error patterns."""
    if not start_time_dict or not end_time_dict:
        log.warning("dmesg scan skipped: missing start or end timestamps")
        return {}
    from cvs.lib.verify_lib import verify_dmesg_for_errors

    return verify_dmesg_for_errors(orch, start_time_dict, end_time_dict, till_end_flag=False)
