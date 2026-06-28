'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Model-cache size probing helpers (no pytest dependency).
'''

from __future__ import annotations

import shlex


def du_bytes(orch, path):
    """Bytes under ``path`` in the container.

    Returns 0 when absent or empty. Returns ``None`` when ``du`` cannot run so
    callers do not treat infrastructure failure as "model not present".
    """
    quoted = shlex.quote(path)
    cmd = (
        f"if [ ! -e {quoted} ]; then echo __MISSING__; "
        f"elif bytes=$(du -sb {quoted} 2>/dev/null | cut -f1) && [ -n \"$bytes\" ]; "
        f"then echo \"$bytes\"; else echo __DU_ERROR__; fi"
    )
    out = orch.exec(f"bash -c {shlex.quote(cmd)}")
    total = 0
    saw_marker = False
    for text in (out or {}).values():
        text = (text or "").strip()
        if text == "__DU_ERROR__":
            return None
        if text == "__MISSING__":
            saw_marker = True
            continue
        if text.isdigit():
            total += int(text)
    if saw_marker and total == 0:
        return 0
    return total
