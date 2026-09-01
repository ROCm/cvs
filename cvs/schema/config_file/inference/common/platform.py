"""Optional platform / post-run checks for inference suites."""

from __future__ import annotations

from cvs.schema.base import _Forbid


class PlatformConfig(_Forbid):
    dmesg_scan: bool = False
    gpu_metrics_poll: bool = False
