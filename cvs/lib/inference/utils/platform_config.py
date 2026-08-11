'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Optional platform / post-run checks for inference suites (INF-6/7).
'''

from __future__ import annotations

from cvs.lib.utils.config_loader import _Forbid


class PlatformConfig(_Forbid):
    dmesg_scan: bool = False
    gpu_metrics_poll: bool = False
