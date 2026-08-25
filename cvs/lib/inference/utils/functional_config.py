'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Optional functional smoke gates for inference suites (FUNC-1/2).
'''

from __future__ import annotations

from cvs.lib.utils.config_loader import _Forbid


class FunctionalConfig(_Forbid):
    api_smoke: bool = False
    health_check: bool = False
