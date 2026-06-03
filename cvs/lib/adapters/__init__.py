"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Adapter package: importing this module triggers registration of every
shipped adapter via the ``@register_adapter`` decorator.

The spine's contract is "any code path that looks up an adapter by name
imports ``cvs.lib.adapters`` first" -- the same lazy-load pattern the
config loader uses (``_ensure_frameworks_loaded``). The conftest's
``get_adapter(cfg.framework)`` call relies on this.
"""

from __future__ import annotations

from cvs.lib.adapters import vllm_adapter  # noqa: F401  -- side-effect: register

__all__ = ["vllm_adapter"]
