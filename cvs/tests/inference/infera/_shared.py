'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.
'''

from __future__ import annotations

import os
from typing import Any, Mapping

from cvs.lib import globals

log = globals.log

INFERA_DISAGG_TEST_ORDER = {
    "test_cleanup_stale_containers": 0,
    "test_launch_inference_containers": 1,
    "test_start_etcd": 2,
    "test_setup_worker_env": 3,
    "test_launch_frontend": 4,
    "test_launch_prefill_servers": 5,
    "test_launch_decode_servers": 6,
    "test_poll_for_server_ready": 7,
    "test_openai_compatible_http_endpoints": 8,
    "test_teardown": 9,
}


def resolve_benchmark_variant_key(root: Mapping[str, Any], config_path: str) -> str:
    env_key = (os.environ.get("INFERA_BENCHMARK_KEY") or "").strip()
    bp = root.get("benchmark_params") or {}
    if not isinstance(bp, dict) or not bp:
        raise ValueError(f"benchmark_params missing or empty in {config_path!r}")

    if env_key:
        if env_key not in bp:
            raise ValueError(
                f"INFERA_BENCHMARK_KEY={env_key!r} not found in benchmark_params "
                f"({config_path}); valid: {sorted(bp)!r}"
            )
        return env_key

    explicit = root.get("active_benchmark")
    if explicit is not None:
        if explicit not in bp:
            raise ValueError(
                f"active_benchmark={explicit!r} not found in benchmark_params "
                f"({config_path}); valid: {sorted(bp)!r}"
            )
        return str(explicit)

    if len(bp) == 1:
        return str(next(iter(bp)))

    raise ValueError(
        f"Multiple benchmark_params keys in {config_path!r}: {sorted(bp)!r}. "
        'Set top-level "active_benchmark" or export INFERA_BENCHMARK_KEY.'
    )
