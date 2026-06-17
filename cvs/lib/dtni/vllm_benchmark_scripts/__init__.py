'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Canonical **vLLM benchmark server** shell helpers used by:

* :mod:`cvs.lib.inference.vllm_orch` (``VllmJob`` — ``variant.paths.benchmark_scripts_dir`` should
  include these files on the host path that is bind-mounted into the container), and
* :mod:`cvs.lib.inference.inferencemax_orch` (host-mounted / staged server scripts when
  ``host_benchmark_scripts_relpath`` defaults to ``lib/dtni/vllm_benchmark_scripts``).

Keeping scripts here avoids drift between InferenceMax wrappers and vLLM single orchestration.
'''

from __future__ import annotations

from pathlib import Path

# Same fork used by legacy ``InferenceBaseJob`` / cluster configs (calibration-bearing client).
BENCH_SERVING_GIT_URL = "https://github.com/kimbochen/bench_serving.git"


def bundled_scripts_dir() -> Path:
    """Directory containing checked-in ``*.sh`` server entrypoints (package-relative)."""
    return Path(__file__).resolve().parent
