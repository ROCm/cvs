'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Canonical **vLLM benchmark server** shell helpers used by:

* :mod:`cvs.lib.inference.vllm_orch` (``VllmJob`` — ``variant.paths.benchmark_scripts_dir`` should
  include these files on the host path that is bind-mounted into the container), and
* :mod:`cvs.lib.inference.inferencemax_orch` (host-mounted / staged server scripts when
  ``host_benchmark_scripts_relpath`` defaults to ``lib/dtni/vllm_benchmark_scripts``).

Keeping scripts here avoids drift between InferenceMax wrappers and vLLM single orchestration.

Client load generation uses the **vLLM install’s** ``benchmarks/<script>`` (no third-party git clone).
'''

from __future__ import annotations

import re
import shlex
from pathlib import Path


def bundled_scripts_dir() -> Path:
    """Directory containing checked-in ``*.sh`` server entrypoints (package-relative)."""
    return Path(__file__).resolve().parent


def validated_bench_script_basename(name: str) -> str:
    """Return a safe ``*.py`` basename under ``vllm/benchmarks/`` for ``bench_serv_script``."""
    base = (name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.py", base):
        raise ValueError(f"bench_serv_script must be a safe .py basename, got {name!r}")
    return base


def bash_export_bench_script_from_vllm_install(bench_serv_script: str) -> str:
    """
    Shell snippet that ``export``\ s ``BENCH_SCRIPT`` to the absolute path of the packaged
    benchmark driver next to the installed ``vllm`` package (typically ``.../vllm/benchmarks/...``).

    Fails at runtime if the file is missing (use a vLLM image/wheel that ships the ``benchmarks/``
    tree, or ``vllm bench serve`` once CVS migrates its CLI wiring).
    """
    base = validated_bench_script_basename(bench_serv_script)
    py = (
        "import pathlib,sys;import vllm;"
        f"n={base!r};"
        "r=pathlib.Path(vllm.__file__).resolve().parent;"
        "c=r/'benchmarks'/n;"
        "(c.is_file() and print(c)) or "
        "(sys.stderr.write('CVS: missing vLLM benchmark script: '+str(c)+chr(10)) or sys.exit(1))"
    )
    return f'export BENCH_SCRIPT="$(python3 -c {shlex.quote(py)})"'
