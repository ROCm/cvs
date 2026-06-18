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
    Shell snippet that sets ``BENCH_PY`` (``sys.executable`` for the import) and
    ``BENCH_SCRIPT`` (absolute path to ``vllm/benchmarks/<script>``), then ``export``\ s both.

    Tries ``python3.13``, ``python3.12``, …, ``python3`` so ROCm images where ``/usr/bin/python3``
    is not the interpreter that owns the installed ``vllm`` package still resolve the driver.

    Fails at runtime if no interpreter can import ``vllm`` or the benchmark file is missing.
    """
    base = validated_bench_script_basename(bench_serv_script)
    py = (
        "import pathlib,sys;import vllm;"
        f"n={base!r};"
        "r=pathlib.Path(vllm.__file__).resolve().parent;"
        "c=r/'benchmarks'/n;"
        "(c.is_file() or (sys.stderr.write('CVS: missing vLLM benchmark script: '+str(c)+chr(10)) or sys.exit(1)));"
        "print(sys.executable);print(c)"
    )
    py_q = shlex.quote(py)
    # Embedded in ``bash -c`` arguments (use :func:`shlex.quote` on the full script in callers).
    return (
        "BENCH_PY=; BENCH_SCRIPT=; "
        "for _cvs_py in python3.13 python3.12 python3.11 python3.10 python3; do "
        f"{{ read -r BENCH_PY; read -r BENCH_SCRIPT; }} < <($_cvs_py -c {py_q} 2>/dev/null) && break; "
        "done; "
        '[ -n "$BENCH_SCRIPT" ] && [ -n "$BENCH_PY" ] || '
        "{ echo 'CVS: could not resolve vLLM benchmark driver (tried python3.13..python3)' >&2; exit 1; }; "
        "export BENCH_PY BENCH_SCRIPT"
    )
