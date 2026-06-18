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

    Resolution searches several layouts (``site-packages/vllm/benchmarks/``, ancestors, bare
    ``site-packages/benchmarks/``) because some vLLM wheels omit ``benchmarks/`` unless built
    with bench extras; the probe opens the file before printing the path.

    Fails at runtime if no interpreter can import ``vllm`` or no readable benchmark script exists.
    """
    base = validated_bench_script_basename(bench_serv_script)
    # Multiline -c body: shlex.quote wraps it safely for bash -c.
    py = f"""import os, pathlib, site, sys
import vllm
n = {base!r}
root = pathlib.Path(vllm.__file__).resolve().parent
seen = set()
cands = []

def add(p):
    try:
        q = pathlib.Path(p).resolve()
    except Exception:
        return
    if q in seen:
        return
    seen.add(q)
    cands.append(q)

add(root / "benchmarks" / n)
for anc in list(root.parents)[:8]:
    add(anc / "benchmarks" / n)
for sp in site.getsitepackages():
    sp = pathlib.Path(sp)
    add(sp / "vllm" / "benchmarks" / n)
    add(sp / "benchmarks" / n)
chosen = None
for c in cands:
    if not (c.is_file() and os.access(c, os.R_OK)):
        continue
    try:
        with c.open("rb") as f:
            f.read(1)
    except OSError:
        continue
    chosen = c
    break
if not chosen:
    sys.stderr.write(
        "CVS: vLLM benchmark script %r not found after search. "
        "Many wheels omit vllm/benchmarks; install in the image (e.g. pip install 'vllm[bench]') "
        "or bind-mount benchmark_serving.py from a vLLM source tree.\\n" % (n,)
    )
    sys.exit(1)
print(sys.executable)
print(str(chosen))
"""
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
