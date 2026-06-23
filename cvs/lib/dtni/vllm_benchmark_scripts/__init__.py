'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Canonical **vLLM benchmark server** shell helpers retained for legacy
:class:`~cvs.lib.inference.base.InferenceBaseJob` flows. **InferenceX ATOM** and
**vllm_single** (:class:`~cvs.lib.inference.vllm_orch.VllmJob`) build
``vllm serve`` in Python via ``roles.server.serve_args``; they do not stage
these scripts at runtime.

Client load generation uses the **vLLM install’s** ``benchmarks/<script>`` when present, otherwise
``python -m vllm.entrypoints.cli.main bench serve`` (no third-party git clone).
'''

from __future__ import annotations

import re
import shlex
import textwrap
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


def clamped_bench_random_range_ratio_str(
    random_range_ratio: str | float | int,
    input_sequence_length: str | float | int,
    output_sequence_length: str | float | int,
    max_model_length: str | float | int,
) -> tuple[str, bool]:
    """
    Return ``(ratio_str, was_clamped)`` for ``--random-range-ratio`` on vLLM ``random`` datasets.

    Current vLLM bench drivers sample input and output lengths near the configured ISL/OSL with
    spread controlled by ``random_range_ratio`` (approximately ``ISL*(1±r)`` and ``OSL*(1±r)``).
    If the peak combined length ``(ISL+OSL)*(1+r)`` would exceed ``max_model_length``, many
    completion requests are rejected by the server; clamp ``r`` to ``max(0, MML/(ISL+OSL)-1)``.
    """
    raw_s = str(random_range_ratio).strip()
    try:
        r = float(raw_s)
        isl = int(float(str(input_sequence_length).strip()))
        osl = int(float(str(output_sequence_length).strip()))
        mml = int(float(str(max_model_length).strip()))
    except (TypeError, ValueError, ArithmeticError):
        return raw_s, False

    base = isl + osl
    if base <= 0 or mml <= 0:
        return raw_s, False

    r_cap = mml / float(base) - 1.0
    if r_cap < 0.0:
        r_cap = 0.0
    r_eff = min(max(r, 0.0), r_cap)
    # Stable short decimal (avoid 0.30000000004-style floats in argv).
    out = f"{r_eff:.12g}"
    was = abs(r_eff - r) > 1e-12
    return out, was


def bash_export_bench_script_from_vllm_install(bench_serv_script: str) -> str:
    """
    Shell snippet that exports ``BENCH_PY``, ``BENCH_KIND``, ``BENCH_SCRIPT``, and defines
    ``_cvs_run_bench`` — a shell function that invokes either the legacy script or
    ``python -m vllm.entrypoints.cli.main bench serve``.

    Tries ``python3.13``, ``python3.12``, …, ``python3`` so ROCm images where ``/usr/bin/python3``
    is not the interpreter that owns the installed ``vllm`` package still resolve the driver.

    Resolution searches several layouts (``site-packages/vllm/benchmarks/``, ancestors, bare
    ``site-packages/benchmarks/``) because some vLLM wheels omit ``benchmarks/`` unless built
    with bench extras; the probe opens the file before accepting a path.

    If no ``benchmarks/<script>`` file exists but the install exposes the vLLM CLI bench entry
    (``python -m vllm.entrypoints.cli.main bench serve``), ``BENCH_KIND`` is set to
    ``cli_module`` and ``_cvs_run_bench`` forwards the same flags the legacy script accepted.

    Fails at runtime if no interpreter can import ``vllm`` or neither a script nor the bench CLI
    is usable.
    """
    base = validated_bench_script_basename(bench_serv_script)
    py_body = textwrap.dedent(
        f"""
        import importlib.util, os, pathlib, shlex, site, subprocess, sys

        import vllm

        n = {base!r}

        def esc(s):
            return shlex.quote(str(s))

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
        if chosen:
            print("export BENCH_PY=" + esc(sys.executable))
            print("export BENCH_KIND=script")
            print("export BENCH_SCRIPT=" + esc(str(chosen)))
            sys.exit(0)

        if importlib.util.find_spec("vllm.entrypoints.cli.main") is None:
            sys.stderr.write(
                "CVS: vLLM benchmark script %r not found after search and "
                "``vllm.entrypoints.cli.main`` is missing. "
                "Install a build that ships benchmarks (e.g. pip install 'vllm[bench]').\\n" % (n,)
            )
            sys.exit(1)
        try:
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vllm.entrypoints.cli.main",
                    "bench",
                    "serve",
                    "--help",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            r = None
        if r is None or r.returncode != 0:
            sys.stderr.write(
                "CVS: vLLM benchmark script %r not found and "
                "``python -m vllm.entrypoints.cli.main bench serve`` is not available. "
                "Many wheels omit ``vllm/benchmarks/``; install ``vllm[bench]`` or match your "
                "Python interpreter to the vLLM install.\\n" % (n,)
            )
            sys.exit(1)
        print("export BENCH_PY=" + esc(sys.executable))
        print("export BENCH_KIND=cli_module")
        print("export BENCH_SCRIPT=")
        sys.exit(0)
        """
    ).strip()
    py_q = shlex.quote(py_body)
    # ``eval`` consumes ``export VAR=...`` lines from the probe (no ``read`` / process-substitution
    # edge cases). ``_cvs_run_bench`` mirrors ``"$BENCH_PY" "$BENCH_SCRIPT"`` for script mode.
    return (
        "BENCH_PY=; BENCH_KIND=; BENCH_SCRIPT=; "
        "for _cvs_py in python3.13 python3.12 python3.11 python3.10 python3; do "
        f"_cvs_eval=$($_cvs_py -c {py_q} 2>/dev/null) || true; "
        '[ -z "$_cvs_eval" ] && continue; '
        'eval "$_cvs_eval" || continue; '
        '[ -n "$BENCH_PY" ] && [ -n "$BENCH_KIND" ] && break; '
        "done; "
        '[ -n "$BENCH_KIND" ] && [ -n "$BENCH_PY" ] || '
        "{ echo 'CVS: could not resolve vLLM benchmark driver (tried python3.13..python3)' >&2; exit 1; }; "
        'if [ "$BENCH_KIND" = script ] && { [ ! -f "$BENCH_SCRIPT" ] || [ ! -r "$BENCH_SCRIPT" ]; }; then '
        '{ echo "CVS: BENCH_SCRIPT is missing or unreadable: $BENCH_SCRIPT" >&2; exit 1; }; fi; '
        '_cvs_run_bench() { case "$BENCH_KIND" in cli_module) '
        '"$BENCH_PY" -m vllm.entrypoints.cli.main bench serve "$@" '
        ';; *) "$BENCH_PY" "$BENCH_SCRIPT" "$@" ;; esac; }; '
        "export BENCH_PY BENCH_KIND BENCH_SCRIPT"
    )
