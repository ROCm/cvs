'''Launch-command provenance for inferencex_atom suite reports.'''

from __future__ import annotations

import shlex
from types import SimpleNamespace
from typing import Any

from cvs.lib.inference.inferencex_atom.inferencex_atom_orch import InferenceXAtomJob


def _first_sweep_point(variant: Any) -> tuple[str, str, str, str] | None:
    """Return (cell_key, isl, osl, concurrency) for the first sweep run."""
    runs = list(getattr(getattr(variant, "sweep", None), "runs", None) or [])
    if not runs:
        return None
    by_name = {c.name: c for c in variant.sweep.sequence_combinations}
    combo = by_name.get(runs[0].combo)
    if combo is None:
        return None
    isl = str(combo.isl)
    osl = str(combo.osl)
    conc = str(runs[0].concurrency)
    return variant.cell_key(isl, osl, conc), isl, osl, conc


def launch_summary(variant: Any) -> str:
    p = variant.params
    parts = [str(p.driver), f"TP={p.tensor_parallelism}"]
    nnodes = int(getattr(p, "nnodes", 1) or 1)
    if nnodes > 1:
        parts.append(f"nnodes={nnodes}")
        parts.append(f"PP={getattr(p, 'pipeline_parallel_size', 1)}")
    max_len = getattr(p, "max_model_length", None)
    if max_len not in (None, ""):
        parts.append(f"max_model_len={max_len}")
    return " \u00b7 ".join(parts)


def _example_job(variant: Any, *, isl: str, osl: str, concurrency: str) -> InferenceXAtomJob:
    return InferenceXAtomJob(
        orch=SimpleNamespace(),
        variant=variant,
        hf_token="***",
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        num_prompts=variant.params.num_prompts,
    )


def server_command(variant: Any) -> str:
    point = _first_sweep_point(variant)
    if point is None:
        return ""
    _cell, isl, osl, conc = point
    job = _example_job(variant, isl=isl, osl=osl, concurrency=conc)
    argv = job._atom_server_argv() if job.driver == "atom" else job._server_argv()
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def bench_command(variant: Any) -> str:
    point = _first_sweep_point(variant)
    if point is None:
        return ""
    _cell, isl, osl, conc = point
    job = _example_job(variant, isl=isl, osl=osl, concurrency=conc)
    argv = job._atom_client_argv() if job.driver == "atom" else job._vllm_client_argv()
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def build_launch_provenance(variant: Any) -> dict[str, str]:
    out: dict[str, str] = {"launch_summary": launch_summary(variant)}
    point = _first_sweep_point(variant)
    if point is None:
        return out
    cell, isl, osl, conc = point
    job = _example_job(variant, isl=isl, osl=osl, concurrency=conc)
    if job.driver == "atom":
        server_argv = job._atom_server_argv()
        bench_argv = job._atom_client_argv()
    else:
        server_argv = job._server_argv()
        bench_argv = job._vllm_client_argv()
    out["launch_example_cell"] = cell
    out["launch_server_cmd"] = " ".join(shlex.quote(str(arg)) for arg in server_argv)
    out["launch_bench_cmd"] = " ".join(shlex.quote(str(arg)) for arg in bench_argv)
    return out
