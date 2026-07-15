'''Launch-command provenance for inferencex_atom suite reports.'''

from __future__ import annotations

import re
import shlex
from types import SimpleNamespace
from typing import Any

from cvs.lib.inference.inferencex_atom.inferencex_atom_orch import InferenceXAtomJob

_CELL_RE = re.compile(
    r"ISL=(?P<isl>\d+),OSL=(?P<osl>\d+)(?:,TP=\d+)?(?:,PP=\d+)?(?:,NNODES=\d+)?,CONC=(?P<conc>\d+)"
)


def _first_sweep_coords(variant: Any) -> tuple[str, str, str, str]:
    cells = list(getattr(variant, "expected_cells", lambda: [])())
    if cells:
        match = _CELL_RE.search(cells[0])
        if match:
            return cells[0], match.group("isl"), match.group("osl"), match.group("conc")
    return "", "1024", "1024", "128"


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


def server_command(variant: Any, *, rank: int = 0) -> str:
    example_cell, isl, osl, conc = _first_sweep_coords(variant)
    del example_cell
    job = _example_job(variant, isl=isl, osl=osl, concurrency=conc)
    if job.driver == "atom":
        argv = job._atom_server_argv(rank)
    else:
        argv = job._server_argv(rank)
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def bench_command(variant: Any) -> str:
    example_cell, isl, osl, conc = _first_sweep_coords(variant)
    del example_cell
    job = _example_job(variant, isl=isl, osl=osl, concurrency=conc)
    if job.driver == "atom":
        argv = job._atom_client_argv()
    else:
        argv = job._vllm_client_argv()
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def build_launch_provenance(variant: Any) -> dict[str, str]:
    example_cell, _isl, _osl, _conc = _first_sweep_coords(variant)
    out = {
        "launch_summary": launch_summary(variant),
        "launch_server_cmd": server_command(variant),
        "launch_bench_cmd": bench_command(variant),
    }
    if example_cell:
        out["launch_example_cell"] = example_cell
    return out
