'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Needle-in-a-haystack long-context accuracy against a live ATOM server (ACC-12).
'''

from __future__ import annotations

import base64
import shlex
from typing import Any, Dict

from cvs.lib import globals
from cvs.lib.inference.utils.long_context_accuracy_config import LongContextAccCell
from cvs.lib.utils.model_query_lib import LongContextNiahBenchmark

log = globals.log


def run_niah_cell(
    *,
    orch: Any,
    variant,
    cell: LongContextAccCell,
    expected_pass_rate: float,
    output_dir: str,
) -> Dict[str, float]:
    p = variant.params
    port = int(p.port_no)
    host = (p.base_url or "http://0.0.0.0").replace("http://", "").replace("https://", "")
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"

    i_dict = {
        "num_prompts": cell.num_prompts,
        "seed": cell.seed,
        "local_files_only": True,
        "expected_results": {"auto": {LongContextNiahBenchmark.DEFAULT_METRIC_KEY: expected_pass_rate}},
    }
    _inner_cmd, scoring = LongContextNiahBenchmark.prepare(
        i_dict,
        port=port,
        host=host,
        model_id=variant.model.id,
        isl=cell.isl,
        osl=cell.osl,
        log_dir=output_dir,
        log_basename=f"niah_{cell.id}.log",
    )
    probe_src = LongContextNiahBenchmark.probe_script(**scoring["probe_kwargs"])
    b64 = base64.b64encode(probe_src.encode("utf-8")).decode("ascii")
    probe_path = "/tmp/long_ctx_niah_probe.py"
    cmd = (
        f"source /tmp/server_env_script.sh && "
        f"mkdir -p {shlex.quote(output_dir)}/benchmark_node && "
        f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(probe_path)} && "
        f"python3 {shlex.quote(probe_path)} 2>&1 | tee {shlex.quote(scoring['log_path'])}"
    )
    out = orch.exec_on_head(cmd, timeout=scoring["exec_timeout_sec"])
    text = next(iter(out.values()), "") or ""
    check_kwargs = LongContextNiahBenchmark.check_kwargs_from_scoring(scoring)
    ok, summary, err = LongContextNiahBenchmark.check_results(text, **check_kwargs)
    if not ok:
        raise RuntimeError(err or f"NIAH cell {cell.id!r} failed")
    actual = float(summary["actual"])
    return {f"accuracy.niah_pass_rate__{cell.id}": actual}
