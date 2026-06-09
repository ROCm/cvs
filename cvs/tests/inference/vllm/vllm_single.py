'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Parametrized vLLM single-node benchmark suite (replaces the 4 per-model wrappers).
'''

from cvs.lib import globals
from cvs.lib.dtni.verdict import evaluate_all
from cvs.lib.inference.vllm_orch import VllmJob

import importlib.util as _ilu
import pathlib as _pl
_spec = _ilu.spec_from_file_location("_dtni_vllm_shared", _pl.Path(__file__).with_name("_shared.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
test_print_results_table = _mod.test_print_results_table  # exported as a sibling test  # noqa: F841

log = globals.log


def _num_prompts_for(osl, concurrency):
    return str(concurrency * 20) if int(osl) >= 8192 else str(concurrency * 50)


def test_vllm_inference(orch, variant_config, hf_token, seq_combo, concurrency, inf_res_dict):
    isl = seq_combo["isl"]
    osl = seq_combo["osl"]
    job = VllmJob(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        num_prompts=_num_prompts_for(osl, concurrency),
    )

    job.stop_server()
    job.build_server_cmd()
    job.start_server()
    job.wait_ready()
    job.run_client()
    job.wait_client_complete()
    results = job.parse_results()

    key = (
        variant_config.model.id,
        variant_config.gpu_arch,
        isl,
        osl,
        seq_combo.get("name", "default"),
        concurrency,
    )
    inf_res_dict[key] = results

    # Per-cell thresholds: thresholds.json layout is `{"ISL=...,OSL=...,TP=...,CONC=...": {metric: spec}}`
    cell = f"ISL={isl},OSL={osl},TP={variant_config.params.tensor_parallelism},CONC={concurrency}"
    cell_thresholds = variant_config.thresholds.get(cell, {})
    if not cell_thresholds:
        log.warning("no thresholds for %s; skipping verdict assertion", cell)
        return
    for host, actuals in results.items():
        evaluate_all(actuals, cell_thresholds)
