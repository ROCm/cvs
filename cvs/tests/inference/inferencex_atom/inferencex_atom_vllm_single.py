'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX vLLM parity suite (M4): same sweep layout as ``inferencex_atom``
with ``framework: inferencex_atom_vllm_single`` and ``params.driver: vllm``.
'''

import json
import os
import time

import pytest

from cvs.lib import globals
from cvs.lib.inference.inference_suite_lifecycle import (
    sweep_cell_result_key,
    test_launch_container,
    test_model_fetch,
    test_setup_sshd,
    test_teardown,
)
from cvs.lib.inference.inferencex_atom_orch import InferenceXAtomJob
from cvs.lib.inference.utils.inferencex_atom_config_loader import (
    expand_sweep_parametrize,
    load_parity_reference_cells,
    reuse_server_flag,
    server_session_key,
)
from cvs.lib.inference.utils.inferencex_atom_parsing import (
    CLIENT_METRIC_UNITS as _METRIC_UNITS,
    COMPARE_VLLM_METRIC_UNITS,
    METRIC_TIERS,
    RECORD_METRICS,
    SCALING_METRIC_UNITS,
    VLLM_METRIC_TIER_ORDER,
    compare_vllm_metrics,
    tier_metric_specs,
)
from cvs.lib.utils.verdict import evaluate_all
from cvs.tests.inference.inferencex_atom._shared import test_print_results_table  # noqa: F401

log = globals.log


def _tier_display_metric(tier):
    if tier == "record":
        return RECORD_METRICS[0] if RECORD_METRICS else "output_throughput"
    if tier == "compare":
        from cvs.lib.inference.utils.inferencex_atom_parsing import COMPARE_VLLM_METRICS

        return COMPARE_VLLM_METRICS[0]
    names = METRIC_TIERS.get(tier, ())
    return names[0] if names else tier


@pytest.fixture(scope="module")
def parity_reference_cells(variant_config):
    path = (variant_config.run_card.parity_reference_json or "").strip()
    if not path:
        return {}
    return load_parity_reference_cells(path)


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        raise pytest.UsageError(
            f"--config_file not found or not specified: {config_file!r}"
        )
    with open(config_file) as fp:
        raw = json.load(fp)
    spec = expand_sweep_parametrize(
        raw.get("sweep", {}),
        metafunc.fixturenames,
        metric_tier_order=VLLM_METRIC_TIER_ORDER,
    )
    if spec:
        argnames, argvalues, ids = spec
        metafunc.parametrize(argnames, argvalues, ids=ids)


def test_inferencex_atom_vllm_inference(
    orch,
    variant_config,
    hf_token,
    seq_combo,
    concurrency,
    inf_res_dict,
    server_session,
    lifecycle,
    parity_reference_cells,
    request,
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    isl = seq_combo["isl"]
    osl = seq_combo["osl"]
    p = variant_config.params
    job = InferenceXAtomJob.from_variant(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
    )

    session_key = server_session_key(variant_config, isl, osl)
    reuse = reuse_server_flag(p) and server_session.get("key") == session_key

    try:
        if not reuse:
            job.stop_server()
            job.build_server_cmd()
            t = time.monotonic()
            job.start_server()
            job.wait_ready()
            lifecycle.record(request.node.nodeid, "server_ready", time.monotonic() - t)
            if reuse_server_flag(p):
                server_session["key"] = session_key
        else:
            log.info("reusing vLLM server across sweep cell (key=%s)", session_key)
            job.prepare_cell_out_dir()
        t_client = time.monotonic()
        job.run_client()
        job.wait_client_complete()
        results = job.parse_results()
    except Exception:
        lifecycle.failed = True
        raise

    cell = variant_config.cell_key(isl, osl, concurrency)
    ref_cell = parity_reference_cells.get(cell)
    if ref_cell:
        for host, metrics in results.items():
            metrics.update(compare_vllm_metrics(metrics, ref_cell))

    inf_res_dict[sweep_cell_result_key(variant_config, seq_combo, isl, osl, concurrency)] = results
    lifecycle.record(request.node.nodeid, "client_complete", time.monotonic() - t_client)


def test_cell_metrics(
    seq_combo,
    concurrency,
    metric_tier,
    inf_res_dict,
    variant_config,
    lifecycle,
    request,
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    isl = seq_combo["isl"]
    osl = seq_combo["osl"]
    key = sweep_cell_result_key(variant_config, seq_combo, isl, osl, concurrency)
    if key not in inf_res_dict:
        pytest.skip(f"no recorded results for cell {key!r} (inference did not run)")
    host_dict = inf_res_dict[key]
    _host, actuals = next(iter(host_dict.items()))
    cell = variant_config.cell_key(isl, osl, concurrency)
    thresholds_cell = variant_config.thresholds.get(cell) or {}
    specs = tier_metric_specs(thresholds_cell, metric_tier)

    display = _tier_display_metric(metric_tier)
    if metric_tier == "scaling":
        full = f"scaling.{display}"
        unit = SCALING_METRIC_UNITS.get(display, "%")
    elif metric_tier == "compare":
        full = f"compare.vllm.{display}"
        unit = COMPARE_VLLM_METRIC_UNITS.get(display, "ratio")
    else:
        full = f"client.{display}"
        unit = _METRIC_UNITS.get(display, metric_tier)
    value = actuals.get(full)
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", unit))

    if not variant_config.enforce_thresholds or metric_tier == "record":
        return
    if not specs:
        if metric_tier == "scaling" and int(variant_config.params.nnodes) <= 1:
            pytest.skip("scaling tier not configured for single-node runs")
        if metric_tier == "compare" and not (variant_config.run_card.parity_reference_json or "").strip():
            pytest.skip("compare tier requires run_card.parity_reference_json")
        pytest.fail(f"no threshold specs for tier {metric_tier!r} in cell {cell!r}")
    specs = {
        k: v for k, v in specs.items() if k in actuals and actuals[k] is not None
    }
    if not specs:
        pytest.fail(
            f"no assertable threshold specs for tier {metric_tier!r} in cell {cell!r} "
            f"(metrics missing from benchmark artifact)"
        )
    evaluate_all(actuals, specs)
