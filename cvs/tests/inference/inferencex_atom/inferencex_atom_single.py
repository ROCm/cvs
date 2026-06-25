'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
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
from cvs.lib.inference.utils.inferencex_atom_config_loader import expand_sweep
from cvs.lib.inference.utils.inferencex_atom_parsing import (
    CLIENT_METRIC_UNITS as _METRIC_UNITS,
    METRIC_TIERS,
    METRIC_TIER_ORDER,
    RECORD_METRICS,
    tier_metric_specs,
)
from cvs.lib.utils.verdict import evaluate_all
from cvs.tests.inference.inferencex_atom._shared import test_print_results_table  # noqa: F401

log = globals.log


def _log_variant_run_card(variant_config):
    rc = variant_config.run_card
    parts = [
        f"gpu_arch={variant_config.gpu_arch}",
        f"driver={variant_config.params.driver}",
        f"model={variant_config.model.id}",
    ]
    if variant_config.ix_recipe_id:
        parts.append(f"ix_recipe_id={variant_config.ix_recipe_id}")
    if rc.atom_image_pin:
        parts.append(f"image_pin={rc.atom_image_pin}")
    if rc.upstream_run_url:
        parts.append(f"upstream_run={rc.upstream_run_url}")
    if rc.notes:
        parts.append(f"notes={rc.notes}")
    log.info("InferenceX ATOM run card: %s", "; ".join(parts))


def _reuse_server_flag(params) -> bool:
    raw = str(getattr(params, "reuse_server_across_sweep", "true")).strip().lower()
    return raw in ("true", "1", "yes")


def _server_session_key(variant_config, isl, osl):
    p = variant_config.params
    return (
        variant_config.model.id,
        p.driver,
        str(isl),
        str(osl),
        variant_config.ix_recipe_id or "",
        p.tensor_parallelism,
    )


def _tier_display_metric(tier):
    if tier == "record":
        return RECORD_METRICS[0] if RECORD_METRICS else "output_throughput"
    names = METRIC_TIERS.get(tier, ())
    return names[0] if names else tier


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)
    cases, ids = expand_sweep(raw.get("sweep", {}))
    if "metric_tier" in metafunc.fixturenames:
        if cases:
            tier_cases = []
            tier_ids = []
            for (combo, c), cid in zip(cases, ids):
                for tier in METRIC_TIER_ORDER:
                    tier_cases.append((combo, c, tier))
                    tier_ids.append(f"{cid}-{tier}")
            metafunc.parametrize("seq_combo,concurrency,metric_tier", tier_cases, ids=tier_ids)
    elif "seq_combo" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames and cases:
        metafunc.parametrize("seq_combo,concurrency", cases, ids=ids)


def test_inferencex_atom_inference(
    orch,
    variant_config,
    hf_token,
    seq_combo,
    concurrency,
    inf_res_dict,
    server_session,
    lifecycle,
    request,
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    isl = seq_combo["isl"]
    osl = seq_combo["osl"]
    p = variant_config.params
    _log_variant_run_card(variant_config)
    job = InferenceXAtomJob.from_variant(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
    )

    session_key = _server_session_key(variant_config, isl, osl)
    reuse = _reuse_server_flag(p) and server_session.get("key") == session_key

    try:
        if not reuse:
            job.stop_server()
            job.build_server_cmd()
            t = time.monotonic()
            job.start_server()
            job.wait_ready()
            lifecycle.record(request.node.nodeid, "server_ready", time.monotonic() - t)
            if _reuse_server_flag(p):
                server_session["key"] = session_key
        else:
            log.info("reusing ATOM server across sweep cell (key=%s)", session_key)
            job.prepare_cell_out_dir()
        t_client = time.monotonic()
        job.run_client()
        job.wait_client_complete()
        results = job.parse_results()
    except Exception:
        lifecycle.failed = True
        raise

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
    """One pytest row per metric tier per sweep cell (W1 gate batches)."""
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
    full = f"client.{display}"
    value = actuals.get(full)
    unit = _METRIC_UNITS.get(display, metric_tier)
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", unit))

    if not variant_config.enforce_thresholds or metric_tier == "record":
        return
    if not specs:
        pytest.fail(f"no threshold specs for tier {metric_tier!r} in cell {cell!r}")
    evaluate_all(actuals, specs)
