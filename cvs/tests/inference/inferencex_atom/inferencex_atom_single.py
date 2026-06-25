'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import importlib.util as _ilu
import json
import os
import pathlib as _pl
import shlex
import time

import pytest

from cvs.lib import globals
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

_spec = _ilu.spec_from_file_location("_inferencex_atom_shared", _pl.Path(__file__).with_name("_shared.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
test_print_results_table = _mod.test_print_results_table  # noqa: F841

log = globals.log

_FETCH_POLL_COUNT = 80
_FETCH_POLL_WAIT_S = 30
_FETCH_PRESENCE_RETRIES = 5


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


def _du_bytes(orch, path):
    """Bytes under ``path`` in the container.

    Returns 0 when the path is absent or empty. Returns ``None`` when ``du``
    cannot run (permission, missing binary, etc.) so callers do not treat an
    infrastructure failure as "model not present".
    """
    quoted = shlex.quote(path)
    cmd = (
        f"if [ ! -e {quoted} ]; then echo __MISSING__; "
        f"elif bytes=$(du -sb {quoted} 2>/dev/null | cut -f1) && [ -n \"$bytes\" ]; "
        f"then echo \"$bytes\"; else echo __DU_ERROR__; fi"
    )
    out = orch.exec(f"bash -c {shlex.quote(cmd)}")
    total = 0
    saw_marker = False
    for text in (out or {}).values():
        text = (text or "").strip()
        if text == "__DU_ERROR__":
            return None
        if text == "__MISSING__":
            saw_marker = True
            continue
        if text.isdigit():
            total = max(total, int(text))
    if saw_marker and total == 0:
        return 0
    return total


def test_launch_container(orch, lifecycle, request):
    t = time.monotonic()
    ok = orch.setup_containers()
    lifecycle.record(request.node.nodeid, "container_launch", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        name = orch.get_container_name(orch.container_config, orch.container_config["image"])
        pytest.fail(f"setup_containers() returned False for {name}")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


def test_setup_sshd(orch, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    t = time.monotonic()
    ok = orch.setup_sshd()
    lifecycle.record(request.node.nodeid, "sshd_setup", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        pytest.fail("setup_sshd() returned False")
    if len(orch.hosts) > 1:
        probe = orch.exec("bash -c 'ss -ltn 2>/dev/null | grep -q :2224 && echo OK || echo NO'")
        if not any("OK" in (v or "") for v in (probe or {}).values()):
            lifecycle.failed = True
            pytest.fail("sshd not listening on 2224 after setup_sshd()")


def test_model_fetch(orch, variant_config, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    models_dir = variant_config.paths.models_dir
    if not models_dir:
        pytest.skip("paths.models_dir unset; cannot locate/verify the HF cache")

    remote = getattr(variant_config.model, "remote", 0)
    t = time.monotonic()
    orch.exec(f"mkdir -p {shlex.quote(models_dir)}")

    if not remote:
        final = 0
        for it in range(_FETCH_PRESENCE_RETRIES):
            cur = _du_bytes(orch, models_dir)
            if cur is None:
                lifecycle.failed = True
                pytest.fail(f"could not measure model cache size under {models_dir} (du error)")
            final = cur
            log.info("[fetch presence %d] size=%.1fGB", it, final / 1e9)
            if final > 0:
                break
            time.sleep(_FETCH_POLL_WAIT_S)
    else:
        fetch = (
            f"HF_HUB_CACHE={shlex.quote(models_dir)} "
            f"nohup hf download {shlex.quote(variant_config.model.id)} "
            f"> /tmp/hf_fetch.log 2>&1 &"
        )
        orch.exec("bash -c " + shlex.quote(fetch))
        prev = -1
        stable = 0
        final = _du_bytes(orch, models_dir)
        if final is None:
            lifecycle.failed = True
            pytest.fail(f"could not measure model cache size under {models_dir} (du error)")
        for it in range(_FETCH_POLL_COUNT):
            cur = _du_bytes(orch, models_dir)
            if cur is None:
                lifecycle.failed = True
                pytest.fail(f"could not measure model cache size under {models_dir} (du error)")
            final = cur
            log.info("[fetch poll %d] size=%.1fGB", it, cur / 1e9)
            if cur > 0 and cur == prev:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev = cur
            time.sleep(_FETCH_POLL_WAIT_S)

    lifecycle.record(request.node.nodeid, "model_fetch", time.monotonic() - t)
    lifecycle.record(request.node.nodeid, "model_size", final / 1e9, "GB")
    if final <= 0:
        lifecycle.failed = True
        pytest.fail(f"no model bytes under {models_dir} after fetch")


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

    key = (
        variant_config.model.id,
        variant_config.gpu_arch,
        isl,
        osl,
        seq_combo.get("name", "default"),
        concurrency,
    )
    inf_res_dict[key] = results
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
    key = (
        variant_config.model.id,
        variant_config.gpu_arch,
        isl,
        osl,
        seq_combo.get("name", "default"),
        concurrency,
    )
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


def test_teardown(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
