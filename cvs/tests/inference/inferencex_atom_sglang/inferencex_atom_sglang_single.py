'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX SGLang (ROCm) parity suite — same workload cards as ``inferencex_atom_single``
with ``sglang.launch_server`` + ``sglang.bench_serving``.
'''

import importlib.util as _ilu
import json
import os
import pathlib as _pl
import shlex
import time

import pytest

from cvs.lib import globals
from cvs.lib.inference.inferencex_atom_sglang_orch import InferenceXAtomSglangJob
from cvs.lib.inference.utils.inferencing_config_loader import validate_sweep_selector
from cvs.lib.inference.utils.vllm_parsing import CLIENT_METRICS as _METRICS, CLIENT_METRIC_UNITS as _METRIC_UNITS
from cvs.lib.utils.verdict import evaluate_all

_spec = _ilu.spec_from_file_location("_inferencex_atom_sglang_shared", _pl.Path(__file__).with_name("_shared.py"))
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
        "engine=sglang",
        f"model={variant_config.model.id}",
    ]
    if variant_config.ix_recipe_id:
        parts.append(f"ix_recipe_id={variant_config.ix_recipe_id}")
    if rc.notes:
        parts.append(f"notes={rc.notes}")
    log.info("InferenceX SGLang parity run card: %s", "; ".join(parts))


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)
    sweep = raw.get("sweep", {})
    combos = sweep.get("sequence_combinations", [])
    runs = sweep.get("runs", [])
    validate_sweep_selector([c["name"] for c in combos], [r["combo"] for r in runs])
    by_name = {c["name"]: c for c in combos}
    cases = []
    ids = []
    for run in runs:
        combo = by_name[run["combo"]]
        conc = run["concurrency"]
        cases.append((combo, conc))
        ids.append(run["combo"] + "-conc" + str(conc))
    if "metric" in metafunc.fixturenames:
        if cases:
            metric_cases = []
            metric_ids = []
            for (combo, c), cid in zip(cases, ids):
                for short, _unit in _METRICS:
                    metric_cases.append((combo, c, short))
                    metric_ids.append(cid + "-" + short)
            metafunc.parametrize("seq_combo,concurrency,metric", metric_cases, ids=metric_ids)
    elif "seq_combo" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames and cases:
        metafunc.parametrize("seq_combo,concurrency", cases, ids=ids)


def _du_bytes(orch, path):
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


def test_inferencex_atom_sglang_inference(
    orch,
    variant_config,
    hf_token,
    seq_combo,
    concurrency,
    inf_res_dict,
    lifecycle,
    request,
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    isl = seq_combo["isl"]
    osl = seq_combo["osl"]
    p = variant_config.params
    _log_variant_run_card(variant_config)
    job = InferenceXAtomSglangJob(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        num_prompts=p.num_prompts,
        client_poll_count=int(p.client_poll_count),
        client_poll_wait_s=int(p.client_poll_wait_time),
        bench_max_failed_requests=int(p.bench_max_failed_requests),
    )

    try:
        job.stop_server()
        job.build_server_cmd()
        t = time.monotonic()
        job.start_server()
        job.wait_ready()
        lifecycle.record(request.node.nodeid, "server_ready", time.monotonic() - t)
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


def test_metric(seq_combo, concurrency, metric, inf_res_dict, variant_config, lifecycle, request):
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
    full = "client." + metric
    value = actuals.get(full)
    unit = _METRIC_UNITS.get(metric, "-")
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", unit))

    if not variant_config.enforce_thresholds:
        return
    cell = variant_config.cell_key(isl, osl, concurrency)
    spec = (variant_config.thresholds.get(cell) or {}).get(full)
    if spec is None:
        return
    if full not in actuals or actuals[full] is None:
        return
    evaluate_all(actuals, {full: spec})


def test_teardown(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
