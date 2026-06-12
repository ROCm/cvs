'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Parametrized vLLM single-node benchmark suite (replaces the 4 per-model wrappers).
'''

import json
import os
import shlex
import time

import pytest

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

# Fetch-progress poll: du the cache dir until its size stops growing. The model
# download streams in parallel shards, so size climbs then plateaus at the full
# weight set; a stable size across two polls means the fetch settled.
_FETCH_POLL_COUNT = 80
_FETCH_POLL_WAIT_S = 30


def pytest_generate_tests(metafunc):
    """Parametrize test_vllm_inference over sequence_combinations × concurrency_levels.

    Lives in the suite module (not conftest) because it parametrizes fixtures
    only test_vllm_inference consumes -- co-locating the parametrization with
    its sole consumer. It runs at collection time, before fixtures exist, so it
    reads the raw config_file JSON directly (it cannot use the variant_config
    fixture / the typed loader).
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)
    sweep = raw.get("sweep", {})
    combos = sweep.get("sequence_combinations", [])
    concs = sweep.get("concurrency_levels", [])
    cases = []
    ids = []
    for combo in combos:
        default_name = "isl" + combo["isl"] + "_osl" + combo["osl"]
        for c in concs:
            cases.append((combo, c))
            ids.append(combo.get("name", default_name) + "-conc" + str(c))
    if "seq_combo" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames and cases:
        metafunc.parametrize("seq_combo,concurrency", cases, ids=ids)


def _num_prompts_for(osl, concurrency):
    return str(concurrency * 20) if int(osl) >= 8192 else str(concurrency * 50)


def _du_bytes(orch, path):
    """Total bytes under `path` inside the container, or 0 if it doesn't exist yet."""
    out = orch.exec(f"bash -c {shlex.quote(f'du -sb {shlex.quote(path)} 2>/dev/null | cut -f1')}")
    total = 0
    for text in (out or {}).values():
        for tok in (text or "").split():
            if tok.isdigit():
                total = max(total, int(tok))
    return total


def test_launch_container(orch, variant_config, lifecycle, request):
    """Stage 1: launch the container. Asserts it is independently observed running."""
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
    """Stage 2: start sshd in the container. Asserts the daemon is reachable on 2224."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    t = time.monotonic()
    ok = orch.setup_sshd()
    lifecycle.record(request.node.nodeid, "sshd_setup", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        pytest.fail("setup_sshd() returned False")
    probe = orch.exec("bash -c 'ss -ltn 2>/dev/null | grep -q :2224 && echo OK || echo NO'")
    if not any("OK" in (v or "") for v in (probe or {}).values()):
        lifecycle.failed = True
        pytest.fail("sshd not listening on 2224 after setup_sshd()")


def test_model_fetch(orch, variant_config, lifecycle, request):
    """Stage 3: ensure the model is present in the HF cache (mounted models dir).

    For a remote pull this is the ~152GB download; the row shows its real
    duration and final size. For an offline/pre-staged model it returns near
    instantly. Skips (never silently passes) if the cache dir is unconfigured
    -- without it the fetch target is meaningless. Progress is polled via
    `du -sb` (size on disk), the robust size-poll proven in the validation run.
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    models_dir = variant_config.paths.models_dir
    if not models_dir:
        pytest.skip("paths.models_dir unset; cannot locate/verify the HF cache")

    remote = getattr(variant_config.model, "remote", 0)
    t = time.monotonic()
    orch.exec(f"mkdir -p {shlex.quote(models_dir)}")

    if remote:
        # Kick a background download into the pinned cache, then poll size.
        fetch = (
            f"HF_HUB_CACHE={shlex.quote(models_dir)} "
            f"nohup hf download {shlex.quote(variant_config.model.id)} "
            f"> /tmp/hf_fetch.log 2>&1 &"
        )
        orch.exec("bash -c " + shlex.quote(fetch))

    prev = -1
    stable = 0
    final = _du_bytes(orch, models_dir)
    for it in range(_FETCH_POLL_COUNT):
        cur = _du_bytes(orch, models_dir)
        final = cur
        log.info("[fetch poll %d] size=%.1fGB", it, cur / 1e9)
        if cur > 0 and cur == prev:
            stable += 1
            if stable >= 2 or not remote:
                break
        else:
            stable = 0
        prev = cur
        if not remote:
            break
        time.sleep(_FETCH_POLL_WAIT_S)

    lifecycle.record(request.node.nodeid, "model_fetch", time.monotonic() - t)
    lifecycle.record(request.node.nodeid, "model_size", final / 1e9, "GB")
    if final <= 0:
        lifecycle.failed = True
        pytest.fail(f"no model bytes under {models_dir} after fetch")


def test_vllm_inference(orch, variant_config, hf_token, seq_combo, concurrency, inf_res_dict, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
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
        client_poll_count=int(variant_config.params.client_poll_count),
    )

    # A failure mid-sweep flips lifecycle.failed so the remaining cells skip
    # cleanly (instead of each re-failing) AND the orch leak-guard finalizer
    # still tears the container down. The explicit teardown row may not run on
    # the failure path, which is exactly what the finalizer covers.
    try:
        job.stop_server()
        job.build_server_cmd()
        t = time.monotonic()
        job.start_server()
        job.wait_ready()
        lifecycle.record(request.node.nodeid, "server_ready", time.monotonic() - t)
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

    # Per-cell thresholds: thresholds layout is `{"ISL=...,OSL=...,TP=...,CONC=...": {metric: spec}}`.
    # The key is built by VariantConfig.cell_key (the same builder the loader uses for its
    # coverage check). When enforce_thresholds is true a missing cell is a hard error -- never
    # a silent skip that would report a green PASS with no assertions. When it is false the
    # config is a record-only scaffold (un-calibrated thresholds): capture the metrics and
    # skip the verdict.
    if not variant_config.enforce_thresholds:
        log.info("enforce_thresholds=false; recorded metrics for cell, skipping verdict")
        return
    cell = variant_config.cell_key(isl, osl, concurrency)
    cell_thresholds = variant_config.thresholds.get(cell)
    if not cell_thresholds:
        raise AssertionError(f"no thresholds for cell {cell!r}; threshold file is out of sync with the sweep")
    for host, actuals in results.items():
        evaluate_all(actuals, cell_thresholds)


def test_teardown(orch, lifecycle, request):
    """Final stage: explicit container teardown, timed, asserting it is gone.

    Sets lifecycle.torn_down so the orch fixture's leak-guard finalizer no-ops
    (avoids a double teardown). Runs even if an earlier stage failed -- teardown
    must happen regardless -- so it does NOT skip on lifecycle.failed.
    """
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    lifecycle.torn_down = True
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
