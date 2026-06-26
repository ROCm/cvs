'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Parametrized vLLM multinode distributed benchmark suite (replaces the 4 per-model wrappers).
'''

import json
import os
import shlex
import time

import pytest

from cvs.lib import globals
from cvs.lib.inference.utils.inferencing_config_loader import GoodputSlo, validate_sweep_selector
from cvs.lib.utils.verdict import evaluate_all
from cvs.lib.inference.utils.vllm_parsing import CLIENT_METRICS as _METRICS, CLIENT_METRIC_UNITS as _METRIC_UNITS
from cvs.lib.inference.vllm_distributed import VllmDistributedJob

import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location("_vllm_shared", _pl.Path(__file__).parent.parent / "vllm" / "_shared.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
test_print_results_table = _mod.test_print_results_table  # exported as a sibling test  # noqa: F841

log = globals.log

# Fetch-progress poll: du the cache dir until its size stops growing. The model
# download streams in parallel shards, so size climbs then plateaus at the full
# weight set; a stable size across two polls means the fetch settled.
_FETCH_POLL_COUNT = 80
_FETCH_POLL_WAIT_S = 30
_FETCH_PRESENCE_RETRIES = 5


def pytest_generate_tests(metafunc):
    """Parametrize test_vllm_inference from the sweep's named-combo + runs selector.

    Lives in the suite module (not conftest) because it parametrizes fixtures
    only test_vllm_inference consumes -- co-locating the parametrization with
    its sole consumer. It runs at collection time, before fixtures exist, so it
    reads the raw config_file JSON directly (it cannot use the variant_config
    fixture / the typed loader).

    The sweep lists `sequence_combinations` (each with a `name`) once and a
    `runs` array of `{combo, concurrency}` pairs; one case is emitted per run.
    No NxM cartesian -- exactly the cells `runs` enumerates.
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)
    sweep = raw.get("sweep", {})
    combos = sweep.get("sequence_combinations", [])
    runs = sweep.get("runs", [])
    # Validate each raw goodput_slo dict through the same _Forbid model the
    # typed loader uses. pytest_generate_tests bypasses load_variant (it reads
    # raw JSON at collection time), so without this a typo'd SLO key would be
    # silently dropped and a wrong goodput gate would run on hardware.
    for combo in combos:
        if combo.get("goodput_slo") is not None:
            GoodputSlo(**combo["goodput_slo"])
    # Mirror the typed Sweep validator here (this path reads raw JSON before
    # load_variant runs) via the shared rule so the two cannot drift: a
    # duplicate combo name or a run referencing an unknown combo must fail
    # collection, not silently drop.
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

    if not remote:
        # Pre-staged model: nothing to download. Confirm bytes are present,
        # retrying a few times so a cold/slow mount that reads 0 on the first
        # du does not false-fail a model that is actually there.
        final = 0
        for it in range(_FETCH_PRESENCE_RETRIES):
            final = _du_bytes(orch, models_dir)
            log.info("[fetch presence %d] size=%.1fGB", it, final / 1e9)
            if final > 0:
                break
            time.sleep(_FETCH_POLL_WAIT_S)
    else:
        # Kick a background download into the pinned cache, then poll size until
        # it stops growing (two equal readings) or we exhaust the poll budget.
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


def test_vllm_inference(orch, variant_config, hf_token, seq_combo, concurrency, inf_res_dict, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    isl = seq_combo["isl"]
    osl = seq_combo["osl"]
    job = VllmDistributedJob(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        num_prompts=_num_prompts_for(osl, concurrency),
        goodput_slo=seq_combo.get("goodput_slo"),
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
    # Verdict is no longer asserted here: each metric is its own test (test_metric,
    # one HTML row per metric per cell). This test only runs the benchmark and
    # records the cell's results into the module-scoped inf_res_dict.


def test_metric(seq_combo, concurrency, metric, inf_res_dict, variant_config, lifecycle, request):
    """One pytest test (= one HTML row) per perf metric per cell.

    The benchmark already ran once in test_vllm_inference and stashed its results
    in the module-scoped inf_res_dict; this reads a single cached metric and
    surfaces it as its own pass/fail row. The value is rendered inline via the
    Value/Unit table columns (pytest_html_results_table_row in conftest). No GPU
    work. Skips cleanly when the cell's inference failed/skipped so a missing cell
    never reports a false green.

    Verdict: when enforce_thresholds is true AND a spec exists for this cell+metric
    the value is asserted via the shared evaluate_all; otherwise the row is a
    record-only PASS that simply displays the number. evaluate_all is handed the
    full per-cell actuals (not just this one metric) so a min_ratio spec can still
    resolve its reference metric.
    """
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
    evaluate_all(actuals, {full: spec})


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
    if orch.verify_containers_running(name):
        # Leave torn_down False so the orch finalizer retries the teardown.
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
