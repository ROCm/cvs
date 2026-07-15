'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unified vLLM benchmark suite for single-node and multinode distributed runs.

Replaces vllm_single.py (single-node) and tests/inference/vllm_distributed/
vllm_distributed.py (distributed) with one parametrized suite.

The topology is determined entirely by the config file:
  nnodes=1 (default) -> single-node, no distributed flags
  nnodes=2 + pipeline_parallel_size=2 -> 2-node PP distributed

IB device discovery (test_discover_topology) runs once per lifecycle for
distributed runs, before the benchmark sweep. Results are stored in the
lifecycle object and passed into VllmJob per cell.
'''

import json
import os
import shlex
import time

import pytest

from cvs.lib import globals
from cvs.lib.inference.utils.vllm_config_loader import GoodputSlo, validate_sweep_selector
from cvs.lib.utils.verdict import evaluate_all
from cvs.lib.inference.utils.vllm_parsing import CLIENT_METRICS as _METRICS, CLIENT_METRIC_UNITS as _METRIC_UNITS
from cvs.lib.inference.vllm_job import VllmJob

import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location("_vllm_shared", _pl.Path(__file__).with_name("_shared.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
test_print_results_table = _mod.test_print_results_table  # exported as a sibling test  # noqa: F841

log = globals.log

_FETCH_POLL_COUNT = 80
_FETCH_POLL_WAIT_S = 30
_FETCH_PRESENCE_RETRIES = 5


def pytest_generate_tests(metafunc):
    """Parametrize test_vllm_inference from the sweep's named-combo + runs selector.

    Runs at collection time (before fixtures exist), so it reads the raw
    config_file JSON directly. Validates GoodputSlo and sweep selector against
    the same rules the typed loader uses so collection-time and load-time checks
    cannot drift.
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)
    sweep = raw.get("sweep", {})
    combos = sweep.get("sequence_combinations", [])
    runs = sweep.get("runs", [])
    for combo in combos:
        if combo.get("goodput_slo") is not None:
            GoodputSlo(**combo["goodput_slo"])
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
    """Stage 2: no-op for vllm — distributed execution uses NCCL/gloo over host network, not MPI/sshd."""
    pytest.skip("vllm uses --distributed-executor-backend mp + NCCL; no inter-container sshd needed")


def test_discover_topology(orch, variant_config, lifecycle, request):
    """Stage 3: discover IB HCA devices on all nodes.

    Skipped for single-node runs (nnodes=1) since IB HCA selection is not
    needed for NCCL_IB_HCA on single-node.

    For distributed runs:
    - Runs ibv_devinfo -l on all nodes
    - If ib_hca_devices in config is an explicit list, validates it against
      the discovered devices (fails loudly if a named device is absent)
    - If ib_hca_devices is absent or "auto", uses the full discovered list
    - Stores the resolved HCA list in lifecycle.ib_hcas for use per cell
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    nn = int(variant_config.params.nnodes)
    if nn == 1:
        lifecycle.ib_hcas = []
        return

    from cvs.lib.utils.ib_discovery import discover_ib_hca_names, validate_ib_hca_preflight

    t = time.monotonic()
    try:
        discovered = discover_ib_hca_names(orch)
    except RuntimeError as e:
        lifecycle.failed = True
        lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - t)
        pytest.fail(str(e))

    requested = variant_config.roles.server.ib_hca_devices
    if requested and requested != "auto":
        try:
            validate_ib_hca_preflight(discovered, requested)
        except RuntimeError as e:
            lifecycle.failed = True
            lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - t)
            pytest.fail(str(e))
        resolved = requested
    else:
        # "auto" or absent: use whatever the first host reported (symmetry verified above).
        resolved = next(iter(discovered.values()))

    lifecycle.ib_hcas = resolved
    lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - t)
    log.info("test_discover_topology: resolved HCAs=%s", resolved)


def test_model_fetch(orch, variant_config, lifecycle, request):
    """Stage 4: ensure the model is present in the HF cache (mounted models dir)."""
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
            final = _du_bytes(orch, models_dir)
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
    job = VllmJob(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        num_prompts=variant_config.params.num_prompts,
        ib_hcas=getattr(lifecycle, "ib_hcas", []),
        goodput_slo=seq_combo.get("goodput_slo"),
        client_poll_count=int(variant_config.params.client_poll_count),
    )

    try:
        # Reuse the already-running server when this cell needs an identical one
        # (cells that differ only in concurrency share a server signature, since
        # concurrency is a client-only knob). This skips a full stop + weight
        # reload + warmup between such cells. The server keeps serving on the
        # same port; only the client args change.
        sig = job.server_signature()
        if getattr(lifecycle, "live_server_sig", None) == sig:
            log.info("reusing running vllm server (same server args); skipping restart")
            lifecycle.record(request.node.nodeid, "server_ready", 0.0)
        else:
            job.stop_server()
            job.build_server_cmd()
            t = time.monotonic()
            job.start_server()
            job.wait_ready()
            lifecycle.record(request.node.nodeid, "server_ready", time.monotonic() - t)
            lifecycle.live_server_sig = sig
        job.run_client()
        job.wait_client_complete()
        results = job.parse_results()
    except Exception:
        lifecycle.failed = True
        # A failed cell may have left the server in a bad state; force the next
        # cell to do a clean bringup rather than reuse a possibly-dead server.
        lifecycle.live_server_sig = None
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


def test_metric(seq_combo, concurrency, metric, inf_res_dict, variant_config, lifecycle, request):
    """One pytest test (= one HTML row) per perf metric per cell."""
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
    """Final stage: explicit container teardown, timed, asserting it is gone."""
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
