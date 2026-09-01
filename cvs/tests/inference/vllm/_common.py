'''Shared vLLM lifecycle tests for the explicit single and distributed suites.'''

import json
import os
import pathlib
import shlex
import time

import pytest

from cvs.lib import globals
from cvs.lib.inference.utils.inference_suite_lifecycle import test_accuracy_eval  # noqa: F401
from cvs.lib.inference.utils.vllm_config_loader import GoodputSlo, validate_sweep_selector
from cvs.lib.inference.utils.vllm_parsing import CLIENT_METRICS, CLIENT_METRIC_UNITS
from cvs.lib.inference.utils.vllm_server_metrics import PROM_METRICS, PROM_METRIC_UNITS, to_prom_metrics
from cvs.lib.inference.vllm_job import VllmJob, scrape_vllm_metrics
from cvs.lib.utils.gpu import (
    GPU_METRICS,
    GPU_METRIC_UNITS,
    agg_readings,
    capture_gpu_metrics,
    start_gpu_poller,
    stop_and_collect_gpu_poller,
)
from cvs.lib.utils.verdict import evaluate_all

from ._shared import test_print_results_table  # noqa: F401

log = globals.log

_FETCH_PRESENCE_RETRIES = 5
_FETCH_POLL_WAIT_S = 30
_SMOKE_ISL = 128
_SMOKE_OSL = 32
_SMOKE_MAX_MODEL_LEN = 512


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file, encoding="utf-8") as fp:
        raw = json.load(fp)
    sweep = raw.get("sweep", {})
    combos = sweep.get("sequence_combinations", [])
    runs = sweep.get("runs", [])
    for combo in combos:
        if combo.get("goodput_slo") is not None:
            GoodputSlo(**combo["goodput_slo"])
    validate_sweep_selector([c["name"] for c in combos], [r["combo"] for r in runs])
    by_name = {c["name"]: c for c in combos}
    cases = [(by_name[run["combo"]], run["concurrency"]) for run in runs]
    ids = [f'{run["combo"]}-conc{run["concurrency"]}' for run in runs]
    if "metric" in metafunc.fixturenames and cases:
        params = [(combo, conc, metric) for (combo, conc) in cases for metric, _ in CLIENT_METRICS]
        metafunc.parametrize(
            "seq_combo,concurrency,metric",
            params,
            ids=[f"{case_id}-{metric}" for case_id in ids for metric, _ in CLIENT_METRICS],
        )
    elif "gpu_metric" in metafunc.fixturenames and cases:
        params = [(combo, conc, metric) for (combo, conc) in cases for metric, _ in GPU_METRICS]
        metafunc.parametrize(
            "seq_combo,concurrency,gpu_metric",
            params,
            ids=[f"{case_id}-{metric}" for case_id in ids for metric, _ in GPU_METRICS],
        )
    elif "prom_metric" in metafunc.fixturenames and cases:
        params = [(combo, conc, metric) for (combo, conc) in cases for metric, _ in PROM_METRICS]
        metafunc.parametrize(
            "seq_combo,concurrency,prom_metric",
            params,
            ids=[f"{case_id}-{metric}" for case_id in ids for metric, _ in PROM_METRICS],
        )
    elif "accuracy_task" in metafunc.fixturenames:
        tasks = [task["id"] for task in raw.get("accuracy", {}).get("tasks", [])]
        metafunc.parametrize("accuracy_task", tasks, ids=tasks)
    elif "seq_combo" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames and cases:
        metafunc.parametrize("seq_combo,concurrency", cases, ids=ids)


def _cell_result_key(variant, combo, concurrency):
    return (variant.model.id, variant.gpu_arch, combo["isl"], combo["osl"], combo.get("name", "default"), concurrency)


def test_launch_container(orch, vllm_targets, lifecycle, request):
    started = time.monotonic()
    launched = orch.setup_containers()
    lifecycle.record(request.node.nodeid, "container_launch", time.monotonic() - started)
    if not launched:
        lifecycle.failed = True
        pytest.fail("setup_containers() returned False")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


def test_setup_sshd():
    pytest.skip("vLLM uses host-network NCCL/gloo rather than in-container sshd")


def test_discover_topology(orch, variant_config, vllm_targets, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if len(vllm_targets[0]) == 1:
        lifecycle.ib_hcas = []
        return
    from cvs.lib.utils.ib_discovery import discover_ib_hca_names, validate_ib_hca_preflight

    started = time.monotonic()
    try:
        discovered = discover_ib_hca_names(orch)
    except RuntimeError as exc:
        lifecycle.failed = True
        lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - started)
        pytest.fail(str(exc))
    requested = variant_config.roles.server.ib_hca_devices
    if requested and requested != "auto":
        try:
            validate_ib_hca_preflight(discovered, requested)
        except RuntimeError as exc:
            lifecycle.failed = True
            lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - started)
            pytest.fail(str(exc))
        lifecycle.ib_hcas = requested
    else:
        lifecycle.ib_hcas = next(iter(discovered.values()))
    lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - started)


def test_model_fetch(orch, variant_config, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    path = variant_config.paths.models_dir
    if not path:
        pytest.skip("paths.models_dir unset; cannot verify model cache")
    cmd = f"du -sb {shlex.quote(path)} 2>/dev/null | cut -f1"
    started = time.monotonic()
    final = {}
    for _ in range(_FETCH_PRESENCE_RETRIES):
        out = orch.exec(cmd)
        final = {host: int(text.strip()) if (text or "").strip().isdigit() else 0 for host, text in out.items()}
        if final and all(final.values()):
            break
        time.sleep(_FETCH_POLL_WAIT_S)
    if not final or not all(final.values()):
        lifecycle.failed = True
        missing = [host for host, size in final.items() if not size]
        pytest.fail(f"no model bytes under {path} on {missing or 'any host'}")
    lifecycle.record(request.node.nodeid, "model_fetch", time.monotonic() - started)
    lifecycle.record(request.node.nodeid, "model_size", max(final.values()) / 1e9, "GB")


def _gpu_snap(orch):
    try:
        return capture_gpu_metrics(orch)
    except Exception:
        return {}


def test_openai_compatible_smoke(orch, variant_config, hf_token, vllm_targets, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    job = VllmJob(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=_SMOKE_ISL,
        osl=_SMOKE_OSL,
        concurrency=1,
        num_prompts=1,
        ib_hcas=getattr(lifecycle, "ib_hcas", []),
        client_poll_count=int(variant_config.params.client_poll_count),
    )
    job.serve_args.setdefault("max-model-len", str(_SMOKE_MAX_MODEL_LEN))
    started = time.monotonic()
    try:
        job.stop_server()
        job.build_server_cmd()
        job.start_server()
        job.wait_ready()
        summary = job.probe_openai_endpoints()
    except Exception:
        lifecycle.failed = True
        job.dump_server_log()
        raise
    finally:
        job.stop_server()
    lifecycle.record(request.node.nodeid, "openai_smoke", time.monotonic() - started)
    log.info("OpenAI-compatible smoke results:\n%s", "\n".join(summary))


def test_vllm_inference(
    orch, variant_config, hf_token, vllm_targets, seq_combo, concurrency, inf_res_dict, lifecycle, request
):
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
    load_s = None
    load_mb = None
    poll_readings = []
    try:
        signature = job.server_signature()
        if getattr(lifecycle, "live_server_sig", None) == signature:
            lifecycle.record(request.node.nodeid, "server_ready", 0.0)
        else:
            job.stop_server()
            job.build_server_cmd()
            lifecycle.live_server_job = job
            before = _gpu_snap(orch)
            started = time.monotonic()
            job.start_server()
            job.wait_ready()
            load_s = time.monotonic() - started
            lifecycle.record(request.node.nodeid, "server_ready", load_s)
            lifecycle.live_server_sig = signature
            after = _gpu_snap(orch)
            load_mb = ((after.get("gpu.used_vram") or 0) - (before.get("gpu.used_vram") or 0)) or None

        html_path = getattr(request.config.option, "htmlpath", None)
        html_dir = getattr(request.config, "_test_html_dir", "test_html")
        gpu_log = (
            pathlib.Path(html_path).parent / html_dir / f"gpu_poll_isl{isl}_osl{osl}_conc{concurrency}.log"
            if html_path
            else None
        )
        handle = start_gpu_poller(
            orch,
            run_id=f"{request.node.nodeid}_{isl}_{osl}_{concurrency}",
            nodes=None if int(job.nnodes) == 1 else list(job.hosts),
        )
        before_prom = scrape_vllm_metrics(orch, job.base_url, job.port_no)
        try:
            job.run_client()
            job.wait_client_complete()
        finally:
            poll_readings = stop_and_collect_gpu_poller(
                orch,
                handle,
                log_path=str(gpu_log) if gpu_log else None,
                model_load_s=load_s,
                model_load_memory_mb=load_mb,
            )
        after_prom = scrape_vllm_metrics(orch, job.base_url, job.port_no)
        results = job.parse_results()
    except Exception:
        lifecycle.failed = True
        lifecycle.live_server_sig = None
        getattr(lifecycle, "live_server_job", job).dump_server_log()
        raise

    aggregate = agg_readings(poll_readings)
    gpu_results = {
        "gpu.peak_gpu_memory_mb": aggregate.get("peak_gpu_memory_mb"),
        "gpu.model_load_memory_mb": load_mb,
        "gpu.model_load_s": load_s,
        "gpu.gpu_bandwidth_util_pct": aggregate.get("gpu_bandwidth_util_pct"),
        "gpu.gpu_compute_util_pct": aggregate.get("gpu_compute_util_pct"),
    }
    prom_results = to_prom_metrics(before_prom, after_prom)
    for actuals in results.values():
        actuals.update(gpu_results)
        actuals.update(prom_results)
    inf_res_dict[_cell_result_key(variant_config, seq_combo, concurrency)] = results


def _test_metric(
    seq_combo, concurrency, metric, prefix, units, inf_res_dict, variant_config, vllm_targets, lifecycle, request
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    key = _cell_result_key(variant_config, seq_combo, concurrency)
    host_dict = inf_res_dict.get(key)
    if not host_dict:
        pytest.skip(f"no recorded results for {key!r}")
    full = prefix + metric
    _host, actuals = next(iter(host_dict.items()))
    value = actuals.get(full)
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", units.get(metric, "-")))
    if prefix != "client." and value is None:
        pytest.skip(f"{full}: no value recorded")
    cell = variant_config.cell_key(seq_combo["isl"], seq_combo["osl"], concurrency)
    spec = (variant_config.thresholds.get(cell) or {}).get(full)
    if not variant_config.enforce_thresholds or spec is None:
        return
    evaluate_all(actuals, {full: spec})


def test_metric(seq_combo, concurrency, metric, inf_res_dict, variant_config, vllm_targets, lifecycle, request):
    _test_metric(
        seq_combo,
        concurrency,
        metric,
        "client.",
        CLIENT_METRIC_UNITS,
        inf_res_dict,
        variant_config,
        vllm_targets,
        lifecycle,
        request,
    )


def test_gpu_metric(seq_combo, concurrency, gpu_metric, inf_res_dict, variant_config, vllm_targets, lifecycle, request):
    _test_metric(
        seq_combo,
        concurrency,
        gpu_metric,
        "gpu.",
        GPU_METRIC_UNITS,
        inf_res_dict,
        variant_config,
        vllm_targets,
        lifecycle,
        request,
    )


def test_prom_metric(
    seq_combo, concurrency, prom_metric, inf_res_dict, variant_config, vllm_targets, lifecycle, request
):
    _test_metric(
        seq_combo,
        concurrency,
        prom_metric,
        "prom.",
        PROM_METRIC_UNITS,
        inf_res_dict,
        variant_config,
        vllm_targets,
        lifecycle,
        request,
    )


def test_teardown(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    started = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - started)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
