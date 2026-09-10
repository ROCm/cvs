'''Shared vLLM lifecycle tests for the explicit single and distributed suites.'''

import json
import pathlib
import shlex
import time

import pytest

from cvs.lib import globals
from cvs.lib.inference.utils.inference_suite_lifecycle import test_accuracy_eval  # noqa: F401
from cvs.lib.inference.utils.vllm_config_loader import load_variant
from cvs.lib.inference.utils.vllm_metrics import (
    METRIC_REGISTRY,
    VLLM_RESULTS_COLUMNS,
    is_finite_number,
    merge_metric_sources,
    metric_contract,
)
from cvs.lib.inference.utils.vllm_server_metrics import to_prom_metrics
from cvs.lib.inference.utils.vllm_verification import evaluate_metric_verdicts
from cvs.lib.inference.vllm_job import VllmJob, scrape_vllm_metrics
from cvs.lib.utils.gpu import (
    agg_readings,
    capture_gpu_metrics,
    start_gpu_poller,
    stop_and_collect_gpu_poller,
)
from cvs.lib.report.benchmark_metric_registry import record_benchmark_metric_rows

from ._shared import test_print_results_table  # noqa: F401

log = globals.log

_FETCH_PRESENCE_RETRIES = 5
_FETCH_POLL_WAIT_S = 30
_SMOKE_ISL = 128
_SMOKE_OSL = 32
_SMOKE_MAX_MODEL_LEN = 512
VLLM_JUNIT_PROPERTY = "cvs_vllm_metrics_v1"


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file:
        return
    variant = load_variant(config_file, {})
    runs = variant.resolved_runs()
    ids = [run.cell.key for run in runs]
    if "accuracy_task" in metafunc.fixturenames:
        tasks = [task.id for task in variant.accuracy.tasks]
        metafunc.parametrize("accuracy_task", tasks, ids=tasks)
    elif "run" in metafunc.fixturenames and runs:
        metafunc.parametrize("run", runs, ids=ids)


def _cell_result_key(variant, run):
    return (
        variant.model_id,
        "",
        str(run.cell.isl),
        str(run.cell.osl),
        run.cell.key,
        run.cell.concurrency,
    )


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
    container_hcas = variant_config.container.nccl_ib_hcas
    requested = container_hcas if container_hcas is not None else variant_config.ib_hca_devices
    if requested and requested != "auto":
        try:
            validate_ib_hca_preflight(discovered, requested)
        except RuntimeError as exc:
            lifecycle.failed = True
            lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - started)
            pytest.fail(str(exc))
        lifecycle.ib_hcas = [] if container_hcas is not None else requested
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


def _clear_live_server(lifecycle):
    lifecycle.live_server_state = None
    lifecycle.live_server_job = None


def _live_server_measurements(lifecycle, signature):
    state = getattr(lifecycle, "live_server_state", None)
    if not isinstance(state, tuple) or len(state) != 3 or state[0] != signature:
        return None
    return state[1], state[2]


def _load_measurements(before, after, elapsed):
    before_vram = before.get("gpu.used_vram")
    after_vram = after.get("gpu.used_vram")
    if not all(is_finite_number(value) for value in (before_vram, after_vram, elapsed)):
        return None
    return elapsed, after_vram - before_vram


def _record_junit_metrics(node, actuals_by_host):
    actuals = {}
    for host in sorted(actuals_by_host, key=str):
        host_actuals = actuals_by_host[host]
        actuals[str(host)] = {
            definition.name: host_actuals[definition.name]
            for definition in METRIC_REGISTRY
            if is_finite_number(host_actuals.get(definition.name))
        }
    payload = json.dumps(
        {
            "actuals_by_host": actuals,
            "metric_contract": metric_contract(),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    properties = [
        (key, value)
        for key, value in getattr(node, "user_properties", [])
        if key != VLLM_JUNIT_PROPERTY
    ]
    properties.append((VLLM_JUNIT_PROPERTY, payload))
    node.user_properties = properties


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
    )
    job.serve_args.setdefault("max_model_len", str(_SMOKE_MAX_MODEL_LEN))
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


def test_vllm_inference(orch, variant_config, hf_token, vllm_targets, run, inf_res_dict, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    isl = run.cell.isl
    osl = run.cell.osl
    job = VllmJob(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=isl,
        osl=osl,
        concurrency=run.cell.concurrency,
        num_prompts=run.benchmark_params["num_prompts"],
        benchmark_params=run.benchmark_params,
        ib_hcas=getattr(lifecycle, "ib_hcas", []),
    )
    load_s = None
    load_mb = None
    poll_readings = []
    try:
        signature = job.server_signature()
        measurements = _live_server_measurements(lifecycle, signature)
        if measurements is not None:
            load_s, load_mb = measurements
            lifecycle.record(request.node.nodeid, "server_ready", 0.0)
        else:
            _clear_live_server(lifecycle)
            job.stop_server()
            job.build_server_cmd()
            lifecycle.live_server_job = job
            before = _gpu_snap(orch)
            started = time.monotonic()
            job.start_server()
            job.wait_ready()
            elapsed = time.monotonic() - started
            lifecycle.record(request.node.nodeid, "server_ready", elapsed)
            after = _gpu_snap(orch)
            measurements = _load_measurements(before, after, elapsed)
            if measurements is not None:
                load_s, load_mb = measurements
                lifecycle.live_server_state = (signature, load_s, load_mb)

        html_path = getattr(request.config.option, "htmlpath", None)
        html_dir = getattr(request.config, "_test_html_dir", "test_html")
        gpu_log = (
            pathlib.Path(html_path).parent / html_dir / f"gpu_poll_isl{isl}_osl{osl}_conc{run.cell.concurrency}.log"
            if html_path
            else None
        )
        handle = start_gpu_poller(
            orch,
            run_id=f"{request.node.nodeid}_{isl}_{osl}_{run.cell.concurrency}",
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
        dump_job = getattr(lifecycle, "live_server_job", None) or job
        _clear_live_server(lifecycle)
        dump_job.dump_server_log()
        raise

    aggregate = agg_readings(poll_readings)
    gpu_results = {
        "peak_gpu_memory_mb": aggregate.get("peak_gpu_memory_mb"),
        "model_load_memory_mb": load_mb,
        "model_load_s": load_s,
        "gpu_bandwidth_util_pct": aggregate.get("gpu_bandwidth_util_pct"),
        "gpu_compute_util_pct": aggregate.get("gpu_compute_util_pct"),
    }
    prom_results = to_prom_metrics(before_prom, after_prom)
    for host, actuals in results.items():
        results[host] = merge_metric_sources(actuals, gpu_results, prom_results)
    inf_res_dict[_cell_result_key(variant_config, run)] = results


def test_verify_cell_metrics(run, inf_res_dict, variant_config, lifecycle, request, subtests):
    """Report configured metrics and verify active gates as pytest subtests."""
    key = _cell_result_key(variant_config, run)
    host_dict = inf_res_dict.get(key)
    if not host_dict:
        pytest.skip(f"no recorded inference result for {key!r}")

    verdicts = evaluate_metric_verdicts(
        host_dict,
        variant_config.thresholds.get(run.cell.key) or {},
        enforce_thresholds=variant_config.enforce_thresholds,
    )
    if not verdicts:
        pytest.skip(f"no configured metric specs for {run.cell.key}")

    record_benchmark_metric_rows(request.node, verdicts, columns=VLLM_RESULTS_COLUMNS)
    _record_junit_metrics(request.node, host_dict)
    asserted_verdicts = [verdict for verdict in verdicts if verdict["enforced"]]
    started = time.monotonic()
    for verdict in asserted_verdicts:
        with subtests.test(node=verdict["node"], metric=verdict["metric"]):
            assert verdict["status"] == "pass", verdict["reason"]
    lifecycle.record(request.node.nodeid, "metric_verification", time.monotonic() - started)
    if not asserted_verdicts:
        pytest.skip(f"metrics recorded without active threshold gates for {run.cell.key}")


def test_teardown(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    started = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - started)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
