'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import json
import os
import shlex
import time

import pytest

from cvs.lib import globals
from cvs.lib.inference.utils.inference_suite_lifecycle import (
    sweep_cell_result_key,
    test_accuracy_eval,  # noqa: F401
    test_launch_container,  # noqa: F401
    test_model_fetch,  # noqa: F401
    test_setup_sshd,  # noqa: F401
    test_teardown,  # noqa: F401
)
from cvs.lib.inference.atom.atom_dmesg import verify_dmesg_window
from cvs.lib.inference.atom.atom_orch import AtomJob
from cvs.lib.inference.atom.atom_niah_job import run_niah_cell
from cvs.lib.inference.atom.atom_mtp_quality import (
    chat_template_ok,
    chat_template_sha256,
    degenerate_decode_ratio,
    extract_completion_text,
    parse_mtp_log_metrics,
)
from cvs.lib.inference.atom.atom_config_loader import (
    expand_sweep_parametrize,
    reuse_server_flag,
    server_session_key,
)
from cvs.lib.inference.atom.atom_parsing import (
    CLIENT_METRIC_UNITS as _METRIC_UNITS,
    METRIC_TIERS,
    RECORD_METRICS,
    SCALING_METRIC_UNITS,
    tier_metric_specs,
)
from cvs.lib.utils.verdict import evaluate_all
from cvs.tests.inference.atom._shared import test_print_results_table  # noqa: F401

log = globals.log

_SMOKE_ISL = 128
_SMOKE_OSL = 32
_SMOKE_MAX_MODEL_LEN = 512


def _tier_display_metric(tier):
    if tier == "record":
        return RECORD_METRICS[0] if RECORD_METRICS else "output_throughput"
    names = METRIC_TIERS.get(tier, ())
    return names[0] if names else tier


def test_discover_topology(orch, variant_config, lifecycle, request):
    """Discover IB HCAs and socket netdev on all nodes before the benchmark sweep."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    nn = int(variant_config.params.nnodes)
    if nn == 1:
        lifecycle.ib_hcas = []
        lifecycle.ib_netdev = ""
        return

    from cvs.lib.utils.ib_discovery import resolve_multinode_fabric

    t = time.monotonic()
    master_addr = (variant_config.params.master_addr or "").strip() or orch.hosts[0]
    try:
        resolved_hcas, resolved_netdev = resolve_multinode_fabric(
            orch,
            ib_hca_devices=variant_config.roles.server.ib_hca_devices,
            ib_netdev=variant_config.roles.server.ib_netdev,
            master_addr=master_addr,
        )
    except RuntimeError as e:
        lifecycle.failed = True
        lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - t)
        pytest.fail(str(e))

    lifecycle.ib_hcas = resolved_hcas
    lifecycle.ib_netdev = resolved_netdev
    lifecycle.record(request.node.nodeid, "topology_discovery", time.monotonic() - t)
    log.info(
        "test_discover_topology: resolved netdev=%s HCAs=%s",
        resolved_netdev,
        resolved_hcas,
    )


def _mtp_quality_requested(variant_config) -> bool:
    if variant_config.mtp_quality.enabled:
        return True
    args = variant_config.roles.server.atom_args or []
    joined = " ".join(str(a) for a in args).lower()
    return "--method" in joined and "mtp" in joined


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        raise pytest.UsageError(f"--config_file not found or not specified: {config_file!r}")
    with open(config_file) as fp:
        raw = json.load(fp)
    if "accuracy_task" in metafunc.fixturenames:
        task_ids = [t["id"] for t in raw.get("accuracy", {}).get("tasks", [])]
        metafunc.parametrize("accuracy_task", task_ids, ids=task_ids)
        return
    if "long_context_acc_cell" in metafunc.fixturenames:
        cell_ids = [c["id"] for c in raw.get("long_context_accuracy", {}).get("cells", [])]
        metafunc.parametrize("long_context_acc_cell", cell_ids, ids=cell_ids)
        return
    spec = expand_sweep_parametrize(raw.get("sweep", {}), metafunc.fixturenames)
    if spec:
        argnames, argvalues, ids = spec
        metafunc.parametrize(argnames, argvalues, ids=ids)


def test_openai_compatible_smoke(orch, variant_config, hf_token, lifecycle, request):
    """FUNC-1: OpenAI-compatible HTTP smoke when ``functional.api_smoke`` is enabled."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not variant_config.functional.api_smoke:
        pytest.skip("functional.api_smoke not enabled for this variant")

    job = AtomJob.from_variant(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=_SMOKE_ISL,
        osl=_SMOKE_OSL,
        concurrency=1,
        num_prompts=1,
    )
    if variant_config.params.driver == "atom" and "--max-model-len" not in job.atom_server_args:
        job.atom_server_args = list(job.atom_server_args) + [
            "--max-model-len",
            str(_SMOKE_MAX_MODEL_LEN),
        ]
    t = time.monotonic()
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
    lifecycle.record(request.node.nodeid, "openai_smoke", time.monotonic() - t)
    log.info("OpenAI-compatible smoke results:\n%s", "\n".join(summary))


def test_server_health(orch, variant_config, hf_token, lifecycle, request):
    """FUNC-2: /health, model list, and max_tokens=1 liveness when ``functional.health_check`` is enabled."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not variant_config.functional.health_check:
        pytest.skip("functional.health_check not enabled for this variant")

    job = AtomJob.from_variant(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=_SMOKE_ISL,
        osl=_SMOKE_OSL,
        concurrency=1,
        num_prompts=1,
    )
    if variant_config.params.driver == "atom" and "--max-model-len" not in job.atom_server_args:
        job.atom_server_args = list(job.atom_server_args) + [
            "--max-model-len",
            str(_SMOKE_MAX_MODEL_LEN),
        ]
    t = time.monotonic()
    try:
        job.stop_server()
        job.build_server_cmd()
        job.start_server()
        job.wait_ready()
        summary = job.probe_server_health()
    except Exception:
        lifecycle.failed = True
        job.dump_server_log()
        raise
    finally:
        job.stop_server()
    lifecycle.record(request.node.nodeid, "server_health", time.monotonic() - t)
    log.info("Server health check results:\n%s", "\n".join(summary))


def test_atom_inference(
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
    job = AtomJob.from_variant(
        orch=orch,
        variant=variant_config,
        hf_token=hf_token,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        ib_hcas=getattr(lifecycle, "ib_hcas", []),
        ib_netdev=getattr(lifecycle, "ib_netdev", None),
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
            ready_s = time.monotonic() - t
            lifecycle.record(request.node.nodeid, "server_ready", ready_s)
            lifecycle.record(request.node.nodeid, "server.time_to_ready_s", ready_s, "s")
            if reuse_server_flag(p):
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
    client_s = time.monotonic() - t_client
    lifecycle.record(request.node.nodeid, "client_complete", client_s)
    lifecycle.record(request.node.nodeid, "server.client_wall_s", client_s, "s")


def test_cell_metrics(
    seq_combo,
    concurrency,
    metric_tier,
    inf_res_dict,
    variant_config,
    lifecycle,
    request,
):
    """One pytest row per metric tier per sweep cell (W1 gate batches).

    Fails when ``enforce_thresholds`` is on and either (1) the cell has no
    threshold specs for the requested tier, or (2) specs exist but every gated
    metric for that tier is missing from the benchmark artifact (ATOM may omit
    tail percentiles even when ``metric_percentiles`` requests them).
    """
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
        pytest.fail(f"no threshold specs for tier {metric_tier!r} in cell {cell!r}")
    # ATOM benchmark_serving may omit some tail percentiles even when
    # metric_percentiles requests them; only gate metrics present in actuals.
    specs = {k: v for k, v in specs.items() if k in actuals and actuals[k] is not None}
    if not specs:
        pytest.fail(
            f"no assertable threshold specs for tier {metric_tier!r} in cell {cell!r} "
            f"(metrics missing from benchmark artifact)"
        )
    evaluate_all(actuals, specs)


def test_atom_long_context_accuracy(orch, variant_config, long_context_acc_cell, lifecycle, request):
    """ACC-12: needle-in-a-haystack long-context accuracy (NIAH) per configured cell."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    lca = variant_config.long_context_accuracy
    cells_by_id = {c.id: c for c in (lca.cells if lca else [])}
    cell = cells_by_id.get(long_context_acc_cell)
    if cell is None:
        pytest.skip(f"long_context_accuracy cell {long_context_acc_cell!r} not configured")

    lc_thresholds = (variant_config.thresholds or {}).get("long_context_accuracy", {})
    cell_specs = lc_thresholds.get(long_context_acc_cell, {})
    metric_key = f"accuracy.niah_pass_rate__{cell.id}"
    expected = 1.0
    min_spec = cell_specs.get(metric_key)
    if isinstance(min_spec, dict) and min_spec.get("kind") == "min":
        expected = float(min_spec["value"])

    output_dir = f"{variant_config.paths.log_dir}/long_context_accuracy"
    t = time.monotonic()
    try:
        actuals = run_niah_cell(
            orch=orch,
            variant=variant_config,
            cell=cell,
            expected_pass_rate=expected,
            output_dir=output_dir,
        )
    except RuntimeError as e:
        lifecycle.record(request.node.nodeid, "long_context_accuracy", time.monotonic() - t)
        pytest.fail(str(e))
    lifecycle.record(request.node.nodeid, "long_context_accuracy", time.monotonic() - t)
    for key, value in actuals.items():
        lifecycle.record(request.node.nodeid, key, value, "")

    if not variant_config.enforce_thresholds or not cell_specs:
        return
    specs = {k: v for k, v in cell_specs.items() if k in actuals and actuals[k] is not None}
    if specs:
        evaluate_all(actuals, specs)


def test_atom_mtp_quality(orch, variant_config, lifecycle, request):
    """ACC-4/5/13: MTP acceptance, degenerate decode, and chat-template checks."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not _mtp_quality_requested(variant_config):
        pytest.skip("mtp_quality not enabled for this variant")

    p = variant_config.params
    mq = variant_config.mtp_quality
    combo = variant_config.sweep.sequence_combinations[0]
    run = variant_config.sweep.runs[0]
    job = AtomJob.from_variant(
        orch=orch,
        variant=variant_config,
        hf_token="",
        isl=combo.isl,
        osl=combo.osl,
        concurrency=run.concurrency,
    )
    base = f"{p.base_url}:{p.port_no}".replace("0.0.0.0", "127.0.0.1")
    model_id = variant_config.model.id

    t = time.monotonic()
    log_out = orch.exec_on_head(f"tail -500 {shlex.quote(job.server_log)} 2>/dev/null || true", timeout=30)
    log_text = next(iter(log_out.values()), "") or ""
    client_out = orch.exec_on_head(f"tail -500 {shlex.quote(job.client_log)} 2>/dev/null || true", timeout=30)
    client_text = next(iter(client_out.values()), "") or ""

    actuals = parse_mtp_log_metrics(log_text + "\n" + client_text)
    actuals["mtp.empty_or_repeat_ratio"] = degenerate_decode_ratio(client_text)

    probe_body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": mq.chat_template_prompt}],
            "max_tokens": 64,
        }
    )
    curl_cmd = (
        f"curl -sS -X POST {shlex.quote(base + '/v1/chat/completions')} "
        f"-H 'Content-Type: application/json' -d {shlex.quote(probe_body)}"
    )
    probe_out = orch.exec_on_head(curl_cmd, timeout=120)
    probe_text = next(iter(probe_out.values()), "") or ""
    completion = extract_completion_text(probe_text)
    actuals["accuracy.chat_template_sha256"] = chat_template_sha256(completion)
    ok = chat_template_ok(completion, mq.chat_template_expected_sha256)
    if ok is not None:
        actuals["accuracy.chat_template_ok"] = ok

    lifecycle.record(request.node.nodeid, "mtp_quality", time.monotonic() - t)
    for metric_key, value in actuals.items():
        lifecycle.record(request.node.nodeid, metric_key, value, "")

    if not variant_config.enforce_thresholds:
        return
    mtp_thresholds = (variant_config.thresholds or {}).get("mtp_quality", {})
    if mtp_thresholds:
        specs = {k: v for k, v in mtp_thresholds.items() if k in actuals and actuals[k] is not None}
        if specs:
            evaluate_all(actuals, specs)


def test_verify_dmesg(orch, variant_config, lifecycle, request):
    """INF-6: time-bounded kernel log scan when ``platform.dmesg_scan`` is enabled."""
    if not variant_config.platform.dmesg_scan:
        pytest.skip("platform.dmesg_scan not enabled for this variant")
    start_time = getattr(lifecycle, "dmesg_start", None)
    if not start_time:
        pytest.skip("dmesg start timestamps were not captured")

    t = time.monotonic()
    end_time = orch.exec('date +"%a %b %e %H:%M:%S"') or {}
    verify_dmesg_window(orch, start_time, end_time)
    lifecycle.record(request.node.nodeid, "dmesg_scan", time.monotonic() - t)
