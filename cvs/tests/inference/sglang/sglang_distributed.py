'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Multi-node unified SGLang benchmark: one sharded ``sglang.launch_server`` across all
server nodes (TP/PP + ``nnodes``). No PD disaggregation, no proxy router.

Run:
  pytest cvs/tests/inference/sglang/sglang_distributed.py \\
    --cluster_file <cluster.json> \\
    --config_file <sglang_config.json> \\
    --html=~/cvs_results/sglang_distributed.html

Set ``server_node_list`` (or ``prefill_node_list`` + ``decode_node_list`` whose union
is every server rank) and matching ``nnodes`` in the inference config. All listed
nodes get a container and participate in the unified server. ``benchmark_serv_node``
runs smoke/bench/lm-eval (defaults to rank-0 when omitted).

With ``--html``, session end also writes ``sglang_run_deck.html`` (plus JSON
and interactive viewer) via ``cvs/lib/report/profiles/sglang.json`` (all SGLang stems).
'''

import pytest
import time
from cvs.lib.inference.sglang.sglang_common import cleanup_sglang_log_dir
from cvs.lib import globals
from cvs.lib.verify_lib import verify_dmesg_for_errors

log = globals.log


def test_launch_container(orch, variant_config, lifecycle, request):
    """Stage 1: launch containers and reset log directory on all server nodes."""
    log.info("Testcase launch SGLang container (distributed unified server)")
    globals.error_list = []
    t0 = time.monotonic()

    if not orch.setup_containers():
        lifecycle.failed = True
        lifecycle.complete_stage(request, "container_launch", t0)
        pytest.fail("setup_containers() returned False")

    cleanup_sglang_log_dir(orch, variant_config.paths.log_dir)

    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        lifecycle.complete_stage(request, "container_launch", t0)
        pytest.fail(f"container {name} not running after setup_containers()")

    lifecycle.complete_stage(request, "container_launch", t0)


def test_setup_ibv_devices(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.exec_nic_setup_scripts()
    im_obj.check_ibv_devices()
    lifecycle.complete_stage(request, "ibv_setup", t0)


def test_rms_norm(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.run_test_rmsnorm()
    lifecycle.complete_stage(request, "rms_norm", t0)


def test_launch_server(im_obj, lifecycle, request):
    """Stage: setup env and launch unified multi-node ``sglang.launch_server``."""
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_server_container_env()
    im_obj.launch_server()
    lifecycle.complete_stage(request, "server_launch", t0)


def test_poll_for_server_ready(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.poll_and_check_server_ready()
    lifecycle.complete_stage(request, "server_ready", t0)


def test_openai_compatible_http_endpoints(im_obj, inf_res_dict, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    results = im_obj.verify_openai_compatible_endpoints()
    lifecycle.smoke_results = results
    lifecycle.complete_stage(request, "smoke_endpoints", t0)


def test_run_lm_eval_hellaswag_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_benchmark_serv_container_env()
    h = im_obj.run_lm_eval_hellaswag_benchmark_test()
    lifecycle.phase_labels["accuracy_hellaswag"] = h
    lifecycle.complete_stage(request, "lm_eval_hellaswag", t0)


def test_run_lm_eval_gsm8k_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_benchmark_serv_container_env()
    g = im_obj.run_lm_eval_gsm8k_benchmark_test()
    lifecycle.phase_labels["accuracy_gsm8k"] = g
    lifecycle.complete_stage(request, "lm_eval_gsm8k", t0)


def test_run_performance_benchmark_test(im_obj, inf_res_dict, lifecycle, request, perf_cell, subtests):
    globals.error_list = []
    t0 = time.monotonic()
    bench = im_obj.bp_dict["inference_tests"]["bench_serv_random"]
    bench["input_length"] = perf_cell["isl"]
    bench["output_length"] = perf_cell["osl"]
    bench.setdefault("expected_results", {})["auto"] = dict(perf_cell["specs"])
    im_obj.bp_dict["max_concurrency"] = perf_cell["conc"]
    im_obj.setup_benchmark_serv_container_env()

    im_obj.benchserv_test_random(d_type="auto", verify=False)
    metrics_ok = im_obj.verify_inference_results_subtests(
        subtests,
        "bench_serv",
        bench["expected_results"]["auto"],
        lifecycle=lifecycle,
        report_nodeid=request.node.nodeid,
    )
    from cvs.lib.inference.sglang.sglang_parsing import SGLANG_RESULTS_COLUMNS
    from cvs.lib.report.benchmark_metric_registry import record_benchmark_metric_rows

    record_benchmark_metric_rows(
        request.node,
        lifecycle.perf_metric_rows.get(request.node.nodeid, []),
        columns=SGLANG_RESULTS_COLUMNS,
    )

    key = (
        im_obj.model_name,
        im_obj.gpu_type,
        perf_cell["isl"],
        perf_cell["osl"],
        "bench_serv_random",
        str(perf_cell["conc"]),
    )
    lifecycle.phase_labels.setdefault("performance_by_cell", {})[perf_cell["cell_key"]] = (
        "PASS" if metrics_ok and not globals.error_list else "FAIL"
    )
    inf_res_dict[key] = dict(im_obj.inference_results_dict or {})
    lifecycle.complete_stage(request, f"bench_serv_random[{perf_cell['isl']}/{perf_cell['osl']}]", t0)


def test_verify_dmesg_after_benchmark(im_obj, lifecycle, request):
    globals.error_list = []
    if not im_obj.inference_end_time:
        pytest.skip("benchmark did not complete; no dmesg window")
    t0 = time.monotonic()
    time.sleep(2)
    verify_dmesg_for_errors(im_obj.orch.all, im_obj.inference_start_time, im_obj.inference_end_time)
    lifecycle.complete_stage(request, "verify_dmesg", t0)


def test_distributed_gpu_topology(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.sglang_distributed_gpu_counts()
    lifecycle.complete_stage(request, "gpu_topology", t0)


def test_print_results_table(inf_res_dict, lifecycle, variant_config):
    from cvs.lib.report.registry import bind_session_results
    from cvs.tests.inference.sglang._shared import test_print_results_table as _print

    bind_session_results(
        inf_res_dict=inf_res_dict,
        variant_config=variant_config,
        lifecycle=lifecycle,
    )
    _print(inf_res_dict, lifecycle, variant_config)


def test_teardown(orch, variant_config, lifecycle, request):
    """Final stage: tear down containers and logs. Runs even if a prior stage failed."""
    t0 = time.monotonic()
    orch.teardown_containers()
    cleanup_sglang_log_dir(orch, variant_config.paths.log_dir)
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t0)
    lifecycle.torn_down = True
