'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Single SGLang disaggregated (PD) benchmark module: model is selected from
``benchmark_params`` via ``active_benchmark`` / env / single-key auto (see ``_shared``).
'''

import time

from cvs.lib import globals
from cvs.tests.inference.sglang.conftest import flat_expected_from_specs

log = globals.log

def test_cleanup_stale_containers(orch, lifecycle, request):
    """Stage 0: remove stale containers and logs from a prior run."""
    globals.error_list = []
    t0 = time.monotonic()
    orch.teardown_containers()
    orch.cleanup_log_dir()
    time.sleep(5)
    lifecycle.complete_stage(request, "stale_cleanup", t0)


def test_launch_inference_containers(orch, lifecycle, request):
    """Stage 1: launch SGLang containers on prefill/decode/router/bench nodes."""
    log.info("Testcase launch SGLang containers")
    globals.error_list = []
    t0 = time.monotonic()
    if not orch.setup_containers():
        lifecycle.failed = True
    lifecycle.complete_stage(request, "container_launch", t0)


def test_setup_ibv_devices(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.check_ibv_devices()
    im_obj.exec_nic_setup_scripts()
    lifecycle.complete_stage(request, "ibv_setup", t0)


def test_rms_norm(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.run_test_rmsnorm()
    lifecycle.complete_stage(request, "rms_norm", t0)


def test_launch_prefill_servers(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_prefill_container_env()
    im_obj.launch_prefill_servers()
    lifecycle.complete_stage(request, "prefill_launch", t0)


def test_launch_decode_servers(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_decode_container_env()
    im_obj.launch_decode_servers()
    lifecycle.complete_stage(request, "decode_launch", t0)


def test_poll_for_server_ready(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.poll_and_check_server_ready()
    lifecycle.complete_stage(request, "server_ready", t0)


def test_launch_proxy_router(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_proxy_router_container_env()
    im_obj.launch_proxy_router()
    lifecycle.complete_stage(request, "proxy_router_launch", t0)


def test_openai_compatible_http_endpoints(im_obj, inf_res_dict, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    results = im_obj.verify_openai_compatible_endpoints()
    lifecycle.smoke_results = results
    lifecycle.complete_stage(request, "smoke_endpoints", t0)
    

# def test_run_long_context_accuracy(im_obj, lifecycle, request, acc_cell):
#     globals.error_list = []
#     t0 = time.monotonic()
#     bench = im_obj.bp_dict["inference_tests"]["long_ctx_niah"]
#     bench["input_length"] = acc_cell["isl"]
#     bench["output_length"] = acc_cell["osl"]
#     bench.setdefault("expected_results", {})["auto"] = flat_expected_from_specs(acc_cell["specs"])
#     im_obj.bp_dict["max_concurrency"] = "1"
#     im_obj.setup_benchmark_serv_container_env()
#     summary = im_obj.run_long_context_niah_accuracy(
#         isl=int(acc_cell["isl"]),
#         osl=int(acc_cell["osl"]),
#         d_type="auto",
#     )
#     lifecycle.phase_labels[f"accuracy_long_ctx_{acc_cell['isl']}"] = summary
#     lifecycle.phase_labels.setdefault("accuracy_by_cell", {})[acc_cell["cell_key"]] = (
#         "PASS" if summary.get("passed") else "FAIL"
#     )
#     lifecycle.complete_stage(
#         request,
#         f"long_ctx_niah[{acc_cell['isl']}/{acc_cell['osl']}]",
#         t0,
#     )


# def test_run_lm_eval_hellaswag_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
#     globals.error_list = []
#     t0 = time.monotonic()
#     im_obj.setup_benchmark_serv_container_env()
#     h = im_obj.run_lm_eval_hellaswag_benchmark_test()
#     lifecycle.phase_labels["accuracy_hellaswag"] = h
#     lifecycle.complete_stage(request, "lm_eval_hellaswag", t0)


def test_run_lm_eval_gsm8k_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_benchmark_serv_container_env()
    g = im_obj.run_lm_eval_gsm8k_benchmark_test()
    lifecycle.phase_labels["accuracy_gsm8k"] = g
    lifecycle.complete_stage(request, "lm_eval_gsm8k", t0)


# def test_run_lm_eval_mmlu_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
#     globals.error_list = []
#     t0 = time.monotonic()
#     im_obj.setup_benchmark_serv_container_env()
#     m = im_obj.run_lm_eval_mmlu_benchmark_test()
#     lifecycle.phase_labels["accuracy_mmlu"] = m
#     lifecycle.complete_stage(request, "lm_eval_mmlu", t0)


def test_run_performance_benchmark_test(im_obj, inf_res_dict, lifecycle, request, perf_cell):
    globals.error_list = []
    t0 = time.monotonic()
    bench = im_obj.bp_dict["inference_tests"]["bench_serv_random"]
    bench["input_length"] = perf_cell["isl"]
    bench["output_length"] = perf_cell["osl"]
    bench.setdefault("expected_results", {})["auto"] = flat_expected_from_specs(perf_cell["specs"])
    im_obj.bp_dict["max_concurrency"] = perf_cell["conc"]
    im_obj.setup_benchmark_serv_container_env()
    im_obj.benchserv_test_random(d_type="auto")
    key = (
        im_obj.model_name,
        im_obj.gpu_type,
        perf_cell["isl"],
        perf_cell["osl"],
        "bench_serv_random",
        str(perf_cell["conc"]),
    )
    lifecycle.phase_labels.setdefault("performance_by_cell", {})[perf_cell["cell_key"]] = (
        "PASS" if not globals.error_list else "FAIL"
    )
    inf_res_dict[key] = dict(im_obj.inference_results_dict or {})
    lifecycle.complete_stage(request, f"bench_serv_random[{perf_cell['isl']}/{perf_cell['osl']}]", t0)


def test_disagg_gpu_topology(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.sglang_disagg_gpu_counts()
    lifecycle.complete_stage(request, "gpu_topology", t0)

def test_print_results_table(inf_res_dict, lifecycle, variant_config=None):
    from cvs.lib.report.registry import bind_session_results
    from cvs.tests.inference.sglang._shared import test_print_results_table as _print

    bind_session_results(
        inf_res_dict=inf_res_dict,
        variant_config=variant_config,
        lifecycle=lifecycle,
    )
    _print(inf_res_dict, lifecycle)


def test_teardown(orch, lifecycle, request):
    """Final stage: tear down containers and logs. Runs even if a prior stage failed."""
    t0 = time.monotonic()
    orch.teardown_containers()
    orch.cleanup_log_dir()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t0)
    lifecycle.torn_down = True