'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Single SGLang disaggregated (PD) benchmark module: model is selected from
``benchmark_params`` via ``active_benchmark`` / env / single-key auto (see ``_shared``).
'''

import re
import time

import pytest

from cvs.lib import docker_lib, globals
from cvs.lib.utils_lib import fail_test, update_test_result
from cvs.tests.inference.sglang._shared import test_print_results_table

log = globals.log


def _skip_if_prior_failure(lifecycle):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")


def _complete_stage(lifecycle, request, label, t0):
    lifecycle.record(request.node.nodeid, label, time.monotonic() - t0)
    if globals.error_list:
        lifecycle.failed = True
    update_test_result()


def _container_launch_handles(inference_dict):
    """Return the PSSH handle list needed to launch inference containers."""
    hdl_list = ["p_phdl", "d_phdl"]
    proxy = inference_dict["proxy_router_node"]
    bench = inference_dict["benchmark_serv_node"]
    prefill = inference_dict["prefill_node_list"]
    decode = inference_dict["decode_node_list"]

    if proxy == bench:
        if proxy not in prefill and proxy not in decode:
            hdl_list.append("r_phdl")
    else:
        if proxy not in prefill and proxy not in decode:
            hdl_list.append("r_phdl")
        if bench not in prefill and bench not in decode:
            hdl_list.append("b_phdl")
    return hdl_list


def test_cleanup_stale_containers(p_phdl, d_phdl, r_phdl, b_phdl, inference_dict, lifecycle, request):
    """Stage 0: remove stale containers and logs from a prior run."""
    globals.error_list = []
    t0 = time.monotonic()
    container_name = inference_dict["container_name"]
    for a_phdl in (p_phdl, d_phdl, r_phdl, b_phdl):
        docker_lib.kill_docker_container(a_phdl, container_name)
        docker_lib.delete_all_containers_and_volumes(a_phdl)
    log.info("Cleaning up log directory")
    r_phdl.exec(f"sudo rm -rf {inference_dict['log_dir']}")
    time.sleep(5)
    _complete_stage(lifecycle, request, "stale_cleanup", t0)


def test_launch_inference_containers(p_phdl, d_phdl, r_phdl, b_phdl, inference_dict, lifecycle, request):
    """Stage 1: launch SGLang containers on prefill/decode/router/bench nodes."""
    _skip_if_prior_failure(lifecycle)
    log.info("Testcase launch SGLang containers")
    globals.error_list = []
    t0 = time.monotonic()
    container_name = inference_dict["container_name"]
    handles = {
        "p_phdl": p_phdl,
        "d_phdl": d_phdl,
        "r_phdl": r_phdl,
        "b_phdl": b_phdl,
    }
    for key in _container_launch_handles(inference_dict):
        a_phdl = handles[key]
        docker_lib.launch_docker_container(
            a_phdl,
            container_name,
            inference_dict["container_image"],
            inference_dict["container_config"]["device_list"],
            inference_dict["container_config"]["volume_dict"],
            inference_dict["container_config"]["env_dict"],
            shm_size="48G",
            timeout=60 * 20,
        )
    time.sleep(30)
    log.info("Verify if the containers have been launched properly")
    for a_phdl in (p_phdl, d_phdl, r_phdl, b_phdl):
        out_dict = a_phdl.exec("docker ps")
        for node, out in out_dict.items():
            if not re.search(re.escape(container_name), out or "", re.I):
                fail_test(f"Failed to launch container on node {node}")
    _complete_stage(lifecycle, request, "container_launch", t0)


def test_setup_ibv_devices(im_obj, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.check_ibv_devices()
    im_obj.exec_nic_setup_scripts()
    _complete_stage(lifecycle, request, "ibv_setup", t0)


def test_rms_norm(im_obj, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.run_test_rmsnorm()
    _complete_stage(lifecycle, request, "rms_norm", t0)


def test_launch_prefill_servers(im_obj, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_prefill_container_env()
    im_obj.launch_prefill_servers()
    _complete_stage(lifecycle, request, "prefill_launch", t0)


def test_launch_decode_servers(im_obj, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_decode_container_env()
    im_obj.launch_decode_servers()
    _complete_stage(lifecycle, request, "decode_launch", t0)


def test_poll_for_server_ready(im_obj, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.poll_and_check_server_ready()
    _complete_stage(lifecycle, request, "server_ready", t0)


def test_launch_proxy_router(im_obj, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_proxy_router_container_env()
    im_obj.launch_proxy_router()
    _complete_stage(lifecycle, request, "proxy_router_launch", t0)


def test_openai_compatible_http_endpoints(im_obj, inf_res_dict, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    results = im_obj.verify_openai_compatible_endpoints()
    inf_res_dict["__smoke_probe_results__"] = results
    _complete_stage(lifecycle, request, "smoke_endpoints", t0)


def test_run_lm_eval_hellaswag_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_benchmark_serv_container_env()
    h = im_obj.run_lm_eval_hellaswag_benchmark_test()
    inf_res_dict.setdefault("__phase_labels__", {})["accuracy_hellaswag"] = h
    _complete_stage(lifecycle, request, "lm_eval_hellaswag", t0)


def test_run_lm_eval_gsm8k_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_benchmark_serv_container_env()
    g = im_obj.run_lm_eval_gsm8k_benchmark_test()
    inf_res_dict.setdefault("__phase_labels__", {})["accuracy_gsm8k"] = g
    _complete_stage(lifecycle, request, "lm_eval_gsm8k", t0)


def test_run_lm_eval_mmlu_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_benchmark_serv_container_env()
    m = im_obj.run_lm_eval_mmlu_benchmark_test()
    inf_res_dict.setdefault("__phase_labels__", {})["accuracy_mmlu"] = m
    _complete_stage(lifecycle, request, "lm_eval_mmlu", t0)


def test_run_performance_benchmark_test(im_obj, inf_res_dict, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_benchmark_serv_container_env()
    im_obj.benchserv_test_random(d_type="auto")

    bench = (im_obj.bp_dict.get("inference_tests") or {}).get("bench_serv_random") or {}
    expected = (bench.get("expected_results") or {}).get("auto") or {}

    key = (
        im_obj.model_name,
        im_obj.gpu_type,
        str(bench.get("input_length", "-")),
        str(bench.get("output_length", "-")),
        "bench_serv_random",
        str(im_obj.bp_dict.get("max_concurrency", "-")),
    )
    labels = inf_res_dict.setdefault("__phase_labels__", {})
    labels["performance_expected"] = expected
    labels["performance_test"] = "PASS" if not globals.error_list else "FAIL"

    inf_res_dict[key] = dict(im_obj.inference_results_dict or {})
    _complete_stage(lifecycle, request, "bench_serv_random", t0)


def test_disagg_gpu_topology(im_obj, lifecycle, request):
    _skip_if_prior_failure(lifecycle)
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.sglang_disagg_gpu_counts()
    _complete_stage(lifecycle, request, "gpu_topology", t0)


def test_teardown(p_phdl, d_phdl, r_phdl, b_phdl, inference_dict, lifecycle, request):
    """Final stage: tear down containers and logs. Runs even if a prior stage failed."""
    container_name = inference_dict["container_name"]
    t0 = time.monotonic()
    for a_phdl in (p_phdl, d_phdl, r_phdl, b_phdl):
        docker_lib.kill_docker_container(a_phdl, container_name)
        docker_lib.delete_all_containers_and_volumes(a_phdl)
    log.info("Teardown: cleaning up log directory")
    r_phdl.exec(f"sudo rm -rf {inference_dict['log_dir']}")
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t0)
    lifecycle.torn_down = True