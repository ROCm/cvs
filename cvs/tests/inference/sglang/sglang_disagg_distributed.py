'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Single SGLang disaggregated (PD) benchmark module: model is selected from
``benchmark_params`` via ``active_benchmark`` / env / single-key auto (see ``_shared``).
'''

import importlib.util as _ilu
import pathlib as _pl
import re
import time

import pytest

from cvs.lib import docker_lib, globals
from cvs.lib.utils_lib import fail_test, update_test_result

_spec = _ilu.spec_from_file_location(
    "_sglang_disagg_shared", _pl.Path(__file__).with_name("_shared.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

test_print_results_table = _mod.test_print_results_table

log = globals.log


def test_cleanup_stale_containers(p_phdl, d_phdl, r_phdl, b_phdl, inference_dict):
    container_name = inference_dict["container_name"]
    for a_phdl in (p_phdl, d_phdl, r_phdl, b_phdl):
        docker_lib.kill_docker_container(a_phdl, container_name)
        docker_lib.delete_all_containers_and_volumes(a_phdl)
    log.info("Cleaning up log directory")
    r_phdl.exec(f"sudo rm -rf {inference_dict['log_dir']}")
    time.sleep(5)


def test_launch_inference_containers(p_phdl, d_phdl, r_phdl, b_phdl, inference_dict):
    log.info("Testcase launch SGLang containers")
    globals.error_list = []
    container_name = inference_dict["container_name"]
    hdl_list = [p_phdl, d_phdl]

    if inference_dict["proxy_router_node"] == inference_dict["benchmark_serv_node"]:
        if (inference_dict["proxy_router_node"] in inference_dict["prefill_node_list"]) or (
            inference_dict["proxy_router_node"] in inference_dict["decode_node_list"]
        ):
            log.info("Already part of the handle list, no need to add")
        else:
            hdl_list.append(r_phdl)
    else:
        if (inference_dict["proxy_router_node"] in inference_dict["prefill_node_list"]) or (
            inference_dict["proxy_router_node"] in inference_dict["decode_node_list"]
        ):
            log.info("Already part of the handle list, no need to add")
        else:
            hdl_list.append(r_phdl)
        if (inference_dict["benchmark_serv_node"] in inference_dict["prefill_node_list"]) or (
            inference_dict["benchmark_serv_node"] in inference_dict["decode_node_list"]
        ):
            log.info("Already part of the handle list, no need to add")
        else:
            hdl_list.append(b_phdl)

    for a_phdl in hdl_list:
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
    update_test_result()


def test_setup_ibv_devices(im_obj):
    globals.error_list = []
    im_obj.check_ibv_devices()
    im_obj.exec_nic_setup_scripts()
    update_test_result()


def test_rms_norm(im_obj):
    globals.error_list = []
    im_obj.run_test_rmsnorm()
    update_test_result()


def test_launch_prefill_servers(im_obj):
    globals.error_list = []
    im_obj.setup_prefill_container_env()
    im_obj.launch_prefill_servers()
    update_test_result()


def test_launch_decode_servers(im_obj):
    globals.error_list = []
    im_obj.setup_decode_container_env()
    im_obj.launch_decode_servers()
    update_test_result()


def test_poll_for_server_ready(im_obj):
    globals.error_list = []
    im_obj.poll_and_check_server_ready()
    update_test_result()


def test_launch_proxy_router(im_obj):
    globals.error_list = []
    im_obj.setup_proxy_router_container_env()
    im_obj.launch_proxy_router()
    update_test_result()


def test_openai_compatible_http_endpoints(im_obj, inf_res_dict):
    globals.error_list = []
    results = im_obj.verify_openai_compatible_endpoints()
    inf_res_dict["__smoke_probe_results__"] = results  
    update_test_result()


def test_run_lm_eval_hellaswag_benchmark_test(im_obj, inf_res_dict):
    globals.error_list = []
    im_obj.setup_benchmark_serv_container_env()
    h = im_obj.run_lm_eval_hellaswag_benchmark_test()
    inf_res_dict.setdefault("__phase_labels__", {})["accuracy_hellaswag"] = h
    update_test_result()


def test_run_lm_eval_gsm8k_benchmark_test(im_obj, inf_res_dict):
    globals.error_list = []
    im_obj.setup_benchmark_serv_container_env()
    g = im_obj.run_lm_eval_gsm8k_benchmark_test()
    inf_res_dict.setdefault("__phase_labels__", {})["accuracy_gsm8k"] = g
    update_test_result()


def test_run_lm_eval_mmlu_benchmark_test(im_obj, inf_res_dict):
    globals.error_list = []
    im_obj.setup_benchmark_serv_container_env()
    m = im_obj.run_lm_eval_mmlu_benchmark_test()
    inf_res_dict.setdefault("__phase_labels__", {})["accuracy_mmlu"] = m
    update_test_result()


def test_run_performance_benchmark_test(im_obj, inf_res_dict):
    globals.error_list = []
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
    update_test_result()


def test_disagg_gpu_topology(im_obj):
    globals.error_list = []
    im_obj.sglang_disagg_gpu_counts()
    update_test_result()