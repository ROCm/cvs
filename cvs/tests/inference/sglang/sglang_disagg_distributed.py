'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Single SGLang disaggregated (PD) benchmark module: model is selected from
``benchmark_params`` via ``active_benchmark`` / env / single-key auto (see ``_shared``).
'''

import pathlib
import re
import threading
import time

import pytest

from cvs.lib import docker_lib, globals
from cvs.lib.utils_lib import fail_test, update_test_result
from cvs.lib.utils.gpu import (
    GPU_METRICS,
    GPU_METRIC_UNITS,
    agg_readings,
    capture_gpu_metrics,
    poll_gpu_metrics,
)
from cvs.tests.inference.sglang._shared import test_print_results_table

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


def test_run_performance_benchmark_test(im_obj, inf_res_dict, gpu_metrics_snap, request):
    globals.error_list = []
    im_obj.setup_benchmark_serv_container_env()

    # --- GPU polling setup (Pattern B: client runs synchronously, poll in thread) ---
    nodes = [("prefill-0", im_obj.p_phdl), ("decode-0", im_obj.d_phdl)]
    bench_log = f"{im_obj.log_dir}/benchmark_node/benchmark_results.log"

    def _is_done() -> bool:
        try:
            out = im_obj.b_phdl.exec(f"cat {bench_log}")
            return any(
                "Serving Benchmark Result" in (text or "")
                for text in out.values()
            )
        except Exception:
            return False

    def _snap():
        try:
            return capture_gpu_metrics(None, nodes=nodes)
        except Exception:
            return {}

    # Pre-load snapshot (before client starts)
    pre_snap = _snap()

    _htmlpath = getattr(request.config.option, "htmlpath", None)
    _html_dir = getattr(request.config, "_test_html_dir", "test_html")
    _gpu_log = (
        pathlib.Path(_htmlpath).parent / _html_dir / "gpu_poll_disagg.log"
        if _htmlpath else None
    )

    poll_readings: list = []
    done_flag = threading.Event()

    def _poll_thread():
        poll_readings.extend(
            poll_gpu_metrics(
                None,
                is_done_fn=done_flag.is_set,
                nodes=nodes,
                log_path=str(_gpu_log) if _gpu_log else None,
            )
        )

    poll_t = threading.Thread(target=_poll_thread, daemon=True)
    poll_t.start()

    # --- Run client (synchronous, blocks until done) ---
    im_obj.benchserv_test_random(d_type="auto")
    done_flag.set()
    poll_t.join(timeout=60)

    # --- Store GPU metrics ---
    agg = agg_readings(poll_readings)
    gpu_metrics_snap["pre_snap"] = pre_snap
    inf_res_dict["gpu.peak_gpu_memory_mb"] = agg.get("peak_gpu_memory_mb")
    inf_res_dict["gpu.model_load_memory_mb"] = (
        ((pre_snap.get("gpu.used_vram") or 0)) or None
    )
    inf_res_dict["gpu.gpu_bandwidth_util_pct"] = agg.get("gpu_bandwidth_util_pct")
    inf_res_dict["gpu.gpu_compute_util_pct"] = agg.get("gpu_compute_util_pct")

    # --- Existing results wiring ---
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


def test_gpu_metric(gpu_metric, inf_res_dict, request):
    val = inf_res_dict.get(gpu_metric)
    unit = GPU_METRIC_UNITS.get(gpu_metric, "")

    request.node.user_properties.append(("metric_value", val))
    request.node.user_properties.append(("metric_unit", unit))

    if val is None:
        pytest.skip(f"{gpu_metric}: no value recorded (amd-smi unavailable or polling failed)")


def test_disagg_gpu_topology(im_obj):
    globals.error_list = []
    im_obj.sglang_disagg_gpu_counts()
    update_test_result()