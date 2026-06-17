'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import copy
import importlib.util as _ilu
import pathlib as _pl
import time

import pytest

from cvs.lib import globals
from cvs.lib.inference.inferencemax_orch import InferenceMaxJob
from cvs.lib.verify_lib import update_test_result

_spec = _ilu.spec_from_file_location("_inferencemax_shared", _pl.Path(__file__).with_name("_shared.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
test_print_results_table = _mod.test_print_results_table  # noqa: F841

log = globals.log


def test_launch_container(orch, lifecycle, request):
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
    time.sleep(30)
    update_test_result()


def test_inferencemax_inference(
    orch,
    hf_token,
    gpu_type,
    inference_dict,
    benchmark_params_dict,
    model_name,
    seq_combo,
    concurrency,
    inf_res_dict,
    lifecycle,
    request,
):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    bp_run = copy.deepcopy(benchmark_params_dict)
    cell = bp_run[model_name]
    cell["input_sequence_length"] = str(seq_combo["isl"])
    cell["output_sequence_length"] = str(seq_combo["osl"])
    cell["max_concurrency"] = str(concurrency)

    globals.error_list = []
    im_obj = InferenceMaxJob(
        c_phdl=orch.all,
        s_phdl=orch.all,
        model_name=model_name,
        inference_config_dict=inference_dict,
        benchmark_params_dict=bp_run,
        hf_token=hf_token,
        gpu_type=gpu_type,
        distributed_inference=False,
        orch=orch,
    )
    try:
        t_server = time.monotonic()
        im_obj.build_server_inference_job_cmd()
        im_obj.start_inference_server_job()
        lifecycle.record(request.node.nodeid, "server_ready", time.monotonic() - t_server)
        t_client = time.monotonic()
        im_obj.start_inference_client_job()
        poll_status = im_obj.poll_for_inference_completion()
        assert poll_status.get("status") == "success", f"Inference did not complete: {poll_status}"
        im_obj.verify_inference_results()
        assert im_obj.inference_results_dict, (
            "inference_results_dict empty after benchmark; log parsing or client run likely failed silently"
        )
        for _node, metrics in im_obj.inference_results_dict.items():
            assert metrics, f"no per-metric rows parsed for node {_node}; check bench_serv_script.log"
    except Exception:
        lifecycle.failed = True
        raise

    display_model = str(cell.get("model") or model_name)
    key = (
        display_model,
        gpu_type,
        str(seq_combo["isl"]),
        str(seq_combo["osl"]),
        seq_combo.get("name", "default"),
        concurrency,
    )
    inf_res_dict[key] = getattr(im_obj, "inference_results_dict", {}) or {}
    lifecycle.record(request.node.nodeid, "client_complete", time.monotonic() - t_client)
    update_test_result()


def test_teardown(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
