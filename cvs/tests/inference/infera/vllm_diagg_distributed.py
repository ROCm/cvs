'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Minimal Infera vLLM 1P1D smoke suite: GET /v1/models and POST /v1/completions
via infera.server (MooncakeConnector + etcd).
'''

import time

from cvs.lib import globals

log = globals.log


def test_cleanup_stale_containers(orch, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    orch.teardown_containers()
    orch.cleanup_log_dir()
    time.sleep(5)
    lifecycle.complete_stage(request, "stale_cleanup", t0)


def test_launch_inference_containers(orch, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    if not orch.setup_containers():
        lifecycle.failed = True
    lifecycle.complete_stage(request, "container_launch", t0)


def test_start_etcd(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.start_etcd()
    im_obj.write_mooncake_config()
    lifecycle.complete_stage(request, "etcd_start", t0)


def test_setup_worker_env(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.setup_worker_env()
    lifecycle.complete_stage(request, "worker_env", t0)


def test_launch_frontend(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.launch_frontend()
    lifecycle.complete_stage(request, "frontend_launch", t0)


def test_launch_prefill_servers(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.launch_prefill_servers()
    lifecycle.complete_stage(request, "prefill_launch", t0)


def test_launch_decode_servers(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.launch_decode_servers()
    lifecycle.complete_stage(request, "decode_launch", t0)


def test_poll_for_server_ready(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    im_obj.poll_and_check_server_ready()
    lifecycle.complete_stage(request, "server_ready", t0)


def test_openai_compatible_http_endpoints(im_obj, lifecycle, request):
    globals.error_list = []
    t0 = time.monotonic()
    results = im_obj.verify_openai_compatible_endpoints()
    lifecycle.smoke_results = results
    lifecycle.complete_stage(request, "smoke_endpoints", t0)


def test_teardown(orch, lifecycle, request):
    t0 = time.monotonic()
    orch.teardown_containers()
    orch.cleanup_log_dir()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t0)
    lifecycle.torn_down = True
