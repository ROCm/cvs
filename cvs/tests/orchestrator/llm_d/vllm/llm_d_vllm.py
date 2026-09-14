'''Two-or-more vLLM replicas behind an llm-d EPP and Envoy gateway.'''

import time

from cvs.lib import globals
from cvs.lib.verify_lib import verify_dmesg_for_errors

log = globals.log


class _WorkerExecutor:
    def __init__(self, orch, hosts):
        self.orch = orch
        self.hosts = hosts

    def exec(self, command):
        return self.orch.exec(command, hosts=self.hosts)


def test_resolve_topology(llmd_topology, lifecycle, request):
    started = time.monotonic()
    for worker in llmd_topology.workers:
        log.info(
            "llm-d worker name=%s node=%s endpoint=%s:%s",
            worker["name"],
            worker["node"],
            worker["address"],
            worker["port"],
        )
    lifecycle.record(request.node.nodeid, "topology", started)


def test_stage_gateway_config(stack, lifecycle, request):
    started = time.monotonic()
    stack.stage_gateway_config()
    lifecycle.record(request.node.nodeid, "gateway_config", started)


def test_launch_vllm_workers(stack, lifecycle, request):
    started = time.monotonic()
    stack.launch_workers()
    lifecycle.record(request.node.nodeid, "worker_launch", started)


def test_poll_workers_ready(stack, lifecycle, request):
    started = time.monotonic()
    stack.wait_workers_ready()
    lifecycle.record(request.node.nodeid, "worker_ready", started)


def test_launch_epp(stack, lifecycle, request):
    started = time.monotonic()
    stack.launch_epp()
    lifecycle.record(request.node.nodeid, "epp_launch", started)


def test_launch_envoy(stack, lifecycle, request):
    started = time.monotonic()
    stack.launch_envoy()
    lifecycle.record(request.node.nodeid, "envoy_launch", started)


def test_gateway_ready(stack, lifecycle, request):
    started = time.monotonic()
    stack.wait_gateway_ready()
    lifecycle.record(request.node.nodeid, "gateway_ready", started)


def test_openai_compatible_http_endpoints(stack, lifecycle, request):
    started = time.monotonic()
    lifecycle.smoke_results = stack.probe_openai_endpoints()
    lifecycle.record(request.node.nodeid, "openai_smoke", started)


def test_verify_dmesg(stack, llmd_topology, lifecycle, request):
    started = time.monotonic()
    worker_hosts = list(dict.fromkeys(worker["node"] for worker in llmd_topology.workers))
    end_time = stack.orch.exec('date +"%a %b %e %H:%M:%S"', hosts=worker_hosts)
    executor = _WorkerExecutor(stack.orch, worker_hosts)
    start_time = {host: stack.start_time[host] for host in worker_hosts}
    verify_dmesg_for_errors(executor, start_time, end_time, till_end_flag=False)
    lifecycle.record(request.node.nodeid, "verify_dmesg", started)


def test_teardown(stack, lifecycle, request):
    started = time.monotonic()
    stack.cleanup()
    lifecycle.torn_down = True
    lifecycle.record(request.node.nodeid, "teardown", started)
