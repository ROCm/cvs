"""
Shared lifecycle stages for the xDiT inference suites.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import inspect
import shlex
import time

import pytest
from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.inference.xdit.pytorch_xdit_flux import FluxOutputParser
from cvs.lib.inference.xdit.pytorch_xdit_flux_job import (
    build_output_cleanup_cmd,
    launch_flux_benchmark,
    validate_flux_parallelism_config,
)
from cvs.lib.inference.xdit.pytorch_xdit_model_verify import (
    build_diffusers_local_model_required_checks,
    resolve_wan_local_model_required_checks,
    verify_required_checks_on_nodes,
)
from cvs.lib.inference.xdit.pytorch_xdit_wan import WanOutputParser
from cvs.lib.inference.xdit.pytorch_xdit_wan_i2v import WanI2vOutputParser
from cvs.lib.inference.xdit.pytorch_xdit_wan_job import (
    build_wan_output_cleanup_cmd,
    launch_wan_benchmark,
    validate_wan_parallelism_config,
)
from cvs.lib.utils_lib import fail_test, update_test_result

log = globals.log

XDIT_TEST_ORDER = {
    "test_launch_container": 0,
    "test_verify_prerequisites": 1,
    "test_verify_model": 2,
    "test_verify_parallelism": 3,
    "test_run_benchmark": 4,
    "test_parse_thresholds": 5,
    "test_print_results": 6,
    "test_teardown": 7,
}


class Lifecycle:
    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}
        self.report_results = {}
        self.benchmark_host = ""
        self.results = []

    def record(self, request, label, started):
        elapsed = time.monotonic() - started
        self.report.setdefault(request.node.nodeid, []).append((label, elapsed, "s"))

    def skip_if_failed(self):
        if self.failed:
            pytest.skip("a prior xDiT lifecycle stage failed")


def value_from_variant(variant, name, default=None):
    value = getattr(variant, name, None)
    if value is not None:
        return value
    if isinstance(variant, dict):
        return variant.get(name, default)
    return default


def inference_from_variant(variant):
    for name in ("inference", "config", "inference_config"):
        value = value_from_variant(variant, name)
        if value is not None:
            return value
    raise ValueError("xDiT variant must expose inference/config")


def benchmark_params_from_variant(variant):
    value = value_from_variant(variant, "benchmark_params")
    if value is None:
        raise ValueError("xDiT variant must expose benchmark_params")
    return value


def suite_spec(module_name):
    name = module_name.rsplit(".", 1)[-1].lower()
    return {
        "family": "flux" if "flux" in name else "wan",
        "distributed": name.endswith("_distributed"),
        "diffusers": "diffusers" in name,
    }


def resolve_execution_hosts(cluster_dict, inference, distributed):
    node_dict = cluster_dict.get("node_dict") or {}
    if distributed:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import resolve_nnodes, resolve_server_nodes

        nodes = resolve_server_nodes(cluster_dict, inference)
        nnodes = resolve_nnodes(inference, nodes)
        if nnodes < 2:
            raise ValueError(f"distributed xDiT suite requires nnodes >= 2, got {nnodes}")
        if len(nodes) < nnodes:
            raise ValueError(f"xDiT config requests {nnodes} nodes but only {len(nodes)} are available")
        hosts = nodes[:nnodes]
    else:
        hosts = list(node_dict)

    hosts = [host for host in hosts if host]
    missing = [host for host in hosts if host not in node_dict]
    if not hosts:
        raise ValueError("xDiT suite could not resolve an execution host")
    if missing:
        raise ValueError(f"xDiT execution hosts are absent from cluster node_dict: {missing}")
    return hosts


def scoped_cluster_dict(cluster_dict, hosts):
    scoped = dict(cluster_dict)
    scoped["node_dict"] = {host: cluster_dict["node_dict"][host] for host in hosts}
    scoped["head_node_dict"] = {"mgmt_ip": hosts[0]}
    return scoped


def _complete(lifecycle, request, label, started):
    lifecycle.record(request, label, started)
    if globals.error_list:
        lifecycle.failed = True
    update_test_result()


def launch_container_stage(orch, lifecycle, request):
    globals.error_list = []
    started = time.monotonic()
    if not orch.setup_containers():
        lifecycle.failed = True
        lifecycle.record(request, "container_launch", started)
        pytest.fail("setup_containers() returned False")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        lifecycle.record(request, "container_launch", started)
        pytest.fail(f"container {name} not running after setup_containers()")
    _complete(lifecycle, request, "container_launch", started)


def verify_prerequisites_stage(orch, lifecycle, request):
    lifecycle.skip_if_failed()
    globals.error_list = []
    started = time.monotonic()
    checks = orch.exec("bash -c 'test -e /dev/kfd && command -v torchrun >/dev/null && echo XDIT_PREREQS_OK'")
    bad = [host for host, output in checks.items() if "XDIT_PREREQS_OK" not in str(output)]
    if bad:
        fail_test(f"xDiT prerequisites missing in containers on: {', '.join(bad)}")
    _complete(lifecycle, request, "prerequisites", started)


def _model_host_path(orch, inference):
    explicit = inference.get("_resolved_model_mount_host") or inference.get("model_repo")
    if isinstance(explicit, str) and explicit.startswith("/"):
        return explicit
    repo = inference.get("model_repo")
    revision = inference.get("model_rev")
    hf_home = inference.get("hf_home")
    if repo and revision and hf_home:
        return f"{hf_home}/hub/models--{repo.replace('/', '--')}/snapshots/{revision}"
    if repo and hf_home:
        snapshots = f"{hf_home}/hub/models--{repo.replace('/', '--')}/snapshots"
        first_host = orch.hosts[0]
        output = orch.all.exec(
            f"test -d {shlex.quote(snapshots)} && ls -1 {shlex.quote(snapshots)} | head -1 || true",
            print_console=False,
        )
        snapshot = ((output.get(first_host, "") or "").strip().splitlines() or [""])[0]
        if snapshot:
            return f"{snapshots}/{snapshot}"
    return None


def verify_model_stage(orch, variant, spec, lifecycle, request):
    lifecycle.skip_if_failed()
    globals.error_list = []
    started = time.monotonic()
    inference = inference_from_variant(variant)
    model_path = _model_host_path(orch, inference)
    if not model_path:
        fail_test("xDiT model must be an absolute local path or a pinned offline HF snapshot")
        _complete(lifecycle, request, "model_verify", started)
        return

    exists = orch.all.exec(f"test -d {shlex.quote(model_path)} && echo MODEL_OK || echo MODEL_MISSING")
    missing = [host for host, output in exists.items() if "MODEL_OK" not in str(output)]
    if missing:
        fail_test(f"xDiT model path {model_path!r} is missing on: {', '.join(missing)}")
        _complete(lifecycle, request, "model_verify", started)
        return

    if spec["family"] == "flux" or spec["diffusers"]:
        required = build_diffusers_local_model_required_checks(model_path)
        description = "FLUX Diffusers" if spec["family"] == "flux" else "WAN Diffusers"
    else:
        required = resolve_wan_local_model_required_checks(
            model_path,
            model_format=inference.get("_resolved_wan_model_format"),
            model_repo=inference.get("model_repo", ""),
        )
        description = "WAN"
    error = verify_required_checks_on_nodes(
        orch.all,
        model_path,
        required,
        layout_description=description,
    )
    if error:
        fail_test(error)
    else:
        local_model = str(inference.get("model_repo", "")).startswith("/")
        if local_model:
            inference["_resolved_model_mount_host"] = model_path
        if spec["family"] == "flux":
            if not local_model:
                inference["_resolved_model_path_container"] = model_path.replace(
                    inference["hf_home"].rstrip("/"),
                    inference.get("hf_home_container", "/hf_home").rstrip("/"),
                    1,
                )
        else:
            if not local_model:
                inference["_resolved_ckpt_dir_container"] = model_path.replace(
                    inference["hf_home"].rstrip("/"),
                    inference.get("hf_home_container", "/hf_home").rstrip("/"),
                    1,
                )
    _complete(lifecycle, request, "model_verify", started)


def _parallel_degrees(workload, spec):
    if spec["family"] == "flux":
        return [
            ("ulysses", int(workload["ulysses_degree"])),
            ("ring", int(workload["ring_degree"])),
            ("pipefusion", int(workload.get("pipefusion_parallel_degree", 1))),
            ("tp", int(workload.get("tensor_parallel_degree", 1))),
            ("dp", int(workload.get("data_parallel_degree", 1))),
        ]
    nproc = int(workload["torchrun_nproc"])
    return [
        ("ulysses", int(workload.get("ulysses_size", nproc))),
        ("ring", int(workload.get("ring_size", 1))),
    ]


def _topology_nodes(variant, cluster_dict, spec):
    inference = inference_from_variant(variant)
    if not spec["distributed"]:
        return list(inference.get("_execution_hosts") or cluster_dict.get("node_dict") or [])

    from cvs.lib.inference.xdit.pytorch_xdit_flux_job import resolve_nnodes, resolve_server_nodes

    nodes = resolve_server_nodes(cluster_dict, inference)
    return nodes[: resolve_nnodes(inference, nodes)]


def log_topology(variant, cluster_dict, spec):
    inference = inference_from_variant(variant)
    workload = _workload_params(variant, spec)
    nodes = _topology_nodes(variant, cluster_dict, spec)
    nproc = int(workload["torchrun_nproc"])
    degrees = _parallel_degrees(workload, spec)
    product = 1
    for _, degree in degrees:
        product *= degree
    nnodes = len(nodes) if spec["distributed"] else 1
    world_size = nnodes * nproc
    family = "FLUX" if spec["family"] == "flux" else "WAN"
    mode = "Distributed" if spec["distributed"] else "Single-node"

    log.info("=" * 60)
    log.info("%s %s topology", mode, family)
    log.info("=" * 60)
    if spec["distributed"]:
        log.info("Participating nodes: %d", nnodes)
        for rank, node in enumerate(nodes):
            log.info("  rank %d -> %s (%d GPUs)", rank, node, nproc)
    else:
        log.info("Independent jobs: %d (one per node)", len(nodes))
        for node in nodes:
            log.info("  %s (%d GPUs)", node, nproc)
    log.info("GPUs per node (torchrun_nproc): %d", nproc)
    log.info("Total GPU ranks (world_size): %d = %d nodes × %d nproc", world_size, nnodes, nproc)
    log.info(
        "xDiT parallel layout: %s = %d",
        " × ".join(f"{name}={degree}" for name, degree in degrees),
        product,
    )
    log.info("Sequence-parallel size (ulysses × ring): %d", degrees[0][1] * degrees[1][1])
    if spec["distributed"]:
        log.info(
            "Rendezvous: %s:%s",
            inference.get("master_addr") or "<auto rank-0>",
            inference.get("master_port", 29500),
        )
    if product == world_size:
        log.info("Parallelism check: PASS (product %d == world_size %d)", product, world_size)
    log.info("=" * 60)


def verify_parallelism_stage(variant, cluster_dict, spec, lifecycle, request):
    lifecycle.skip_if_failed()
    globals.error_list = []
    started = time.monotonic()
    inference = inference_from_variant(variant)
    params = benchmark_params_from_variant(variant)
    log_topology(variant, cluster_dict, spec)
    validator = validate_flux_parallelism_config if spec["family"] == "flux" else validate_wan_parallelism_config
    error = validator(
        inference,
        params,
        distributed=spec["distributed"],
        cluster_dict=cluster_dict if spec["distributed"] else None,
    )
    if error:
        fail_test(error)
    _complete(lifecycle, request, "parallelism", started)


def _launch_accepts_orchestrator(launcher):
    parameters = inspect.signature(launcher).parameters
    return "orch" in parameters or "executor" in parameters


def run_benchmark_stage(orch, variant, hf_token, cluster_dict, spec, lifecycle, request):
    lifecycle.skip_if_failed()
    globals.error_list = []
    started = time.monotonic()
    inference = inference_from_variant(variant)
    params = benchmark_params_from_variant(variant)
    launcher = launch_flux_benchmark if spec["family"] == "flux" else launch_wan_benchmark
    output_base_dir = inference.get("output_base_dir")
    if output_base_dir:
        cleanup_builder = build_output_cleanup_cmd if spec["family"] == "flux" else build_wan_output_cleanup_cmd
        orch.all.exec(cleanup_builder(output_base_dir, use_sudo=True))
    lifecycle.benchmark_host = orch.hosts[0]

    kwargs = {
        "distributed": spec["distributed"],
        "cluster_dict": cluster_dict if spec["distributed"] else None,
    }
    if _launch_accepts_orchestrator(launcher):
        executor_name = "orch" if "orch" in inspect.signature(launcher).parameters else "executor"
        kwargs[executor_name] = orch
        errors = launcher(
            inference_dict=inference,
            benchmark_params_dict=params,
            hf_token=hf_token,
            **kwargs,
        )
    else:
        errors = launcher(orch, inference, params, hf_token, **kwargs)

    for error in errors or []:
        fail_test(error)
    _complete(lifecycle, request, "benchmark", started)


def _workload_params(variant, spec):
    params = benchmark_params_from_variant(variant)
    return params["flux1_dev_t2i" if spec["family"] == "flux" else "wan22_i2v_a14b"]


def _threshold_inputs(variant, spec):
    workload = _workload_params(variant, spec)
    return workload, workload["expected_results"]


def _metric_name(spec):
    return "avg_total_time_s" if spec["family"] == "wan" and not spec["diffusers"] else "avg_pipe_time_s"


def _output_parser(spec, params, output_dir):
    if spec["family"] == "flux":
        return FluxOutputParser(output_dir, expected_image_pattern="flux_*.png")
    if spec["diffusers"]:
        return WanI2vOutputParser(
            output_dir,
            require_video_artifact=bool(params.get("require_video_artifact", True)),
        )
    artifact = "video.mp4" if params.get("require_video_artifact", True) else ""
    return WanOutputParser(output_dir, expected_artifact=artifact)


def _output_dirs_by_host(inference, lifecycle):
    by_host = inference.get("_test_output_dirs_by_node") or {}
    if by_host:
        grouped = {}
        for host, output_dir in by_host.items():
            grouped.setdefault(output_dir, []).append(host)
        return {", ".join(hosts): output_dir for output_dir, hosts in grouped.items()}
    output_dir = inference.get("_test_output_dir")
    if output_dir:
        return {lifecycle.benchmark_host or "unknown": output_dir}
    return {}


def _report_dimensions(variant, params, spec):
    model = value_from_variant(variant, "model")
    model_id = getattr(model, "id", None) or inference_from_variant(variant).get("model_repo", "unknown")
    if spec["family"] == "flux":
        shape = f"{params.get('height', '-')}x{params.get('width', '-')}"
        steps = params.get("num_inference_steps", "-")
        backend = "diffusers"
    else:
        shape = params.get("size", "-")
        steps = params.get("frame_num", "-")
        backend = "diffusers" if spec["diffusers"] else "native"
    workers = params.get("torchrun_nproc", 1)
    if spec["distributed"]:
        workers = int(workers) * int(inference_from_variant(variant).get("nnodes", 1))
    cell_id = f"ISL={shape},OSL={steps},C={workers}"
    return str(model_id), shape, steps, backend, workers, cell_id


def _report_threshold(thresholds, gpu_type, metric):
    selected = thresholds.get(gpu_type) or thresholds.get("auto") or {}
    threshold_key = f"max_{metric}"
    value = selected.get(threshold_key)
    if value is None:
        return None
    return {"kind": "max", "value": value}


def parse_thresholds_stage(variant, gpu_type, spec, lifecycle, request):
    lifecycle.skip_if_failed()
    globals.error_list = []
    started = time.monotonic()
    inference = inference_from_variant(variant)
    outputs = _output_dirs_by_host(inference, lifecycle)
    if not outputs:
        fail_test("xDiT benchmark did not publish any output directory")
        _complete(lifecycle, request, "parse_thresholds", started)
        return

    params, thresholds = _threshold_inputs(variant, spec)
    metric = _metric_name(spec)
    enforce_thresholds = bool(value_from_variant(variant, "enforce_thresholds", True))
    model_id, shape, steps, backend, workers, cell_id = _report_dimensions(variant, params, spec)
    topology = "distributed" if spec["distributed"] else "single"
    cell = lifecycle.report_results.setdefault((model_id, gpu_type, shape, steps, cell_id, str(workers)), {})
    report_spec = _report_threshold(thresholds, gpu_type, metric)
    if report_spec is not None:
        variant.thresholds[cell_id] = {metric: report_spec}

    failures = []
    for host, output_dir in outputs.items():
        parser = _output_parser(spec, params, output_dir)
        result, errors = parser.parse()
        if result is None:
            failures.append(f"Failed to parse xDiT output on {host} from {output_dir}: {errors}")
            continue
        passed, message = parser.validate_threshold(result, thresholds, gpu_type)
        value = getattr(result, metric)
        cell[host] = {
            metric: value,
            "backend": backend,
            "topology": topology,
            "sample_count": getattr(result, "repetition_count", getattr(result, "step_count", 0)),
            "output_dir": output_dir,
        }
        lifecycle.results.append(
            (
                host,
                spec["family"],
                topology,
                output_dir,
                metric,
                value,
                passed if enforce_thresholds else None,
                message,
            )
        )
        if enforce_thresholds and not passed:
            failures.append(f"{host}: {message}")

    for failure in failures:
        fail_test(failure)
    _complete(lifecycle, request, "parse_thresholds", started)


def print_results_stage(lifecycle):
    if not lifecycle.results:
        log.info("No xDiT benchmark results to print")
        return
    rows = [
        [
            host,
            family,
            topology,
            output,
            metric,
            f"{value:.3f}",
            "RECORDED" if passed is None else "PASS" if passed else "FAIL",
            message,
        ]
        for host, family, topology, output, metric, value, passed, message in lifecycle.results
    ]
    log.info(
        "\n======== xDiT benchmark results ========\n%s",
        tabulate(
            rows,
            headers=["Host", "Family", "Topology", "Output", "Metric", "Value", "Result", "Threshold"],
            tablefmt="github",
        ),
    )


def teardown_stage(orch, lifecycle, request):
    started = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request, "teardown", started)
    lifecycle.torn_down = True
