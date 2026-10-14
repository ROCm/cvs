"""
Shared lifecycle stages for the xDiT inference suites.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import base64
import inspect
import json
import os
import shlex
import time
from pathlib import Path

import pytest
from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.inference.xdit.xdit_benchmark_job import (
    _ARTIFACT_COLLECT_TIMEOUT_S,
    _materialize_artifacts,
    _parse_artifact_payload,
    cleanup_staged_artifacts,
    remote_benchmark_collect_cmd,
)
from cvs.lib.inference.xdit.xdit_clipscore import (
    CLIP_SCORE_TIMEOUT_S,
    build_clip_model_probe_cmd,
    build_clipscore_cmd,
    clip_model_missing_message,
    clipscore_output_path,
    parse_clipscore_text,
    resolve_clip_model,
    resolve_clip_sample_frames,
    validate_clip_score,
    validate_mean_frame_clip_score,
    wan_clip_video_name,
)
from cvs.lib.inference.xdit.xdit_flux import FluxOutputParser
from cvs.lib.inference.xdit.xdit_flux_job import (
    build_output_cleanup_cmd,
    is_flux2_model,
    launch_flux_benchmark,
    resolve_flux_model_type,
    validate_flux_parallelism_config,
)
from cvs.lib.inference.xdit.xdit_model_verify import (
    build_diffusers_local_model_required_checks,
    container_snapshot_to_host,
    download_hf_snapshot,
    is_local_model_path,
    resolve_wan_local_model_required_checks,
    verify_required_checks_on_nodes,
)
from cvs.lib.inference.xdit.xdit_wan import WanOutputParser
from cvs.lib.inference.xdit.xdit_wan_i2v import WanI2vOutputParser
from cvs.lib.inference.xdit.xdit_wan_job import (
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
    "test_clip_score": 6,
    "test_print_results": 7,
    "test_teardown": 8,
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


def hf_remote_download_enabled(variant):
    model = value_from_variant(variant, "model")
    if model is not None:
        remote = model.get("remote") if isinstance(model, dict) else getattr(model, "remote", None)
        if remote is not None:
            return int(remote) == 1
    return int(inference_from_variant(variant).get("model_remote") or 0) == 1


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
        from cvs.lib.inference.xdit.xdit_flux_job import resolve_distributed_execution_hosts

        hosts = resolve_distributed_execution_hosts(cluster_dict, inference)
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


class _SecretValue(str):
    def __repr__(self):
        return "<redacted>"


def hf_token_from_variant(variant):
    path = ""
    paths = value_from_variant(variant, "paths")
    if paths is not None:
        path = getattr(paths, "hf_token_file", None) or ""
        if not path and isinstance(paths, dict):
            path = paths.get("hf_token_file") or ""
    if not path:
        path = inference_from_variant(variant).get("hf_token_file") or ""
    if not path:
        return ""
    if not os.path.isfile(path):
        log.warning("HF token file missing: %s", path)
        return ""
    with open(path, encoding="utf-8") as fp:
        token = fp.read().strip()
    if not token:
        return ""
    return _SecretValue(token)


def verify_model_stage(orch, variant, spec, lifecycle, request):
    lifecycle.skip_if_failed()
    globals.error_list = []
    started = time.monotonic()
    inference = inference_from_variant(variant)
    model_id = inference.get("_resolved_model_mount_host") or inference.get("model_repo")
    if is_local_model_path(model_id):
        model_path = model_id
    elif inference.get("model_repo"):
        if hf_remote_download_enabled(variant):
            snapshots, errors = download_hf_snapshot(orch, inference)
            if errors:
                fail_test("; ".join(errors))
                _complete(lifecycle, request, "model_verify", started)
                return
            container_snapshot = next(iter(snapshots.values()), "")
            inference["_resolved_model_path_container"] = container_snapshot
            inference["_resolved_ckpt_dir_container"] = container_snapshot
            model_path = container_snapshot_to_host(container_snapshot, inference) or _model_host_path(orch, inference)
        else:
            model_path = _model_host_path(orch, inference)
            if not model_path:
                fail_test(
                    f"Pre-populate HF cache under {inference.get('hf_home')} (no downloads are performed by this test)."
                )
                _complete(lifecycle, request, "model_verify", started)
                return
    else:
        model_path = None
    if not model_path:
        fail_test("xDiT server_params.model must be a Hugging Face repo id or an absolute host path")
        _complete(lifecycle, request, "model_verify", started)
        return

    exists = orch.all.exec(f"test -d {shlex.quote(model_path)} && echo MODEL_OK || echo MODEL_MISSING")
    missing = [host for host, output in exists.items() if "MODEL_OK" not in str(output)]
    if missing:
        if not hf_remote_download_enabled(variant) and not is_local_model_path(model_id):
            fail_test(
                f"Pre-populate HF cache under {inference.get('hf_home')} (no downloads are performed by this test)."
            )
        else:
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
        local_model = is_local_model_path(inference.get("model_repo", ""))
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
        clip_error = _verify_clip_model_offline(orch, variant, spec)
        if clip_error:
            fail_test(clip_error)
    _complete(lifecycle, request, "model_verify", started)


def _clip_model_for_variant(variant, spec):
    params = {}
    benchmark = value_from_variant(variant, "benchmark_params")
    if isinstance(benchmark, dict):
        key = "flux1_dev_t2i" if spec.get("family") == "flux" else "wan22_i2v_a14b"
        workload = benchmark.get(key)
        if isinstance(workload, dict):
            params = workload
    return resolve_clip_model(params)


def _clip_probe_ok(output):
    if isinstance(output, dict):
        text = str(output.get("output") or "")
    else:
        text = str(output or "")
    return any(line.strip() == "CLIP_OK" for line in text.splitlines())


def _verify_clip_model_offline(orch, variant, spec):
    """Fail when the CLIP checkpoint is not already in the container. Never downloads it."""
    model = _clip_model_for_variant(variant, spec)
    inference = inference_from_variant(variant)
    hf_home = inference.get("hf_home_container") or "/hf_home"
    try:
        results = orch.exec(build_clip_model_probe_cmd(model, hf_home))
    except Exception as exc:
        return clip_model_missing_message(model, []) + f" Probe failed: {exc}"
    missing = [host for host, output in (results or {}).items() if not _clip_probe_ok(output)]
    if missing:
        return clip_model_missing_message(model, missing)
    return None


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


def _ulysses_ring(workload, spec):
    degrees = _parallel_degrees(workload, spec)
    return degrees[0][1], degrees[1][1]


def _topology_nodes(variant, cluster_dict, spec):
    inference = inference_from_variant(variant)
    if not spec["distributed"]:
        return list(inference.get("_execution_hosts") or cluster_dict.get("node_dict") or [])
    from cvs.lib.inference.xdit.xdit_flux_job import resolve_distributed_execution_hosts

    return resolve_distributed_execution_hosts(cluster_dict, inference)


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


def run_benchmark_stage(orch, variant, cluster_dict, spec, lifecycle, request):
    lifecycle.skip_if_failed()
    globals.error_list = []
    started = time.monotonic()
    inference = inference_from_variant(variant)
    params = benchmark_params_from_variant(variant)
    hf_token = hf_token_from_variant(variant)
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
    _complete(lifecycle, request, "generate_and_benchmark", started)


def _workload_params(variant, spec):
    params = benchmark_params_from_variant(variant)
    return params["flux1_dev_t2i" if spec["family"] == "flux" else "wan22_i2v_a14b"]


def _threshold_inputs(variant, spec):
    workload = _workload_params(variant, spec)
    return workload, workload["expected_results"]


def _metric_name(spec):
    return "avg_total_time_s" if spec["family"] == "wan" and not spec["diffusers"] else "avg_pipe_time_s"


def _configured_count(value):
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _is_flux2_workload(variant, params):
    inference = inference_from_variant(variant)
    model = value_from_variant(variant, "model")
    model_id = None
    if isinstance(model, dict):
        model_id = model.get("id")
    elif model is not None:
        model_id = getattr(model, "id", None)
    model_type = resolve_flux_model_type(
        params.get("model_type") or inference.get("_resolved_flux_model_type"),
        inference.get("model_repo"),
        model_id,
    )
    return is_flux2_model(model_type)


def _output_parser(spec, params, output_dir, variant):
    if spec["family"] == "flux":
        expected_repetitions = None
        if not _is_flux2_workload(variant, params):
            expected_repetitions = _configured_count(params.get("num_repetitions"))
        return FluxOutputParser(
            output_dir,
            expected_image_pattern="flux_*.png",
            expected_repetitions=expected_repetitions,
        )
    if spec["diffusers"]:
        return WanI2vOutputParser(
            output_dir,
            require_video_artifact=bool(params.get("require_video_artifact", True)),
        )
    artifact = "video.mp4" if params.get("require_video_artifact", True) else ""
    return WanOutputParser(
        output_dir,
        expected_artifact=artifact,
        expected_steps=_configured_count(params.get("num_benchmark_steps")),
    )


def _is_report_artifact(name):
    if name in {"timing.json", "clipscore.json"} or name.endswith((".png", ".mp4")):
        return True
    return name.startswith("rank0") and name.endswith(".json")


def _iter_report_artifacts(output_dir):
    root = Path(output_dir)
    found = []

    def _onerror(exc):
        log.warning("Cannot read xDiT output path %s: %s", getattr(exc, "filename", output_dir), exc)

    for dirpath, _dirnames, filenames in os.walk(root, onerror=_onerror):
        for name in filenames:
            if _is_report_artifact(name):
                found.append(Path(dirpath) / name)
    return sorted(found)


def _attach_benchmark_artifacts(request, host, output_dir):
    config = getattr(request, "config", None)
    mgr = getattr(config, "_html_report_manager", None)
    if mgr is None or not getattr(mgr, "is_enabled", False):
        return []
    copied = []
    safe_host = str(host).replace("/", "_").replace(" ", "_")
    root = Path(output_dir)
    for path in _iter_report_artifacts(output_dir):
        if path.suffix in {".png", ".mp4"} and path.is_file() and path.stat().st_size == 0:
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = Path(path.name)
        dest_name = f"{safe_host}_{str(rel).replace(os.sep, '_')}"
        rel_path = mgr.add_html_to_report(
            str(path),
            dest_name=dest_name,
            track_in_reports=False,
        )
        if rel_path:
            copied.append(rel_path)
    if copied:
        log.info("Added %d xDiT artifact(s) from %s to the HTML report", len(copied), output_dir)
    return copied


def _output_dirs_by_host(inference, lifecycle):
    by_host = inference.get("_test_output_dirs_by_node") or {}
    if by_host:
        grouped = {}
        for host, output_dir in by_host.items():
            grouped.setdefault(output_dir, []).append(host)
        labeled = {}
        for output_dir, hosts in grouped.items():
            if lifecycle.benchmark_host and lifecycle.benchmark_host in hosts:
                label = lifecycle.benchmark_host
            else:
                label = hosts[0]
            labeled[label] = output_dir
        return labeled
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
        benchmark_steps = params.get("num_repetitions", "-")
        backend = "diffusers"
        cell_id = f"SIZE={shape},STEPS={steps},BENCH={benchmark_steps}"
    else:
        shape = params.get("size", "-")
        frames = params.get("frame_num", "-")
        steps = params.get("num_inference_steps")
        if steps is None:
            steps = 40 if spec["diffusers"] else "-"
        benchmark_steps = params.get("num_benchmark_steps", "-")
        backend = "diffusers" if spec["diffusers"] else "native"
        cell_id = f"SIZE={shape},FRAMES={frames},STEPS={steps},BENCH={benchmark_steps}"
    nnodes = _nnodes_count(variant, spec)
    return str(model_id), shape, steps, backend, nnodes, cell_id


def _nnodes_count(variant, spec):
    if not spec["distributed"]:
        return 1
    inference = inference_from_variant(variant)
    nnodes = inference.get("nnodes")
    if nnodes is not None and str(nnodes).strip() != "":
        return int(nnodes)
    hosts = inference.get("_execution_hosts") or inference.get("server_node_list") or []
    return max(len(hosts), 1)


def _report_threshold(thresholds, gpu_type, metric):
    selected = thresholds.get(gpu_type) or thresholds.get("auto") or {}
    for prefix, kind in (("max_", "max"), ("min_", "min")):
        value = selected.get(prefix + metric)
        if value is not None:
            return {"kind": kind, "value": value}
    return None


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
    model_id, shape, steps, backend, nnodes, cell_id = _report_dimensions(variant, params, spec)
    topology = "distributed" if spec["distributed"] else "single"
    ulysses_degree, ring_degree = _ulysses_ring(params, spec)
    cell = lifecycle.report_results.setdefault((model_id, gpu_type, shape, steps, cell_id, str(nnodes)), {})
    report_spec = _report_threshold(thresholds, gpu_type, metric)
    if report_spec is not None:
        variant.thresholds[cell_id] = {metric: report_spec}

    reported_by_host = _output_dirs_by_host(
        {
            "_test_output_dirs_by_node": inference.get("_test_node_output_dirs_by_node"),
            "_test_output_dir": inference.get("_test_node_output_dir"),
        },
        lifecycle,
    )
    failures = []
    for host, output_dir in outputs.items():
        reported_dir = reported_by_host.get(host) or output_dir
        _attach_benchmark_artifacts(request, host, output_dir)
        parser = _output_parser(spec, params, output_dir, variant)
        if spec["family"] == "flux" and not _is_flux2_workload(variant, params) and parser.expected_repetitions is None:
            failures.append(f"{host}: FLUX.1 requires num_repetitions")
            continue
        if spec["family"] == "wan" and not spec["diffusers"] and parser.expected_steps is None:
            failures.append(f"{host}: native WAN requires num_benchmark_steps")
            continue
        result, errors = parser.parse()
        if result is None:
            staged_names = {str(output_dir), str(Path(output_dir))}
            detail = []
            for err in errors:
                text = str(err)
                for staged_name in staged_names:
                    text = text.replace(staged_name, reported_dir)
                detail.append(text)
            failures.append(f"Failed to parse xDiT output on {host} from {reported_dir}: {detail}")
            continue
        passed, message = parser.validate_threshold(result, thresholds, gpu_type)
        value = getattr(result, metric)
        cell[host] = {
            metric: value,
            "backend": backend,
            "topology": topology,
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "sample_count": getattr(result, "repetition_count", getattr(result, "step_count", 0)),
            "output_dir": reported_dir,
        }
        lifecycle.results.append(
            (
                host,
                model_id,
                topology,
                ulysses_degree,
                ring_degree,
                reported_dir,
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


def _container_output_groups(inference, lifecycle):
    by_node = inference.get("_test_container_output_dirs_by_node") or {}
    groups = {}
    order = []
    for host, output_dir in by_node.items():
        if output_dir not in groups:
            order.append(output_dir)
            groups[output_dir] = []
        groups[output_dir].append(host)
    labeled = []
    for output_dir in order:
        hosts = groups[output_dir]
        if lifecycle.benchmark_host and lifecycle.benchmark_host in hosts:
            label = lifecycle.benchmark_host
        else:
            label = hosts[0]
        labeled.append((label, output_dir, hosts))
    return labeled


def _exec_detail(value):
    if isinstance(value, dict):
        return str(value.get("output") or ""), int(value.get("exit_code", -1))
    return str(value or ""), None


def _score_one_directory(orch, hosts, output_dir, prompt, model, sample_frames=None, video_name=None):
    cmd = build_clipscore_cmd(
        prompt,
        output_dir,
        clipscore_output_path(output_dir),
        model,
        sample_frames=sample_frames,
        video_name=video_name,
    )
    errors = []
    for host in hosts:
        try:
            raw = orch.exec(cmd, hosts=[host], timeout=CLIP_SCORE_TIMEOUT_S, detailed=True, print_console=False)
        except Exception as exc:
            errors.append(f"{host}: {exc}")
            continue
        text, exit_code = _exec_detail((raw or {}).get(host))
        payload = None
        try:
            payload = parse_clipscore_text(text)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{host}: {exc}")
            continue
        if payload is not None and exit_code in (None, 0):
            return host, payload, None
        tail = " | ".join(line.strip() for line in text.splitlines() if line.strip())[-500:]
        errors.append(f"{host}: CLIP score command failed ({tail or 'no output'})")
    return None, None, "; ".join(errors) or "CLIP score command failed"


def _clipscore_json(directory):
    path = Path(directory) / "results" / "clipscore.json"
    if path.is_file():
        return path
    return None


def _local_clipscore_dir(inference, host):
    """Return a directory the controller can already read, when one is mounted."""
    seen = []
    for key in ("_test_node_output_dirs_by_node", "_test_output_dirs_by_node"):
        path = (inference.get(key) or {}).get(host)
        if path and path not in seen:
            seen.append(path)
    for directory in seen:
        if _clipscore_json(directory):
            return directory
    return None


def _materialize_clipscore_payload(payload):
    blob = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    return _materialize_artifacts([{"rel": "results/clipscore.json", "b64": base64.b64encode(blob).decode("ascii")}])


def _fetch_remote_clipscore(orch, host, container_dir):
    """Read clipscore.json from the node. Node-local output dirs are not on NFS."""
    cmd = remote_benchmark_collect_cmd(container_dir)
    try:
        raw = orch.exec(
            cmd,
            hosts=[host],
            timeout=_ARTIFACT_COLLECT_TIMEOUT_S,
            detailed=True,
            print_console=False,
        )
    except Exception as exc:
        log.warning("Could not read CLIP score file from %s: %s", host, exc)
        return None
    text, _exit_code = _exec_detail((raw or {}).get(host))
    items = _parse_artifact_payload(text)
    if not items:
        return None
    local = _materialize_artifacts(items)
    if local and _clipscore_json(local):
        return local
    return None


def _publish_clipscore(orch, writer, container_dir, inference, label, payload):
    existing = _local_clipscore_dir(inference, label) or _local_clipscore_dir(inference, writer)
    if existing:
        return existing
    fetched = _fetch_remote_clipscore(orch, writer, container_dir)
    if fetched:
        return fetched
    return _materialize_clipscore_payload(payload)


def _merge_clip_threshold(variant, cell_id, metric, spec):
    slot = variant.thresholds.get(cell_id)
    if isinstance(slot, dict) and slot and all(isinstance(value, dict) and "kind" in value for value in slot.values()):
        slot[metric] = spec
        return
    variant.thresholds[cell_id] = {metric: spec}


def clip_score_stage(orch, variant, gpu_type, spec, lifecycle, request):
    if spec["family"] not in {"flux", "wan"}:
        return
    lifecycle.skip_if_failed()
    globals.error_list = []
    started = time.monotonic()
    inference = inference_from_variant(variant)
    groups = _container_output_groups(inference, lifecycle)
    if not groups:
        fail_test("xDiT benchmark did not publish container output directories for CLIP score")
        _complete(lifecycle, request, "clip_score", started)
        return

    params, thresholds = _threshold_inputs(variant, spec)
    scoring_video = spec["family"] == "wan"
    prompt = params.get("prompt")
    if not prompt:
        fail_test(f"{'WAN' if scoring_video else 'FLUX'} CLIP score requires benchmark_params prompt")
        _complete(lifecycle, request, "clip_score", started)
        return
    try:
        sample_frames = resolve_clip_sample_frames(params) if scoring_video else None
    except ValueError as exc:
        fail_test(str(exc))
        _complete(lifecycle, request, "clip_score", started)
        return
    video_name = wan_clip_video_name(spec["diffusers"]) if scoring_video else None
    metric = "mean_frame_clip_score" if scoring_video else "clip_score"
    subject = "WAN video" if scoring_video else "FLUX images"
    model = resolve_clip_model(params)
    enforce_thresholds = bool(value_from_variant(variant, "enforce_thresholds", True))
    model_id, shape, steps, _backend, nnodes, cell_id = _report_dimensions(variant, params, spec)
    topology = "distributed" if spec["distributed"] else "single"
    ulysses_degree, ring_degree = _ulysses_ring(params, spec)
    cell = lifecycle.report_results.setdefault((model_id, gpu_type, shape, steps, cell_id, str(nnodes)), {})
    report_spec = _report_threshold(thresholds, gpu_type, metric)
    if report_spec is not None:
        _merge_clip_threshold(variant, cell_id, metric, report_spec)

    node_dirs = inference.get("_test_node_output_dirs_by_node") or {}
    failures = []
    for label, output_dir, hosts in groups:
        writer, payload, error = _score_one_directory(
            orch,
            hosts,
            output_dir,
            prompt,
            model,
            sample_frames=sample_frames,
            video_name=video_name,
        )
        if payload is None:
            failures.append(f"Failed to score {subject} for {label} under {output_dir}: {error}")
            continue
        if metric not in payload:
            failures.append(f"Failed to score {subject} for {label} under {output_dir}: payload missing {metric}")
            continue
        local_dir = _publish_clipscore(orch, writer, output_dir, inference, label, payload)
        if local_dir:
            _attach_benchmark_artifacts(request, label, local_dir)
        score = float(payload[metric])
        if scoring_video:
            passed, message = validate_mean_frame_clip_score(payload, thresholds, gpu_type)
        else:
            passed, message = validate_clip_score(payload, thresholds, gpu_type)
        host_row = cell.setdefault(label, {})
        host_row[metric] = score
        reported_dir = node_dirs.get(label) or output_dir
        host_row.setdefault("output_dir", reported_dir)
        lifecycle.results.append(
            (
                label,
                model_id,
                topology,
                ulysses_degree,
                ring_degree,
                reported_dir,
                metric,
                score,
                passed if enforce_thresholds else None,
                message,
            )
        )
        logged = "Mean frame CLIPScore" if scoring_video else "CLIPScore"
        log.info("%s on %s: %.2f (%s)", logged, label, score, message)
        if enforce_thresholds and not passed:
            failures.append(f"{label}: {message}")

    for failure in failures:
        fail_test(failure)
    _complete(lifecycle, request, "clip_score", started)


def print_results_stage(lifecycle):
    if not lifecycle.results:
        log.info("No xDiT benchmark results to print")
        return
    rows = [
        [
            host,
            model_id,
            topology,
            ulysses_degree,
            ring_degree,
            output,
            metric,
            f"{value:.3f}",
            "RECORDED" if passed is None else "PASS" if passed else "FAIL",
            message,
        ]
        for host, model_id, topology, ulysses_degree, ring_degree, output, metric, value, passed, message in lifecycle.results
    ]
    log.info(
        "\n======== xDiT benchmark results ========\n%s",
        tabulate(
            rows,
            headers=[
                "Host",
                "Model",
                "Topology",
                "Ulysses",
                "Ring",
                "Output",
                "Metric",
                "Value",
                "Result",
                "Threshold",
            ],
            tablefmt="github",
        ),
    )


def teardown_stage(orch, lifecycle, request):
    started = time.monotonic()
    orch.teardown_containers()
    cleanup_staged_artifacts()
    lifecycle.record(request, "teardown", started)
    lifecycle.torn_down = True
