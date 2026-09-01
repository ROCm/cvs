"""
PyTorch XDit WAN 2.2 I2V-A14B Diffusers (xFuser) unified multi-node test.

Runs one coordinated xFuser ``wan_i2v_example.py`` torchrun job across ``nnodes``
inside the ufb-private container and validates ``results/timing.json`` and
``results/video_i2v.mp4`` on rank 0.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import pytest
import shlex

from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    fail_test,
    update_test_result,
    get_model_from_rocm_smi_output,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib import docker_lib
from cvs.lib import globals
from cvs.parsers.schemas import ClusterConfigFile, PytorchXditWanConfigFile
from cvs.lib.inference.xdit.pytorch_xdit_model_verify import (
    build_diffusers_local_model_required_checks,
    verify_required_checks_on_nodes,
)
from cvs.lib.inference.xdit.pytorch_xdit_wan_i2v import (
    WanI2vOutputParser,
    log_results_summary,
)
from cvs.lib.inference.xdit.pytorch_xdit_wan_job import (
    build_wan_output_cleanup_cmd,
    compute_world_size,
    launch_wan_benchmark,
    parallel_product,
    resolve_nnodes,
    resolve_server_nodes,
    store_resolved_wan_model_format_from_index,
    validate_wan_parallelism_config,
)

log = globals.log


class _SecretValue:
    """Wrapper to avoid leaking secrets in pytest tracebacks."""

    def __init__(self, value: str):
        self.value = value or ""

    def __bool__(self) -> bool:
        return bool(self.value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return "<redacted>"


def _distributed_server_nodes(cluster_dict, inference_dict):
    """Return (server_nodes[:nnodes], nnodes) for this distributed job."""
    nodes = resolve_server_nodes(cluster_dict, inference_dict)
    nnodes = resolve_nnodes(inference_dict, nodes)
    if nnodes < 2:
        raise ValueError(f"Distributed test requires config nnodes >= 2, got {nnodes}")
    if len(nodes) < nnodes:
        raise ValueError(f"Cluster/server_node_list has {len(nodes)} node(s) but nnodes={nnodes}")
    return nodes[:nnodes], nnodes


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def training_config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file, encoding="utf-8") as json_file:
        loaded = json.load(json_file)

    loaded = resolve_cluster_config_placeholders(loaded)

    try:
        validated = ClusterConfigFile.model_validate(loaded)
        log.info("Cluster config validated successfully: %d nodes", len(validated.node_dict))
    except Exception as exc:
        log.error("Cluster config validation failed: %s", exc)
        pytest.fail(f"Invalid cluster configuration: {exc}")

    return loaded


@pytest.fixture(scope="module")
def wan_config_dict(training_config_file, cluster_dict):
    with open(training_config_file, encoding="utf-8") as json_file:
        raw_config = json.load(json_file)

    try:
        validated_config = PytorchXditWanConfigFile.model_validate(raw_config)
        log.info("WAN diffusers xFuser distributed config validated successfully")
        log.info("  Container: %s", validated_config.config.container_image)
        log.info("  Model: %s", validated_config.config.model_repo)
        log.info("  nnodes: %s", validated_config.config.nnodes)
    except Exception as exc:
        log.error("WAN config validation failed: %s", exc)
        pytest.fail(f"Invalid WAN configuration: {exc}")

    validated_dict = validated_config.model_dump()
    config_dict = resolve_test_config_placeholders(validated_dict["config"], cluster_dict)
    benchmark_params = resolve_test_config_placeholders(validated_dict["benchmark_params"], cluster_dict)
    return {"config": config_dict, "benchmark_params": benchmark_params}


@pytest.fixture(scope="module")
def inference_dict(wan_config_dict):
    return wan_config_dict["config"]


@pytest.fixture(scope="module")
def benchmark_params_dict(wan_config_dict):
    return wan_config_dict["benchmark_params"]


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    hf_token_file = inference_dict.get("hf_token_file") or ""
    if not hf_token_file:
        return _SecretValue("")
    try:
        with open(hf_token_file, encoding="utf-8") as fp:
            return _SecretValue(fp.read().rstrip("\n"))
    except FileNotFoundError:
        log.warning("HF token file not found: %s", hf_token_file)
        return _SecretValue("")


@pytest.fixture(scope="module")
def s_phdl(cluster_dict, inference_dict):
    try:
        node_list, nnodes = _distributed_server_nodes(cluster_dict, inference_dict)
    except ValueError as exc:
        pytest.fail(str(exc))

    env_vars = cluster_dict.get("env_vars")
    log.info(
        "Using parallel-ssh execution mode for distributed diffusers job on %d node(s): %s",
        nnodes,
        node_list,
    )
    return Pssh(
        log,
        node_list,
        user=cluster_dict.get("username"),
        password=cluster_dict.get("password"),
        pkey=cluster_dict.get("priv_key_file"),
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def gpu_type(s_phdl):
    head_node = s_phdl.host_list[0]
    smi_out = s_phdl.exec("rocm-smi -a | head -30")[head_node]
    detected = get_model_from_rocm_smi_output(smi_out)
    log.info("Detected GPU type: %s", detected)
    return detected


def test_cleanup_stale_containers(s_phdl, inference_dict, cluster_dict):
    """Clean Docker state and remove stale WAN output dirs before the run."""
    container_name = inference_dict["container_name"]
    _, nnodes = _distributed_server_nodes(cluster_dict, inference_dict)

    log.info(
        "Cleaning up stale containers: %s (+ ranks 0..%d) on %d node(s)",
        container_name,
        nnodes - 1,
        len(s_phdl.host_list),
    )

    docker_lib.kill_docker_container(s_phdl, container_name)
    for rank in range(nnodes):
        docker_lib.kill_docker_container(s_phdl, f"{container_name}-rank{rank}")
    docker_lib.delete_all_containers_and_volumes(s_phdl)

    output_base_dir = inference_dict.get("output_base_dir")
    if output_base_dir:
        cleanup_cmd = build_wan_output_cleanup_cmd(output_base_dir, use_sudo=True)
        log.info("Cleaning stale WAN outputs under %s", output_base_dir)
        s_phdl.exec(cleanup_cmd)

    log.info("Container and output cleanup completed on all server nodes")
    update_test_result()


def _read_model_index_from_node(s_phdl, node: str, model_dir: str):
    index_path = f"{model_dir.rstrip('/')}/model_index.json"
    output = s_phdl.exec(f"cat {shlex.quote(index_path)}", print_console=False).get(node, "")
    if not (output or "").strip():
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        log.warning("Could not parse model_index.json from %s on %s", index_path, node)
        return None


def test_verify_model_on_nodes(s_phdl, inference_dict):
    """Verify the Diffusers checkpoint exists locally on every participating node."""
    globals.error_list = []

    model_repo = inference_dict["model_repo"]
    if not (isinstance(model_repo, str) and model_repo.strip().startswith("/")):
        fail_test(
            "Diffusers xFuser distributed WAN test requires config.model_repo as an "
            "explicit local path on every server node."
        )
        update_test_result()
        return

    host_model_path = model_repo.strip()
    check_cmd = f"test -d {shlex.quote(host_model_path)} && echo EXISTS || echo MISSING"
    check_result = s_phdl.exec(check_cmd)

    missing_nodes = [node for node, output in check_result.items() if "EXISTS" not in (output or "")]
    if missing_nodes:
        fail_test(
            f"Local model path not found on {len(missing_nodes)} node(s): {', '.join(missing_nodes)}. "
            f"Path: {host_model_path}"
        )
        update_test_result()
        return

    verify_err = verify_required_checks_on_nodes(
        s_phdl,
        host_model_path,
        build_diffusers_local_model_required_checks(host_model_path),
        layout_description="WAN Diffusers",
    )
    if verify_err:
        fail_test(verify_err)
        update_test_result()
        return

    inference_dict["_resolved_model_mount_host"] = host_model_path
    inference_dict["_resolved_ckpt_dir_container"] = "/model"
    model_index = _read_model_index_from_node(s_phdl, s_phdl.host_list[0], host_model_path)
    if model_index:
        store_resolved_wan_model_format_from_index(inference_dict, model_index)

    update_test_result()


def test_verify_parallelism_config(cluster_dict, inference_dict, benchmark_params_dict):
    """Fail fast if xDiT parallel degrees do not match nnodes × torchrun_nproc."""
    globals.error_list = []

    wan_params = benchmark_params_dict["wan22_i2v_a14b"]
    server_nodes = resolve_server_nodes(cluster_dict, inference_dict)
    nnodes = resolve_nnodes(inference_dict, server_nodes)
    participating = server_nodes[:nnodes]
    nproc = int(wan_params["torchrun_nproc"])
    world_size = compute_world_size(nnodes, nproc)
    product = parallel_product(wan_params)

    ulysses = int(wan_params["ulysses_size"])
    ring = int(wan_params["ring_size"])

    log.info("=" * 60)
    log.info("Distributed WAN diffusers xFuser topology")
    log.info("=" * 60)
    log.info("Participating nodes: %d", nnodes)
    for rank, node in enumerate(participating):
        log.info("  rank %d -> %s (%d GPUs)", rank, node, nproc)
    log.info("GPUs per node (torchrun_nproc): %d", nproc)
    log.info("Total GPU ranks (world_size): %d = %d nodes × %d nproc", world_size, nnodes, nproc)
    log.info("xDiT parallel layout: ulysses=%d × ring=%d = %d", ulysses, ring, product)
    log.info(
        "Rendezvous: %s:%s",
        inference_dict.get("master_addr") or "<auto rank-0>",
        inference_dict.get("master_port", 29500),
    )
    if product == world_size:
        log.info("Parallelism check: PASS (product %d == world_size %d)", product, world_size)
    log.info("=" * 60)

    err = validate_wan_parallelism_config(
        inference_dict,
        benchmark_params_dict,
        distributed=True,
        cluster_dict=cluster_dict,
    )
    if err:
        fail_test(err)
    update_test_result()


def test_run_wan22_diffusers_benchmark(s_phdl, cluster_dict, inference_dict, benchmark_params_dict, hf_token):
    """Run unified multi-node Diffusers xFuser WAN benchmark."""
    globals.error_list = []

    errors = launch_wan_benchmark(
        s_phdl,
        inference_dict,
        benchmark_params_dict,
        hf_token,
        distributed=True,
        cluster_dict=cluster_dict,
    )
    if errors:
        for err in errors:
            fail_test(err)

    update_test_result()


def test_parse_and_validate_results(s_phdl, cluster_dict, inference_dict, benchmark_params_dict, gpu_type):
    """
    Parse rank-0 xFuser outputs and validate against pipe_time thresholds.

    Unified distributed runs write ``results/timing.json`` and ``results/video_i2v.mp4``
    to ``wan_22_{rank0_hostname}_outputs`` on rank 0.
    """
    globals.error_list = []

    output_base_dir = inference_dict.get("output_base_dir")
    if not output_base_dir:
        fail_test("output_base_dir not set in config; cannot locate benchmark results")
        update_test_result()
        return

    output_dir = inference_dict.get("_test_output_dir")
    if not output_dir:
        try:
            server_nodes, _ = _distributed_server_nodes(cluster_dict, inference_dict)
            rank0_node = server_nodes[0]
            hostname_out = s_phdl.exec("hostname", print_console=False)
            rank0_hostname = (hostname_out.get(rank0_node, "") or "").strip() or rank0_node
            output_dir = f"{output_base_dir}/wan_22_{rank0_hostname}_outputs"
            log.info("Derived rank-0 output directory: %s", output_dir)
        except Exception as exc:
            fail_test(f"Could not determine rank-0 WAN output directory: {exc}")
            update_test_result()
            return

    wan_params = benchmark_params_dict["wan22_i2v_a14b"]
    expected_results = wan_params["expected_results"]
    require_video = bool(wan_params.get("require_video_artifact", True))

    log.info("Parsing diffusers xFuser results from: %s", output_dir)
    parser = WanI2vOutputParser(output_dir, require_video_artifact=require_video)
    result, errors = parser.parse()

    for error in errors:
        log.warning("Parse warning: %s", error)

    if result is None:
        fail_test(f"Failed to parse benchmark results from {output_dir}: {errors}")
        update_test_result()
        return

    if require_video and not result.video_path:
        fail_test(f"Artifact video_i2v.mp4 not found under {output_dir}")
        update_test_result()
        return

    if result.video_path:
        log.info("Video artifact found: %s", result.video_path)

    log.info(
        "Benchmark results: repetitions=%d avg_pipe_time=%.2fs",
        result.repetition_count,
        result.avg_pipe_time_s,
    )

    passed, message = parser.validate_threshold(result, expected_results, gpu_type)
    log.info("%s", message)

    try:
        server_nodes, _ = _distributed_server_nodes(cluster_dict, inference_dict)
        hostname_out = s_phdl.exec("hostname", print_console=False)
        results_summary = []
        for node in server_nodes:
            label = (hostname_out.get(node, "") or "").strip() or node
            results_summary.append(
                {
                    "label": label,
                    "avg_pipe_time_s": result.avg_pipe_time_s,
                    "passed": passed,
                }
            )
        log_results_summary(
            results_summary,
            metric_key="avg_pipe_time_s",
            title="Distributed diffusers xFuser results summary",
        )
    except Exception as exc:
        log.warning("Could not build distributed results summary: %s", exc)

    if not passed:
        fail_test(message)

    update_test_result()
