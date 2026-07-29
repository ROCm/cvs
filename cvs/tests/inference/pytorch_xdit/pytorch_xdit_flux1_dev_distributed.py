"""
PyTorch XDit FLUX.1-dev unified multi-node distributed inference test.

Runs one coordinated FLUX.1-dev torchrun job across ``nnodes`` inside the
amdsiloai/pytorch-xdit container and validates results against configured thresholds.

Requires:
  - config ``nnodes >= 2`` with matching parallel-degree product
  - full model staged on every participating server node
  - NCCL/network settings appropriate for the cluster

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import pytest
import re
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
from cvs.parsers.schemas import ClusterConfigFile, PytorchXditFluxConfigFile
from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux import FluxOutputParser
from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import (
    launch_flux_benchmark,
    resolve_nnodes,
    resolve_server_nodes,
    validate_flux_parallelism_config,
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
        raise ValueError(
            f"Cluster/server_node_list has {len(nodes)} node(s) but nnodes={nnodes}"
        )
    return nodes[:nnodes], nnodes


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """Retrieve the --cluster_file CLI option provided to pytest."""
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def training_config_file(pytestconfig):
    """Retrieve the --config_file CLI option provided to pytest."""
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """Load and validate cluster configuration."""
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    try:
        validated = ClusterConfigFile.model_validate(cluster_dict)
        log.info(f"Cluster config validated successfully: {len(validated.node_dict)} nodes")
    except Exception as e:
        log.error(f"Cluster config validation failed: {e}")
        pytest.fail(f"Invalid cluster configuration: {e}")

    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def flux_config_dict(training_config_file, cluster_dict):
    """Load and validate Flux inference configuration."""
    with open(training_config_file) as json_file:
        raw_config = json.load(json_file)

    try:
        validated_config = PytorchXditFluxConfigFile.model_validate(raw_config)
        log.info("Flux config validated successfully")
        log.info(f"  Container: {validated_config.config.container_image}")
        log.info(f"  Model: {validated_config.config.model_repo}")
        if validated_config.config.model_rev:
            log.info(f"  Revision: {validated_config.config.model_rev}")
        log.info(f"  nnodes: {validated_config.config.nnodes}")
        log.info(f"  Repetitions: {validated_config.benchmark_params.flux1_dev_t2i.num_repetitions}")
    except Exception as e:
        log.error(f"Flux config validation failed: {e}")
        pytest.fail(f"Invalid Flux configuration: {e}")

    config_dict = raw_config["config"]
    benchmark_params = raw_config["benchmark_params"]

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    benchmark_params = resolve_test_config_placeholders(benchmark_params, cluster_dict)

    return {"config": config_dict, "benchmark_params": benchmark_params}


@pytest.fixture(scope="module")
def inference_dict(flux_config_dict):
    """Extract main config section."""
    return flux_config_dict["config"]


@pytest.fixture(scope="module")
def benchmark_params_dict(flux_config_dict):
    """Extract benchmark params section."""
    return flux_config_dict["benchmark_params"]


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    """Load the Hugging Face access token from the file path specified in config."""
    hf_token_file = inference_dict["hf_token_file"]
    if not hf_token_file:
        return _SecretValue("")
    try:
        with open(hf_token_file, "r") as fp:
            token = fp.read().rstrip("\n")
        log.info("HF token loaded successfully")
        return _SecretValue(token)
    except FileNotFoundError:
        log.warning(f"HF token file not found: {hf_token_file}")
        return _SecretValue("")
    except Exception as e:
        log.error(f"Error reading HF token file: {e}")
        return _SecretValue("")


@pytest.fixture(scope="module")
def s_phdl(cluster_dict, inference_dict):
    """
    Command handle scoped to participating distributed server nodes only.

    Uses ``server_node_list`` / ``nnodes`` from the inference config, not every
    node that happens to appear in cluster.json.
    """
    try:
        node_list, nnodes = _distributed_server_nodes(cluster_dict, inference_dict)
    except ValueError as e:
        pytest.fail(str(e))

    env_vars = cluster_dict.get("env_vars")
    log.info(
        "Using parallel-ssh execution mode for distributed job on %d node(s): %s",
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
    """Detect GPU type from rocm-smi output on rank-0 server node."""
    head_node = s_phdl.host_list[0]
    smi_out_dict = s_phdl.exec("rocm-smi -a | head -30")
    smi_out = smi_out_dict[head_node]
    gpu_type = get_model_from_rocm_smi_output(smi_out)
    log.info(f"Detected GPU type: {gpu_type}")
    return gpu_type


# =============================================================================
# Test Cases
# =============================================================================


def test_cleanup_stale_containers(s_phdl, inference_dict, cluster_dict):
    """
    Clean up potentially stale Docker containers before tests on server nodes.

    Distributed runs use ranked container names ``{container_name}-rankN``.
    """
    globals.error_list = []

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

    log.info("Container cleanup completed on all server nodes")
    update_test_result()


def test_verify_hf_cache_or_download(s_phdl, inference_dict):
    """
    Verify the model is present locally on all participating server nodes (no downloads).

    Users should provide either:
    - an explicit local filesystem path in config['model_repo'] (preferred), or
    - a Hugging Face repo id with the model already pre-cached under config['hf_home'].
    """
    globals.error_list = []

    model_repo = inference_dict["model_repo"]
    model_rev = inference_dict.get("model_rev", "")
    hf_home = inference_dict["hf_home"]

    log.info(f"Verifying model presence on {len(s_phdl.host_list)} server node(s)")

    if isinstance(model_repo, str) and model_repo.strip().startswith("/"):
        host_model_path = model_repo.strip()
        check_cmd = f"test -d {shlex.quote(host_model_path)} && echo 'EXISTS' || echo 'MISSING'"
        check_result = s_phdl.exec(check_cmd)

        missing_nodes = []
        for node, output in check_result.items():
            if "EXISTS" not in (output or ""):
                missing_nodes.append(node)
                log.error(f"Local model path not found on {node}: {host_model_path}")
            else:
                log.info(f"Model found on {node}: {host_model_path}")

        if missing_nodes:
            fail_test(
                f"Local model path not found on {len(missing_nodes)} node(s): {', '.join(missing_nodes)}. "
                f"Pre-stage the model on all server nodes and set config['model_repo'] to that path."
            )
            update_test_result()
            return

        required_checks = {
            "model_index.json": f"test -f {shlex.quote(host_model_path + '/model_index.json')} && echo OK || echo MISSING",
            "transformer/config.json": f"test -f {shlex.quote(host_model_path + '/transformer/config.json')} && echo OK || echo MISSING",
            "transformer weights": (
                f"test -f {shlex.quote(host_model_path + '/transformer/diffusion_pytorch_model.safetensors')} "
                f"-o -f {shlex.quote(host_model_path + '/transformer/diffusion_pytorch_model.safetensors.index.json')} "
                f"-o -f {shlex.quote(host_model_path + '/transformer/pytorch_model.bin')} "
                f"-o -f {shlex.quote(host_model_path + '/transformer/pytorch_model.bin.index.json')} "
                f"&& echo OK || echo MISSING"
            ),
            "vae/config.json": f"test -f {shlex.quote(host_model_path + '/vae/config.json')} && echo OK || echo MISSING",
            "vae weights": (
                f"test -f {shlex.quote(host_model_path + '/vae/diffusion_pytorch_model.safetensors')} "
                f"-o -f {shlex.quote(host_model_path + '/vae/diffusion_pytorch_model.safetensors.index.json')} "
                f"-o -f {shlex.quote(host_model_path + '/vae/pytorch_model.bin')} "
                f"-o -f {shlex.quote(host_model_path + '/vae/pytorch_model.bin.index.json')} "
                f"&& echo OK || echo MISSING"
            ),
        }

        for label, cmd in required_checks.items():
            res = s_phdl.exec(cmd, print_console=False)
            bad = [n for n, out in (res or {}).items() if "OK" not in (out or "")]
            if bad:
                fail_test(
                    "Local FLUX model directory appears incomplete for diffusers loading. "
                    f"Missing/invalid '{label}' on {len(bad)} node(s): {', '.join(bad)}. "
                    f"Model path: {host_model_path}. "
                    "Ensure the repo contains full weights (especially transformer weights), not just configs."
                )
                update_test_result()
                return

        inference_dict["_resolved_model_mount_host"] = host_model_path
        inference_dict["_resolved_model_path_container"] = "/model"
        log.info(f"Using local model path: {host_model_path} (mounted to /model in container) on all server nodes")
        update_test_result()
        return

    model_path_safe = model_repo.replace("/", "--")
    model_dir_host = f"{hf_home}/hub/models--{model_path_safe}"
    snapshots_dir_host = f"{model_dir_host}/snapshots"

    if model_rev:
        snapshot_dir_host = f"{snapshots_dir_host}/{model_rev}"
        log.info(f"Checking for pre-cached snapshot at: {snapshot_dir_host} on all server nodes")
        check_cmd = f"test -d {shlex.quote(snapshot_dir_host)} && echo 'EXISTS' || echo 'MISSING'"
        check_result = s_phdl.exec(check_cmd)

        missing_nodes = []
        for node, output in check_result.items():
            if "EXISTS" not in (output or ""):
                missing_nodes.append(node)
                log.error(f"Pre-cached model snapshot not found on {node}: {snapshot_dir_host}")
            else:
                log.info(f"Pre-cached model snapshot found on {node}: {snapshot_dir_host}")

        if missing_nodes:
            fail_test(
                f"Pre-cached model snapshot not found on {len(missing_nodes)} node(s): {', '.join(missing_nodes)}. "
                f"Pre-populate HF cache under {hf_home} (no downloads are performed by this test)."
            )
            update_test_result()
            return

        inference_dict["_resolved_model_path_container"] = (
            f"/hf_home/hub/models--{model_path_safe}/snapshots/{model_rev}"
        )
        log.info(f"Using pre-cached snapshot: {inference_dict['_resolved_model_path_container']} on all server nodes")
        update_test_result()
        return

    head_node = s_phdl.host_list[0]
    log.info(f"Checking for any pre-cached snapshot under: {snapshots_dir_host} on {head_node}")
    list_cmd = f"test -d {shlex.quote(snapshots_dir_host)} && ls -1 {shlex.quote(snapshots_dir_host)} | head -1 || true"
    list_out = s_phdl.exec(list_cmd).get(head_node, "") or ""
    snapshot_id = (list_out.strip().splitlines() or [""])[0].strip()
    if not snapshot_id:
        fail_test(
            f"No pre-cached snapshots found under {snapshots_dir_host} on {head_node}. "
            f"Pre-populate HF cache under {hf_home} or set config['model_repo'] to a local model path."
        )
        update_test_result()
        return

    snapshot_dir_host = f"{snapshots_dir_host}/{snapshot_id}"
    log.info(f"Verifying snapshot {snapshot_id} exists on all server nodes")
    check_cmd = f"test -d {shlex.quote(snapshot_dir_host)} && echo 'EXISTS' || echo 'MISSING'"
    check_result = s_phdl.exec(check_cmd)

    missing_nodes = []
    for node, output in check_result.items():
        if "EXISTS" not in (output or ""):
            missing_nodes.append(node)
            log.error(f"Snapshot {snapshot_id} not found on {node}")
        else:
            log.info(f"Snapshot {snapshot_id} found on {node}")

    if missing_nodes:
        fail_test(
            f"Snapshot {snapshot_id} not found on {len(missing_nodes)} node(s): {', '.join(missing_nodes)}. "
            f"Pre-populate HF cache on all server nodes."
        )
        update_test_result()
        return

    inference_dict["_resolved_model_path_container"] = (
        f"/hf_home/hub/models--{model_path_safe}/snapshots/{snapshot_id}"
    )
    log.info(f"Using pre-cached snapshot: {inference_dict['_resolved_model_path_container']} on all server nodes")

    update_test_result()


def test_verify_parallelism_config(cluster_dict, inference_dict, benchmark_params_dict):
    """Fail fast if xDiT parallel degrees do not match nnodes × torchrun_nproc."""
    globals.error_list = []
    err = validate_flux_parallelism_config(
        inference_dict,
        benchmark_params_dict,
        distributed=True,
        cluster_dict=cluster_dict,
    )
    if err:
        fail_test(err)
    update_test_result()


def test_run_flux1_benchmark(s_phdl, cluster_dict, inference_dict, benchmark_params_dict, hf_token):
    """
    Run unified multi-node FLUX.1-dev benchmark via torchrun across server nodes.
    """
    globals.error_list = []
    for msg in launch_flux_benchmark(
        s_phdl,
        inference_dict,
        benchmark_params_dict,
        hf_token,
        distributed=True,
        cluster_dict=cluster_dict,
    ):
        fail_test(msg)
    update_test_result()


def test_parse_and_validate_results(s_phdl, cluster_dict, inference_dict, benchmark_params_dict, gpu_type):
    """
    Parse rank-0 benchmark output and validate against thresholds.

    Unified distributed runs write timing.json/images to a single shared output dir
    on rank 0: ``flux_{rank0_hostname}_outputs``.
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
            output_dir = f"{output_base_dir}/flux_{rank0_hostname}_outputs"
            log.info(f"Derived rank-0 output directory: {output_dir}")
        except Exception as e:
            fail_test(f"Could not determine rank-0 Flux output directory: {e}")
            update_test_result()
            return

    log.info(f"Parsing results from: {output_dir}")
    parser = FluxOutputParser(output_dir, expected_image_pattern="flux_*.png")
    result, errors = parser.parse()

    for error in errors:
        log.warning(f"Parse warning: {error}")

    if result is None:
        fail_test(f"Failed to parse benchmark results from {output_dir}: {errors}")
        update_test_result()
        return

    if not result.image_paths:
        log.warning(f"No images (flux_*.png) found under {output_dir}")
    else:
        log.info(f"Found {len(result.image_paths)} generated images")

    log.info("Benchmark results:")
    log.info(f"  Repetitions parsed: {result.repetition_count}")
    log.info(f"  Average pipe_time: {result.avg_pipe_time_s:.2f}s")

    flux_params = benchmark_params_dict["flux1_dev_t2i"]
    expected_results = flux_params["expected_results"]
    passed, message = parser.validate_threshold(result, expected_results, gpu_type)
    log.info("%s", message)

    if not passed:
        fail_test(message)

    update_test_result()