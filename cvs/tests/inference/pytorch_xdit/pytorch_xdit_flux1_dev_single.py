"""
PyTorch XDit FLUX.1-dev Text-to-Image inference test (single master node).

Runs one independent FLUX.1-dev torchrun job on the cluster head/master node only
via ``launch_flux_benchmark(distributed=False)``.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import pytest
import re
import socket
import shlex
import subprocess

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
    validate_flux_parallelism_config,
    compute_world_size,
    parallel_product,
    build_output_cleanup_cmd, 
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

def _is_local_target(target: str) -> bool:
    if not target:
        return False
    target_norm = target.strip().lower()
    if target_norm in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        if target_norm in {socket.gethostname().lower(), socket.getfqdn().lower()}:
            return True
    except Exception:
        pass
    try:
        target_ip = socket.gethostbyname(target)
    except Exception:
        target_ip = None
    if target_ip:
        local_ips = {"127.0.0.1", "::1"}
        try:
            for fam, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
                if fam in (socket.AF_INET, socket.AF_INET6) and sockaddr:
                    local_ips.add(sockaddr[0])
        except Exception:
            pass
        if target_ip in local_ips:
            return True
    return False

class LocalPssh:
    """Minimal local drop-in for Pssh when the master node is this machine."""

    def __init__(self, host: str):
        self.host_list = [host]

    def exec(self, cmd: str, timeout=None, print_console=True):
        completed = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout if timeout is None else int(timeout),
        )
        out = (completed.stdout or "") + (completed.stderr or "")
        if print_console:
            log.info("cmd = %s", re.sub(r"(HF_TOKEN=)[^\s]+", r"\1<redacted>", cmd))
            log.info("%s", out)
        return {self.host_list[0]: out}

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        out = {}
        for host, cmd in zip(self.host_list, cmd_list):
            completed = subprocess.run(
                cmd,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout if timeout is None else int(timeout),
            )
            out_str = (completed.stdout or "") + (completed.stderr or "")
            if print_console:
                log.info("cmd = %s", re.sub(r"(HF_TOKEN=)[^\s]+", r"\1<redacted>", cmd))
                log.info("%s", out_str)
            out[host] = out_str
        return out

def _master_node(cluster_dict) -> str:
    """Return the cluster head/master node key from cluster.json."""
    head = (cluster_dict.get("head_node_dict") or {}).get("mgmt_ip")
    node_dict = cluster_dict.get("node_dict") or {}
    if head and head in node_dict:
        return head
    nodes = list(node_dict.keys())
    if not nodes:
        raise ValueError("cluster node_dict is empty")
    return nodes[0]

# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")

@pytest.fixture(scope="module")
def training_config_file(pytestconfig):
    return pytestconfig.getoption("config_file")

@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    try:
        validated = ClusterConfigFile.model_validate(cluster_dict)
        log.info("Cluster config validated successfully: %d nodes", len(validated.node_dict))
    except Exception as e:
        pytest.fail(f"Invalid cluster configuration: {e}")
    log.info("%s", cluster_dict)
    return cluster_dict

@pytest.fixture(scope="module")
def flux_config_dict(training_config_file, cluster_dict):
    with open(training_config_file) as json_file:
        raw_config = json.load(json_file)
    try:
        validated_config = PytorchXditFluxConfigFile.model_validate(raw_config)
        log.info("Flux config validated successfully")
        log.info("  Container: %s", validated_config.config.container_image)
        log.info("  Model: %s", validated_config.config.model_repo)
        if validated_config.config.nnodes:
            log.info("  Config nnodes: %s (single test uses master node only)", validated_config.config.nnodes)
        log.info("  Repetitions: %s", validated_config.benchmark_params.flux1_dev_t2i.num_repetitions)
    except Exception as e:
        pytest.fail(f"Invalid Flux configuration: {e}")

    config_dict = raw_config["config"]
    benchmark_params = raw_config["benchmark_params"]
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    benchmark_params = resolve_test_config_placeholders(benchmark_params, cluster_dict)
    return {"config": config_dict, "benchmark_params": benchmark_params}

@pytest.fixture(scope="module")
def inference_dict(flux_config_dict):
    return flux_config_dict["config"]

@pytest.fixture(scope="module")
def benchmark_params_dict(flux_config_dict):
    return flux_config_dict["benchmark_params"]

@pytest.fixture(scope="module")
def hf_token(inference_dict):
    hf_token_file = inference_dict.get("hf_token_file") or ""
    if not hf_token_file:
        return _SecretValue("")
    try:
        with open(hf_token_file, "r") as fp:
            token = fp.read().rstrip("\n")
        log.info("HF token loaded successfully")
        return _SecretValue(token)
    except FileNotFoundError:
        log.warning("HF token file not found: %s", hf_token_file)
        return _SecretValue("")
    except Exception as e:
        log.error("Error reading HF token file: %s", e)
        return _SecretValue("")

@pytest.fixture(scope="module")
def master_node(cluster_dict):
    node = _master_node(cluster_dict)
    log.info("Single-node FLUX will run on master node: %s", node)
    return node

@pytest.fixture(scope="module")
def s_phdl(cluster_dict, master_node):
    """Command handle scoped to the cluster master/head node only."""
    env_vars = cluster_dict.get("env_vars")
    if _is_local_target(master_node):
        log.info("Using local execution mode for master node %s", master_node)
        return LocalPssh(host=master_node)
    log.info("Using parallel-ssh execution mode for master node %s", master_node)
    return Pssh(
        log,
        [master_node],
        user=cluster_dict.get("username"),
        password=cluster_dict.get("password"),
        pkey=cluster_dict.get("priv_key_file"),
        env_vars=env_vars,
    )

@pytest.fixture(scope="module")
def gpu_type(s_phdl):
    head_node = s_phdl.host_list[0]
    smi_out = s_phdl.exec("rocm-smi -a | head -30")[head_node]
    gpu_type = get_model_from_rocm_smi_output(smi_out)
    log.info("Detected GPU type: %s", gpu_type)
    return gpu_type

# =============================================================================
# Test Cases
# =============================================================================

def test_cleanup_stale_containers(s_phdl, inference_dict):
    globals.error_list = []
    container_name = inference_dict["container_name"]
    log.info("Cleaning up stale containers: %s on master node %s", container_name, s_phdl.host_list[0])
    docker_lib.kill_docker_container(s_phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(s_phdl)

    output_base_dir = inference_dict.get("output_base_dir")
    if output_base_dir:
        cleanup_cmd = build_output_cleanup_cmd(output_base_dir, use_sudo=True)
        log.info("Cleaning stale FLUX outputs under %s", output_base_dir)
        s_phdl.exec(cleanup_cmd)
    
    log.info("Container cleanup completed")
    update_test_result()

def test_verify_hf_cache_or_download(s_phdl, inference_dict):
    """Verify model on master node only (no downloads)."""
    globals.error_list = []

    model_repo = inference_dict["model_repo"]
    model_rev = inference_dict.get("model_rev", "")
    hf_home = inference_dict["hf_home"]
    log.info("Verifying model presence on master node %s", s_phdl.host_list[0])

    if isinstance(model_repo, str) and model_repo.strip().startswith("/"):
        host_model_path = model_repo.strip()
        check_cmd = f"test -d {shlex.quote(host_model_path)} && echo 'EXISTS' || echo 'MISSING'"
        check_result = s_phdl.exec(check_cmd)
        missing_nodes = [n for n, out in check_result.items() if "EXISTS" not in (out or "")]
        if missing_nodes:
            fail_test(
                f"Local model path not found on master node: {host_model_path}. "
                "Pre-stage the model and set config['model_repo'] to that path."
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
                    f"Local FLUX model directory incomplete ('{label}') at {host_model_path} on master node."
                )
                update_test_result()
                return

        inference_dict["_resolved_model_mount_host"] = host_model_path
        inference_dict["_resolved_model_path_container"] = "/model"
        log.info("Using local model path: %s (mounted to /model)", host_model_path)
        update_test_result()
        return

    # HF cache path (unchanged logic, master node only via s_phdl)
    model_path_safe = model_repo.replace("/", "--")
    snapshots_dir_host = f"{hf_home}/hub/models--{model_path_safe}/snapshots"
    head_node = s_phdl.host_list[0]

    if model_rev:
        snapshot_dir_host = f"{snapshots_dir_host}/{model_rev}"
        check_cmd = f"test -d {shlex.quote(snapshot_dir_host)} && echo 'EXISTS' || echo 'MISSING'"
        if "EXISTS" not in (s_phdl.exec(check_cmd).get(head_node, "") or ""):
            fail_test(f"Pre-cached snapshot not found on master: {snapshot_dir_host}")
            update_test_result()
            return
        inference_dict["_resolved_model_path_container"] = (
            f"/hf_home/hub/models--{model_path_safe}/snapshots/{model_rev}"
        )
        update_test_result()
        return

    list_cmd = f"test -d {shlex.quote(snapshots_dir_host)} && ls -1 {shlex.quote(snapshots_dir_host)} | head -1 || true"
    snapshot_id = (s_phdl.exec(list_cmd).get(head_node, "") or "").strip().splitlines()[:1]
    snapshot_id = snapshot_id[0].strip() if snapshot_id else ""
    if not snapshot_id:
        fail_test(f"No pre-cached snapshots under {snapshots_dir_host} on master node")
        update_test_result()
        return

    snapshot_dir_host = f"{snapshots_dir_host}/{snapshot_id}"
    check_cmd = f"test -d {shlex.quote(snapshot_dir_host)} && echo 'EXISTS' || echo 'MISSING'"
    if "EXISTS" not in (s_phdl.exec(check_cmd).get(head_node, "") or ""):
        fail_test(f"Snapshot {snapshot_id} not found on master node")
        update_test_result()
        return

    inference_dict["_resolved_model_path_container"] = (
        f"/hf_home/hub/models--{model_path_safe}/snapshots/{snapshot_id}"
    )
    update_test_result()

def test_verify_parallelism_config(master_node, inference_dict, benchmark_params_dict):
    """Fail fast if xDiT parallel degrees do not match 1 node × torchrun_nproc."""
    globals.error_list = []

    flux_params = benchmark_params_dict["flux1_dev_t2i"]
    nproc = int(flux_params["torchrun_nproc"])
    world_size = compute_world_size(1, nproc)
    product = parallel_product(flux_params)

    ulysses = int(flux_params["ulysses_degree"])
    ring = int(flux_params["ring_degree"])
    pipefusion = int(flux_params.get("pipefusion_parallel_degree", 1))
    tp = int(flux_params.get("tensor_parallel_degree", 1))
    dp = int(flux_params.get("data_parallel_degree", 1))

    log.info("=" * 60)
    log.info("Single-node FLUX topology")
    log.info("=" * 60)
    log.info("Master node: %s", master_node)
    log.info("Participating nodes: 1 (single-node mode; config nnodes ignored for execution)")
    log.info("  rank 0 -> %s (%d GPUs)", master_node, nproc)
    log.info("GPUs per node (torchrun_nproc): %d", nproc)
    log.info("Total GPU ranks (world_size): %d = 1 node × %d nproc", world_size, nproc)
    log.info(
        "xDiT parallel layout: ulysses=%d × ring=%d × pipefusion=%d × tp=%d × dp=%d = %d",
        ulysses, ring, pipefusion, tp, dp, product,
    )
    log.info("Sequence-parallel size (ulysses × ring): %d", ulysses * ring)
    if inference_dict.get("nnodes") and int(inference_dict["nnodes"]) > 1:
        log.info(
            "Note: config nnodes=%s is for distributed runs; single test uses master only",
            inference_dict["nnodes"],
        )
    if product == world_size:
        log.info("Parallelism check: PASS (product %d == world_size %d)", product, world_size)
    log.info("=" * 60)

    err = validate_flux_parallelism_config(
        inference_dict,
        benchmark_params_dict,
        distributed=False,
    )
    if err:
        fail_test(err)
    update_test_result()

def test_run_flux1_benchmark(s_phdl, inference_dict, benchmark_params_dict, hf_token):
    """Run FLUX.1-dev benchmark on master node via shared job lib."""
    globals.error_list = []
    for msg in launch_flux_benchmark(
        s_phdl,
        inference_dict,
        benchmark_params_dict,
        hf_token,
        distributed=False,
    ):
        fail_test(msg)
    update_test_result()

def test_parse_and_validate_results(s_phdl, inference_dict, benchmark_params_dict, gpu_type):
    """Parse master-node output and validate against thresholds."""
    globals.error_list = []

    output_base_dir = inference_dict.get("output_base_dir")
    if not output_base_dir:
        fail_test("output_base_dir not set in config")
        update_test_result()
        return

    output_dir = inference_dict.get("_test_output_dir")
    if not output_dir:
        head_node = s_phdl.host_list[0]
        hostname_out = s_phdl.exec("hostname", print_console=False)
        hostname = (hostname_out.get(head_node, "") or "").strip() or head_node
        output_dir = f"{output_base_dir}/flux_{hostname}_outputs"
        log.info("Derived output directory: %s", output_dir)

    log.info("Parsing results from: %s", output_dir)
    parser = FluxOutputParser(output_dir, expected_image_pattern="flux_*.png")
    result, errors = parser.parse()

    for error in errors:
        log.warning("Parse warning: %s", error)

    if result is None:
        fail_test(f"Failed to parse benchmark results from {output_dir}: {errors}")
        update_test_result()
        return

    if not result.image_paths:
        log.warning("No images (flux_*.png) found under %s", output_dir)
    else:
        log.info("Found %d generated images", len(result.image_paths))

    log.info("Benchmark results:")
    log.info("  Repetitions parsed: %d", result.repetition_count)
    log.info("  Average pipe_time: %.2fs", result.avg_pipe_time_s)

    flux_params = benchmark_params_dict["flux1_dev_t2i"]
    passed, message = parser.validate_threshold(result, flux_params["expected_results"], gpu_type)
    log.info("%s", message)
    if not passed:
        fail_test(message)
    update_test_result()