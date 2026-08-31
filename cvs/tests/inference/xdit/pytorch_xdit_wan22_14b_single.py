"""
PyTorch XDit WAN 2.2 Image-to-Video A14B inference test.

Runs WAN 2.2 I2V-A14B PyTorch inference inside amdsiloai/pytorch-xdit container
and validates results against configured thresholds.

Supports native checkpoints (``Wan-AI/Wan2.2-I2V-A14B``) and Diffusers layouts
(``Wan-AI/Wan2.2-I2V-A14B-Diffusers``).

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import pytest
import re
import shlex
import socket
import subprocess
from typing import Optional

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
    resolve_wan_local_model_required_checks,
    verify_required_checks_on_nodes,
    wan_native_hf_snapshot_required_checks,
)
from cvs.lib.inference.xdit.pytorch_xdit_wan import WanOutputParser
from cvs.lib.inference.xdit.pytorch_xdit_wan_job import (
    launch_wan_benchmark,
    store_resolved_wan_model_format_from_index,
)

log = globals.log


def _is_local_target(target: str) -> bool:
    """
    Best-effort check whether a "target" refers to the current machine.

    Used to decide whether single-node execution should be local (no SSH) or remote via SSH.
    """
    if not target:
        return False

    target_norm = target.strip().lower()
    if target_norm in {"localhost", "127.0.0.1", "::1"}:
        return True

    # Hostname / FQDN match
    try:
        if target_norm in {socket.gethostname().lower(), socket.getfqdn().lower()}:
            return True
    except Exception:
        pass

    # IP address match against locally-resolvable addresses
    try:
        target_ip = socket.gethostbyname(target)
    except Exception:
        target_ip = None

    if target_ip:
        local_ips = set()
        try:
            for fam, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
                if fam in (socket.AF_INET, socket.AF_INET6) and sockaddr:
                    local_ips.add(sockaddr[0])
        except Exception:
            pass

        # Always include loopback
        local_ips.update({"127.0.0.1", "::1"})

        if target_ip in local_ips:
            return True

    return False


class LocalPssh:
    """
    Minimal drop-in replacement for `Pssh` that executes commands locally.

    This is used only when the target resolves to the current machine to avoid
    unnecessary SSH hops for true localhost single-node runs.
    """

    def __init__(self, host: str):
        self.host_list = [host]

    def exec(self, cmd: str, timeout=None, print_console=True):
        # Keep output format similar to Pssh.exec: return dict[host] -> combined output
        completed = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout if timeout is None else int(timeout),
        )
        out = (completed.stdout or "") + (completed.stderr or "")
        if print_console:
            log.info(f"cmd = {_redact_secrets(cmd)}")
            log.info("%s", out)
        return {self.host_list[0]: out}

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        # Run different commands; map 1:1 with host_list ordering
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
                log.info(f"cmd = {_redact_secrets(cmd)}")
                log.info("%s", out_str)
            out[host] = out_str
        return out


def _redact_secrets(s: str) -> str:
    """
    Best-effort redaction for secrets that may appear in command strings/logs.

    Currently redacts:
    - HF_TOKEN=...
    """
    if not s:
        return s
    # Replace HF_TOKEN=<anything until space> with HF_TOKEN=<redacted>
    return re.sub(r"(HF_TOKEN=)\\S+", r"\\1<redacted>", s)


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
    """
    Load and validate cluster configuration.

    Uses Pydantic schema for fail-fast validation.
    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    # Validate with Pydantic schema
    try:
        validated = ClusterConfigFile.model_validate(cluster_dict)
        log.info(f"Cluster config validated successfully: {len(validated.node_dict)} nodes")
    except Exception as e:
        log.error(f"Cluster config validation failed: {e}")
        pytest.fail(f"Invalid cluster configuration: {e}")

    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def wan_config_dict(training_config_file, cluster_dict):
    """
    Load and validate WAN inference configuration.

    Uses Pydantic schema for fail-fast validation of:
    - Required fields
    - Type correctness
    - Value ranges
    - Expected results structure
    """
    with open(training_config_file) as json_file:
        raw_config = json.load(json_file)

    # Validate with Pydantic schema BEFORE placeholder resolution
    # This catches structural issues and typos early
    try:
        validated_config = PytorchXditWanConfigFile.model_validate(raw_config)
        log.info("WAN config validated successfully")
        log.info(f"  Container: {validated_config.config.container_image}")
        log.info(f"  Model: {validated_config.config.model_repo}@{validated_config.config.model_rev}")
        log.info(f"  Benchmark steps: {validated_config.benchmark_params.wan22_i2v_a14b.num_benchmark_steps}")
    except Exception as e:
        log.error(f"WAN config validation failed: {e}")
        pytest.fail(f"Invalid WAN configuration: {e}")

    # Apply schema defaults from validation, then resolve placeholders
    validated_dict = validated_config.model_dump()
    config_dict = resolve_test_config_placeholders(validated_dict["config"], cluster_dict)
    benchmark_params = resolve_test_config_placeholders(validated_dict["benchmark_params"], cluster_dict)

    # Return resolved config
    return {"config": config_dict, "benchmark_params": benchmark_params}


@pytest.fixture(scope="module")
def inference_dict(wan_config_dict):
    """Extract main config section."""
    return wan_config_dict['config']


@pytest.fixture(scope="module")
def benchmark_params_dict(wan_config_dict):
    """Extract benchmark params section."""
    return wan_config_dict['benchmark_params']


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    """
    Load the Hugging Face access token from the file path specified in config.

    Returns empty string if not configured or file not found.
    """
    hf_token_file = inference_dict['hf_token_file']
    if not hf_token_file:
        return ""
    try:
        with open(hf_token_file, 'r') as fp:
            hf_token = fp.read().rstrip("\n")
        log.info("HF token loaded successfully")
        return hf_token
    except FileNotFoundError:
        log.warning(f"HF token file not found: {hf_token_file}")
        return ""
    except Exception as e:
        log.error(f"Error reading HF token file: {e}")
        return ""


@pytest.fixture(scope="module")
def s_phdl(cluster_dict):
    """Create and return a command execution handle for all cluster nodes."""
    node_list = list(cluster_dict['node_dict'].keys())
    env_vars = cluster_dict.get("env_vars")

    # Single-node mode: execute locally ONLY when the target actually refers to this machine.
    #
    # Rationale: users often specify a remote node IP/hostname in cluster.json even for a
    # single-node run. Always forcing local execution will run benchmarks on the login node
    # (no GPUs/ROCm) and silently "pass" until parsing fails.
    if len(node_list) == 1:
        target = node_list[0]
        if _is_local_target(target):
            log.info(f"Using local execution mode for single-node target {target}")
            return LocalPssh(host=target)
        log.info(f"Using parallel-ssh execution mode for single-node target {target}")
        return Pssh(
            log,
            [target],
            user=cluster_dict.get("username"),
            password=cluster_dict.get("password"),
            pkey=cluster_dict.get("priv_key_file"),
            env_vars=env_vars,
        )

    log.info(f"Using parallel-ssh execution mode for {len(node_list)} node(s)")
    return Pssh(
        log,
        node_list,
        user=cluster_dict.get('username'),
        password=cluster_dict.get('password'),
        pkey=cluster_dict.get('priv_key_file'),
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def gpu_type(s_phdl):
    """
    Detect GPU type from rocm-smi output.

    Used to select appropriate performance thresholds.
    """
    head_node = s_phdl.host_list[0]
    smi_out_dict = s_phdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    gpu_type = get_model_from_rocm_smi_output(smi_out)
    log.info(f"Detected GPU type: {gpu_type}")
    return gpu_type


# =============================================================================
# Test Cases
# =============================================================================


def test_cleanup_stale_containers(s_phdl, inference_dict):
    """
    Clean up potentially stale Docker containers before tests on all nodes.

    Kills the specific container and removes all containers/volumes across all nodes.
    """
    container_name = inference_dict['container_name']
    log.info(f"Cleaning up stale containers: {container_name} on {len(s_phdl.host_list)} node(s)")

    # Cleanup runs on all nodes in parallel via Pssh
    docker_lib.kill_docker_container(s_phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(s_phdl)

    log.info("Container cleanup completed on all nodes")


def _read_model_index_from_node(s_phdl, node: str, model_dir: str) -> Optional[dict]:
    """Read model_index.json from a remote model directory on one node."""
    index_path = f"{model_dir.rstrip('/')}/model_index.json"
    output = s_phdl.exec(
        f"cat {shlex.quote(index_path)}",
        print_console=False,
    ).get(node, "")
    if not (output or "").strip():
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        log.warning("Could not parse model_index.json from %s on %s", index_path, node)
        return None


def test_verify_hf_cache_or_download(s_phdl, inference_dict, hf_token):
    """
    Verify the model is present locally on all nodes (no downloads).

    This benchmark is intended for large-scale parallel runs (100s of nodes). We must
    avoid triggering Hugging Face downloads at runtime. Users should provide either:
    - an explicit local filesystem path in config['model_repo'] (preferred), or
    - a Hugging Face repo id in config['model_repo'] with the model already pre-cached
      under config['hf_home'] (offline mode).
    """
    globals.error_list = []

    model_repo = inference_dict['model_repo']
    model_rev = inference_dict['model_rev']
    hf_home = inference_dict['hf_home']

    log.info(f"Verifying model presence on {len(s_phdl.host_list)} node(s)")

    # Preferred mode: config supplies explicit host path to the checkpoint directory.
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
                f"Pre-stage the model on all nodes and set config['model_repo'] to that path."
            )
            update_test_result()
            return

        model_index = _read_model_index_from_node(s_phdl, s_phdl.host_list[0], host_model_path)
        if model_index:
            store_resolved_wan_model_format_from_index(inference_dict, model_index)

        verify_err = verify_required_checks_on_nodes(
            s_phdl,
            host_model_path,
            resolve_wan_local_model_required_checks(
                host_model_path,
                model_format=inference_dict.get("_resolved_wan_model_format"),
                model_repo=model_repo,
            ),
            layout_description="WAN",
        )
        if verify_err:
            fail_test(verify_err)
            update_test_result()
            return

        inference_dict["_resolved_model_mount_host"] = host_model_path
        inference_dict["_resolved_ckpt_dir_container"] = "/model"
        log.info(f"Using local model path: {host_model_path} (mounted to /model in container) on all nodes")
        update_test_result()
        return

    # Backward-compatible offline mode: config supplies HF repo id; model must already be cached under hf_home.
    model_path_safe = model_repo.replace("/", "--")
    snapshot_dir_host = f"{hf_home}/hub/models--{model_path_safe}/snapshots/{model_rev}"
    log.info(f"Checking for pre-cached snapshot at: {snapshot_dir_host} on all nodes")
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

    snapshot_checks = wan_native_hf_snapshot_required_checks(snapshot_dir_host, model_repo)
    if snapshot_checks is None and "diffusers" in str(model_repo).lower():
        snapshot_checks = build_diffusers_local_model_required_checks(snapshot_dir_host)
    if snapshot_checks:
        verify_err = verify_required_checks_on_nodes(
            s_phdl,
            snapshot_dir_host,
            snapshot_checks,
            layout_description="WAN HF snapshot",
        )
        if verify_err:
            fail_test(verify_err)
            update_test_result()
            return

    inference_dict["_resolved_ckpt_dir_container"] = f"/hf_home/hub/models--{model_path_safe}/snapshots/{model_rev}"
    model_index = _read_model_index_from_node(s_phdl, s_phdl.host_list[0], snapshot_dir_host)
    if model_index:
        store_resolved_wan_model_format_from_index(inference_dict, model_index)
    log.info(f"Using pre-cached snapshot: {inference_dict['_resolved_ckpt_dir_container']} on all nodes")

    update_test_result()


def test_run_wan22_benchmark(s_phdl, inference_dict, benchmark_params_dict, hf_token):
    """
    Run WAN 2.2 I2V-A14B benchmark (native or Diffusers layout) on all cluster nodes.
    """
    globals.error_list = []

    errors = launch_wan_benchmark(
        s_phdl,
        inference_dict,
        benchmark_params_dict,
        hf_token,
        distributed=False,
    )
    if errors:
        for err in errors:
            fail_test(err)

    update_test_result()


def test_parse_and_validate_results(s_phdl, inference_dict, benchmark_params_dict, gpu_type):
    """
    Parse benchmark outputs and validate against thresholds.

    Uses WanOutputParser to:
    - Locate rank0_step*.json files
    - Parse total_time values
    - Compute average
    - Verify artifact (video.mp4) exists
    - Validate against GPU-specific threshold
    """
    globals.error_list = []

    output_dir = inference_dict.get('_test_output_dir')
    if not output_dir:
        # Allow running this test standalone by deriving the output directory
        # from the configured output_base_dir and current hostname.
        try:
            head_node = s_phdl.host_list[0]
            hostname_out = s_phdl.exec('hostname', print_console=False)
            hostname = hostname_out.get(head_node, '').strip() or head_node
            output_base_dir = inference_dict.get('output_base_dir')
            if output_base_dir:
                output_dir = f"{output_base_dir}/wan_22_{hostname}_outputs"
                log.info(f"Derived output directory: {output_dir}")
        except Exception:
            output_dir = None

        if not output_dir:
            fail_test("Output directory not set by previous test and could not be derived")
            update_test_result()
            return

    node_count = len(getattr(s_phdl, "host_list", []) or [])

    # If running on multiple nodes, aggregate like `wan.sh`.
    # For single-node runs, do NOT aggregate across output_base_dir because it may contain
    # stale run directories from other nodes / previous executions.
    base_dir = inference_dict.get("output_base_dir")
    wan_params = benchmark_params_dict["wan22_i2v_a14b"]
    expected_results = wan_params["expected_results"]
    require_video = bool(wan_params.get("require_video_artifact", True))
    expected_artifact = "video.mp4" if require_video else ""

    agg, agg_errors = None, []
    if base_dir and node_count > 1:
        # Filter aggregation to the current nodes only (avoid mixing with stale dirs).
        try:
            hostnames = s_phdl.exec("hostname", print_console=False)
            expected_dirnames = []
            for _, hn in (hostnames or {}).items():
                h = (hn or "").strip()
                if h:
                    expected_dirnames.append(f"wan_22_{h}_outputs")
        except Exception:
            expected_dirnames = []

        agg, agg_errors = WanOutputParser.parse_runs_under_base_dir(
            base_dir=base_dir,
            expected_artifact=expected_artifact,
            run_glob="wan_22_*_outputs",
            require_artifact=require_video,
            allowed_run_dir_names=expected_dirnames or None,
        )
    elif not base_dir:
        agg_errors = ["output_base_dir not set in config; cannot aggregate runs"]

    for e in agg_errors:
        log.warning(f"Parse warning: {e}")

    if agg and agg.result_count > 1:
        # Print per-run lines + overall average in the same style as wan.sh
        for r in agg.per_run:
            log.info(f"{r.label} {r.avg_total_time_s:.2f}")
        log.info(f"Average {agg.overall_avg_total_time_s:.2f} 720P - {agg.result_count} results")

        # Validate using overall average
        overall_result = type("Tmp", (), {"avg_total_time_s": agg.overall_avg_total_time_s})()
        parser = WanOutputParser(output_dir, expected_artifact="video.mp4")  # only used for threshold selection
        passed, message = parser.validate_threshold(overall_result, expected_results, gpu_type)
        log.info("%s", message)
        if not passed:
            fail_test(message)
        update_test_result()
        return

    # Fallback: single-run behavior (existing logic)
    log.info(f"Parsing results from: {output_dir}")
    parser = WanOutputParser(output_dir, expected_artifact="video.mp4" if require_video else "video.mp4")
    result, errors = parser.parse()

    for error in errors:
        log.warning(f"Parse warning: {error}")

    if result is None:
        fail_test(f"Failed to parse benchmark results: {errors}")
        update_test_result()
        return

    if require_video and not result.artifact_path:
        fail_test(f"Artifact 'video.mp4' not found under {output_dir}")
    elif result.artifact_path:
        log.info(f"Artifact found: {result.artifact_path}")

    log.info("Benchmark results:")
    log.info(f"  Steps parsed: {result.step_count}")
    log.info(f"  Average total_time: {result.avg_total_time_s:.2f}s")
    log.info(f"  Step times: {[f'{t:.2f}' for t in result.step_times]}")

    passed, message = parser.validate_threshold(result, expected_results, gpu_type)
    log.info("%s", message)
    if not passed:
        fail_test(message)
    update_test_result()
