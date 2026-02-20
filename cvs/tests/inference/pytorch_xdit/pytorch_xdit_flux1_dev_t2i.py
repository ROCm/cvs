"""
PyTorch XDit FLUX.1-dev Text-to-Image inference test.

Runs FLUX.1-dev PyTorch inference inside amdsiloai/pytorch-xdit container
and validates results against configured thresholds.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import pytest
import re
import socket
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from cvs.parsers.pytorch_xdit_flux import FluxOutputParser

log = globals.log


class _SecretValue:
    """
    Wrapper to avoid leaking secrets in pytest tracebacks.

    Pytest will include fixture values in failure reports; by wrapping the token, we ensure
    the token's repr is redacted while still behaving like a string for command building.
    """

    def __init__(self, value: str):
        self.value = value or ""

    def __bool__(self) -> bool:  # truthiness checks like `if hf_token:`
        return bool(self.value)

    def __str__(self) -> str:  # f-strings and command assembly
        return self.value

    def __repr__(self) -> str:  # pytest failure display
        return "<redacted>"


def _redact_secrets(s: str) -> str:
    """
    Best-effort redaction for secrets that may appear in command strings/logs.

    Currently redacts:
    - HF_TOKEN=...
    """
    if not s:
        return s
    # Replace HF_TOKEN=<anything until space> with HF_TOKEN=<redacted>
    return re.sub(r"(HF_TOKEN=)[^\s]+", r"\1<redacted>", s)


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

    This is needed on HPC systems where SSH authentication works interactively
    (Kerberos/certs/hostbased) but libssh2-based clients (parallel-ssh) cannot
    authenticate non-interactively.
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
            print(f"cmd = {_redact_secrets(cmd)}")
            print(out)
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
                print(f"cmd = {_redact_secrets(cmd)}")
                print(out_str)
            out[host] = out_str
        return out


class OpenSshPssh:
    """
    Drop-in replacement for `Pssh` that executes commands via the system `ssh` client.

    This is much more compatible with HPC environments than libssh2-based clients
    (parallel-ssh), and supports:
    - ssh-agent
    - Kerberos/SSSD-style usernames (e.g., user@realm)
    - ProxyJump and other ~/.ssh/config behaviors
    """

    def __init__(self, host: str, user: str | None = None, pkey: str | None = None):
        self.host_list = [host]
        self.user = user
        self.pkey = pkey

    def _dest(self, host: str) -> str:
        return f"{self.user}@{host}" if self.user else host

    def _ssh_args(self, host: str) -> list[str]:
        args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
        ]
        if self.pkey:
            args += ["-i", self.pkey]
        args.append(self._dest(host))
        return args

    def exec(self, cmd: str, timeout=None, print_console=True):
        host = self.host_list[0]
        # IMPORTANT: ssh concatenates argv into a single remote command string without
        # escaping. If we pass ["bash","-lc",cmd], bash will receive only the first word
        # of cmd as the -c payload (e.g., "docker"), and the remaining words as $0/$1...
        # which leads to running `docker` with no args (prints docker help).
        remote_cmd = f"bash -lc {shlex.quote(cmd)}"
        ssh_cmd = self._ssh_args(host) + [remote_cmd]
        completed = subprocess.run(
            ssh_cmd,
            text=True,
            capture_output=True,
            timeout=timeout if timeout is None else int(timeout),
        )
        out = (completed.stdout or "") + (completed.stderr or "")
        if print_console:
            printable = " ".join(shlex.quote(a) for a in ssh_cmd[:-1]) + " " + shlex.quote(_redact_secrets(cmd))
            print(f"cmd = {printable}")
            print(out)
        return {host: out}

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        out = {}
        for host, cmd in zip(self.host_list, cmd_list):
            remote_cmd = f"bash -lc {shlex.quote(cmd)}"
            ssh_cmd = self._ssh_args(host) + [remote_cmd]
            completed = subprocess.run(
                ssh_cmd,
                text=True,
                capture_output=True,
                timeout=timeout if timeout is None else int(timeout),
            )
            out_str = (completed.stdout or "") + (completed.stderr or "")
            if print_console:
                printable = " ".join(shlex.quote(a) for a in ssh_cmd[:-1]) + " " + shlex.quote(_redact_secrets(cmd))
                print(f"cmd = {printable}")
                print(out_str)
            out[host] = out_str
        return out


class OpenSshMultiPssh:
    """
    Multi-host drop-in replacement for `Pssh` using the system `ssh` client.

    parallel-ssh/libssh2 commonly fails to authenticate on HPC environments where OpenSSH works.
    This class runs per-host SSH commands concurrently with a bounded thread pool.
    """

    def __init__(self, hosts: list[str], user: str | None = None, pkey: str | None = None, max_workers: int = 32):
        self.host_list = hosts
        self.user = user
        self.pkey = pkey
        self.max_workers = max_workers

    def _dest(self, host: str) -> str:
        return f"{self.user}@{host}" if self.user else host

    def _ssh_args(self, host: str) -> list[str]:
        args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
        ]
        if self.pkey:
            args += ["-i", self.pkey]
        args.append(self._dest(host))
        return args

    def _run_one(self, host: str, cmd: str, timeout=None, print_console=True) -> tuple[str, str]:
        remote_cmd = f"bash -lc {shlex.quote(cmd)}"
        ssh_cmd = self._ssh_args(host) + [remote_cmd]
        completed = subprocess.run(
            ssh_cmd,
            text=True,
            capture_output=True,
            timeout=timeout if timeout is None else int(timeout),
        )
        out_str = (completed.stdout or "") + (completed.stderr or "")
        if print_console:
            printable = " ".join(shlex.quote(a) for a in ssh_cmd[:-1]) + " " + shlex.quote(_redact_secrets(cmd))
            print(f"cmd = {printable}")
            print(out_str)
        return host, out_str

    def exec(self, cmd: str, timeout=None, print_console=True):
        return self.exec_cmd_list([cmd] * len(self.host_list), timeout=timeout, print_console=print_console)

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        if len(cmd_list) != len(self.host_list):
            raise ValueError(f"cmd_list length ({len(cmd_list)}) must match host_list length ({len(self.host_list)})")
        out: dict[str, str] = {}
        workers = min(int(self.max_workers), max(1, len(self.host_list)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(self._run_one, host, cmd, timeout, print_console)
                for host, cmd in zip(self.host_list, cmd_list)
            ]
            for f in as_completed(futs):
                host, out_str = f.result()
                out[host] = out_str
        return out


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

    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def flux_config_dict(training_config_file, cluster_dict):
    """
    Load and validate Flux inference configuration.

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
        validated_config = PytorchXditFluxConfigFile.model_validate(raw_config)
        log.info("Flux config validated successfully")
        log.info(f"  Container: {validated_config.config.container_image}")
        log.info(f"  Model: {validated_config.config.model_repo}")
        if validated_config.config.model_rev:
            log.info(f"  Revision: {validated_config.config.model_rev}")
        log.info(f"  Repetitions: {validated_config.benchmark_params.flux1_dev_t2i.num_repetitions}")
    except Exception as e:
        log.error(f"Flux config validation failed: {e}")
        pytest.fail(f"Invalid Flux configuration: {e}")

    # Now resolve placeholders in the validated structure
    config_dict = raw_config['config']
    benchmark_params = raw_config['benchmark_params']

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    benchmark_params = resolve_test_config_placeholders(benchmark_params, cluster_dict)

    # Return resolved config
    return {"config": config_dict, "benchmark_params": benchmark_params}


@pytest.fixture(scope="module")
def inference_dict(flux_config_dict):
    """Extract main config section."""
    return flux_config_dict['config']


@pytest.fixture(scope="module")
def benchmark_params_dict(flux_config_dict):
    """Extract benchmark params section."""
    return flux_config_dict['benchmark_params']


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    """
    Load the Hugging Face access token from the file path specified in config.

    Returns empty string if not configured or file not found.
    """
    hf_token_file = inference_dict['hf_token_file']
    if not hf_token_file:
        return _SecretValue("")
    try:
        with open(hf_token_file, 'r') as fp:
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
def s_phdl(cluster_dict):
    """Create and return a command execution handle for all cluster nodes."""
    node_list = list(cluster_dict['node_dict'].keys())

    # Single-node mode: execute locally ONLY when the target actually refers to this machine.
    #
    # Rationale: users often specify a remote node IP/hostname in cluster.json even for a
    # single-node run. Always forcing local execution will run benchmarks on the login node
    # (no GPUs/ROCm) and fail in confusing ways.
    if len(node_list) == 1:
        target = node_list[0]
        if _is_local_target(target):
            log.info(f"Using local execution mode for single-node target {target}")
            return LocalPssh(host=target)
        # parallel-ssh/libssh2 commonly fails in environments where OpenSSH works.
        log.info(f"Using OpenSSH execution mode for single-node target {target}")
        return OpenSshPssh(
            host=target,
            user=cluster_dict.get("username"),
            pkey=cluster_dict.get("priv_key_file"),
        )

    # Multi-node mode: prefer system OpenSSH for robustness; fall back to parallel-ssh only if it works.
    try:
        p = Pssh(
            log,
            node_list,
            user=cluster_dict.get('username'),
            password=cluster_dict.get('password'),
            pkey=cluster_dict.get('priv_key_file'),
        )
        # Smoke-test authentication quickly; if it fails, we'll fall back.
        p.exec("true", timeout=10, print_console=False)
        return p
    except Exception as e:
        log.warning(f"parallel-ssh authentication failed; falling back to OpenSSH for multi-node: {e}")
        return OpenSshMultiPssh(
            hosts=node_list,
            user=cluster_dict.get("username"),
            pkey=cluster_dict.get("priv_key_file"),
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
    model_rev = inference_dict.get('model_rev', '')
    hf_home = inference_dict['hf_home']

    log.info(f"Verifying model presence on {len(s_phdl.host_list)} node(s)")

    # Preferred mode: config supplies explicit host path to the model directory.
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

        # Validate that the local model directory looks complete enough for diffusers to load.
        #
        # We've seen cases where only configs are present (e.g., transformer/config.json) but the
        # heavy weights are missing; then the container crashes and no timing.json is produced.
        required_checks = {
            "model_index.json": f"test -f {shlex.quote(host_model_path + '/model_index.json')} && echo OK || echo MISSING",
            "transformer/config.json": f"test -f {shlex.quote(host_model_path + '/transformer/config.json')} && echo OK || echo MISSING",
            # Weight file(s) may be single safetensors or sharded with an index.json, depending on how it was staged.
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

        # We'll mount this host path into the container at /model for consistent access.
        inference_dict["_resolved_model_mount_host"] = host_model_path
        inference_dict["_resolved_model_path_container"] = "/model"
        log.info(f"Using local model path: {host_model_path} (mounted to /model in container) on all nodes")
        update_test_result()
        return

    # Backward-compatible offline mode: config supplies HF repo id; model must already be cached under hf_home.
    model_path_safe = model_repo.replace("/", "--")
    model_dir_host = f"{hf_home}/hub/models--{model_path_safe}"
    snapshots_dir_host = f"{model_dir_host}/snapshots"

    if model_rev:
        snapshot_dir_host = f"{snapshots_dir_host}/{model_rev}"
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

        inference_dict["_resolved_model_path_container"] = (
            f"/hf_home/hub/models--{model_path_safe}/snapshots/{model_rev}"
        )
        log.info(f"Using pre-cached snapshot: {inference_dict['_resolved_model_path_container']} on all nodes")
        update_test_result()
        return

    # No revision specified: require at least one snapshot and pick the first one.
    # Check on first node only for this case (snapshot discovery)
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

    # Now verify the discovered snapshot exists on all nodes
    snapshot_dir_host = f"{snapshots_dir_host}/{snapshot_id}"
    log.info(f"Verifying snapshot {snapshot_id} exists on all nodes")
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
            f"Pre-populate HF cache on all nodes."
        )
        update_test_result()
        return

    inference_dict["_resolved_model_path_container"] = f"/hf_home/hub/models--{model_path_safe}/snapshots/{snapshot_id}"
    log.info(f"Using pre-cached snapshot: {inference_dict['_resolved_model_path_container']} on all nodes")

    update_test_result()


def test_run_flux1_benchmark(s_phdl, inference_dict, benchmark_params_dict, hf_token):
    """
    Run FLUX.1-dev text-to-image benchmark inside pytorch-xdit container on all nodes in parallel.

    Executes torchrun with configured parameters and mounts:
    - HF cache to /hf_home
    - Output directory to /outputs
    """
    globals.error_list = []

    # Preflight: ensure all nodes have GPU-capable hardware. Running on a login node (no /dev/kfd)
    # will cause ROCm + container init to fail and produce no timing.json.
    log.info(f"Checking /dev/kfd on {len(s_phdl.host_list)} node(s)")
    kfd_check = s_phdl.exec("test -e /dev/kfd && echo KFD_OK || echo KFD_MISSING", print_console=False)
    missing_kfd_nodes = []
    for node, output in kfd_check.items():
        if "KFD_OK" not in (output or ""):
            missing_kfd_nodes.append(node)
            log.error(f"ROCm device node /dev/kfd not found on {node}")
        else:
            log.info(f"/dev/kfd found on {node}")

    if missing_kfd_nodes:
        fail_test(
            f"ROCm device node /dev/kfd not found on {len(missing_kfd_nodes)} node(s): {', '.join(missing_kfd_nodes)}. "
            f"This test must be run on GPU compute nodes (e.g., via an interactive SLURM allocation)."
        )
        update_test_result()
        return

    container_image = inference_dict['container_image']
    container_name = inference_dict['container_name']
    hf_home = inference_dict['hf_home']
    output_base_dir = inference_dict['output_base_dir']
    # Prefer the resolved container model path computed in test_verify_hf_cache_or_download.
    model_repo = inference_dict.get('_resolved_model_path_container') or inference_dict['model_repo']

    # Get benchmark parameters
    flux_params = benchmark_params_dict['flux1_dev_t2i']
    prompt = flux_params['prompt']
    seed = flux_params['seed']
    num_inference_steps = flux_params['num_inference_steps']
    max_sequence_length = flux_params['max_sequence_length']
    no_use_resolution_binning = flux_params['no_use_resolution_binning']
    warmup_steps = flux_params['warmup_steps']
    warmup_calls = flux_params['warmup_calls']
    num_repetitions = flux_params['num_repetitions']
    height = flux_params['height']
    width = flux_params['width']
    ulysses_degree = flux_params['ulysses_degree']
    ring_degree = flux_params['ring_degree']
    use_torch_compile = flux_params['use_torch_compile']
    torchrun_nproc = flux_params['torchrun_nproc']

    # Get hostnames from all nodes
    log.info(f"Getting hostnames from {len(s_phdl.host_list)} node(s)")
    hostname_result = s_phdl.exec('hostname')
    node_to_hostname = {node: hostname_result[node].strip() for node in s_phdl.host_list}

    # Build common docker command components
    device_list = inference_dict['container_config']['device_list']
    volume_dict = inference_dict['container_config']['volume_dict']
    env_dict = inference_dict['container_config']['env_dict']

    # Build device arguments
    device_args = " ".join([f"--device={dev}" for dev in device_list])

    # Build environment arguments (common to all nodes)
    env_dict_full = env_dict.copy()
    env_dict_full['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    env_dict_full['OMP_NUM_THREADS'] = '16'
    env_dict_full['HF_HOME'] = '/hf_home'
    if hf_token:
        env_dict_full['HF_TOKEN'] = hf_token
    env_args = " ".join([f"-e {key}={value}" for key, value in env_dict_full.items()])

    # Build torchrun command (common to all nodes)
    resolution_binning_flag = "" if no_use_resolution_binning else ""
    if no_use_resolution_binning:
        resolution_binning_flag = "--no_use_resolution_binning"

    compile_flag = "--use-torch-compile" if use_torch_compile else ""

    torchrun_cmd = (
        f"torchrun --nproc_per_node={torchrun_nproc} /app/Flux/run_usp.py "
        f"--model \"{model_repo}\" "
        f"--prompt \"{prompt}\" "
        f"--seed {seed} "
        f"--num_inference_steps {num_inference_steps} "
        f"--max_sequence_length {max_sequence_length} "
        f"{resolution_binning_flag} "
        f"--warmup_steps {warmup_steps} "
        f"--warmup_calls {warmup_calls} "
        f"--num_repetitions {num_repetitions} "
        f"--height {height} --width {width} "
        f"--ulysses_degree {ulysses_degree} "
        f"--ring_degree {ring_degree} "
        f"{compile_flag} "
        f"--benchmark_output_directory /outputs"
    )

    # Create per-node output directories and build per-node docker commands
    mkdir_cmds = []
    docker_cmds = []

    for node in s_phdl.host_list:
        hostname = node_to_hostname[node]
        output_dir = f"{output_base_dir}/flux_{hostname}_outputs"

        # Create output directory command
        mkdir_cmds.append(f"mkdir -p {output_dir}")

        # Build volume arguments with per-node output directory
        volume_dict_full = volume_dict.copy()
        volume_dict_full[output_dir] = "/outputs"
        volume_dict_full[hf_home] = "/hf_home"
        # If user provided an explicit local model path, mount it consistently to /model.
        if inference_dict.get("_resolved_model_mount_host"):
            volume_dict_full[inference_dict["_resolved_model_mount_host"]] = "/model"
        volume_args = " ".join(
            [f"--mount type=bind,source={src},target={dst}" for src, dst in volume_dict_full.items()]
        )

        # Full docker command for this node
        docker_cmd = (
            f"docker run "
            f"--cap-add=SYS_PTRACE "
            f"--security-opt seccomp=unconfined "
            f"--user root "
            f"{device_args} "
            f"--ipc=host "
            f"--network host "
            f"--rm "
            f"--privileged "
            f"--name {container_name} "
            f"{volume_args} "
            f"{env_args} "
            f"{container_image} "
            f"{torchrun_cmd}"
        )
        docker_cmds.append(docker_cmd)
        log.info(f"Node {node} ({hostname}) will write to: {output_dir}")

    # Create output directories on all nodes in parallel
    log.info(f"Creating output directories on {len(s_phdl.host_list)} node(s)")
    s_phdl.exec_cmd_list(mkdir_cmds)

    log.info(f"Running FLUX.1-dev benchmark on {len(s_phdl.host_list)} node(s) in parallel")
    log.debug(f"Docker command (sample): {_redact_secrets(docker_cmds[0])}")

    try:
        # Run benchmarks on all nodes in parallel
        log.info("Starting benchmarks (this may take several minutes)...")
        benchmark_results = s_phdl.exec_cmd_list(docker_cmds, timeout=1800)  # 30 min timeout

        log.info("Benchmarks completed on all nodes")

        # Check for common failure patterns on each node and fail fast.
        fatal_patterns = [
            r"\bTraceback\b",
            r"\bModuleNotFoundError\b",
            r"\bChildFailedError\b",
            r"\bOSError:\b",
        ]

        failed_nodes = []
        for node, output in benchmark_results.items():
            if any(re.search(p, output, re.I) for p in fatal_patterns):
                log.error(f"Benchmark output indicates a failure on {node} (see logs above).")
                failed_nodes.append(node)
            else:
                log.info(f"Benchmark on {node} completed successfully")

        if failed_nodes:
            fail_test(f"Benchmark failed on {len(failed_nodes)} node(s): {', '.join(failed_nodes)}")

    except Exception as e:
        fail_test(f"Benchmark execution failed with exception: {e}")

    # For convenience, store the single-node output dir so parsing can be strict and avoid
    # picking up stale outputs from previous runs on other hosts.
    if len(getattr(s_phdl, "host_list", []) or []) == 1:
        only_node = s_phdl.host_list[0]
        inference_dict["_test_output_dir"] = f"{output_base_dir}/flux_{node_to_hostname[only_node]}_outputs"

    update_test_result()


def test_parse_and_validate_results(s_phdl, inference_dict, benchmark_params_dict, gpu_type):
    """
    Parse benchmark outputs and validate against thresholds.

    Handles both single-node and multi-node runs:
    - Single node: parses flux_{hostname}_outputs
    - Multi-node: parses all flux_*_outputs directories and validates each

    Uses FluxOutputParser to:
    - Locate results/timing.json
    - Parse pipe_time values
    - Compute average
    - Verify images (flux_*.png) exist
    - Validate against GPU-specific threshold
    """
    globals.error_list = []

    node_count = len(getattr(s_phdl, "host_list", []) or [])

    # Determine output directories to parse.
    #
    # IMPORTANT: Do not glob all flux_*_outputs under output_base_dir for single-node runs,
    # because output_base_dir often contains stale directories from prior runs on other hosts.
    output_base_dir = inference_dict.get('output_base_dir')
    if not output_base_dir:
        fail_test("output_base_dir not set in config; cannot locate benchmark results")
        update_test_result()
        return

    # If a prior test set an explicit output dir, prefer it (strictest).
    if inference_dict.get("_test_output_dir"):
        output_dirs = [inference_dict["_test_output_dir"]]
    else:
        # Otherwise, derive expected output dirs from the current nodes' hostnames.
        try:
            head_node = s_phdl.host_list[0]
            hostname_out = s_phdl.exec('hostname', print_console=False)
            expected_hostnames = []
            for node in s_phdl.host_list:
                hn = (hostname_out.get(node, "") or "").strip() or node
                expected_hostnames.append(hn)
        except Exception:
            # Fallback to head node only
            expected_hostnames = [head_node] if 'head_node' in locals() else []

        if not expected_hostnames:
            fail_test("Could not determine node hostnames to locate Flux outputs")
            update_test_result()
            return

        # Single-node: parse only that node's directory.
        if node_count <= 1:
            output_dirs = [f"{output_base_dir}/flux_{expected_hostnames[0]}_outputs"]
        else:
            output_dirs = [f"{output_base_dir}/flux_{hn}_outputs" for hn in expected_hostnames]

    log.info(f"Found {len(output_dirs)} output directory(ies) to parse")

    # Parse and validate each output directory
    all_passed = True
    flux_params = benchmark_params_dict['flux1_dev_t2i']
    expected_results = flux_params['expected_results']

    results_summary = []

    for output_dir in output_dirs:
        # Extract hostname label from directory name
        dir_name = output_dir.split('/')[-1]
        label = dir_name.replace('flux_', '').replace('_outputs', '')

        log.info(f"Parsing results from: {output_dir} ({label})")
        parser = FluxOutputParser(output_dir, expected_image_pattern="flux_*.png")
        result, errors = parser.parse()

        for error in errors:
            log.warning(f"Parse warning ({label}): {error}")

        if result is None:
            log.error(f"Failed to parse benchmark results for {label}: {errors}")
            all_passed = False
            continue

        if not result.image_paths:
            log.warning(f"No images (flux_*.png) found under {output_dir}")
        else:
            log.info(f"Found {len(result.image_paths)} generated images for {label}")

        log.info(f"Benchmark results ({label}):")
        log.info(f"  Repetitions parsed: {result.repetition_count}")
        log.info(f"  Average pipe_time: {result.avg_pipe_time_s:.2f}s")

        # Validate against threshold
        passed, message = parser.validate_threshold(result, expected_results, gpu_type)
        log.info(f"{label}: {message}")

        results_summary.append({'label': label, 'avg_pipe_time_s': result.avg_pipe_time_s, 'passed': passed})

        if not passed:
            all_passed = False

    # Print summary for multi-node runs
    if len(results_summary) > 1:
        log.info("=" * 60)
        log.info("Multi-node results summary:")
        for r in results_summary:
            status = "PASS" if r['passed'] else "FAIL"
            log.info(f"  {r['label']}: {r['avg_pipe_time_s']:.2f}s [{status}]")

        # Compute overall average
        avg_pipe_times = [r['avg_pipe_time_s'] for r in results_summary]
        overall_avg = sum(avg_pipe_times) / len(avg_pipe_times) if avg_pipe_times else 0
        log.info(f"  Overall average: {overall_avg:.2f}s")
        log.info(f"  Nodes passed: {sum(1 for r in results_summary if r['passed'])}/{len(results_summary)}")
        log.info("=" * 60)

    if not all_passed:
        fail_test("One or more nodes failed benchmark validation")

    update_test_result()
