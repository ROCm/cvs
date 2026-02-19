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

    Returns empty string if file not found (will be caught later if download needed).
    """
    hf_token_file = inference_dict['hf_token_file']
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

    # Default: use parallel-ssh for remote execution
    return Pssh(
        log,
        node_list,
        user=cluster_dict.get('username'),
        password=cluster_dict.get('password'),
        pkey=cluster_dict.get('priv_key_file'),
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
    Clean up potentially stale Docker containers before tests.

    Kills the specific container and removes all containers/volumes.
    """
    container_name = inference_dict['container_name']
    log.info(f"Cleaning up stale containers: {container_name}")

    docker_lib.kill_docker_container(s_phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(s_phdl)

    log.info("Container cleanup completed")


def test_verify_hf_cache_or_download(s_phdl, inference_dict, hf_token):
    """
    Verify HF model cache exists, or download if missing.

    Uses short-lived container with 'hf download' to populate cache.
    Fails if HF_TOKEN is missing and download is required.
    """
    globals.error_list = []

    model_repo = inference_dict['model_repo']
    model_rev = inference_dict.get('model_rev', '')
    hf_home = inference_dict['hf_home']

    # Construct expected model directory path
    model_path_safe = model_repo.replace("/", "--")
    model_dir = f"{hf_home}/hub/models--{model_path_safe}"

    head_node = s_phdl.host_list[0]

    # Check if model_rev is specified
    if model_rev:
        # Require specific snapshot directory
        snapshot_dir = f"{model_dir}/snapshots/{model_rev}"
        log.info(f"Checking for model cache at: {snapshot_dir}")

        check_cmd = f"test -d {snapshot_dir} && echo 'EXISTS' || echo 'MISSING'"
        check_result = s_phdl.exec(check_cmd)

        if "EXISTS" in check_result[head_node]:
            log.info(f"Model cache found at {snapshot_dir}")
            update_test_result()
            return

        log.info(f"Model cache not found, downloading {model_repo}@{model_rev}...")
    else:
        # No specific revision - check if any snapshot exists
        snapshots_dir = f"{model_dir}/snapshots"
        log.info(f"Checking for any model snapshot under: {snapshots_dir}")

        check_cmd = f"test -d {snapshots_dir} && ls {snapshots_dir} | head -1 && echo 'EXISTS' || echo 'MISSING'"
        check_result = s_phdl.exec(check_cmd)

        if "EXISTS" in check_result[head_node] and check_result[head_node].strip() != "MISSING":
            log.info(f"Model cache found under {snapshots_dir}")
            update_test_result()
            return

        log.info(f"Model cache not found, downloading {model_repo}...")

    # Require HF token for download
    if not hf_token:
        fail_test(
            f"HF token required to download {model_repo}. "
            f"Please ensure {inference_dict['hf_token_file']} exists and contains a valid token."
        )
        update_test_result()
        return

    # Run download container
    container_image = inference_dict['container_image']
    if model_rev:
        download_cmd = (
            f"docker run --rm "
            f"--mount type=bind,source={hf_home},target=/hf_home "
            f"-e HF_HOME=/hf_home "
            f"-e HF_TOKEN={hf_token} "
            f"{container_image} "
            f"hf download {model_repo} --revision {model_rev}"
        )
    else:
        download_cmd = (
            f"docker run --rm "
            f"--mount type=bind,source={hf_home},target=/hf_home "
            f"-e HF_HOME=/hf_home "
            f"-e HF_TOKEN={hf_token} "
            f"{container_image} "
            f"hf download {model_repo}"
        )

    log.info(f"Running download: {_redact_secrets(download_cmd)}")

    try:
        download_result = s_phdl.exec(download_cmd, timeout=1800)  # 30 min timeout
        log.info(f"Download output: {download_result[head_node]}")

        # Verify download succeeded
        if model_rev:
            verify_cmd = f"test -d {snapshot_dir} && echo 'EXISTS' || echo 'MISSING'"
        else:
            verify_cmd = f"test -d {snapshots_dir} && ls {snapshots_dir} | head -1 && echo 'EXISTS' || echo 'MISSING'"

        verify_result = s_phdl.exec(verify_cmd)
        if "EXISTS" not in verify_result[head_node]:
            fail_test("Model download failed: expected directory still missing after download")
    except Exception as e:
        fail_test(f"Model download failed with exception: {e}")

    update_test_result()


def test_run_flux1_benchmark(s_phdl, inference_dict, benchmark_params_dict, hf_token):
    """
    Run FLUX.1-dev text-to-image benchmark inside pytorch-xdit container.

    Executes torchrun with configured parameters and mounts:
    - HF cache to /hf_home
    - Output directory to /outputs
    """
    globals.error_list = []

    # Preflight: ensure we're on a GPU-capable host. Running on a login node (no /dev/kfd)
    # will cause ROCm + container init to fail and produce no timing.json.
    head_node = s_phdl.host_list[0]
    kfd_check = s_phdl.exec("test -e /dev/kfd && echo KFD_OK || echo KFD_MISSING", print_console=False)
    if "KFD_OK" not in (kfd_check.get(head_node, "") or ""):
        fail_test(
            "ROCm device node /dev/kfd not found on this host. "
            "This test must be run on a GPU compute node (e.g., via an interactive SLURM allocation)."
        )
        update_test_result()
        return

    container_image = inference_dict['container_image']
    container_name = inference_dict['container_name']
    hf_home = inference_dict['hf_home']
    output_base_dir = inference_dict['output_base_dir']
    model_repo = inference_dict['model_repo']

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

    # Create output directory
    head_node = s_phdl.host_list[0]
    hostname_result = s_phdl.exec('hostname')
    hostname = hostname_result[head_node].strip()
    output_dir = f"{output_base_dir}/flux_{hostname}_outputs"

    log.info(f"Creating output directory: {output_dir}")
    s_phdl.exec(f"mkdir -p {output_dir}")

    # Build docker run command
    device_list = inference_dict['container_config']['device_list']
    volume_dict = inference_dict['container_config']['volume_dict']
    env_dict = inference_dict['container_config']['env_dict']

    # Build device arguments
    device_args = " ".join([f"--device={dev}" for dev in device_list])

    # Build volume arguments (add our required mounts)
    volume_dict_full = volume_dict.copy()
    volume_dict_full[output_dir] = "/outputs"
    volume_dict_full[hf_home] = "/hf_home"
    volume_args = " ".join([f"--mount type=bind,source={src},target={dst}" for src, dst in volume_dict_full.items()])

    # Build environment arguments
    env_dict_full = env_dict.copy()
    env_dict_full['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    env_dict_full['OMP_NUM_THREADS'] = '16'
    env_dict_full['HF_HOME'] = '/hf_home'
    if hf_token:
        env_dict_full['HF_TOKEN'] = hf_token
    env_args = " ".join([f"-e {key}={value}" for key, value in env_dict_full.items()])

    # Build torchrun command
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

    # Full docker command
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

    log.info(f"Running FLUX.1-dev benchmark on {head_node}")
    log.info(f"Output directory: {output_dir}")
    log.debug(f"Docker command: {_redact_secrets(docker_cmd)}")

    try:
        # Run with generous timeout (benchmarks can take 10+ minutes)
        log.info("Starting benchmark (this may take several minutes)...")
        benchmark_result = s_phdl.exec(docker_cmd, timeout=1800)  # 30 min timeout

        log.info("Benchmark completed")
        log.debug(f"Benchmark output:\n{benchmark_result[head_node]}")

        # Check for common failure patterns
        output = benchmark_result[head_node]
        if re.search(r'error|fail|exception|traceback', output, re.I):
            log.warning("Detected potential errors in benchmark output")

    except Exception as e:
        fail_test(f"Benchmark execution failed with exception: {e}")

    # Store output directory for next test
    inference_dict['_test_output_dir'] = output_dir

    update_test_result()


def test_parse_and_validate_results(s_phdl, inference_dict, benchmark_params_dict, gpu_type):
    """
    Parse benchmark outputs and validate against thresholds.

    Uses FluxOutputParser to:
    - Locate results/timing.json
    - Parse pipe_time values
    - Compute average
    - Verify images (flux_*.png) exist
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
                output_dir = f"{output_base_dir}/flux_{hostname}_outputs"
                log.info(f"Derived output directory: {output_dir}")
        except Exception:
            output_dir = None

        if not output_dir:
            fail_test("Output directory not set by previous test and could not be derived")
            update_test_result()
            return

    log.info(f"Parsing results from: {output_dir}")
    parser = FluxOutputParser(output_dir, expected_image_pattern="flux_*.png")
    result, errors = parser.parse()

    for error in errors:
        log.warning(f"Parse warning: {error}")

    if result is None:
        fail_test(f"Failed to parse benchmark results: {errors}")
        update_test_result()
        return

    if not result.image_paths:
        fail_test(f"No images (flux_*.png) found under {output_dir}")
    else:
        log.info(f"Found {len(result.image_paths)} generated images")
        for img_path in result.image_paths[:5]:  # Log first 5
            log.info(f"  - {img_path}")

    log.info("Benchmark results:")
    log.info(f"  Repetitions parsed: {result.repetition_count}")
    log.info(f"  Average pipe_time: {result.avg_pipe_time_s:.2f}s")
    log.info(f"  Pipe times: {[f'{t:.2f}' for t in result.pipe_times[:10]]}")  # Log first 10

    # Validate against threshold
    flux_params = benchmark_params_dict['flux1_dev_t2i']
    expected_results = flux_params['expected_results']

    passed, message = parser.validate_threshold(result, expected_results, gpu_type)
    log.info(message)
    if not passed:
        fail_test(message)
    update_test_result()
