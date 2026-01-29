"""
PyTorch XDit WAN 2.2 Image-to-Video A14B inference test.

Runs WAN 2.2 I2V-A14B PyTorch inference inside amdsiloai/pytorch-xdit container
and validates results against configured thresholds.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import pytest
import re

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
from cvs.parsers.pytorch_xdit_wan import WanOutputParser

log = globals.log


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

    # Now resolve placeholders in the validated structure
    config_dict = raw_config['config']
    benchmark_params = raw_config['benchmark_params']

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    benchmark_params = resolve_test_config_placeholders(benchmark_params, cluster_dict)

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

    Returns empty string if file not found (will be caught later if download needed).
    """
    hf_token_file = inference_dict['hf_token_file']
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
    """Create and return a parallel SSH handle for all cluster nodes."""
    node_list = list(cluster_dict['node_dict'].keys())
    s_phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return s_phdl


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
    model_rev = inference_dict['model_rev']
    hf_home = inference_dict['hf_home']

    # Construct expected snapshot directory path
    model_path_safe = model_repo.replace("/", "--")
    snapshot_dir = f"{hf_home}/hub/models--{model_path_safe}/snapshots/{model_rev}"

    log.info(f"Checking for model cache at: {snapshot_dir}")

    # Check if snapshot directory exists
    head_node = s_phdl.host_list[0]
    check_cmd = f"test -d {snapshot_dir} && echo 'EXISTS' || echo 'MISSING'"
    check_result = s_phdl.exec(check_cmd)

    if "EXISTS" in check_result[head_node]:
        log.info(f"Model cache found at {snapshot_dir}")
        update_test_result()
        return

    log.info(f"Model cache not found, downloading {model_repo}@{model_rev}...")

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
    download_cmd = (
        f"docker run --rm "
        f"--mount type=bind,source={hf_home},target=/hf_home "
        f"-e HF_HOME=/hf_home "
        f"-e HF_TOKEN={hf_token} "
        f"{container_image} "
        f"hf download {model_repo} --revision {model_rev}"
    )

    log.info(f"Running download: {download_cmd}")

    try:
        download_result = s_phdl.exec(download_cmd, timeout=1800)  # 30 min timeout
        log.info(f"Download output: {download_result[head_node]}")

        # Verify download succeeded
        verify_result = s_phdl.exec(check_cmd)
        if "EXISTS" not in verify_result[head_node]:
            fail_test("Model download failed: snapshot directory still missing after download")
    except Exception as e:
        fail_test(f"Model download failed with exception: {e}")

    update_test_result()


def test_run_wan22_benchmark(s_phdl, inference_dict, benchmark_params_dict, hf_token):
    """
    Run WAN 2.2 I2V-A14B benchmark inside pytorch-xdit container.

    Executes torchrun with configured parameters and mounts:
    - HF cache to /hf_home
    - Output directory to /outputs
    """
    globals.error_list = []

    container_image = inference_dict['container_image']
    container_name = inference_dict['container_name']
    hf_home = inference_dict['hf_home']
    output_base_dir = inference_dict['output_base_dir']
    model_repo = inference_dict['model_repo']
    model_rev = inference_dict['model_rev']

    # Get benchmark parameters
    wan_params = benchmark_params_dict['wan22_i2v_a14b']
    prompt = wan_params['prompt']
    size = wan_params['size']
    frame_num = wan_params['frame_num']
    num_benchmark_steps = wan_params['num_benchmark_steps']
    compile_flag = "--compile" if wan_params['compile'] else ""
    torchrun_nproc = wan_params['torchrun_nproc']

    # Create output directory
    head_node = s_phdl.host_list[0]
    hostname_result = s_phdl.exec('hostname')
    hostname = hostname_result[head_node].strip()
    output_dir = f"{output_base_dir}/wan_22_{hostname}_outputs"
    outputs_dir = f"{output_dir}/outputs"

    log.info(f"Creating output directory: {outputs_dir}")
    s_phdl.exec(f"mkdir -p {outputs_dir}")

    # Construct checkpoint directory path
    model_path_safe = model_repo.replace("/", "--")
    ckpt_dir = f"/hf_home/hub/models--{model_path_safe}/snapshots/{model_rev}"

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
    torchrun_cmd = (
        f"torchrun --nproc_per_node={torchrun_nproc} /app/Wan2.2/run.py "
        f"--task i2v-A14B "
        f"--size \"{size}\" "
        f"--ckpt_dir \"{ckpt_dir}\" "
        f"--image /app/Wan2.2/examples/i2v_input.JPG "
        f"--save_file /outputs/outputs/video.mp4 "
        f"--ulysses_size 8 "
        f"--ring_size 1 "
        f"--vae_dtype bfloat16 "
        f"--frame_num {frame_num} "
        f"--prompt \"{prompt}\" "
        f"--benchmark_output_directory /outputs "
        f"--num_benchmark_steps {num_benchmark_steps} "
        f"--offload_model 0 "
        f"--allow_tf32 "
        f"{compile_flag}"
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

    log.info(f"Running WAN 2.2 benchmark on {head_node}")
    log.info(f"Output directory: {output_dir}")
    log.debug(f"Docker command: {docker_cmd}")

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
        fail_test("Output directory not set by previous test")
        update_test_result()
        return

    log.info(f"Parsing results from: {output_dir}")

    # Parse outputs
    parser = WanOutputParser(output_dir, expected_artifact="video.mp4")
    result, errors = parser.parse()

    # Report parse errors
    for error in errors:
        log.warning(f"Parse warning: {error}")

    # Check if parsing succeeded
    if result is None:
        fail_test(f"Failed to parse benchmark results: {errors}")
        update_test_result()
        return

    # Verify artifact exists
    if not result.artifact_path:
        fail_test(f"Artifact 'video.mp4' not found under {output_dir}")
    else:
        log.info(f"Artifact found: {result.artifact_path}")

    # Log results
    log.info("Benchmark results:")
    log.info(f"  Steps parsed: {result.step_count}")
    log.info(f"  Average total_time: {result.avg_total_time_s:.2f}s")
    log.info(f"  Step times: {[f'{t:.2f}' for t in result.step_times]}")

    # Validate against threshold
    wan_params = benchmark_params_dict['wan22_i2v_a14b']
    expected_results = wan_params['expected_results']

    passed, message = parser.validate_threshold(result, expected_results, gpu_type)
    log.info(message)

    if not passed:
        fail_test(message)

    update_test_result()
