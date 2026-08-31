"""
PyTorch XDit WAN 2.2 I2V-A14B Diffusers (xFuser) inference test.

Runs Wan2.2-I2V-A14B-Diffusers via ``wan_i2v_example.py`` inside the ufb-private
container, validates ``results/timing.json`` and ``results/video_i2v.mp4``.

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
    verify_required_checks_on_nodes,
)
from cvs.lib.inference.xdit.pytorch_xdit_wan_i2v import (
    WanI2vOutputParser,
    log_results_summary,
)
from cvs.lib.inference.xdit.pytorch_xdit_wan_job import (
    build_wan_output_cleanup_cmd,
    launch_wan_benchmark,
    store_resolved_wan_model_format_from_index,
)

log = globals.log


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
            log.info("cmd = %s", _redact_secrets(cmd))
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
                log.info("cmd = %s", _redact_secrets(cmd))
                log.info("%s", out_str)
            out[host] = out_str
        return out


def _redact_secrets(s: str) -> str:
    if not s:
        return s
    return re.sub(r"(HF_TOKEN=)\S+", r"\1<redacted>", s)


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def training_config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file, encoding="utf-8") as json_file:
        cluster_dict = json.load(json_file)

    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    try:
        validated = ClusterConfigFile.model_validate(cluster_dict)
        log.info("Cluster config validated successfully: %d nodes", len(validated.node_dict))
    except Exception as exc:
        log.error("Cluster config validation failed: %s", exc)
        pytest.fail(f"Invalid cluster configuration: {exc}")

    return cluster_dict


@pytest.fixture(scope="module")
def wan_config_dict(training_config_file, cluster_dict):
    with open(training_config_file, encoding="utf-8") as json_file:
        raw_config = json.load(json_file)

    try:
        validated_config = PytorchXditWanConfigFile.model_validate(raw_config)
        log.info("WAN diffusers xFuser config validated successfully")
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
        return ""
    try:
        with open(hf_token_file, encoding="utf-8") as fp:
            return fp.read().rstrip("\n")
    except FileNotFoundError:
        log.warning("HF token file not found: %s", hf_token_file)
        return ""


@pytest.fixture(scope="module")
def s_phdl(cluster_dict):
    node_list = list(cluster_dict["node_dict"].keys())
    env_vars = cluster_dict.get("env_vars")

    if len(node_list) == 1 and _is_local_target(node_list[0]):
        log.info("Using local execution mode for single-node target %s", node_list[0])
        return LocalPssh(host=node_list[0])

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


def test_cleanup_stale_containers(s_phdl, inference_dict):
    container_name = inference_dict["container_name"]
    docker_lib.kill_docker_container(s_phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(s_phdl)

    output_base_dir = inference_dict.get("output_base_dir")
    if output_base_dir:
        cleanup_cmd = build_wan_output_cleanup_cmd(output_base_dir, use_sudo=True)
        log.info("Cleaning stale WAN outputs under %s", output_base_dir)
        s_phdl.exec(cleanup_cmd)

    update_test_result()


def _read_model_index_from_node(s_phdl, node: str, model_dir: str) -> Optional[dict]:
    index_path = f"{model_dir.rstrip('/')}/model_index.json"
    output = s_phdl.exec(f"cat {shlex.quote(index_path)}", print_console=False).get(node, "")
    if not (output or "").strip():
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def test_verify_model_on_nodes(s_phdl, inference_dict):
    globals.error_list = []

    model_repo = inference_dict["model_repo"]
    if not (isinstance(model_repo, str) and model_repo.strip().startswith("/")):
        fail_test("Diffusers xFuser WAN test requires config.model_repo as an explicit local path on every node.")
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


def test_run_wan22_diffusers_benchmark(s_phdl, inference_dict, benchmark_params_dict, hf_token):
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
    globals.error_list = []

    output_base_dir = inference_dict.get("output_base_dir")
    if not output_base_dir:
        fail_test("output_base_dir not set in config; cannot locate benchmark results")
        update_test_result()
        return

    if inference_dict.get("_test_output_dir"):
        output_dirs = [inference_dict["_test_output_dir"]]
    else:
        try:
            hostname_out = s_phdl.exec("hostname", print_console=False)
            expected_hostnames = [(hostname_out.get(node, "") or "").strip() or node for node in s_phdl.host_list]
        except Exception:
            expected_hostnames = list(s_phdl.host_list)

        output_dirs = [f"{output_base_dir}/wan_22_{hn}_outputs" for hn in expected_hostnames]

    wan_params = benchmark_params_dict["wan22_i2v_a14b"]
    expected_results = wan_params["expected_results"]
    require_video = bool(wan_params.get("require_video_artifact", True))

    all_passed = True
    results_summary = []

    for output_dir in output_dirs:
        label = output_dir.split("/")[-1].replace("wan_22_", "").replace("_outputs", "")
        parser = WanI2vOutputParser(output_dir, require_video_artifact=require_video)
        result, errors = parser.parse()

        for error in errors:
            log.warning("Parse warning (%s): %s", label, error)

        if result is None:
            log.error("Failed to parse benchmark results for %s: %s", label, errors)
            all_passed = False
            continue

        if require_video and not result.video_path:
            log.error("Artifact video_i2v.mp4 not found under %s", output_dir)
            all_passed = False
            continue

        if result.video_path:
            log.info("Video artifact found for %s: %s", label, result.video_path)

        log.info(
            "Benchmark results (%s): repetitions=%d avg_pipe_time=%.2fs",
            label,
            result.repetition_count,
            result.avg_pipe_time_s,
        )

        passed, message = parser.validate_threshold(result, expected_results, gpu_type)
        log.info("%s: %s", label, message)
        results_summary.append({"label": label, "avg_pipe_time_s": result.avg_pipe_time_s, "passed": passed})
        if not passed:
            all_passed = False

    log_results_summary(results_summary, metric_key="avg_pipe_time_s")

    if not all_passed:
        fail_test("One or more nodes failed Wan diffusers xFuser benchmark validation")

    update_test_result()
