"""
PyTorch XDit WAN 2.2 benchmark launcher (single-node scale-out + unified distributed).

Single mode:
  - One independent torchrun job per node in ``s_phdl.host_list``.
  - Each node writes to ``wan_22_{hostname}_outputs``.

Distributed mode:
  - One coordinated torchrun job across ``nnodes`` with distinct ``--node_rank``.
  - All nodes share rank-0 output dir ``wan_22_{rank0_hostname}_outputs``.
  - Requires ulysses_size × ring_size == nnodes × torchrun_nproc.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cvs.lib import globals
from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import (
    DEFAULT_BENCHMARK_TIMEOUT_S,
    DEFAULT_MASTER_PORT,
    build_nccl_env,
    compute_world_size,
    resolve_master_addr,
    resolve_nnodes,
    resolve_server_nodes,
    scan_fatal_output,
    verify_distributed_logs,
    log_benchmark_failure_excerpt,
    _exec_cmd_list_on_nodes,
    _exec_on_nodes,
)

log = globals.log

RUN_WAN_PATH = "/app/Wan2.2/run.py"
I2V_INPUT_IMAGE = "/app/Wan2.2/examples/i2v_input.JPG"
CONTAINER_OUTPUT_MOUNT = "/outputs"

WAN_FATAL_OUTPUT_PATTERNS_EXTRA = (
    r"No AMD GPU detected",
    r"0 active drivers \(\[\]\)\. There should only be one\.",
)


def _secret_str(value: Any) -> str:
    return "" if value is None else str(value)


def parallel_product(wan_params: Mapping[str, Any]) -> int:
    return int(wan_params["ulysses_size"]) * int(wan_params["ring_size"])


def validate_parallelism(
    nnodes: int,
    wan_params: Mapping[str, Any],
) -> Tuple[int, int, Optional[str]]:
    nproc = int(wan_params["torchrun_nproc"])
    world_size = compute_world_size(nnodes, nproc)
    product = parallel_product(wan_params)
    if product != world_size:
        return (
            world_size,
            product,
            (
                f"Parallel degree product {product} != world_size {world_size} "
                f"(nnodes={nnodes} × nproc={nproc}). "
                f"Check ulysses_size and ring_size."
            ),
        )
    return world_size, product, None


def build_run_wan_args(
    wan_params: Mapping[str, Any],
    *,
    ckpt_dir: str,
    output_dir_container: str = CONTAINER_OUTPUT_MOUNT,
) -> str:
    compile_flag = "--compile" if wan_params.get("compile") else ""
    return (
        f"--task i2v-A14B "
        f"--size {shlex.quote(str(wan_params['size']))} "
        f"--ckpt_dir {shlex.quote(ckpt_dir)} "
        f"--image {I2V_INPUT_IMAGE} "
        f"--save_file {CONTAINER_OUTPUT_MOUNT}/outputs/video.mp4 "
        f"--ulysses_size {int(wan_params['ulysses_size'])} "
        f"--ring_size {int(wan_params['ring_size'])} "
        f"--vae_dtype bfloat16 "
        f"--frame_num {int(wan_params['frame_num'])} "
        f"--prompt {shlex.quote(str(wan_params['prompt']))} "
        f"--benchmark_output_directory {shlex.quote(output_dir_container)} "
        f"--num_benchmark_steps {int(wan_params['num_benchmark_steps'])} "
        f"--offload_model 0 "
        f"--allow_tf32 "
        f"{compile_flag}"
    ).strip()


def build_torchrun_cmd(
    wan_params: Mapping[str, Any],
    *,
    ckpt_dir: str,
    distributed: bool,
    node_rank: int = 0,
    nnodes: int = 1,
    nproc_per_node: Optional[int] = None,
    master_addr: str = "127.0.0.1",
    master_port: int = DEFAULT_MASTER_PORT,
) -> str:
    nproc = int(nproc_per_node or wan_params["torchrun_nproc"])
    run_args = build_run_wan_args(wan_params, ckpt_dir=ckpt_dir)

    if distributed:
        return (
            f"torchrun "
            f"--nnodes={nnodes} "
            f"--node_rank={node_rank} "
            f"--nproc_per_node={nproc} "
            f"--master_addr={shlex.quote(master_addr)} "
            f"--master_port={master_port} "
            f"{RUN_WAN_PATH} "
            f"{run_args}"
        )

    return f"torchrun --nproc_per_node={nproc} {RUN_WAN_PATH} {run_args}"


def scan_wan_fatal_output(output: str) -> bool:
    if scan_fatal_output(output):
        return True
    return any(re.search(p, output or "", re.I) for p in WAN_FATAL_OUTPUT_PATTERNS_EXTRA)


def build_wan_output_cleanup_cmd(output_base_dir: str, *, use_sudo: bool = True) -> str:
    prefix = "sudo " if use_sudo else ""
    return f"bash -c {shlex.quote(f'{prefix}rm -rf {output_base_dir}/wan_22_*_outputs')}"


@dataclass
class WanLaunchPlan:
    mkdir_cmds: List[str] = field(default_factory=list)
    docker_cmds: List[str] = field(default_factory=list)
    node_order: List[str] = field(default_factory=list)
    node_to_hostname: Dict[str, str] = field(default_factory=dict)
    output_dirs_by_node: Dict[str, str] = field(default_factory=dict)
    primary_output_dir: str = ""
    distributed: bool = False
    world_size: int = 0


class WanBenchmarkJob:
    """Build and run WAN 2.2 docker+torchrun commands via a Pssh-like handle."""

    def __init__(
        self,
        s_phdl,
        inference_dict: Dict[str, Any],
        benchmark_params_dict: Mapping[str, Any],
        hf_token: Any = "",
        *,
        distributed: bool = False,
        cluster_dict: Optional[Mapping[str, Any]] = None,
    ):
        self.s_phdl = s_phdl
        self.inference_dict = inference_dict
        self.wan_params = benchmark_params_dict["wan22_i2v_a14b"]
        self.hf_token = hf_token
        self.distributed = distributed
        self.cluster_dict = cluster_dict or {}

        self.nproc_per_node = int(self.wan_params["torchrun_nproc"])
        self.server_nodes = self._resolve_execution_nodes()
        self.nnodes = len(self.server_nodes) if self.distributed else 1

    def _resolve_execution_nodes(self) -> List[str]:
        if self.distributed:
            if not self.cluster_dict:
                raise ValueError("distributed=True requires cluster_dict")
            nodes = resolve_server_nodes(self.cluster_dict, self.inference_dict)
            nnodes = resolve_nnodes(self.inference_dict, nodes)
            if nnodes < 2:
                raise ValueError(f"Distributed mode requires nnodes >= 2, got {nnodes}")
            if len(nodes) < nnodes:
                raise ValueError(f"Cluster/server_node_list has {len(nodes)} node(s) but nnodes={nnodes}")
            return nodes[:nnodes]
        return list(self.s_phdl.host_list)

    def validate_parallelism(self) -> Optional[str]:
        if not self.distributed:
            _, _, err = validate_parallelism(1, self.wan_params)
        else:
            _, _, err = validate_parallelism(self.nnodes, self.wan_params)
        if err:
            return err

        world_size, product, _ = validate_parallelism(
            self.nnodes if self.distributed else 1,
            self.wan_params,
        )
        log.info(
            "Parallelism OK (%s): world_size=%s product=%s (ulysses=%s ring=%s)",
            "distributed" if self.distributed else "single-node",
            world_size,
            product,
            self.wan_params["ulysses_size"],
            self.wan_params["ring_size"],
        )
        return None

    def check_kfd(self) -> List[str]:
        log.info("Checking /dev/kfd on %d node(s)", len(self.server_nodes))
        kfd_check = _exec_on_nodes(
            self.s_phdl,
            self.server_nodes,
            "test -e /dev/kfd && echo KFD_OK || echo KFD_MISSING",
            print_console=False,
        )
        missing = []
        for node in self.server_nodes:
            output = kfd_check.get(node, "")
            if "KFD_OK" not in (output or ""):
                missing.append(node)
                log.error("ROCm device node /dev/kfd not found on %s", node)
            else:
                log.info("/dev/kfd found on %s", node)
        return missing

    def _fetch_hostnames(self) -> Dict[str, str]:
        log.info("Getting hostnames from %d node(s)", len(self.server_nodes))
        hostname_result = _exec_on_nodes(self.s_phdl, self.server_nodes, "hostname")
        return {node: (hostname_result.get(node, "") or "").strip() or node for node in self.server_nodes}

    def _resolved_ckpt_dir(self) -> str:
        ckpt_dir = self.inference_dict.get("_resolved_ckpt_dir_container")
        if ckpt_dir:
            return ckpt_dir
        model_repo = self.inference_dict["model_repo"]
        model_rev = self.inference_dict["model_rev"]
        model_path_safe = model_repo.replace("/", "--")
        return f"/hf_home/hub/models--{model_path_safe}/snapshots/{model_rev}"

    def _build_env_args(self) -> str:
        user_env = dict(self.inference_dict["container_config"].get("env_dict") or {})
        env_dict: Dict[str, str] = {}
        if self.distributed:
            env_dict.update(build_nccl_env(self.inference_dict))
        env_dict.update(user_env)
        env_dict["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.nproc_per_node))
        env_dict["OMP_NUM_THREADS"] = "16"
        env_dict["HF_HOME"] = "/hf_home"
        if self.hf_token:
            env_dict["HF_TOKEN"] = _secret_str(self.hf_token)
        return " ".join(f"-e {key}={value}" for key, value in env_dict.items())

    def _build_volume_args(self, host_output_dir: str) -> str:
        volume_dict = dict(self.inference_dict["container_config"].get("volume_dict") or {})
        volume_dict[host_output_dir] = CONTAINER_OUTPUT_MOUNT
        volume_dict[self.inference_dict["hf_home"]] = "/hf_home"
        mount_host = self.inference_dict.get("_resolved_model_mount_host")
        if mount_host:
            volume_dict[mount_host] = "/model"
        return " ".join(f"--mount type=bind,source={src},target={dst}" for src, dst in volume_dict.items())

    def _build_docker_cmd(
        self,
        *,
        node_rank: int,
        host_output_dir: str,
        master_addr: str,
        master_port: int,
    ) -> str:
        device_list = self.inference_dict["container_config"]["device_list"]
        device_args = " ".join(f"--device={dev}" for dev in device_list)
        env_args = self._build_env_args()
        volume_args = self._build_volume_args(host_output_dir)

        torchrun_cmd = build_torchrun_cmd(
            self.wan_params,
            ckpt_dir=self._resolved_ckpt_dir(),
            distributed=self.distributed,
            node_rank=node_rank,
            nnodes=self.nnodes if self.distributed else 1,
            nproc_per_node=self.nproc_per_node,
            master_addr=master_addr,
            master_port=master_port,
        )

        container_name = self.inference_dict["container_name"]
        if self.distributed:
            container_name = f"{container_name}-rank{node_rank}"

        return (
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
            f"{self.inference_dict['container_image']} "
            f"{torchrun_cmd}"
        )

    def build_launch_plan(self) -> WanLaunchPlan:
        node_to_hostname = self._fetch_hostnames()
        output_base_dir = self.inference_dict["output_base_dir"]
        master_port = int(self.inference_dict.get("master_port") or DEFAULT_MASTER_PORT)

        plan = WanLaunchPlan(
            distributed=self.distributed,
            node_order=list(self.server_nodes),
            node_to_hostname=dict(node_to_hostname),
        )

        if self.distributed:
            rank0_node = self.server_nodes[0]
            master_addr = resolve_master_addr(
                self.inference_dict,
                node_to_hostname,
                rank0_node,
                s_phdl=self.s_phdl,
            )
            primary_output_dir = f"{output_base_dir}/wan_22_{node_to_hostname[rank0_node]}_outputs"
            plan.primary_output_dir = primary_output_dir
            plan.world_size = compute_world_size(self.nnodes, self.nproc_per_node)

            for node_rank, node in enumerate(self.server_nodes):
                plan.mkdir_cmds.append(f"mkdir -p {shlex.quote(primary_output_dir + '/outputs')}")
                plan.output_dirs_by_node[node] = primary_output_dir
                plan.docker_cmds.append(
                    self._build_docker_cmd(
                        node_rank=node_rank,
                        host_output_dir=primary_output_dir,
                        master_addr=master_addr,
                        master_port=master_port,
                    )
                )
                log.info(
                    "Distributed node %s (%s) rank=%d master=%s:%d output=%s",
                    node,
                    node_to_hostname[node],
                    node_rank,
                    master_addr,
                    master_port,
                    primary_output_dir,
                )
            return plan

        for node in self.server_nodes:
            hostname = node_to_hostname[node]
            host_output_dir = f"{output_base_dir}/wan_22_{hostname}_outputs"
            plan.mkdir_cmds.append(f"mkdir -p {shlex.quote(host_output_dir + '/outputs')}")
            plan.output_dirs_by_node[node] = host_output_dir
            plan.docker_cmds.append(
                self._build_docker_cmd(
                    node_rank=0,
                    host_output_dir=host_output_dir,
                    master_addr="127.0.0.1",
                    master_port=master_port,
                )
            )
            log.info("Single-node job on %s (%s) output=%s", node, hostname, host_output_dir)

        if len(self.server_nodes) == 1:
            only_node = self.server_nodes[0]
            plan.primary_output_dir = plan.output_dirs_by_node[only_node]
        else:
            plan.primary_output_dir = ""

        plan.world_size = self.nproc_per_node
        return plan

    def run(
        self,
        *,
        timeout: int = DEFAULT_BENCHMARK_TIMEOUT_S,
    ) -> Tuple[Dict[str, str], WanLaunchPlan, List[str]]:
        errors: List[str] = []

        par_err = self.validate_parallelism()
        if par_err:
            errors.append(par_err)
            return {}, WanLaunchPlan(), errors

        missing_kfd = self.check_kfd()
        if missing_kfd:
            errors.append(
                f"ROCm device node /dev/kfd not found on {len(missing_kfd)} node(s): "
                f"{', '.join(missing_kfd)}. Run on GPU compute nodes."
            )
            return {}, WanLaunchPlan(), errors

        plan = self.build_launch_plan()
        if not plan.docker_cmds:
            errors.append("No docker commands generated")
            return {}, plan, errors

        log.info("Creating output directories on %d node(s)", len(plan.node_order))
        try:
            _exec_cmd_list_on_nodes(self.s_phdl, plan.node_order, plan.mkdir_cmds)
        except Exception as exc:
            errors.append(f"Failed to create output directories: {exc}")
            return {}, plan, errors

        mode_label = "distributed unified" if self.distributed else "single-node"
        log.info(
            "Running WAN 2.2 benchmark (%s) on %d node command(s)",
            mode_label,
            len(plan.docker_cmds),
        )

        try:
            results = _exec_cmd_list_on_nodes(
                self.s_phdl,
                plan.node_order,
                plan.docker_cmds,
                timeout=timeout,
            )
        except Exception as exc:
            errors.append(f"Benchmark execution failed with exception: {exc}")
            return {}, plan, errors

        combined_output = "\n".join((results or {}).values())
        if self.distributed:
            ok, msg = verify_distributed_logs(combined_output, world_size=plan.world_size)
            log.info("Distributed log proof: %s", msg)
            if not ok:
                errors.append(msg)

        failed_nodes = []
        for node in plan.node_order:
            output = (results or {}).get(node, "")
            if scan_wan_fatal_output(output):
                log.error("Benchmark output indicates failure on %s", node)
                log_benchmark_failure_excerpt(node, output)
                failed_nodes.append(node)
            else:
                log.info("Benchmark on %s completed successfully", node)

        if failed_nodes:
            errors.append(f"Benchmark failed on {len(failed_nodes)} node(s): {', '.join(failed_nodes)}")

        return results or {}, plan, errors

    def store_output_dir_hint(self, plan: WanLaunchPlan) -> None:
        if plan.primary_output_dir:
            self.inference_dict["_test_output_dir"] = plan.primary_output_dir
            return

        if not self.distributed and len(plan.node_order) == 1:
            node = plan.node_order[0]
            self.inference_dict["_test_output_dir"] = plan.output_dirs_by_node[node]


def launch_wan_benchmark(
    s_phdl,
    inference_dict: Dict[str, Any],
    benchmark_params_dict: Mapping[str, Any],
    hf_token: Any = "",
    *,
    distributed: bool = False,
    cluster_dict: Optional[Mapping[str, Any]] = None,
    timeout: int = DEFAULT_BENCHMARK_TIMEOUT_S,
) -> List[str]:
    """
    Run the WAN benchmark and store ``_test_output_dir`` on success.

    Returns a list of error messages (empty == success).
    """
    job = WanBenchmarkJob(
        s_phdl,
        inference_dict,
        benchmark_params_dict,
        hf_token,
        distributed=distributed,
        cluster_dict=cluster_dict,
    )
    _, plan, errors = job.run(timeout=timeout)
    if not errors:
        job.store_output_dir_hint(plan)
    return errors


def validate_wan_parallelism_config(
    inference_dict: Mapping[str, Any],
    benchmark_params_dict: Mapping[str, Any],
    *,
    distributed: bool,
    cluster_dict: Optional[Mapping[str, Any]] = None,
    node_count: Optional[int] = None,
) -> Optional[str]:
    """Standalone parallelism validation for a dedicated pytest preflight."""
    wan_params = benchmark_params_dict["wan22_i2v_a14b"]
    if distributed:
        if not cluster_dict:
            return "distributed parallelism validation requires cluster_dict"
        nodes = resolve_server_nodes(cluster_dict, inference_dict)
        nnodes = resolve_nnodes(inference_dict, nodes)
        _, _, err = validate_parallelism(nnodes, wan_params)
        return err
    if node_count is not None and node_count > 1:
        return None
    _, _, err = validate_parallelism(1, wan_params)
    return err
