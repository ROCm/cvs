"""
PyTorch XDit FLUX.1-dev benchmark launcher (single-node + unified distributed).

Single mode:
  - One independent torchrun job per node in ``s_phdl.host_list``.
  - Each node writes to ``flux_{hostname}_outputs``.

Distributed mode:
  - One coordinated torchrun job across ``nnodes`` with distinct ``--node_rank``.
  - All nodes share rank-0 output dir ``flux_{rank0_hostname}_outputs``.
  - Requires parallel-degree product == nnodes × torchrun_nproc.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cvs.lib import globals
from cvs.lib.parallel_ssh_lib import Pssh

log = globals.log

FATAL_OUTPUT_PATTERNS = (
    r"\bTraceback\b",
    r"\bModuleNotFoundError\b",
    r"\bChildFailedError\b",
    r"\bOSError:\b",
)

DEFAULT_BENCHMARK_TIMEOUT_S = 1800
DEFAULT_MASTER_PORT = 29500
RUN_USP_PATH = "/app/Flux/run_usp.py"
CONTAINER_OUTPUT_MOUNT = "/outputs"


def as_node_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _redact_secrets(text: str) -> str:
    if not text:
        return text
    return re.sub(r"(HF_TOKEN=)[^\s]+", r"\1<redacted>", text)


def _secret_str(value: Any) -> str:
    return "" if value is None else str(value)


def _phdl_connection_kwargs(s_phdl) -> Dict[str, Any]:
    """Best-effort SSH connection kwargs for a scoped one-node Pssh handle."""
    return {
        "user": getattr(s_phdl, "user", None),
        "password": getattr(s_phdl, "password", None),
        "pkey": getattr(s_phdl, "pkey", "id_rsa"),
        "env_vars": getattr(s_phdl, "env_vars", None),
    }


def _exec_on_single_node(
    s_phdl,
    node: str,
    cmd: str,
    *,
    timeout: Optional[int] = None,
    print_console: bool = True,
) -> str:
    """Run ``cmd`` on exactly one node, even when ``s_phdl`` covers more hosts."""
    phdl_hosts = list(getattr(s_phdl, "host_list", []) or [])
    if phdl_hosts == [node]:
        out = s_phdl.exec(cmd, timeout=timeout, print_console=print_console)
        return (out or {}).get(node, "")

    scoped = Pssh(
        getattr(s_phdl, "log", log),
        [node],
        **_phdl_connection_kwargs(s_phdl),
    )
    out = scoped.exec(cmd, timeout=timeout, print_console=print_console)
    return (out or {}).get(node, "")


def _exec_on_nodes(
    s_phdl,
    nodes: Sequence[str],
    cmd: str,
    *,
    timeout: Optional[int] = None,
    print_console: bool = True,
) -> Dict[str, str]:
    """Run the same command on an explicit node subset."""
    node_list = list(nodes)
    phdl_hosts = list(getattr(s_phdl, "host_list", []) or [])

    if phdl_hosts == node_list:
        return s_phdl.exec(cmd, timeout=timeout, print_console=print_console) or {}

    results: Dict[str, str] = {}
    for node in node_list:
        results[node] = _exec_on_single_node(
            s_phdl,
            node,
            cmd,
            timeout=timeout,
            print_console=print_console,
        )
    return results


def _exec_cmd_list_on_nodes(
    s_phdl,
    nodes: Sequence[str],
    cmd_list: Sequence[str],
    *,
    timeout: Optional[int] = None,
    print_console: bool = True,
) -> Dict[str, str]:
    """
    Run per-node commands on an explicit node subset.

    ``Pssh.exec_cmd_list`` maps commands to ``s_phdl.host_list`` order. This helper
    avoids mis-launch when the participating node set is a subset or reordered.
    """
    node_list = list(nodes)
    commands = list(cmd_list)
    if len(node_list) != len(commands):
        raise ValueError(
            f"node/cmd length mismatch: {len(node_list)} nodes vs {len(commands)} commands"
        )

    phdl_hosts = list(getattr(s_phdl, "host_list", []) or [])
    if phdl_hosts == node_list:
        return s_phdl.exec_cmd_list(commands, timeout=timeout, print_console=print_console) or {}

    results: Dict[str, str] = {}
    for node, cmd in zip(node_list, commands):
        results[node] = _exec_on_single_node(
            s_phdl,
            node,
            cmd,
            timeout=timeout,
            print_console=print_console,
        )
    return results


def resolve_server_nodes(cluster_dict: Mapping[str, Any], inference_dict: Mapping[str, Any]) -> List[str]:
    explicit = inference_dict.get("server_node_list")
    if explicit:
        return as_node_list(explicit)
    return list(cluster_dict["node_dict"].keys())


def resolve_nnodes(inference_dict: Mapping[str, Any], server_nodes: Sequence[str]) -> int:
    configured = inference_dict.get("nnodes")
    if configured is not None and str(configured).strip() != "":
        return int(configured)
    return len(server_nodes)


def resolve_master_addr(
    inference_dict: Mapping[str, Any],
    node_to_hostname: Mapping[str, str],
    rank0_node: str,
    *,
    s_phdl=None,
) -> str:
    """
    Resolve torchrun rendezvous address.

    Prefer explicit config. Otherwise use rank-0 IP when possible, then hostname.
    """
    addr = (inference_dict.get("master_addr") or "").strip()
    if addr:
        return addr

    if s_phdl is not None:
        ip_cmd = "hostname -I | awk '{print $1}'"
        ip_out = _exec_on_single_node(s_phdl, rank0_node, ip_cmd, print_console=False).strip()
        first_ip = (ip_out.split() or [""])[0].strip()
        if first_ip:
            log.info("Resolved master_addr from rank-0 node %s: %s", rank0_node, first_ip)
            return first_ip

    hostname = node_to_hostname.get(rank0_node, rank0_node)
    log.info("Using hostname for master_addr on rank-0 node %s: %s", rank0_node, hostname)
    return hostname


def parallel_product(flux_params: Mapping[str, Any]) -> int:
    return (
        int(flux_params["ulysses_degree"])
        * int(flux_params["ring_degree"])
        * int(flux_params.get("pipefusion_parallel_degree", 1))
        * int(flux_params.get("tensor_parallel_degree", 1))
        * int(flux_params.get("data_parallel_degree", 1))
    )


def compute_world_size(nnodes: int, nproc_per_node: int) -> int:
    return nnodes * nproc_per_node


def validate_parallelism(
    nnodes: int,
    flux_params: Mapping[str, Any],
) -> Tuple[int, int, Optional[str]]:
    nproc = int(flux_params["torchrun_nproc"])
    world_size = compute_world_size(nnodes, nproc)
    product = parallel_product(flux_params)
    if product != world_size:
        return world_size, product, (
            f"Parallel degree product {product} != world_size {world_size} "
            f"(nnodes={nnodes} × nproc={nproc}). "
            f"Check ulysses/ring/pipefusion/tensor_parallel/data_parallel."
        )
    return world_size, product, None


def build_nccl_env(inference_dict: Mapping[str, Any]) -> Dict[str, str]:
    env: Dict[str, str] = {"HSA_FORCE_FINE_GRAIN_PCIE": "1"}
    mapping = {
        "nccl_ib_hca": "NCCL_IB_HCA",
        "nccl_socket_ifname": "NCCL_SOCKET_IFNAME",
        "gloo_socket_ifname": "GLOO_SOCKET_IFNAME",
        "nccl_debug": "NCCL_DEBUG",
    }
    for src, dst in mapping.items():
        val = inference_dict.get(src)
        if val:
            env[dst] = str(val)
    gid = inference_dict.get("nccl_ib_gid_index")
    if gid is not None and str(gid).strip() != "":
        env["NCCL_IB_GID_INDEX"] = str(gid)
    return env


def build_run_usp_args(
    flux_params: Mapping[str, Any],
    *,
    model_repo: str,
    output_dir_container: str = CONTAINER_OUTPUT_MOUNT,
) -> str:
    flags: List[str] = []
    if flux_params.get("no_use_resolution_binning"):
        flags.append("--no_use_resolution_binning")
    if flux_params.get("use_torch_compile"):
        flags.append("--use-torch-compile")

    pf = int(flux_params.get("pipefusion_parallel_degree", 1))
    tp = int(flux_params.get("tensor_parallel_degree", 1))
    dp = int(flux_params.get("data_parallel_degree", 1))

    return (
        f"--model {shlex.quote(model_repo)} "
        f"--prompt {shlex.quote(str(flux_params['prompt']))} "
        f"--seed {int(flux_params['seed'])} "
        f"--num_inference_steps {int(flux_params['num_inference_steps'])} "
        f"--max_sequence_length {int(flux_params['max_sequence_length'])} "
        f"{' '.join(flags)} "
        f"--warmup_steps {int(flux_params['warmup_steps'])} "
        f"--warmup_calls {int(flux_params['warmup_calls'])} "
        f"--num_repetitions {int(flux_params['num_repetitions'])} "
        f"--height {int(flux_params['height'])} "
        f"--width {int(flux_params['width'])} "
        f"--ulysses_degree {int(flux_params['ulysses_degree'])} "
        f"--ring_degree {int(flux_params['ring_degree'])} "
        f"--pipefusion_parallel_degree {pf} "
        f"--tensor_parallel_degree {tp} "
        f"--data_parallel_degree {dp} "
        f"--benchmark_output_directory {shlex.quote(output_dir_container)}"
    )


def build_torchrun_cmd(
    flux_params: Mapping[str, Any],
    *,
    model_repo: str,
    distributed: bool,
    node_rank: int = 0,
    nnodes: int = 1,
    nproc_per_node: Optional[int] = None,
    master_addr: str = "127.0.0.1",
    master_port: int = DEFAULT_MASTER_PORT,
) -> str:
    nproc = int(nproc_per_node or flux_params["torchrun_nproc"])
    run_usp_args = build_run_usp_args(flux_params, model_repo=model_repo)

    if distributed:
        return (
            f"torchrun "
            f"--nnodes={nnodes} "
            f"--node_rank={node_rank} "
            f"--nproc_per_node={nproc} "
            f"--master_addr={shlex.quote(master_addr)} "
            f"--master_port={master_port} "
            f"{RUN_USP_PATH} "
            f"{run_usp_args}"
        )

    return f"torchrun --nproc_per_node={nproc} {RUN_USP_PATH} {run_usp_args}"


def verify_distributed_logs(output: str, *, world_size: int) -> Tuple[bool, str]:
    if not output:
        return False, "Empty benchmark output"

    if re.search(rf"\bworld[_ ]?size[=:\s]+{world_size}\b", output, re.I):
        return True, f"Saw world_size={world_size} in logs"

    if re.search(r"Initialized process group|process group initialized|c10d", output, re.I):
        return True, "Saw distributed process-group initialization in logs"

    rank_refs = len(re.findall(r"\brank[=:\s]+\d+\b", output, re.I))
    if rank_refs >= 2:
        return True, f"Saw {rank_refs} rank references in logs"

    return False, (
        f"No distributed proof in logs for world_size={world_size}. "
        "Try NCCL_DEBUG=INFO and verify IB/socket interface settings."
    )


def scan_fatal_output(output: str) -> bool:
    return any(re.search(p, output or "", re.I) for p in FATAL_OUTPUT_PATTERNS)


@dataclass
class FluxLaunchPlan:
    mkdir_cmds: List[str] = field(default_factory=list)
    docker_cmds: List[str] = field(default_factory=list)
    node_order: List[str] = field(default_factory=list)
    node_to_hostname: Dict[str, str] = field(default_factory=dict)
    output_dirs_by_node: Dict[str, str] = field(default_factory=dict)
    primary_output_dir: str = ""
    distributed: bool = False
    world_size: int = 0


class FluxBenchmarkJob:
    """Build and run FLUX.1-dev docker+torchrun commands via a Pssh-like handle."""

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
        self.flux_params = benchmark_params_dict["flux1_dev_t2i"]
        self.hf_token = hf_token
        self.distributed = distributed
        self.cluster_dict = cluster_dict or {}

        self.nproc_per_node = int(self.flux_params["torchrun_nproc"])
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
                raise ValueError(
                    f"Cluster/server_node_list has {len(nodes)} node(s) but nnodes={nnodes}"
                )
            return nodes[:nnodes]
        return list(self.s_phdl.host_list)

    def validate_parallelism(self) -> Optional[str]:
        if not self.distributed:
            _, _, err = validate_parallelism(1, self.flux_params)
        else:
            _, _, err = validate_parallelism(self.nnodes, self.flux_params)
        if err:
            return err

        world_size, product, _ = validate_parallelism(
            self.nnodes if self.distributed else 1,
            self.flux_params,
        )
        log.info(
            "Parallelism OK (%s): world_size=%s product=%s "
            "(ulysses=%s ring=%s pipefusion=%s tp=%s dp=%s)",
            "distributed" if self.distributed else "single-node",
            world_size,
            product,
            self.flux_params["ulysses_degree"],
            self.flux_params["ring_degree"],
            self.flux_params.get("pipefusion_parallel_degree", 1),
            self.flux_params.get("tensor_parallel_degree", 1),
            self.flux_params.get("data_parallel_degree", 1),
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
        return {
            node: (hostname_result.get(node, "") or "").strip() or node
            for node in self.server_nodes
        }

    def _resolved_model_repo(self) -> str:
        return self.inference_dict.get("_resolved_model_path_container") or self.inference_dict["model_repo"]

    def _build_env_args(self) -> str:
        env_dict = dict(self.inference_dict["container_config"].get("env_dict") or {})
        env_dict["OMP_NUM_THREADS"] = "16"
        env_dict["HF_HOME"] = "/hf_home"
        env_dict["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.nproc_per_node))
        if self.distributed:
            env_dict.update(build_nccl_env(self.inference_dict))
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
        return " ".join(
            f"--mount type=bind,source={src},target={dst}"
            for src, dst in volume_dict.items()
        )

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
            self.flux_params,
            model_repo=self._resolved_model_repo(),
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

    def build_launch_plan(self) -> FluxLaunchPlan:
        node_to_hostname = self._fetch_hostnames()
        output_base_dir = self.inference_dict["output_base_dir"]
        master_port = int(self.inference_dict.get("master_port") or DEFAULT_MASTER_PORT)

        plan = FluxLaunchPlan(
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
            primary_output_dir = f"{output_base_dir}/flux_{node_to_hostname[rank0_node]}_outputs"
            plan.primary_output_dir = primary_output_dir
            plan.world_size = compute_world_size(self.nnodes, self.nproc_per_node)

            for node_rank, node in enumerate(self.server_nodes):
                plan.mkdir_cmds.append(f"mkdir -p {shlex.quote(primary_output_dir)}")
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
            host_output_dir = f"{output_base_dir}/flux_{hostname}_outputs"
            plan.mkdir_cmds.append(f"mkdir -p {shlex.quote(host_output_dir)}")
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
    ) -> Tuple[Dict[str, str], FluxLaunchPlan, List[str]]:
        errors: List[str] = []

        par_err = self.validate_parallelism()
        if par_err:
            errors.append(par_err)
            return {}, FluxLaunchPlan(), errors

        missing_kfd = self.check_kfd()
        if missing_kfd:
            errors.append(
                f"ROCm device node /dev/kfd not found on {len(missing_kfd)} node(s): "
                f"{', '.join(missing_kfd)}. Run on GPU compute nodes."
            )
            return {}, FluxLaunchPlan(), errors

        plan = self.build_launch_plan()
        if not plan.docker_cmds:
            errors.append("No docker commands generated")
            return {}, plan, errors

        log.info(
            "Creating output directories on %d node(s)",
            len(plan.node_order),
        )
        try:
            _exec_cmd_list_on_nodes(
                self.s_phdl,
                plan.node_order,
                plan.mkdir_cmds,
            )
        except Exception as exc:
            errors.append(f"Failed to create output directories: {exc}")
            return {}, plan, errors

        mode_label = "distributed unified" if self.distributed else "single-node"
        log.info(
            "Running FLUX.1-dev benchmark (%s) on %d node command(s)",
            mode_label,
            len(plan.docker_cmds),
        )
        log.debug("Docker command (sample): %s", _redact_secrets(plan.docker_cmds[0]))

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
            if scan_fatal_output(output):
                log.error("Benchmark output indicates failure on %s", node)
                failed_nodes.append(node)
            else:
                log.info("Benchmark on %s completed successfully", node)

        if failed_nodes:
            errors.append(f"Benchmark failed on {len(failed_nodes)} node(s): {', '.join(failed_nodes)}")

        return results or {}, plan, errors

    def store_output_dir_hint(self, plan: FluxLaunchPlan) -> None:
        if plan.primary_output_dir:
            self.inference_dict["_test_output_dir"] = plan.primary_output_dir
            return

        if not self.distributed and len(plan.node_order) == 1:
            node = plan.node_order[0]
            self.inference_dict["_test_output_dir"] = plan.output_dirs_by_node[node]


def launch_flux_benchmark(
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
    Run the FLUX benchmark and store ``_test_output_dir`` on success.

    Returns a list of error messages (empty == success). Intended for tests to
    map into ``fail_test`` / ``update_test_result``.
    """
    job = FluxBenchmarkJob(
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


def validate_flux_parallelism_config(
    inference_dict: Mapping[str, Any],
    benchmark_params_dict: Mapping[str, Any],
    *,
    distributed: bool,
    cluster_dict: Optional[Mapping[str, Any]] = None,
    node_count: Optional[int] = None,
) -> Optional[str]:
    """Standalone parallelism validation for a dedicated pytest preflight."""
    flux_params = benchmark_params_dict["flux1_dev_t2i"]
    if distributed:
        if not cluster_dict:
            return "distributed parallelism validation requires cluster_dict"
        nodes = resolve_server_nodes(cluster_dict, inference_dict)
        nnodes = resolve_nnodes(inference_dict, nodes)
        _, _, err = validate_parallelism(nnodes, flux_params)
        return err
    if node_count is not None and node_count > 1:
        return None
    _, _, err = validate_parallelism(1, flux_params)
    return err