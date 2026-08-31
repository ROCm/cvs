"""
Shared PyTorch XDit docker+torchrun benchmark job base (FLUX, WAN, etc.).

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cvs.lib import globals

log = globals.log

CONTAINER_OUTPUT_MOUNT = "/outputs"


@dataclass
class BenchmarkLaunchPlan:
    mkdir_cmds: List[str] = field(default_factory=list)
    docker_cmds: List[str] = field(default_factory=list)
    node_order: List[str] = field(default_factory=list)
    node_to_hostname: Dict[str, str] = field(default_factory=dict)
    output_dirs_by_node: Dict[str, str] = field(default_factory=dict)
    primary_output_dir: str = ""
    distributed: bool = False
    world_size: int = 0


class PytorchXditBenchmarkJob(ABC):
    """Build and run PyTorch XDit docker+torchrun benchmark commands via a Pssh-like handle."""

    def __init__(
        self,
        s_phdl,
        inference_dict: Dict[str, Any],
        hf_token: Any = "",
        *,
        distributed: bool = False,
        cluster_dict: Optional[Mapping[str, Any]] = None,
        nproc_per_node: int,
    ):
        self.s_phdl = s_phdl
        self.inference_dict = inference_dict
        self.hf_token = hf_token
        self.distributed = distributed
        self.cluster_dict = cluster_dict or {}
        self.nproc_per_node = nproc_per_node
        self.server_nodes = self._resolve_execution_nodes()
        self.nnodes = len(self.server_nodes) if self.distributed else 1

    def _resolve_execution_nodes(self) -> List[str]:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import (
            resolve_nnodes,
            resolve_server_nodes,
        )

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

    @abstractmethod
    def validate_parallelism(self) -> Optional[str]:
        """Return an error message when parallelism config is invalid, else None."""

    def check_kfd(self) -> List[str]:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import _exec_on_nodes

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
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import _exec_on_nodes

        log.info("Getting hostnames from %d node(s)", len(self.server_nodes))
        hostname_result = _exec_on_nodes(self.s_phdl, self.server_nodes, "hostname")
        return {node: (hostname_result.get(node, "") or "").strip() or node for node in self.server_nodes}

    def _build_volume_args(self, host_output_dir: str) -> str:
        volume_dict = dict(self.inference_dict["container_config"].get("volume_dict") or {})
        volume_dict[host_output_dir] = CONTAINER_OUTPUT_MOUNT
        volume_dict[self.inference_dict["hf_home"]] = "/hf_home"
        mount_host = self.inference_dict.get("_resolved_model_mount_host")
        if mount_host:
            volume_dict[mount_host] = "/model"
        return " ".join(f"--mount type=bind,source={src},target={dst}" for src, dst in volume_dict.items())

    @abstractmethod
    def _build_env_args(self) -> str:
        """Return docker ``-e KEY=VALUE`` arguments for the benchmark container."""

    @abstractmethod
    def _build_torchrun_cmd(
        self,
        *,
        node_rank: int,
        host_output_dir: str,
        master_addr: str,
        master_port: int,
    ) -> str:
        """Return the in-container torchrun command for this benchmark."""

    @abstractmethod
    def _host_output_dir(self, output_base_dir: str, hostname: str) -> str:
        """Return the host-side output directory for a node hostname."""

    def _mkdir_cmd(self, host_output_dir: str) -> str:
        return f"mkdir -p {shlex.quote(host_output_dir)}"

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
        torchrun_cmd = self._build_torchrun_cmd(
            node_rank=node_rank,
            host_output_dir=host_output_dir,
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

    def build_launch_plan(self) -> BenchmarkLaunchPlan:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import (
            DEFAULT_MASTER_PORT,
            compute_world_size,
            resolve_master_addr,
        )

        node_to_hostname = self._fetch_hostnames()
        output_base_dir = self.inference_dict["output_base_dir"]
        master_port = int(self.inference_dict.get("master_port") or DEFAULT_MASTER_PORT)

        plan = BenchmarkLaunchPlan(
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
            primary_output_dir = self._host_output_dir(output_base_dir, node_to_hostname[rank0_node])
            plan.primary_output_dir = primary_output_dir
            plan.world_size = compute_world_size(self.nnodes, self.nproc_per_node)

            for node_rank, node in enumerate(self.server_nodes):
                plan.mkdir_cmds.append(self._mkdir_cmd(primary_output_dir))
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
            host_output_dir = self._host_output_dir(output_base_dir, hostname)
            plan.mkdir_cmds.append(self._mkdir_cmd(host_output_dir))
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

    def _pre_launch_validation(self, plan: BenchmarkLaunchPlan) -> List[str]:
        return []

    def _resolve_run_timeout(self, timeout: Optional[int]) -> int:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import DEFAULT_BENCHMARK_TIMEOUT_S

        return timeout if timeout is not None else DEFAULT_BENCHMARK_TIMEOUT_S

    def _benchmark_mode_label(self) -> str:
        return "distributed unified" if self.distributed else "single-node"

    @abstractmethod
    def _benchmark_name(self) -> str:
        """Short benchmark label used in run() log messages."""

    def _create_output_directories(self, plan: BenchmarkLaunchPlan) -> Optional[str]:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import _exec_cmd_list_on_nodes

        log.info("Creating output directories on %d node(s)", len(plan.node_order))
        try:
            _exec_cmd_list_on_nodes(self.s_phdl, plan.node_order, plan.mkdir_cmds)
        except Exception as exc:
            return f"Failed to create output directories: {exc}"
        return None

    def _verify_distributed_output(self, plan: BenchmarkLaunchPlan, results: Mapping[str, str]) -> Optional[str]:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import verify_distributed_logs

        if not self.distributed:
            return None
        combined_output = "\n".join(results.values())
        ok, msg = verify_distributed_logs(combined_output, world_size=plan.world_size)
        log.info("Distributed log proof: %s", msg)
        if not ok:
            return msg
        return None

    def _collect_benchmark_failures(
        self,
        raw_results: Mapping[str, Any],
        plan: BenchmarkLaunchPlan,
    ) -> Tuple[List[str], List[str]]:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import (
            _exec_result_exit_code,
            _exec_result_output,
            log_benchmark_failure_excerpt,
        )

        failed_nodes: List[str] = []
        for node in plan.node_order:
            raw = (raw_results or {}).get(node)
            output = _exec_result_output(raw)
            exit_code = _exec_result_exit_code(raw)
            if exit_code != 0:
                log.error("Benchmark exited with code %s on %s", exit_code, node)
                log_benchmark_failure_excerpt(node, output)
                failed_nodes.append(node)
                self._on_benchmark_node_failure(node, output)
            else:
                log.info("Benchmark on %s completed successfully (exit 0)", node)
        return failed_nodes, []

    def _on_benchmark_node_failure(self, node: str, output: str) -> None:
        """Hook for subclass-specific failure hints after a non-zero benchmark exit."""

    def _handle_benchmark_exec_exception(
        self,
        exc: Exception,
        plan: BenchmarkLaunchPlan,
        results: Mapping[str, str],
    ) -> Tuple[List[str], bool]:
        """Return (errors, treat_as_success). Default: fail the run."""
        return [f"Benchmark execution failed with exception: {exc}"], False

    def run(
        self,
        *,
        timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, str], BenchmarkLaunchPlan, List[str]]:
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import (
            _exec_cmd_list_on_nodes,
            _normalize_exec_results,
            _redact_secrets,
        )

        errors: List[str] = []
        empty_plan = BenchmarkLaunchPlan()

        par_err = self.validate_parallelism()
        if par_err:
            errors.append(par_err)
            return {}, empty_plan, errors

        missing_kfd = self.check_kfd()
        if missing_kfd:
            errors.append(
                f"ROCm device node /dev/kfd not found on {len(missing_kfd)} node(s): "
                f"{', '.join(missing_kfd)}. Run on GPU compute nodes."
            )
            return {}, empty_plan, errors

        plan = self.build_launch_plan()
        if not plan.docker_cmds:
            errors.append("No docker commands generated")
            return {}, plan, errors

        pre_launch_errors = self._pre_launch_validation(plan)
        if pre_launch_errors:
            errors.extend(pre_launch_errors)
            return {}, plan, errors

        mkdir_err = self._create_output_directories(plan)
        if mkdir_err:
            errors.append(mkdir_err)
            return {}, plan, errors

        effective_timeout = self._resolve_run_timeout(timeout)
        log.info(
            "Running %s benchmark (%s) on %d node command(s)%s",
            self._benchmark_name(),
            self._benchmark_mode_label(),
            len(plan.docker_cmds),
            f" [timeout={effective_timeout}s]" if effective_timeout else "",
        )
        if plan.docker_cmds:
            log.debug("Docker command (sample): %s", _redact_secrets(plan.docker_cmds[0]))

        results: Dict[str, str] = {}
        raw_results: Dict[str, Any] = {}
        exec_error: Optional[Exception] = None
        try:
            raw_results = _exec_cmd_list_on_nodes(
                self.s_phdl,
                plan.node_order,
                plan.docker_cmds,
                timeout=effective_timeout,
                detailed=True,
            )
            results = _normalize_exec_results(raw_results, plan.node_order)
        except Exception as exc:
            exec_error = exc
            log.warning("Benchmark docker exec ended with exception: %s", exc)
            results = _normalize_exec_results(raw_results, plan.node_order)

        if exec_error is not None:
            exec_errors, treat_as_success = self._handle_benchmark_exec_exception(exec_error, plan, results)
            if treat_as_success:
                return results, plan, []
            errors.extend(exec_errors)
            return results, plan, errors

        dist_err = self._verify_distributed_output(plan, results)
        if dist_err:
            errors.append(dist_err)

        failed_nodes, extra_errors = self._collect_benchmark_failures(raw_results, plan)
        errors.extend(extra_errors)
        if failed_nodes:
            errors.append(f"Benchmark failed on {len(failed_nodes)} node(s): {', '.join(failed_nodes)}")

        return results or {}, plan, errors

    def store_output_dir_hint(self, plan: BenchmarkLaunchPlan) -> None:
        if plan.primary_output_dir:
            self.inference_dict["_test_output_dir"] = plan.primary_output_dir
            return

        if not self.distributed and len(plan.node_order) == 1:
            node = plan.node_order[0]
            self.inference_dict["_test_output_dir"] = plan.output_dirs_by_node[node]
