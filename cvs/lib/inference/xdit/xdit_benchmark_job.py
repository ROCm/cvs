"""
Shared PyTorch xDiT orchestrated torchrun benchmark job base (FLUX, WAN, etc.).

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import base64
import binascii
import json
import shlex
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cvs.lib import globals

log = globals.log

CONTAINER_OUTPUT_MOUNT = "/outputs"
CONTAINER_HF_TOKEN_PATH = "/run/secrets/hf_token"
_ARTIFACT_BEGIN = "XDIT_ARTIFACTS_BEGIN"
_ARTIFACT_END = "XDIT_ARTIFACTS_END"
_ARTIFACT_COLLECT_TIMEOUT_S = 300
_STAGED_ARTIFACT_DIRS = []

# Controller parse has no view of node-local storage. JSON is copied in full.
# Image and video bodies are omitted so stdout stays under the agent inline cap;
# parse only checks that those files exist.
_REMOTE_COLLECT_SCRIPT = """
import base64, json, os, sys
root = sys.argv[1]
items = []
if os.path.isdir(root):
    for dirpath, _, names in os.walk(root):
        for name in names:
            keep = (
                name == "timing.json"
                or name.endswith(".png")
                or name.endswith(".mp4")
                or (name.startswith("rank0") and name.endswith(".json"))
            )
            if not keep:
                continue
            path = os.path.join(dirpath, name)
            media = name.endswith(".png") or name.endswith(".mp4")
            try:
                with open(path, "rb") as handle:
                    blob = b"" if media else handle.read()
            except OSError:
                continue
            items.append({"rel": os.path.relpath(path, root), "b64": base64.b64encode(blob).decode("ascii")})
sys.stdout.write("XDIT_ARTIFACTS_BEGIN\\n")
sys.stdout.write(json.dumps(items))
sys.stdout.write("\\nXDIT_ARTIFACTS_END\\n")
"""


def remote_benchmark_collect_cmd(output_dir):
    return "python3 -c " + shlex.quote(_REMOTE_COLLECT_SCRIPT) + " " + shlex.quote(str(output_dir))


def _parse_artifact_payload(text):
    if _ARTIFACT_BEGIN not in text or _ARTIFACT_END not in text:
        return None
    raw = text.split(_ARTIFACT_BEGIN, 1)[1].split(_ARTIFACT_END, 1)[0].strip()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list) or not items:
        return None
    return items


def _materialize_artifacts(items):
    root = Path(tempfile.mkdtemp(prefix="xdit-results-"))
    wrote = False
    for item in items:
        rel = str(item.get("rel") or "")
        rel_path = Path(rel)
        if not rel or rel_path.is_absolute() or ".." in rel_path.parts:
            continue
        try:
            blob = base64.b64decode(item.get("b64") or "")
        except (binascii.Error, ValueError, TypeError):
            continue
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        wrote = True
    if not wrote:
        shutil.rmtree(root, ignore_errors=True)
        return None
    staged = str(root)
    _STAGED_ARTIFACT_DIRS.append(staged)
    return staged


def cleanup_staged_artifacts():
    while _STAGED_ARTIFACT_DIRS:
        path = _STAGED_ARTIFACT_DIRS.pop()
        shutil.rmtree(path, ignore_errors=True)
        log.info("Removed staged xDiT artifacts at %s", path)


def stage_remote_benchmark_outputs(s_phdl, output_dirs_by_node):
    """Read benchmark JSON and media on each writer node and stage them locally.

    Returns a node -> local directory map for nodes whose files were readable.
    Nodes that share one remote directory reuse the copy from the node that had it.
    """
    # xdit_flux_job imports this module at the bottom; a top-level import here cycles.
    from cvs.lib.inference.xdit.xdit_flux_job import _exec_cmd_list_on_nodes, _exec_result_output

    nodes = [node for node, path in output_dirs_by_node.items() if path]
    if not nodes:
        return {}
    commands = [remote_benchmark_collect_cmd(output_dirs_by_node[node]) for node in nodes]
    try:
        raw = _exec_cmd_list_on_nodes(s_phdl, nodes, commands, timeout=_ARTIFACT_COLLECT_TIMEOUT_S, print_console=False)
    except Exception as exc:
        log.warning("Could not read xDiT benchmark output from nodes: %s", exc)
        return {}

    payloads = {}
    for node in nodes:
        text = _exec_result_output((raw or {}).get(node))
        items = _parse_artifact_payload(text)
        if items:
            payloads[node] = items
        elif text.strip() and _ARTIFACT_BEGIN not in text:
            log.warning("xDiT benchmark read on %s did not return artifacts: %s", node, text.strip().splitlines()[-1])

    local_by_remote = {}
    local_by_node = {}
    for node in nodes:
        remote_dir = output_dirs_by_node[node]
        items = payloads.get(node)
        if not items or remote_dir in local_by_remote:
            continue
        local_dir = _materialize_artifacts(items)
        if not local_dir:
            continue
        local_by_remote[remote_dir] = local_dir
        log.info("Staged xDiT benchmark files from %s at %s", node, local_dir)
    for node in nodes:
        local_dir = local_by_remote.get(output_dirs_by_node[node])
        if local_dir:
            local_by_node[node] = local_dir
    return local_by_node


def _hf_home_token_install_cmd(hf_home="/hf_home", token_path=CONTAINER_HF_TOKEN_PATH):
    token_q = shlex.quote(token_path)
    dest_dir_q = shlex.quote(str(hf_home).rstrip("/"))
    home_q = shlex.quote(str(hf_home).rstrip("/") + "/token")
    return (
        f"if [ -f {token_q} ]; then (umask 077; mkdir -p {dest_dir_q}; cp {token_q} {home_q}; chmod 600 {home_q}); fi"
    )


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
    """Build and run PyTorch XDit torchrun commands in an externally managed container."""

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
        self.orch = s_phdl
        self.uses_container_orchestrator = isinstance(getattr(s_phdl, "hosts", None), (list, tuple)) and callable(
            getattr(s_phdl, "exec_on_host", None)
        )
        self.inference_dict = inference_dict
        self.hf_token = hf_token
        self.distributed = distributed
        self.cluster_dict = cluster_dict or {}
        self.nproc_per_node = nproc_per_node
        self.server_nodes = self._resolve_execution_nodes()
        self.nnodes = len(self.server_nodes) if self.distributed else 1

    def _resolve_execution_nodes(self) -> List[str]:
        from cvs.lib.inference.xdit.xdit_flux_job import resolve_distributed_execution_hosts

        if self.distributed:
            if not self.cluster_dict:
                raise ValueError("distributed=True requires cluster_dict")
            return resolve_distributed_execution_hosts(self.cluster_dict, self.inference_dict)
        if self.uses_container_orchestrator:
            return list(self.orch.hosts)
        return list(self.s_phdl.host_list)

    @abstractmethod
    def validate_parallelism(self) -> Optional[str]:
        """Return an error message when parallelism config is invalid, else None."""

    def check_kfd(self) -> List[str]:
        from cvs.lib.inference.xdit.xdit_flux_job import _exec_on_nodes

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
        from cvs.lib.inference.xdit.xdit_flux_job import _exec_on_nodes

        log.info("Getting hostnames from %d node(s)", len(self.server_nodes))
        hostname_result = _exec_on_nodes(self.s_phdl, self.server_nodes, "hostname")
        return {node: (hostname_result.get(node, "") or "").strip() or node for node in self.server_nodes}

    def _build_volume_args(self, host_output_dir: str) -> str:
        volume_dict = dict(self.inference_dict["container_config"].get("volume_dict") or {})
        volume_dict[host_output_dir] = CONTAINER_OUTPUT_MOUNT
        volume_dict[self.inference_dict["hf_home"]] = "/hf_home"
        token_file = str(self.inference_dict.get("hf_token_file") or "").strip()
        if token_file:
            volume_dict[token_file] = CONTAINER_HF_TOKEN_PATH
        mount_host = self.inference_dict.get("_resolved_model_mount_host")
        if mount_host:
            volume_dict[mount_host] = "/model"
        return " ".join(f"--mount type=bind,source={src},target={dst}" for src, dst in volume_dict.items())

    @abstractmethod
    def _build_env_dict(self):
        """Return environment values needed by the benchmark command."""

    def _build_env_args(self):
        env_dict = self._build_env_dict()
        return " ".join(f"-e {key}={value}" for key, value in env_dict.items())

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
        token_setup = _hf_home_token_install_cmd(self.inference_dict.get("hf_home_container", "/hf_home"))
        inner_cmd = f"{token_setup}; {torchrun_cmd}"

        if self.uses_container_orchestrator:
            exports = " ".join(
                f"export {key}={shlex.quote(str(value))};" for key, value in self._build_env_dict().items()
            )
            return f"bash -c {shlex.quote(f'{exports} {inner_cmd}')}"

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
            f"bash -c {shlex.quote(inner_cmd)}"
        )

    def build_launch_plan(self) -> BenchmarkLaunchPlan:
        from cvs.lib.inference.xdit.xdit_flux_job import (
            DEFAULT_MASTER_PORT,
            compute_world_size,
            resolve_master_addr,
        )

        node_to_hostname = self._fetch_hostnames()
        output_base_dir = self.inference_dict["output_base_dir"]
        if self.uses_container_orchestrator:
            output_base_dir = self.inference_dict.get("output_base_dir_container") or output_base_dir
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
        from cvs.lib.inference.xdit.xdit_flux_job import DEFAULT_BENCHMARK_TIMEOUT_S

        return timeout if timeout is not None else DEFAULT_BENCHMARK_TIMEOUT_S

    def _benchmark_mode_label(self) -> str:
        return "distributed unified" if self.distributed else "single-node"

    @abstractmethod
    def _benchmark_name(self) -> str:
        """Short benchmark label used in run() log messages."""

    def _create_output_directories(self, plan: BenchmarkLaunchPlan) -> Optional[str]:
        from cvs.lib.inference.xdit.xdit_flux_job import _exec_cmd_list_on_nodes

        log.info("Creating output directories on %d node(s)", len(plan.node_order))
        try:
            _exec_cmd_list_on_nodes(self.s_phdl, plan.node_order, plan.mkdir_cmds)
        except Exception as exc:
            return f"Failed to create output directories: {exc}"
        return None

    def _verify_distributed_output(self, plan: BenchmarkLaunchPlan, results: Mapping[str, str]) -> Optional[str]:
        from cvs.lib.inference.xdit.xdit_flux_job import verify_distributed_logs

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
        from cvs.lib.inference.xdit.xdit_flux_job import (
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
        from cvs.lib.inference.xdit.xdit_flux_job import (
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
            log.debug("Benchmark command (sample): %s", _redact_secrets(plan.docker_cmds[0]))

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
            log.warning("Benchmark container exec ended with exception: %s", exc)
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

    def _host_output_path(self, output_dir: str) -> str:
        if not self.uses_container_orchestrator:
            return output_dir
        container_base = str(self.inference_dict.get("output_base_dir_container") or "").rstrip("/")
        host_base = str(self.inference_dict.get("output_base_dir") or "").rstrip("/")
        if container_base and host_base and output_dir.startswith(container_base + "/"):
            return host_base + output_dir[len(container_base) :]
        return output_dir

    def store_output_dir_hint(self, plan: BenchmarkLaunchPlan) -> None:
        remote_by_node = {node: path for node, path in plan.output_dirs_by_node.items() if path}
        staged = stage_remote_benchmark_outputs(self.s_phdl, remote_by_node)
        by_node = {node: staged.get(node) or self._host_output_path(path) for node, path in remote_by_node.items()}
        if by_node:
            self.inference_dict["_test_output_dirs_by_node"] = by_node

        if plan.primary_output_dir:
            writer = next(
                (node for node, path in remote_by_node.items() if path == plan.primary_output_dir and node in staged),
                None,
            )
            if writer:
                self.inference_dict["_test_output_dir"] = staged[writer]
            else:
                self.inference_dict["_test_output_dir"] = self._host_output_path(plan.primary_output_dir)
            return

        if not self.distributed and len(plan.node_order) == 1 and plan.node_order[0] in by_node:
            self.inference_dict["_test_output_dir"] = by_node[plan.node_order[0]]
