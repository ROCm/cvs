"""Aorta execution and artifacts through the shared orchestrator.

Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import logging
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from cvs.lib.benchmark.aorta import aorta_artifacts
from cvs.lib.utils.log_poller import LogPoller
from cvs.lib import verify_lib

log = logging.getLogger(__name__)
_COMMAND_OK = "CVS_AORTA_COMMAND_OK"


def _text(result):
    """Normalize orchestrator text and detailed results."""
    return result if isinstance(result, str) else (result or {}).get("output", "")


class AortaJob:
    """Run Aorta without owning container or transport lifetimes.

    The artifact accessors also satisfy the existing parsers' input protocol;
    execution does not depend on the legacy runner/result hierarchy. Duration
    is zero until benchmark execution starts.
    """

    def __init__(self, orch, variant_config, node_vpc_ips=None):
        self.orch = orch
        self.config = variant_config
        self.hosts = list(orch.hosts)
        if not self.hosts:
            raise ValueError("Aorta requires at least one cluster node")
        self.head_node = self.hosts[0]
        self.node_vpc_ips = node_vpc_ips or {}
        self.mode = self.config.multi_node.master_launch_mode
        if self.mode == "auto":
            self.mode = "script" if len(self.hosts) == 1 else "torchrun"
        if self.mode == "script" and len(self.hosts) > 1:
            raise ValueError("master_launch_mode='script' requires one node; use auto or torchrun")
        self.run_id = uuid.uuid4().hex
        self.mount = PurePosixPath(self.config.container_mount_path)
        self.relative_work = PurePosixPath(".cvs-aorta") / self.run_id
        self.work_dirs = [self.mount / self.relative_work / f"node_{i}" for i in range(len(self.hosts))]
        self.output_dir = Path(self.config.output_dir) / self.run_id
        self.artifacts = {}
        self.collection_errors = {}
        self.trace_trees = {}
        self.owners = {}
        self.launch_commands = []
        self.log_paths = []
        self.pid_paths = []
        self.phase_logs = {}
        self.prepared = False
        self.started = False
        self.start_time = None
        self.end_time = None
        self.trace_floors = {}
        self.kernel_start = {}
        self.render_gids = set()
        self.status = "pending"
        self.error_message = ""

    @property
    def succeeded(self):
        return self.status == "completed"

    @property
    def duration_seconds(self):
        if self.start_time is None:
            return 0
        return (self.end_time or time.time()) - self.start_time

    def get_artifact(self, name):
        """Return a local parser artifact."""
        return self.artifacts.get(name)

    def _exec(self, command, hosts=None, on_host=False, timeout=60):
        """Run a checked command; blocking operations must override the short timeout."""
        hosts = self.hosts if hosts is None else hosts
        execute = self.orch.exec_on_host if on_host else self.orch.exec
        results = execute(command, hosts=hosts, timeout=timeout, detailed=True, print_console=False)
        failed = [host for host in hosts if (results or {}).get(host, {}).get("exit_code") != 0]
        if failed:
            detail = {host: _text((results or {}).get(host))[-2000:] for host in failed}
            raise RuntimeError(f"Aorta command failed on {failed}: {detail}")
        return {host: _text(results[host]) for host in hosts}

    def _exec_list(self, commands, timeout=60):
        # exec_cmd_list has no detailed-result option. A marker emitted only on
        # success distinguishes empty output from a failed or missing node.
        wrapped = [f"bash -c {shlex.quote(cmd)} && printf '\\n{_COMMAND_OK}\\n'" for cmd in commands]
        results = self.orch.exec_cmd_list(wrapped, timeout=timeout, print_console=False)
        failed = [host for host in self.hosts if not _text((results or {}).get(host)).rstrip().endswith(_COMMAND_OK)]
        if failed:
            detail = {host: _text((results or {}).get(host))[-2000:] for host in failed}
            raise RuntimeError(f"Aorta per-node command failed on {failed}: {detail}")
        return {host: _text(results[host]).rstrip()[: -len(_COMMAND_OK)].strip() for host in self.hosts}

    def prepare_hosts(self):
        """Record host identity, resolve device groups, and prepare the repository."""
        owners = self._exec("id -u && id -g && (getent group render | cut -d: -f3 || true)", on_host=True)
        render_gids = {}
        for host, output in owners.items():
            identity = output.split()
            uid, gid = identity[:2]
            self.owners[host] = f"{int(uid)}:{int(gid)}"
            if len(identity) > 2 and identity[2].isdigit():
                render_gids[host] = identity[2]
        self.render_gids = set(render_gids.values())
        path = PurePosixPath(self.config.aorta_path)
        if self.config.aorta_auto_clone:
            command = (
                f"mkdir -p {shlex.quote(str(path.parent))} && "
                f"if ! test -f {shlex.quote(str(path / self.config.base_config))}; then "
                f"git clone -- {shlex.quote(self.config.aorta_clone_url)} {shlex.quote(str(path))}; fi"
            )
            self._exec(command, on_host=True, timeout=600)
        else:
            self._exec(f"mkdir -p {shlex.quote(str(path))}", on_host=True)
        check = " && ".join(f"test -f {shlex.quote(str(path / entry))}" for entry in self._required_entry_points())
        self._exec(check, on_host=True, timeout=600)

    def container_groups(self):
        """Return configured groups plus host render GIDs for container launch."""
        configured = self.config.container.runtime.args.get("group_add", [])
        return list(dict.fromkeys(["video", *configured, *sorted(self.render_gids)]))

    def _required_entry_points(self):
        required = [self.config.base_config]
        if not self.config.skip_rccl_build:
            required.append(self.config.build_script)
        required.append(self.config.experiment_script if self.mode == "script" else self.config.multi_node.train_script)
        return required

    def clone_or_verify_aorta_repo(self):
        """Verify the selected entry points through the mounted repository."""
        command = " && ".join(
            f"test -f {shlex.quote(str(self.mount / entry))}" for entry in self._required_entry_points()
        )
        self._exec(command, timeout=600)

    def setup_distributed(self):
        """Verify torchrun and the explicitly configured RDMA devices."""
        command = "command -v torchrun"
        devices = self.config.container.runtime.args.get("devices", [])
        for device in devices:
            target = device.split(":")[1] if ":" in device else device
            if target.startswith("/dev/infiniband/"):
                command += f" && test -e {shlex.quote(target)}"
        self._exec(command)

    def _prepare_work_dirs(self):
        if self.prepared:
            return
        source = Path(aorta_artifacts.__file__).read_text()
        self._exec_list(
            [
                f"mkdir -p {shlex.quote(str(path))} && "
                f"printf %s {shlex.quote(source)} > {shlex.quote(str(path / 'artifacts.py'))}"
                for path in self.work_dirs
            ]
        )
        self.prepared = True

    def _build_base_env(self):
        env = dict(self.config.container.env)
        env.setdefault("NCCL_MAX_NCHANNELS", "112")
        env.setdefault("NCCL_MAX_P2P_NCHANNELS", "112")
        env.setdefault("NCCL_DEBUG", "VERSION")
        env.setdefault("TORCH_NCCL_HIGH_PRIORITY", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("RCCL_MSCCL_ENABLE", "0")
        env["rccl_path"] = self.config.rccl.build_path
        env["RCCL_CLONE_URL"] = self.config.rccl.clone_url
        env["RCCL_BRANCH"] = self.config.rccl.branch
        if self.config.training_overrides:
            tokens = " ".join(f"{key}={value}" for key, value in self.config.training_overrides.items())
            env["AORTA_OVERRIDE_ARGS"] = f"--override {tokens}"
        env.update({key: str(value) for key, value in self.config.multi_node.extra_env.items()})
        env.setdefault("TENSILE_STREAMK_MAX_CUS", str(256 - int(env["NCCL_MAX_NCHANNELS"])))
        return env

    def _build_experiment_command(self):
        parts = ["bash", str(self.mount / self.config.experiment_script), str(self.mount / self.config.base_config)]
        if self.config.training_overrides:
            parts += ["--override", *(f"{key}={value}" for key, value in self.config.training_overrides.items())]
        return shlex.join(parts)

    def _build_torchrun_command(self, node_rank, master_addr, master_port):
        mn = self.config.multi_node
        parts = [
            "torchrun",
            f"--nnodes={len(self.hosts)}",
            f"--node_rank={node_rank}",
            f"--nproc_per_node={mn.nproc_per_node or self.config.gpus_per_node}",
            f"--master_addr={shlex.quote(master_addr)}",
            f"--master_port={master_port}",
            *mn.extra_torchrun_args,
            shlex.quote(str(self.mount / mn.train_script)),
            "--config",
            shlex.quote(str(self.mount / self.config.base_config)),
        ]
        if self.config.training_overrides:
            parts.append("--override")
            parts.extend(f"{key}={shlex.quote(str(value))}" for key, value in self.config.training_overrides.items())
        parts.extend(mn.extra_train_args)
        return "bash -lc " + shlex.quote(" ".join(parts))

    def build_launch_cmd(self):
        """Resolve rendezvous on the first cluster node and build every rank's command."""
        if self.mode == "script":
            self.launch_commands = [self._build_experiment_command()]
            return self.launch_commands
        mn = self.config.multi_node
        self.master_addr = mn.master_addr or self.node_vpc_ips.get(self.head_node) or self.head_node
        self.master_port = mn.master_port
        if not self.master_port:
            snippet = "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
            output = self._exec(f"python3 -c {shlex.quote(snippet)}", hosts=[self.head_node])
            self.master_port = int(output[self.head_node].strip())
            if not 1024 <= self.master_port <= 65535:
                raise ValueError("Invalid rendezvous port returned by head node")
        self.launch_commands = [
            self._build_torchrun_command(rank, self.master_addr, self.master_port) for rank in range(len(self.hosts))
        ]
        return self.launch_commands

    def _start_phase(self, commands, label):
        self._prepare_work_dirs()
        env = self._build_base_env()
        exports = [f"export {key}={shlex.quote(str(value))}" for key, value in env.items()]
        if "LD_LIBRARY_PATH" not in env:
            prefix = (
                f"{self.config.rccl.build_path}/build/release/:/opt/rocm/lib:/opt/rocm/lib64:"
                "/opt/openmpi/lib:/opt/rccl-tests/build:"
            )
            exports.append(f"export LD_LIBRARY_PATH={shlex.quote(prefix)}\"${{LD_LIBRARY_PATH:-}}\"")
        scripts = []
        self.log_paths = [str(path / f"{label}.log") for path in self.work_dirs]
        self.phase_logs[label] = list(self.log_paths)
        self.pid_paths = [str(path / f"{label}.pid") for path in self.work_dirs]
        for rank, (path, command) in enumerate(zip(self.work_dirs, commands)):
            script = "\n".join(
                [
                    "#!/bin/bash",
                    "trap 'rc=$?; printf \"\\nCVS_AORTA_EXIT=%s\\n\" \"$rc\"' EXIT",
                    "set -e",
                    f"cd {shlex.quote(str(self.mount))}",
                    *exports,
                    f"export NODE_RANK={rank}",
                    f"export MASTER_ADDR={shlex.quote(getattr(self, 'master_addr', self.head_node))}",
                    f"export MASTER_PORT={getattr(self, 'master_port', 0)}",
                    command,
                ]
            )
            scripts.append(f"printf %s {shlex.quote(script)} > {shlex.quote(str(path / (label + '.sh')))}")
        self._exec_list(scripts)
        self._exec_list(
            [
                f"nohup setsid bash {shlex.quote(str(path / (label + '.sh')))} "
                f"> {shlex.quote(self.log_paths[rank])} 2>&1 < /dev/null & "
                f"printf '%s\\n' \"$!\" > {shlex.quote(self.pid_paths[rank])}"
                for rank, path in enumerate(self.work_dirs)
            ]
        )

    def _phase_exit_codes(self):
        results = self._exec_list(
            [
                f"awk -F= '/^CVS_AORTA_EXIT=[0-9]+$/ {{code=$2}} END {{if (code != \"\") print code}}' "
                f"{shlex.quote(path)} 2>/dev/null"
                for path in self.log_paths
            ]
        )
        return {host: int(value) for host, value in results.items() if value.isdigit()}

    def _phase_finished(self):
        return len(self._phase_exit_codes()) == len(self.hosts)

    def _poll_phase(self):
        warned = set()

        def is_complete():
            failed_now = {host for host, code in self._phase_exit_codes().items() if code != 0} - warned
            for host in failed_now:
                log.warning("Aorta command failed early on %s; waiting for remaining hosts", host)
            warned.update(failed_now)
            return self._phase_finished()

        try:
            LogPoller(
                self.orch,
                self.log_paths,
                is_complete=is_complete,
                timeout_s=self.config.timeout_seconds,
                error_label="Aorta",
                label="Aorta in progress",
                log=log,
            ).poll()
        except RuntimeError as exc:
            if "did not complete within" in str(exc):
                raise TimeoutError(str(exc)) from exc
            raise
        failed = {host: code for host, code in self._phase_exit_codes().items() if code != 0}
        if failed:
            raise RuntimeError(f"Aorta command failed on nodes: {failed}")

    def build_rccl(self):
        """Run the configured RCCL build script with bounded shared log polling."""
        if self.config.skip_rccl_build:
            return
        command = f"bash {shlex.quote(str(self.mount / self.config.build_script))}"
        try:
            self._start_phase([command] * len(self.hosts), "rccl")
            self._poll_phase()
        finally:
            try:
                self.stop_processes()
            except RuntimeError as exc:
                log.warning("Failed to stop RCCL build process groups: %s", exc)

    def start_job(self):
        """Launch all rank groups concurrently, recording node-local freshness floors."""
        if not self.launch_commands:
            raise RuntimeError("Call build_launch_cmd before start_job")
        self.trace_floors = {host: float(value.strip()) for host, value in self._exec("date +%s.%N").items()}
        self.start_time = time.time()
        self.status = "running"
        self.started = True
        try:
            self._start_phase(self.launch_commands, "benchmark")
        except Exception as exc:
            self.status = "failed"
            self.error_message = str(exc)
            self.end_time = time.time()
            raise

    def poll_for_completion(self):
        """Wait for every rank group to exit successfully."""
        try:
            self._poll_phase()
            self.status = "completed"
        except TimeoutError as exc:
            self.status = "timeout"
            self.error_message = str(exc)
            raise
        except Exception as exc:
            self.status = "failed"
            self.error_message = str(exc)
            raise
        finally:
            self.end_time = time.time()

    def record_kernel_start(self):
        """Record each host's own clock before the GPU workload."""
        self.kernel_start = self._exec("date +'%a %b %e %H:%M:%S'", on_host=True)

    def check_kernel_errors(self):
        """Scan the workload's kernel window through the shared verifier."""
        ends = self._exec("date +'%a %b %e %H:%M:%S'", on_host=True)
        handle = SimpleNamespace(
            exec=lambda cmd: self.orch.exec_on_host(
                cmd,
                hosts=self.hosts,
                timeout=120,
                print_console=False,
            )
        )
        verify_lib.verify_dmesg_for_errors(handle, self.kernel_start, ends, till_end_flag=False)

    def _download_archive(self, host, rank, filename, destination):
        archive = self.work_dirs[rank] / filename
        remote = PurePosixPath(self.config.aorta_path) / archive.relative_to(self.mount)
        owner = self.owners.get(host)
        if owner:
            self._exec(f"chown {owner} {shlex.quote(str(archive))}", hosts=[host])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = self.orch.download_file(str(remote), str(self.output_dir / f"{rank}-{filename}"), hosts=[host])
        if host not in downloaded:
            raise OSError(f"No artifact download returned for {host}")
        local_archive = Path(downloaded[host])
        try:
            # A corrupt transfer must not leave a partial node tree in the
            # combined directory that the parser would treat as usable data.
            with tempfile.TemporaryDirectory(dir=self.output_dir) as staging:
                staged = Path(staging) / "artifacts"
                aorta_artifacts.extract_artifacts(local_archive, staged)
                if staged.exists():
                    destination = Path(destination)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        shutil.rmtree(destination)
                    staged.replace(destination)
            self._exec(f"rm -f -- {shlex.quote(str(archive))}", hosts=[host])
        finally:
            local_archive.unlink(missing_ok=True)

    def collect_traces(self):
        """Download fresh per-node profiler trees, retaining surviving nodes on failure."""
        if not self.started:
            raise RuntimeError("Benchmark has not started")
        self._prepare_work_dirs()
        combined = self.mode == "torchrun" and self.config.multi_node.collect_traces
        root = self.output_dir / ("combined_traces" if combined else "traces")
        self.collection_errors = {}
        for rank, host in enumerate(self.hosts if combined else self.hosts[:1]):
            work = self.work_dirs[rank]
            destination = root / f"node_{rank}" if combined else root
            command = (
                f"python3 {shlex.quote(str(work / 'artifacts.py'))} traces {shlex.quote(str(self.mount))} "
                f"{shlex.quote(str(work / 'traces.tar.gz'))} --min-mtime {self.trace_floors[host]}"
                + ("" if combined else " --latest")
            )
            try:
                output = self._exec(command, hosts=[host], timeout=self.config.timeout_seconds)
                trees = json.loads(output[host])
                if not trees:
                    raise RuntimeError("No fresh torch_profiler files found")
                self._download_archive(host, rank, "traces.tar.gz", destination)
                self.trace_trees[host] = trees
                self.artifacts["torch_traces"] = root if combined else destination / trees[0]
            except (RuntimeError, OSError, ValueError, tarfile.TarError) as exc:
                self.collection_errors[host] = str(exc)
                log.warning("Trace collection failed on %s: %s", host, exc)
        return self.artifacts.get("torch_traces")

    def collect_logs(self):
        """Download each launched phase's logs through the shared transport."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for phase, paths in self.phase_logs.items():
            for rank, (host, path) in enumerate(zip(self.hosts, paths)):
                key = f"{phase}_log_node_{rank}"
                if key in self.artifacts:
                    continue
                remote = PurePosixPath(self.config.aorta_path) / PurePosixPath(path).relative_to(self.mount)
                try:
                    downloaded = self.orch.download_file(
                        str(remote), str(self.output_dir / f"{phase}-node-{rank}.log"), hosts=[host]
                    )
                    if host not in downloaded:
                        raise OSError(f"No log download returned for {host}")
                    self.artifacts[key] = Path(downloaded[host])
                    if phase == "benchmark":
                        self.artifacts.setdefault("training_log", self.artifacts[key])
                except (OSError, ValueError) as exc:
                    log.warning("Could not collect %s log from %s: %s", phase, host, exc)

    def run_analysis(self):
        """Run optional analysis on the head's original trace tree and download reports."""
        trees = self.trace_trees.get(self.head_node)
        if not trees:
            log.warning("No head-node traces available for optional analysis")
            return
        relative_output = PurePosixPath(trees[0]).parent
        output = self.mount / relative_output
        reports = output / "tracelens_analysis"
        work = self.work_dirs[0]
        analysis = self.config.analysis
        for enabled, script, artifact in (
            (analysis.enable_tracelens, analysis.tracelens_script, "tracelens_analysis"),
            (analysis.enable_gemm_analysis, analysis.gemm_script, "gemm_analysis"),
        ):
            if not enabled:
                continue
            try:
                command = f"cd {shlex.quote(str(self.mount))} && "
                if artifact == "tracelens_analysis":
                    command += "python3 -c 'import TraceLens' && "
                command += f"bash {shlex.quote(str(self.mount / script))} {shlex.quote(str(output))}"
                if analysis.skip_if_exists:
                    command = f"test -d {shlex.quote(str(reports))} || ( {command} )"
                self._exec(command, hosts=[self.head_node], timeout=self.config.timeout_seconds)
                archive = work / "reports.tar.gz"
                report_min_mtime = 0 if analysis.skip_if_exists else self.trace_floors[self.head_node]
                self._exec(
                    f"test -d {shlex.quote(str(reports))} && "
                    f"python3 {shlex.quote(str(work / 'artifacts.py'))} reports "
                    f"{shlex.quote(str(reports))} {shlex.quote(str(archive))} "
                    f"--min-mtime {report_min_mtime}",
                    hosts=[self.head_node],
                    timeout=self.config.timeout_seconds,
                )
                local_reports = self.output_dir / "analysis" / artifact
                self._download_archive(self.head_node, 0, "reports.tar.gz", local_reports)
                if local_reports.is_dir():
                    self.artifacts[artifact] = local_reports
            except (RuntimeError, OSError, ValueError, tarfile.TarError) as exc:
                log.warning("Optional %s failed; raw traces remain available: %s", artifact, exc)

    def stop_processes(self):
        """Stop only process groups launched by this job, including on polling failure."""
        if not self.pid_paths:
            return
        commands = []
        for path in self.pid_paths:
            commands.append(
                f"if test -f {shlex.quote(path)}; then pid=$(cat {shlex.quote(path)}); "
                'case "$pid" in ""|*[!0-9]*) exit 1;; esac; '
                'kill -TERM -- -"$pid" 2>/dev/null || true; '
                'kill -KILL -- -"$pid" 2>/dev/null || true; '
                f"rm -f {shlex.quote(path)}; fi"
            )
        self._exec_list(commands)
        self.pid_paths = []

    def teardown(self):
        """Restore repository ownership; container teardown belongs to the fixture."""
        errors = []
        try:
            self.stop_processes()
        except RuntimeError as exc:
            errors.append(str(exc))
        try:
            self.collect_logs()
        except (OSError, ValueError) as exc:
            errors.append(f"Log collection failed: {exc}")
        if self.owners:
            commands = [
                (f"chown -R {self.owners[host]} {shlex.quote(str(self.mount))}" if host in self.owners else "true")
                for host in self.hosts
            ]
            try:
                self._exec_list(commands, timeout=120)
            except RuntimeError as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))
