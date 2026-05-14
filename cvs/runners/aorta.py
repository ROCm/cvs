"""
Aorta PyTorch benchmark runner.

Deploys Docker containers and executes distributed training benchmarks
using the Aorta framework. Uses Docker SDK over SSH for container orchestration.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import logging
import shlex
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import docker
    from docker.models.containers import Container

try:
    import docker
    from docker.models.containers import Container

    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False
    docker = None  # type: ignore
    Container = None  # type: ignore

from cvs.runners._base_runner import BaseRunner, RunConfig, RunResult, RunStatus

log = logging.getLogger(__name__)


def combined_traces_in(path: Path, root: Path) -> bool:
    """Return True if ``path`` lives under ``root/combined_traces``.

    Used to skip already-collected traces when rescanning the head node so
    repeated runs do not nest combined_traces inside itself.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return rel.parts and rel.parts[0] == "combined_traces"


@dataclass
class RcclConfig:
    """RCCL build and runtime configuration."""

    clone_url: str = "https://github.com/ROCmSoftwarePlatform/rccl.git"
    branch: str = "develop"
    build_path: str = "/mnt/rccl"


@dataclass
class AortaDockerConfig:
    """Docker container configuration for Aorta."""

    image: str = "jeffdaily/pytorch:torchrec-dlrm-complete"
    container_name: str = "aorta-benchmark"
    shm_size: str = "17G"
    network_mode: str = "host"
    privileged: bool = True


@dataclass
class AortaEnvironment:
    """Environment variables for RCCL/NCCL tuning."""

    NCCL_MAX_NCHANNELS: int = 112
    NCCL_MAX_P2P_NCHANNELS: int = 112
    NCCL_DEBUG: str = "VERSION"
    TORCH_NCCL_HIGH_PRIORITY: int = 1
    OMP_NUM_THREADS: int = 1
    RCCL_MSCCL_ENABLE: int = 0

    def to_dict(self) -> Dict[str, str]:
        """Convert to environment dict with computed values."""
        nch = self.NCCL_MAX_NCHANNELS
        return {
            "NCCL_MAX_NCHANNELS": str(nch),
            "NCCL_MAX_P2P_NCHANNELS": str(self.NCCL_MAX_P2P_NCHANNELS),
            "TENSILE_STREAMK_MAX_CUS": str(256 - nch),
            "NCCL_DEBUG": self.NCCL_DEBUG,
            "TORCH_NCCL_HIGH_PRIORITY": str(self.TORCH_NCCL_HIGH_PRIORITY),
            "OMP_NUM_THREADS": str(self.OMP_NUM_THREADS),
            "RCCL_MSCCL_ENABLE": str(self.RCCL_MSCCL_ENABLE),
        }


@dataclass
class AortaAnalysisConfig:
    """Configuration for post-benchmark analysis using Aorta's built-in scripts."""

    enable_tracelens: bool = True
    enable_gemm_analysis: bool = False
    tracelens_script: str = "scripts/tracelens_single_config/run_tracelens_single_config.sh"
    gemm_script: str = "scripts/gemm_analysis/run_tracelens_analysis.sh"
    skip_if_exists: bool = False


@dataclass
class AortaMultiNodeConfig:
    """
    Multi-node disaggregated launch configuration.

    See ``AortaMultiNodeConfigFile`` in ``cvs/parsers/schemas.py`` for the
    YAML-facing description of each field.
    """

    master_launch_mode: str = "auto"
    nproc_per_node: Optional[int] = None
    master_port: Optional[int] = None
    master_addr: Optional[str] = None
    train_script: str = "train.py"
    extra_torchrun_args: List[str] = field(default_factory=list)
    extra_train_args: List[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)
    collect_traces: bool = True


@dataclass
class AortaConfig(RunConfig):
    """
    Configuration for Aorta benchmark runner.

    Extends base RunConfig with Aorta-specific settings.
    """

    # Path to aorta repository on host (will be bind-mounted)
    # NOTE: No default - must be explicitly provided via config file
    aorta_path: Path = field(default_factory=Path)

    # Optional: clone repo when aorta_path does not exist
    aorta_auto_clone: bool = False
    aorta_clone_url: Optional[str] = None

    # Mount point inside container
    container_mount_path: str = "/mnt"

    # Aorta base config file (relative to aorta_path)
    base_config: str = "config/distributed.yaml"

    # Training config overrides (passed via --override)
    training_overrides: Dict[str, Any] = field(default_factory=dict)

    # Docker configuration
    docker: AortaDockerConfig = field(default_factory=AortaDockerConfig)

    # RCCL configuration
    rccl: RcclConfig = field(default_factory=RcclConfig)

    # Environment configuration
    environment: AortaEnvironment = field(default_factory=AortaEnvironment)

    # Analysis configuration (use Aorta's built-in scripts)
    analysis: AortaAnalysisConfig = field(default_factory=AortaAnalysisConfig)

    # Multi-node disaggregated launch configuration
    multi_node: AortaMultiNodeConfig = field(default_factory=AortaMultiNodeConfig)

    # Scripts to execute (relative to container mount)
    build_script: str = "scripts/build_rccl.sh"
    experiment_script: str = "scripts/rccl_exp.sh"

    # Number of GPUs per node
    gpus_per_node: int = 8

    # Whether to skip RCCL build (if already built)
    skip_rccl_build: bool = False


class AortaRunner(BaseRunner):
    """
    Runner for Aorta PyTorch distributed benchmarks.

    Uses Docker SDK over SSH to:
    1. Deploy container with GPU access
    2. Build RCCL from source (optional)
    3. Execute distributed training
    4. Collect profiling artifacts
    """

    def __init__(self, config: AortaConfig):
        """
        Initialize Aorta runner.

        Args:
            config: Aorta benchmark configuration
        """
        if not DOCKER_SDK_AVAILABLE:
            raise ImportError("Docker SDK not available. Install with: pip install docker")

        super().__init__(config)
        self.config: AortaConfig = config  # Type hint for IDE

        # Thread-safe storage for parallel deployment
        self._docker_clients: Dict[str, docker.DockerClient] = {}
        self._containers: Dict[str, Container] = {}
        self._lock = Lock()  # Protects _docker_clients and _containers

    def validate_config(self) -> List[str]:
        """Validate Aorta-specific configuration."""
        errors = super().validate_config()

        aorta_exists = self.config.aorta_path.exists()
        if not aorta_exists and not (self.config.aorta_auto_clone and self.config.aorta_clone_url):
            errors.append(
                f"Aorta path does not exist: {self.config.aorta_path} (set aorta_auto_clone and aorta_clone_url to clone)"
            )

        if not aorta_exists:
            return errors  # Will clone in setup(); skip path checks

        config_path = self.config.aorta_path / self.config.base_config
        if not config_path.exists():
            errors.append(f"Base config does not exist: {config_path}")

        build_script = self.config.aorta_path / self.config.build_script
        if not build_script.exists():
            errors.append(f"Build script does not exist: {build_script}")

        exp_script = self.config.aorta_path / self.config.experiment_script
        if not exp_script.exists():
            errors.append(f"Experiment script does not exist: {exp_script}")

        resolved_mode = self._resolve_launch_mode()
        if resolved_mode == "torchrun":
            train_script = self.config.aorta_path / self.config.multi_node.train_script
            if not train_script.exists():
                errors.append(f"multi_node.train_script does not exist: {train_script}")

        return errors

    def _connect_docker(self, node: str) -> docker.DockerClient:
        """
        Connect to Docker daemon on a node via SSH (thread-safe).

        Args:
            node: Hostname or IP of the node

        Returns:
            Docker client connected to the node
        """
        # Check cache first (read without lock for performance)
        if node in self._docker_clients:
            return self._docker_clients[node]

        # Build SSH URL
        ssh_url = f"ssh://{self.config.username}@{node}"
        log.info(f"Connecting to Docker daemon at {ssh_url}")

        client = docker.DockerClient(
            base_url=ssh_url,
            use_ssh_client=True,
        )

        # Verify connection
        client.ping()
        log.info(f"Connected to Docker on {node}")

        # Thread-safe update of cache
        with self._lock:
            self._docker_clients[node] = client
        return client

    def _cleanup_existing_containers(self, client: docker.DockerClient, node: str):
        """Remove any existing containers with our name."""
        container_name = self.config.docker.container_name
        try:
            existing = client.containers.get(container_name)
            log.info(f"Removing existing container {container_name} on {node}")
            existing.stop(timeout=10)
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass  # Container doesn't exist, that's fine
        except Exception as e:
            log.warning(f"Error cleaning up container on {node}: {e}")

    def _collect_multi_node_traces(self, nodes: List[str]) -> Optional[Path]:
        """
        Collect torch_profiler trees from every node into a single tree on the
        head node and return the parent directory.

        Layout::

            <aorta_path>/combined_traces/node_<rank>/<orig_subpath>/torch_profiler/...

        The head node is rsynced locally; non-head nodes are pulled with rsync
        over SSH (``rsync -az`` with the configured ``priv_key_file``). When
        rsync is unavailable we fall back to ``scp -r``. Failures on
        individual nodes are logged but do not abort the overall collection;
        the returned directory is the best-effort union.

        Returns ``None`` only when nothing could be collected at all.
        """
        head = self.head_node
        combined_root = self.config.aorta_path / "combined_traces"
        try:
            combined_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error(f"Cannot create {combined_root}: {e}")
            return None

        any_collected = False
        for rank, node in enumerate(nodes):
            dest = combined_root / f"node_{rank}"
            dest.mkdir(parents=True, exist_ok=True)

            try:
                # First pass: copy from the orchestrator's local filesystem. This handles
                # the head==orchestrator case and any NFS-shared aorta_path.
                found = False
                if node == head:
                    found = self._copy_local_torch_profilers(self.config.aorta_path, dest)
                # Pull over SSH for non-head nodes, and also for the head when the
                # orchestrator's local fs didn't actually have the head's traces (i.e.
                # orchestrator is a separate login node from the head).
                if not found:
                    found = self._copy_remote_torch_profilers(node, dest)
                if found:
                    any_collected = True
                    log.info(f"Collected traces for node_{rank} ({node}) -> {dest}")
                else:
                    log.warning(f"No torch_profiler artifacts found for node {node} (rank {rank})")
            except Exception as e:
                log.warning(f"Failed to collect traces for node {node} (rank {rank}): {e}")

        return combined_root if any_collected else None

    def _copy_local_torch_profilers(self, src_root: Path, dest: Path) -> bool:
        """
        Copy any ``torch_profiler/`` trees under ``src_root`` into ``dest``,
        preserving the relative path. Used for the head node.
        """
        import shutil

        copied = False
        for tp in src_root.glob("**/torch_profiler"):
            if not tp.is_dir():
                continue
            if combined_traces_in(tp, src_root):
                continue
            rel = tp.relative_to(src_root)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(tp, target, symlinks=True, dirs_exist_ok=False)
                copied = True
            except OSError as e:
                log.warning(f"Local copy {tp} -> {target} failed: {e}")
        return copied

    def _copy_remote_torch_profilers(self, node: str, dest: Path) -> bool:
        """
        Pull every ``torch_profiler/`` tree under the remote ``aorta_path`` to
        ``dest`` using rsync over SSH. Falls back to ``scp -r`` if rsync is
        unavailable.
        """
        ssh_user = self.config.username
        remote_root = str(self.config.aorta_path)

        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if self.config.pkey:
            ssh_opts.extend(["-i", self.config.pkey])
        ssh_cmd = "ssh " + " ".join(shlex.quote(p) for p in ssh_opts)

        list_cmd = [
            "ssh",
            *ssh_opts,
            f"{ssh_user}@{node}",
            f"find {shlex.quote(remote_root)} -type d -name torch_profiler -not -path '*/combined_traces/*'",
        ]
        try:
            r = subprocess.run(list_cmd, capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.warning(f"Listing remote torch_profiler dirs on {node} failed: {e}")
            return False

        if r.returncode != 0:
            log.warning(f"find on {node} returned {r.returncode}: {r.stderr.strip()}")
            return False

        remote_paths = [p.strip() for p in r.stdout.splitlines() if p.strip()]
        if not remote_paths:
            return False

        copied = False
        rsync_available = (
            subprocess.run(
                ["bash", "-lc", "command -v rsync >/dev/null"],
                capture_output=True,
            ).returncode
            == 0
        )
        for rp in remote_paths:
            try:
                rel = Path(rp).relative_to(remote_root)
            except ValueError:
                rel = Path(Path(rp).name)
            target_parent = dest / rel.parent
            target_parent.mkdir(parents=True, exist_ok=True)

            if rsync_available:
                cmd = [
                    "rsync",
                    "-az",
                    "-e",
                    ssh_cmd,
                    f"{ssh_user}@{node}:{rp}/",
                    str(target_parent / rel.name) + "/",
                ]
            else:
                cmd = ["scp", "-r", *ssh_opts, f"{ssh_user}@{node}:{rp}", str(target_parent)]

            log.info(f"[{node}] copying {rp} -> {target_parent / rel.name}")
            try:
                rr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if rr.returncode == 0:
                    copied = True
                else:
                    log.warning(f"copy of {rp} from {node} failed (exit {rr.returncode}): {rr.stderr.strip()}")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                log.warning(f"copy of {rp} from {node} failed: {e}")

        return copied

    def _get_remote_uid_gid(self, node: str) -> Optional[Tuple[int, int]]:
        """
        Get UID and GID for config.username on the given node via SSH.
        Used so the container can run as the host user and avoid permission issues (e.g. no chmod 777).
        """
        try:
            uid_out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{self.config.username}@{node}", "id -u"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            gid_out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{self.config.username}@{node}", "id -g"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if (
                uid_out.returncode == 0
                and gid_out.returncode == 0
                and uid_out.stdout.strip()
                and gid_out.stdout.strip()
            ):
                return (int(uid_out.stdout.strip()), int(gid_out.stdout.strip()))
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as e:
            log.debug(f"Could not get remote UID/GID for {node}: {e}")
        return None

    def _launch_container(self, client: docker.DockerClient, node: str) -> Container:
        """
        Launch Aorta container on a node.

        Args:
            client: Docker client for the node
            node: Node hostname

        Returns:
            Running container object
        """
        cfg = self.config.docker

        # Build volume mounts
        volumes = {str(self.config.aorta_path): {"bind": self.config.container_mount_path, "mode": "rw"}}

        # Build device list for GPU access
        devices = ["/dev/kfd", "/dev/dri"]

        # Run as root so the container can access GPUs (/dev/kfd, /dev/dri). After teardown we chown
        # the aorta path to the host user so you don't need chmod 777 for the next run.
        log.info(f"Launching container {cfg.container_name} on {node}")
        log.info(f"  Image: {cfg.image}")
        log.info(f"  Mount: {self.config.aorta_path} -> {self.config.container_mount_path}")

        container = client.containers.run(
            image=cfg.image,
            name=cfg.container_name,
            detach=True,
            network_mode=cfg.network_mode,
            ipc_mode="host",
            privileged=cfg.privileged,
            shm_size=cfg.shm_size,
            volumes=volumes,
            devices=devices,
            working_dir=self.config.container_mount_path,
            user="root",
            group_add=["video", "render"],
            cap_add=["SYS_PTRACE"],
            security_opt=["seccomp=unconfined"],
            ulimits=[
                docker.types.Ulimit(name="memlock", soft=-1, hard=-1),
                docker.types.Ulimit(name="stack", soft=67108864, hard=67108864),
            ],
            stdin_open=True,
            tty=True,
            command="tail -f /dev/null",  # Keep container running
        )

        # Wait for container to be running
        container.reload()
        if container.status != "running":
            raise RuntimeError(f"Container failed to start on {node}: {container.status}")

        log.info(f"Container {cfg.container_name} running on {node} (ID: {container.short_id})")
        return container

    def _exec_in_container(
        self,
        container: Container,
        cmd: str,
        environment: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        stream: bool = False,
    ) -> tuple[int, str]:
        """
        Execute command inside container.

        Args:
            container: Container to execute in
            cmd: Command to run
            environment: Optional environment variables
            workdir: Optional working directory
            stream: If True, stream output in real-time (for long-running commands)

        Returns:
            Tuple of (exit_code, output)
        """
        log.info(f"Executing in container: {cmd[:100]}...")

        if stream:
            return self._exec_in_container_streaming(container, cmd, environment, workdir)

        exit_code, output = container.exec_run(
            cmd,
            environment=environment,
            workdir=workdir,
            stream=False,
        )

        output_str = output.decode("utf-8") if isinstance(output, bytes) else str(output)
        return exit_code, output_str

    def _exec_in_container_streaming(
        self,
        container: Container,
        cmd: str,
        environment: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
    ) -> tuple[int, str]:
        """
        Execute command with real-time streaming output.

        Provides feedback during long-running commands like training.
        """
        # Use exec_run with stream=True to get real-time output
        exec_result = container.client.api.exec_create(
            container.id,
            cmd,
            environment=environment,
            workdir=workdir,
            stdout=True,
            stderr=True,
        )

        output_generator = container.client.api.exec_start(
            exec_result['Id'],
            stream=True,
            demux=True,  # Separate stdout and stderr
        )

        output_lines = []
        line_count = 0

        for stdout_chunk, stderr_chunk in output_generator:
            # Process stdout
            if stdout_chunk:
                text = stdout_chunk.decode('utf-8', errors='replace')
                for line in text.splitlines():
                    if line.strip():
                        line_count += 1
                        # Log every line but summarize for very verbose output
                        if line_count <= 50 or line_count % 20 == 0:
                            log.info(f"  [stdout] {line[:200]}")
                        output_lines.append(line)

            # Process stderr
            if stderr_chunk:
                text = stderr_chunk.decode('utf-8', errors='replace')
                for line in text.splitlines():
                    if line.strip():
                        line_count += 1
                        # Always log stderr (usually important)
                        log.info(f"  [stderr] {line[:200]}")
                        output_lines.append(line)

        if line_count > 50:
            log.info(f"  ... ({line_count} total lines of output)")

        # Get exit code
        exec_info = container.client.api.exec_inspect(exec_result['Id'])
        exit_code = exec_info.get('ExitCode', -1)

        return exit_code, '\n'.join(output_lines)

    def _setup_single_node(self, node: str) -> Tuple[str, bool, Optional[str]]:
        """
        Set up a single node (thread-safe helper for parallel deployment).

        Args:
            node: Hostname or IP of the node

        Returns:
            Tuple of (node, success, error_message)
        """
        try:
            # Connect to Docker
            client = self._connect_docker(node)

            # Cleanup any existing containers
            self._cleanup_existing_containers(client, node)

            # Pull image (will skip if already present)
            log.info(f"Pulling image {self.config.docker.image} on {node}...")
            try:
                client.images.pull(self.config.docker.image)
            except docker.errors.ImageNotFound:
                return (node, False, f"Image not found: {self.config.docker.image}")

            # Launch container
            container = self._launch_container(client, node)

            # Thread-safe update of shared state
            with self._lock:
                self._containers[node] = container

            # Build RCCL if not skipping
            if not self.config.skip_rccl_build:
                log.info(f"Building RCCL on {node}...")
                build_cmd = f"bash {self.config.container_mount_path}/{self.config.build_script}"
                exit_code, output = self._exec_in_container(container, build_cmd)

                if exit_code != 0:
                    return (node, False, f"RCCL build failed:\n{output}")

                log.info(f"RCCL build completed on {node}")

            return (node, True, None)

        except Exception as e:
            return (node, False, str(e))

    def _ensure_aorta_repo(self) -> bool:
        """Clone Aorta repo into aorta_path if it does not exist and auto_clone is enabled."""
        if self.config.aorta_path.exists():
            return True
        if not self.config.aorta_auto_clone or not self.config.aorta_clone_url:
            return False
        path = self.config.aorta_path
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Cloning Aorta from {self.config.aorta_clone_url} into {path}")
        try:
            subprocess.run(
                ["git", "clone", self.config.aorta_clone_url, str(path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            log.info(f"Aorta repository cloned to {path}")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.error(f"Failed to clone Aorta repo: {e}")
            return False

    def setup(self) -> bool:
        """
        Set up Aorta environment using parallel deployment.

        If aorta_path is missing and aorta_auto_clone is true, clones the repo first.
        Then:
        1. Connect to Docker daemon on each node
        2. Pull image if needed
        3. Launch container with GPU access and aorta bind mount
        4. Build RCCL from source (optional)

        This significantly reduces setup time for multi-node clusters.
        """
        if not self.config.aorta_path.exists() and not self._ensure_aorta_repo():
            log.error("Aorta path does not exist and auto-clone failed or is disabled")
            return False

        nodes = self.config.nodes
        num_nodes = len(nodes)

        if num_nodes == 0:
            log.error("No nodes configured")
            return False

        log.info(f"Setting up {num_nodes} node(s) in parallel...")

        # Use ThreadPoolExecutor for parallel deployment
        # Max workers = number of nodes (each node gets its own thread)
        with ThreadPoolExecutor(max_workers=num_nodes) as executor:
            # Submit all setup tasks
            futures = {executor.submit(self._setup_single_node, node): node for node in nodes}

            # Collect results as they complete
            failed_nodes = []
            for future in as_completed(futures):
                node = futures[future]
                try:
                    node_name, success, error_msg = future.result()
                    if not success:
                        log.error(f"Setup failed on {node_name}: {error_msg}")
                        failed_nodes.append((node_name, error_msg))
                    else:
                        log.info(f"Setup completed on {node_name}")
                except Exception as e:
                    log.exception(f"Unexpected error setting up {node}: {e}")
                    failed_nodes.append((node, str(e)))

        if failed_nodes:
            log.error(f"Setup failed on {len(failed_nodes)}/{num_nodes} nodes:")
            for node, error in failed_nodes:
                log.error(f"  {node}: {error}")
            return False

        log.info(f"All {num_nodes} node(s) set up successfully")
        return True

    def _resolve_launch_mode(self) -> str:
        """
        Resolve ``multi_node.master_launch_mode`` to a concrete mode.

        - ``script`` keeps the legacy single-node behavior (delegates to
          ``experiment_script``); fails if the cluster has more than one node.
        - ``torchrun`` builds a multi-node ``torchrun`` command on every node,
          rendezvous-ing on the head node.
        - ``auto`` picks ``script`` for single-node, ``torchrun`` for >1 node.
        """
        mode = self.config.multi_node.master_launch_mode
        if mode == "auto":
            return "script" if len(self.config.nodes) <= 1 else "torchrun"
        return mode

    def _pick_master_port(self) -> int:
        """
        Return ``multi_node.master_port`` if set, otherwise a free ephemeral
        port on the orchestrator host.

        The bound socket is closed before the port is returned, so there is a
        small TOCTOU window. Operators who care can pin ``master_port``
        explicitly in the config.
        """
        configured = self.config.multi_node.master_port
        if configured:
            return int(configured)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return int(s.getsockname()[1])

    def _build_base_env(self) -> Dict[str, str]:
        """Build the env dict shared by every node's launch."""
        env = self.config.environment.to_dict()

        rccl_path = self.config.rccl.build_path
        env["LD_LIBRARY_PATH"] = (
            f"{rccl_path}/build/release/:/opt/rocm/lib:/opt/rocm/lib64:"
            f"/opt/openmpi/lib:/opt/rccl-tests/build:$LD_LIBRARY_PATH"
        )
        env["rccl_path"] = rccl_path

        if self.config.training_overrides:
            # aorta train.py exposes `--override` with `nargs="*"`; multiple
            # `--override` groups collapse to the last group's values. Emit a
            # single `--override` followed by all key=value tokens so that
            # downstream legacy launch scripts also forward them correctly.
            tokens = " ".join(f'{key}="{value}"' for key, value in self.config.training_overrides.items())
            env["AORTA_OVERRIDE_ARGS"] = f"--override {tokens}"

        for k, v in self.config.multi_node.extra_env.items():
            env[str(k)] = str(v)

        return env

    def _build_torchrun_command(
        self,
        *,
        node_rank: int,
        nnodes: int,
        master_addr: str,
        master_port: int,
        nproc_per_node: int,
    ) -> str:
        """
        Build the ``torchrun ... train.py --config <CFG>`` shell command run
        inside each node's container.

        Mirrors ``scripts/multi_node/local_launch.sh`` from Aorta's own
        repository: one rank-group per node, rendezvous on ``master_addr``.
        Training overrides from the config are appended via ``--override`` so
        operators get the same knobs as the single-node ``script`` mode.
        """
        mn = self.config.multi_node
        mount = self.config.container_mount_path
        config_path = f"{mount}/{self.config.base_config}"
        train_script = f"{mount}/{mn.train_script}"

        parts: List[str] = [
            "torchrun",
            f"--nnodes={nnodes}",
            f"--node_rank={node_rank}",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_addr={shlex.quote(str(master_addr))}",
            f"--master_port={int(master_port)}",
        ]
        parts.extend(str(a) for a in mn.extra_torchrun_args)
        parts.append(shlex.quote(train_script))
        parts.extend(["--config", shlex.quote(config_path)])
        # aorta's train.py uses `argparse(--override, nargs="*")` so emitting
        # one `--override` per key only keeps the LAST group's values. Pack
        # all key=value tokens behind a single `--override` so they all stick.
        if self.config.training_overrides:
            parts.append("--override")
            for key, value in self.config.training_overrides.items():
                parts.append(f"{key}={shlex.quote(str(value))}")
        parts.extend(str(a) for a in mn.extra_train_args)

        return "bash -lc " + shlex.quote(" ".join(parts))

    def _run_single_node(
        self,
        *,
        node: str,
        node_rank: int,
        launch_cmd: str,
        env: Dict[str, str],
    ) -> Tuple[str, int, str]:
        """
        Execute ``launch_cmd`` inside ``node``'s container.

        Returns ``(node, exit_code, output)``. Used as the parallel worker for
        multi-node launches.
        """
        container = self._containers.get(node)
        if container is None:
            return (node, -1, f"No container found for {node}")

        log.info(f"[node {node_rank}/{node}] Launching: {launch_cmd[:200]}...")
        exit_code, output = self._exec_in_container(
            container,
            launch_cmd,
            environment=env,
            workdir=self.config.container_mount_path,
            stream=True,
        )
        log.info(f"[node {node_rank}/{node}] Exit code: {exit_code}")
        return (node, exit_code, output)

    def run(self, **kwargs) -> RunResult:
        """
        Execute the Aorta benchmark.

        Two execution modes are supported, selected by
        ``multi_node.master_launch_mode``:

        - ``script`` (single-node, legacy): runs ``experiment_script`` inside
          the head node's container.
        - ``torchrun`` (disaggregated multi-node): launches ``torchrun`` on
          every node in parallel with proper ``--nnodes/--node_rank/
          --master_addr/--master_port`` rendezvous, mirroring Aorta's
          ``scripts/multi_node/local_launch.sh`` pattern. ``auto`` picks
          ``script`` for single-node clusters and ``torchrun`` for
          multi-node clusters.

        Profiling artifacts (``torch_profiler/`` trees) from every node are
        collected into ``<aorta_path>/combined_traces/node_<rank>/`` on the
        head node when running multi-node, and exposed via the
        ``torch_traces`` artifact for downstream parsers.
        """
        start_time = time.time()
        stdout_dict: Dict[str, str] = {}
        stderr_dict: Dict[str, str] = {}
        exit_codes: Dict[str, int] = {}
        artifacts: Dict[str, Path] = {}

        try:
            launch_mode = self._resolve_launch_mode()
            nodes = list(self.config.nodes)

            if launch_mode == "script" and len(nodes) > 1:
                return RunResult(
                    status=RunStatus.FAILED,
                    start_time=start_time,
                    end_time=time.time(),
                    error_message=(
                        "master_launch_mode='script' but cluster has "
                        f"{len(nodes)} nodes; either set master_launch_mode='torchrun' "
                        "or 'auto' in the multi_node block, or shrink the cluster file."
                    ),
                )

            log.info(f"Launch mode: {launch_mode}; nodes={len(nodes)}; head_node={self.head_node}")

            base_env = self._build_base_env()
            if base_env.get("AORTA_OVERRIDE_ARGS"):
                log.info(f"Training overrides: {base_env['AORTA_OVERRIDE_ARGS']}")

            if launch_mode == "script":
                node = self.head_node
                container = self._containers.get(node)
                if not container:
                    return RunResult(
                        status=RunStatus.FAILED,
                        start_time=start_time,
                        end_time=time.time(),
                        error_message=f"No container found for {node}",
                    )

                config_path = f"{self.config.container_mount_path}/{self.config.base_config}"
                exp_cmd = f"bash {self.config.container_mount_path}/{self.config.experiment_script} {config_path}"
                log.info(f"Running experiment: {exp_cmd}")
                log.info("Streaming output (this may take several minutes)...")

                exit_code, output = self._exec_in_container(
                    container,
                    exp_cmd,
                    environment=base_env,
                    stream=True,
                )
                stdout_dict[node] = output
                exit_codes[node] = exit_code

                if exit_code != 0:
                    log.error(f"Experiment failed on {node} with exit code {exit_code}")
                    return RunResult(
                        status=RunStatus.FAILED,
                        start_time=start_time,
                        end_time=time.time(),
                        stdout=stdout_dict,
                        exit_codes=exit_codes,
                        error_message=f"Experiment exited with code {exit_code}",
                    )
            else:
                mn = self.config.multi_node
                nnodes = len(nodes)
                nproc_per_node = mn.nproc_per_node or self.config.gpus_per_node
                master_addr = mn.master_addr or self.head_node
                master_port = self._pick_master_port()

                log.info(
                    f"Disaggregated launch: nnodes={nnodes}, "
                    f"nproc_per_node={nproc_per_node}, "
                    f"master={master_addr}:{master_port}"
                )

                futures = {}
                with ThreadPoolExecutor(max_workers=max(1, nnodes)) as executor:
                    for rank, node in enumerate(nodes):
                        cmd = self._build_torchrun_command(
                            node_rank=rank,
                            nnodes=nnodes,
                            master_addr=master_addr,
                            master_port=master_port,
                            nproc_per_node=nproc_per_node,
                        )
                        fut = executor.submit(
                            self._run_single_node,
                            node=node,
                            node_rank=rank,
                            launch_cmd=cmd,
                            env=base_env,
                        )
                        futures[fut] = (rank, node)

                    for fut in as_completed(futures):
                        rank, node = futures[fut]
                        try:
                            n, ec, out = fut.result()
                        except Exception as e:
                            log.exception(f"Node {node} (rank {rank}) raised: {e}")
                            stdout_dict[node] = str(e)
                            exit_codes[node] = -1
                            continue
                        stdout_dict[n] = out
                        exit_codes[n] = ec

                failed = {n: c for n, c in exit_codes.items() if c != 0}
                if failed:
                    log.error(f"Disaggregated run failed on {len(failed)}/{nnodes} nodes: {failed}")
                    return RunResult(
                        status=RunStatus.FAILED,
                        start_time=start_time,
                        end_time=time.time(),
                        stdout=stdout_dict,
                        exit_codes=exit_codes,
                        error_message=(f"Disaggregated experiment failed on nodes: {sorted(failed.keys())}"),
                    )

                if mn.collect_traces:
                    combined = self._collect_multi_node_traces(nodes)
                    if combined is not None:
                        artifacts["torch_traces"] = combined
                        log.info(f"Combined per-node traces collected at {combined}")

            # Find torch_profiler directory - Aorta saves traces to output_dir/torch_profiler
            # The output_dir is configured in the YAML config (e.g., "overlap_debug_repro")
            # We search for the most recent torch_profiler directory
            nch = self.config.environment.NCCL_MAX_NCHANNELS
            compute_ch = 256 - nch

            trace_dir: Optional[Path] = None
            output_dir: Optional[Path] = None
            trace_mtime: float = -1.0

            if "torch_traces" in artifacts:
                trace_dir = artifacts["torch_traces"]
                output_dir = trace_dir.parent
                # Multi-node combined_traces should win unless a fresher single-node tree
                # is discovered below; seed mtime from this tree so the comparison is valid.
                try:
                    latest_file = max(
                        trace_dir.glob("**/*"),
                        key=lambda p: p.stat().st_mtime if p.is_file() else 0,
                        default=None,
                    )
                    if latest_file is not None and latest_file.is_file():
                        trace_mtime = latest_file.stat().st_mtime
                    else:
                        trace_mtime = trace_dir.stat().st_mtime
                except (ValueError, OSError):
                    trace_mtime = trace_dir.stat().st_mtime

            # Search for torch_profiler directories in aorta_path (handles nested dirs like artifacts/*/torch_profiler).
            # Skip anything inside the combined_traces tree we just collected so the
            # original (older) per-node copies don't shadow the consolidated set.
            combined_root = self.config.aorta_path / "combined_traces"
            for candidate in self.config.aorta_path.glob("**/torch_profiler"):
                if not candidate.is_dir():
                    continue
                if combined_traces_in(candidate, combined_root):
                    continue
                try:
                    latest_file = max(
                        candidate.glob("**/*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, default=None
                    )
                    candidate_mtime = (
                        latest_file.stat().st_mtime
                        if latest_file and latest_file.is_file()
                        else candidate.stat().st_mtime
                    )
                except (ValueError, OSError):
                    candidate_mtime = candidate.stat().st_mtime

                if trace_dir is None or candidate_mtime > trace_mtime:
                    trace_dir = candidate
                    output_dir = candidate.parent
                    trace_mtime = candidate_mtime

            # Required artifact for host-side parsing: torch_traces (parse runs on host, not in container)
            if trace_dir and trace_dir.exists():
                artifacts["torch_traces"] = trace_dir
                log.info(f"Found trace artifacts at {trace_dir} (host_parse_path will use these)")
            else:
                # Fallback to legacy path format
                output_dir_name = f"nodes1_rccl_develop_commsCh{nch}_computeCh{compute_ch}"
                output_dir = self.config.aorta_path / output_dir_name
                trace_dir = output_dir / "torch_profiler"
                if trace_dir.exists():
                    artifacts["torch_traces"] = trace_dir
                    log.info(f"Found trace artifacts at {trace_dir} (host_parse_path will use these)")
                else:
                    log.warning(
                        "No torch_profiler directory found; host cannot produce benchmark metrics without torch_traces"
                    )

            # Optional container_analysis_path: run TraceLens in container only if enabled and deps present.
            # Parsing/validation use host venv by default; container reports are consumed when present.
            # In multi-node mode the head node's container is used; the analysis scripts
            # operate on traces under aorta_path which (for collected traces) is on the head node.
            analysis_container = self._containers.get(self.head_node)
            if (
                self.config.analysis.enable_tracelens
                and trace_dir
                and trace_dir.exists()
                and analysis_container is not None
            ):
                log.info("Container TraceLens analysis (optional): attempting in-container report generation")
                analysis_result = self._run_tracelens_analysis(analysis_container, output_dir)
                if analysis_result:
                    artifacts["tracelens_analysis"] = analysis_result
                    log.info(f"Container TraceLens analysis completed: {analysis_result}")
                else:
                    log.warning("Container TraceLens skipped or failed; host will parse raw traces")

            # Run GEMM analysis if enabled (optional, same as TraceLens)
            if (
                self.config.analysis.enable_gemm_analysis
                and trace_dir
                and trace_dir.exists()
                and analysis_container is not None
            ):
                gemm_result = self._run_gemm_analysis(analysis_container, output_dir)
                if gemm_result:
                    artifacts["gemm_analysis"] = gemm_result
                    log.info(f"GEMM analysis completed: {gemm_result}")

            # Also collect training logs (best-effort, single-node legacy path)
            for log_node in self.config.nodes:
                log_file = self.config.aorta_path / f"training_{log_node}.log"
                if log_file.exists():
                    artifacts.setdefault("training_log", log_file)
                    break

            return RunResult(
                status=RunStatus.COMPLETED,
                start_time=start_time,
                end_time=time.time(),
                stdout=stdout_dict,
                stderr=stderr_dict,
                exit_codes=exit_codes,
                artifacts=artifacts,
                metadata={
                    "nodes": len(self.config.nodes),
                    "gpus_per_node": self.config.gpus_per_node,
                    "nccl_channels": nch,
                    "compute_channels": compute_ch,
                    "launch_mode": launch_mode,
                },
            )

        except Exception as e:
            log.exception(f"Run failed: {e}")
            return RunResult(
                status=RunStatus.FAILED,
                start_time=start_time,
                end_time=time.time(),
                stdout=stdout_dict,
                stderr=stderr_dict,
                exit_codes=exit_codes,
                error_message=str(e),
            )

    def _run_tracelens_analysis(self, container: Container, output_dir: Path) -> Optional[Path]:
        """
        Run Aorta's built-in TraceLens analysis on the collected traces.

        This uses Aorta's `run_tracelens_single_config.sh` script which:
        1. Runs TraceLens on each rank's trace (individual reports)
        2. Generates collective multi-rank reports
        3. Creates gpu_timeline_summary_mean.xlsx with aggregated metrics

        Args:
            container: Docker container to run analysis in
            output_dir: Directory containing torch_profiler traces

        Returns:
            Path to tracelens_analysis directory, or None if analysis failed
        """
        analysis_dir = output_dir / "tracelens_analysis"

        # Skip if already exists and skip_if_exists is set
        if self.config.analysis.skip_if_exists and analysis_dir.exists():
            log.info(f"TraceLens analysis already exists, skipping: {analysis_dir}")
            return analysis_dir

        # Fast dependency check to avoid running long scripts that will fail immediately.
        check_cmd = 'python3 -c "import TraceLens"'
        check_exit, _ = self._exec_in_container(container, check_cmd)
        if check_exit != 0:
            log.warning("TraceLens python package not available in container; skipping TraceLens analysis")
            return None

        # Build the analysis command
        # The script path is relative to container mount
        script_path = f"{self.config.container_mount_path}/{self.config.analysis.tracelens_script}"
        trace_path = str(output_dir).replace(str(self.config.aorta_path), self.config.container_mount_path)

        analysis_cmd = f"bash {script_path} {trace_path}"

        log.info(f"Running TraceLens analysis: {analysis_cmd}")
        log.info("This may take a few minutes...")

        try:
            exit_code, output = self._exec_in_container(
                container,
                analysis_cmd,
                stream=True,  # Stream for real-time feedback
            )

            if exit_code != 0:
                log.error(f"TraceLens analysis failed with exit code {exit_code}")
                log.error(f"Output: {output[:2000]}...")  # Truncate for readability
                return None

            # Verify the output was created
            if analysis_dir.exists():
                # Log what was generated
                individual_count = len(list(analysis_dir.glob("individual_reports/*.xlsx")))
                collective_count = len(list(analysis_dir.glob("collective_reports/*.xlsx")))
                log.info("TraceLens analysis complete:")
                log.info(f"  Individual reports: {individual_count}")
                log.info(f"  Collective reports: {collective_count}")
                return analysis_dir
            else:
                log.warning(f"TraceLens analysis completed but output not found: {analysis_dir}")
                return None

        except Exception as e:
            log.exception(f"TraceLens analysis failed: {e}")
            return None

    def _run_gemm_analysis(self, container: Container, output_dir: Path) -> Optional[Path]:
        """
        Run Aorta's GEMM analysis on the collected traces.

        This uses Aorta's `gemm_analysis/run_tracelens_analysis.sh` script which:
        1. Discovers configurations (thread configs, channel settings)
        2. Generates individual TraceLens reports per rank
        3. Generates collective multi-rank reports
        4. Compares channels across thread configurations

        Args:
            container: Docker container to run analysis in
            output_dir: Directory containing torch_profiler traces

        Returns:
            Path to tracelens_analysis directory, or None if analysis failed
        """
        analysis_dir = output_dir / "tracelens_analysis"

        # Skip if already exists and skip_if_exists is set
        if self.config.analysis.skip_if_exists and analysis_dir.exists():
            log.info(f"GEMM analysis already exists, skipping: {analysis_dir}")
            return analysis_dir

        # Build the analysis command
        # The script path is relative to container mount
        script_path = f"{self.config.container_mount_path}/{self.config.analysis.gemm_script}"
        trace_path = str(output_dir).replace(str(self.config.aorta_path), self.config.container_mount_path)

        # Check if this is a sweep directory structure or single config
        # For single config runs (like our benchmark), use tracelens_single_config
        # The gemm_analysis script expects sweep directory structure with *thread dirs
        torch_profiler_dir = output_dir / "torch_profiler"

        if torch_profiler_dir.exists():
            # Single config - use the parent directory
            analysis_cmd = f"bash {script_path} {trace_path}"
        else:
            # Sweep structure - pass the sweep directory
            analysis_cmd = f"bash {script_path} {trace_path}"

        log.info(f"Running GEMM analysis: {analysis_cmd}")
        log.info("This may take several minutes depending on trace size...")

        try:
            exit_code, output = self._exec_in_container(
                container,
                analysis_cmd,
                stream=True,  # Stream for real-time feedback
            )

            if exit_code != 0:
                log.error(f"GEMM analysis failed with exit code {exit_code}")
                log.error(f"Output: {output[:2000]}...")  # Truncate for readability
                return None

            # Verify the output was created
            if analysis_dir.exists():
                # Log what was generated
                individual_count = len(list(analysis_dir.glob("**/individual_reports/*.xlsx")))
                collective_count = len(list(analysis_dir.glob("**/collective_reports/*.xlsx")))
                comparison_count = len(list(analysis_dir.glob("comparisons/*.xlsx")))
                log.info("GEMM analysis complete:")
                log.info(f"  Individual reports: {individual_count}")
                log.info(f"  Collective reports: {collective_count}")
                log.info(f"  Comparison reports: {comparison_count}")
                return analysis_dir
            else:
                log.warning(f"GEMM analysis completed but output not found: {analysis_dir}")
                return None

        except Exception as e:
            log.exception(f"GEMM analysis failed: {e}")
            return None

    def teardown(self) -> bool:
        """
        Cleanup containers and connections.

        Handles SSH connection cleanup gracefully to avoid BrokenPipeError warnings.
        """
        import warnings

        success = True

        for node, container in self._containers.items():
            try:
                log.info(f"Stopping container on {node}...")
                container.stop(timeout=30)
                container.remove(force=True)
                log.info(f"Container removed on {node}")
                # Chown aorta path to host user so next run does not require chmod 777 (container runs as root for GPU access)
                uid_gid = self._get_remote_uid_gid(node)
                if uid_gid is not None:
                    try:
                        r = subprocess.run(
                            [
                                "ssh",
                                "-o",
                                "BatchMode=yes",
                                "-o",
                                "ConnectTimeout=10",
                                f"{self.config.username}@{node}",
                                "chown",
                                "-R",
                                f"{uid_gid[0]}:{uid_gid[1]}",
                                str(self.config.aorta_path),
                            ],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        if r.returncode == 0:
                            log.info(f"Chowned {self.config.aorta_path} to {self.config.username} on {node}")
                        else:
                            log.debug(f"Chown on {node} returned {r.returncode}: {r.stderr or r.stdout}")
                    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                        log.debug(f"Chown on {node} skipped: {e}")
            except Exception as e:
                log.warning(f"Error removing container on {node}: {e}")
                success = False

        self._containers.clear()

        # Close Docker clients - suppress BrokenPipeError during SSH cleanup
        for node, client in self._docker_clients.items():
            try:
                # Suppress warnings during cleanup as SSH connections may already be closed
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=BrokenPipeError)
                    warnings.filterwarnings("ignore", message=".*Broken pipe.*")
                    try:
                        client.close()
                    except BrokenPipeError:
                        pass  # Expected when SSH connection is already closed
                    except OSError as e:
                        if "Broken pipe" not in str(e):
                            raise
            except Exception as e:
                # Log but don't fail on cleanup errors
                log.debug(f"Docker client cleanup for {node}: {e}")

        self._docker_clients.clear()

        return success
