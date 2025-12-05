"""
Aorta PyTorch benchmark runner.

Deploys Docker containers and executes distributed training benchmarks
using the Aorta framework. Uses Docker SDK over SSH for container orchestration.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import logging

try:
    import docker
    from docker.models.containers import Container
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False
    docker = None
    Container = None

from runners._base_runner import BaseRunner, RunConfig, RunResult, RunStatus

log = logging.getLogger(__name__)


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
class AortaConfig(RunConfig):
    """
    Configuration for Aorta benchmark runner.
    
    Extends base RunConfig with Aorta-specific settings.
    """
    # Path to aorta repository on host (will be bind-mounted)
    aorta_path: Path = field(default_factory=lambda: Path("/home/AMD/speriasw/projects/aorta"))
    
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
            raise ImportError(
                "Docker SDK not available. Install with: pip install docker"
            )
        
        super().__init__(config)
        self.config: AortaConfig = config  # Type hint for IDE
        
        self._docker_clients: Dict[str, docker.DockerClient] = {}
        self._containers: Dict[str, Container] = {}
    
    def validate_config(self) -> List[str]:
        """Validate Aorta-specific configuration."""
        errors = super().validate_config()
        
        if not self.config.aorta_path.exists():
            errors.append(f"Aorta path does not exist: {self.config.aorta_path}")
        
        config_path = self.config.aorta_path / self.config.base_config
        if not config_path.exists():
            errors.append(f"Base config does not exist: {config_path}")
        
        build_script = self.config.aorta_path / self.config.build_script
        if not build_script.exists():
            errors.append(f"Build script does not exist: {build_script}")
        
        exp_script = self.config.aorta_path / self.config.experiment_script
        if not exp_script.exists():
            errors.append(f"Experiment script does not exist: {exp_script}")
        
        return errors
    
    def _connect_docker(self, node: str) -> docker.DockerClient:
        """
        Connect to Docker daemon on a node via SSH.
        
        Args:
            node: Hostname or IP of the node
            
        Returns:
            Docker client connected to the node
        """
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
        volumes = {
            str(self.config.aorta_path): {
                "bind": self.config.container_mount_path,
                "mode": "rw"
            }
        }
        
        # Build device list for GPU access
        devices = ["/dev/kfd", "/dev/dri"]
        
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
            group_add=["video"],
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
    
    def setup(self) -> bool:
        """
        Set up Aorta environment.
        
        1. Connect to Docker daemon on each node
        2. Pull image if needed
        3. Launch container with GPU access and aorta bind mount
        4. Build RCCL from source (optional)
        """
        try:
            for node in self.config.nodes:
                # Connect to Docker
                client = self._connect_docker(node)
                
                # Cleanup any existing containers
                self._cleanup_existing_containers(client, node)
                
                # Pull image (will skip if already present)
                log.info(f"Pulling image {self.config.docker.image} on {node}...")
                try:
                    client.images.pull(self.config.docker.image)
                except docker.errors.ImageNotFound:
                    log.error(f"Image not found: {self.config.docker.image}")
                    return False
                
                # Launch container
                container = self._launch_container(client, node)
                self._containers[node] = container
                
                # Build RCCL if not skipping
                if not self.config.skip_rccl_build:
                    log.info(f"Building RCCL on {node}...")
                    build_cmd = f"bash {self.config.container_mount_path}/{self.config.build_script}"
                    exit_code, output = self._exec_in_container(container, build_cmd)
                    
                    if exit_code != 0:
                        log.error(f"RCCL build failed on {node}:\n{output}")
                        return False
                    
                    log.info(f"RCCL build completed on {node}")
            
            return True
            
        except Exception as e:
            log.exception(f"Setup failed: {e}")
            return False
    
    def run(self, **kwargs) -> RunResult:
        """
        Execute the Aorta benchmark.
        
        Runs the experiment script inside the container and collects
        profiling artifacts.
        """
        start_time = time.time()
        stdout_dict: Dict[str, str] = {}
        stderr_dict: Dict[str, str] = {}
        exit_codes: Dict[str, int] = {}
        artifacts: Dict[str, Path] = {}
        
        try:
            # For now, run on head node only (single node v1)
            node = self.head_node
            container = self._containers.get(node)
            
            if not container:
                return RunResult(
                    status=RunStatus.FAILED,
                    start_time=start_time,
                    end_time=time.time(),
                    error_message=f"No container found for {node}"
                )
            
            # Build environment with computed values
            env = self.config.environment.to_dict()
            
            # Add RCCL library path
            rccl_path = self.config.rccl.build_path
            env["LD_LIBRARY_PATH"] = (
                f"{rccl_path}/build/release/:/opt/rocm/lib:/opt/rocm/lib64:"
                f"/opt/openmpi/lib:/opt/rccl-tests/build:$LD_LIBRARY_PATH"
            )
            env["rccl_path"] = rccl_path
            
            # Build override arguments if any
            override_args = ""
            if self.config.training_overrides:
                for key, value in self.config.training_overrides.items():
                    override_args += f' --override {key}="{value}"'
            
            # Execute experiment script with streaming output for real-time feedback
            exp_cmd = f"bash {self.config.container_mount_path}/{self.config.experiment_script}"
            log.info(f"Running experiment: {exp_cmd}")
            log.info("Streaming output (this may take several minutes)...")
            
            exit_code, output = self._exec_in_container(
                container, 
                exp_cmd,
                environment=env,
                stream=True,  # Stream output for real-time feedback
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
                    error_message=f"Experiment exited with code {exit_code}"
                )
            
            # Determine output directory from environment
            nch = self.config.environment.NCCL_MAX_NCHANNELS
            compute_ch = 256 - nch
            output_dir_name = f"nodes1_rccl_develop_commsCh{nch}_computeCh{compute_ch}"
            
            # Collect artifact paths
            trace_dir = self.config.aorta_path / output_dir_name / "torch_profiler"
            if trace_dir.exists():
                artifacts["torch_traces"] = trace_dir
                log.info(f"Found trace artifacts at {trace_dir}")
            else:
                log.warning(f"Expected trace directory not found: {trace_dir}")
            
            # Also collect training logs
            log_file = self.config.aorta_path / f"training_{node}.log"
            if log_file.exists():
                artifacts["training_log"] = log_file
            
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
                }
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
                error_message=str(e)
            )
    
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

