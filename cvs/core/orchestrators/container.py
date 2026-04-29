'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
import getpass
import re
from cvs.core.runtimes import RuntimeFactory


# Default container configuration - matches the original docker command
DEFAULT_CONTAINER_ARGS = {
    "devices": [
        "/dev/kfd",  # Kernel Fusion Driver (ROCm)
        "/dev/dri",  # Direct Rendering Infrastructure
        "/dev/infiniband",  # All infiniband devices
    ],
    "capabilities": [
        "SYS_PTRACE",  # Debug processes
        "IPC_LOCK",  # Lock memory for RDMA
        "SYS_ADMIN",  # Administrative operations
    ],
    "security_opts": [
        "seccomp=unconfined",  # Disable seccomp for RDMA
        "apparmor=unconfined",  # Disable apparmor
    ],
    "groups": [
        "video",  # Video group for GPU access
    ],
    "ulimits": [
        "memlock=-1",  # Unlimited memory locking for RDMA
    ],
    "environment": {
        "GPUS": "8",  # Number of GPUs (from original command)
        "MULTINODE": "true",  # Multinode mode
    },
    "network_mode": "host",
    "ipc_mode": "host",
    "privileged": True,
}


class ContainerOrchestrator(BaremetalOrchestrator):
    """
    Container-based orchestrator that extends BaremetalOrchestrator for containerized execution.

    Uses SSH transport (via BaremetalOrchestrator) but executes commands in containers
    on cluster nodes. Supports long-running containers for efficient command execution
    with multiple container runtimes (Docker, Enroot, etc.).

    Container lifecycle is managed through setup_containers() and teardown_containers() methods,
    which should be called explicitly by test code when needed.
    """

    def __init__(self, log, config, stop_on_errors=False):
        """
        Initialize container orchestrator.

        Args:
            log: Logger instance
            config: OrchestratorConfig instance
            stop_on_errors: Whether to stop execution on first error

        Note:
            Containers are not automatically launched. Use setup_containers() for explicit control.
        """
        super().__init__(log, config, stop_on_errors)

        # Set orchestrator type for runtime identification
        self.orchestrator_type = "container"

        # Override SSH port for container SSH daemons
        self.ssh_port = 2224

        # Container-specific initialization
        self.container_config = config.get('container', {})
        if not self.container_config:
            raise ValueError("ContainerOrchestrator requires 'container' config in OrchestratorConfig")

        self.container_enabled = True
        self.container_id = None  # Track running container ID

        # Initialize container runtime
        runtime_config = self.container_config.get('runtime', {})
        runtime_name = runtime_config.get('name', 'docker')
        self.runtime = RuntimeFactory.create(runtime_name, log, self)
        self.log.info(f"ContainerOrchestrator initialized with runtime: {runtime_name}")

        # Note: Containers are not auto-launched here. Use setup_containers() for explicit control.

    def get_volumes(self):
        """
        Get volume mounts, merging defaults with config.json runtime args.

        Returns:
            list: List of volume mount strings in format "host_path:container_path"
        """
        import getpass

        user = getpass.getuser()

        # Start with dynamic user-specific mounts (from original command)
        volumes = [
            f"/home/{user}:/workspace",  # User home directory
            f"/home/{user}/.ssh:/host_ssh",  # SSH keys for multinode
        ]

        # Merge with config.json runtime args
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        config_volumes = runtime_args.get('volumes', [])
        volumes.extend(config_volumes)

        return volumes

    def get_devices(self):
        """
        Get device passthroughs, merging defaults with config.json runtime args.

        Returns:
            list: List of device paths to passthrough
        """
        devices = list(DEFAULT_CONTAINER_ARGS['devices'])

        # InfiniBand devices are added via shell expansion in docker run command
        # to ensure per-host discovery at runtime

        # Merge with config.json runtime args
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        config_devices = runtime_args.get('devices', [])
        devices.extend(config_devices)

        return devices

    def get_capabilities(self):
        """
        Get Linux capabilities, merging defaults with config.json runtime args.

        Returns:
            list: List of capability names
        """
        capabilities = list(DEFAULT_CONTAINER_ARGS['capabilities'])

        # Merge with config.json runtime args
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        config_capabilities = runtime_args.get('cap_add', [])
        capabilities.extend(config_capabilities)

        return capabilities

    def get_security_opts(self):
        """
        Get security options, merging defaults with config.json runtime args.

        Returns:
            list: List of security option strings
        """
        security_opts = list(DEFAULT_CONTAINER_ARGS['security_opts'])

        # Merge with config.json runtime args
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        config_security_opts = runtime_args.get('security_opt', [])
        security_opts.extend(config_security_opts)

        return security_opts

    def get_groups(self):
        """
        Get group additions, merging defaults with config.json runtime args.

        Returns:
            list: List of group names to add
        """
        groups = list(DEFAULT_CONTAINER_ARGS['groups'])

        # Merge with config.json runtime args
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        config_groups = runtime_args.get('group_add', [])
        groups.extend(config_groups)

        return groups

    def get_ulimits(self):
        """
        Get ulimits, merging defaults with config.json runtime args.

        Returns:
            list: List of ulimit strings
        """
        ulimits = list(DEFAULT_CONTAINER_ARGS['ulimits'])

        # Merge with config.json runtime args
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        config_ulimits = runtime_args.get('ulimit', [])
        ulimits.extend(config_ulimits)

        return ulimits

    def get_environment(self):
        """
        Get environment variables, merging defaults with config.json.

        Returns:
            dict: Dictionary of environment variable names to values
        """
        environment = dict(DEFAULT_CONTAINER_ARGS['environment'])

        # Merge with config.json container env
        config_env = self.container_config.get('env', {})
        environment.update(config_env)

        return environment

    def get_network_mode(self):
        """
        Get network mode, checking config.json first then defaults.

        Returns:
            str: Network mode string
        """
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        return runtime_args.get('network', DEFAULT_CONTAINER_ARGS['network_mode'])

    def get_ipc_mode(self):
        """
        Get IPC mode, checking config.json first then defaults.

        Returns:
            str: IPC mode string
        """
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        return runtime_args.get('ipc', DEFAULT_CONTAINER_ARGS['ipc_mode'])

    def is_privileged(self):
        """
        Check if privileged mode should be enabled, checking config.json first then defaults.

        Returns:
            bool: True if containers should run in privileged mode
        """
        runtime_config = self.container_config.get('runtime', {})
        runtime_args = runtime_config.get('args', {})
        return runtime_args.get('privileged', DEFAULT_CONTAINER_ARGS['privileged'])

    def setup_containers(
        self,
        volumes=None,
        devices=None,
        capabilities=None,
        security_opts=None,
        environment=None,
        groups=None,
        ulimits=None,
    ):
        """
        Set up containers based on configuration.

        This method should be called explicitly by tests when they need containers.
        It respects the 'enabled' flag and handles container lifecycle appropriately.
        If 'launch' is true, checks that containers are already running and sets container_id.
        If containers are not running when expected, fails and informs the user.

        Args:
            volumes: Optional list of volume mounts (uses standards if not provided)
            devices: Optional list of device passthroughs (uses standards if not provided)
            capabilities: Optional list of Linux capabilities (uses standards if not provided)
            security_opts: Optional list of security options (uses standards if not provided)
            environment: Optional dict of environment variables (uses standards if not provided)
            groups: Optional list of groups to add (uses standards if not provided)
            ulimits: Optional list of ulimits (uses standards if not provided)

        Returns:
            bool: True if containers were set up successfully or no setup needed
        """
        if not self.container_config or not self.container_config.get('enabled', False):
            self.log.debug("Container mode not enabled, skipping setup")
            return True

        if self.container_config.get('launch', False):
            # Launch containers
            self.log.info("Launching containers...")
            container_name = self.get_container_name(self.container_config, self.container_config['image'])
            self.container_id = container_name

            # Use provided parameters or get standards
            volumes = volumes if volumes is not None else self.get_volumes()
            devices = devices if devices is not None else self.get_devices()
            capabilities = capabilities if capabilities is not None else self.get_capabilities()
            security_opts = security_opts if security_opts is not None else self.get_security_opts()
            environment = environment if environment is not None else self.get_environment()
            groups = groups if groups is not None else self.get_groups()
            ulimits = ulimits if ulimits is not None else self.get_ulimits()

            # Create a modified container config with standard settings
            modified_config = dict(self.container_config)

            # Ensure runtime config exists
            if 'runtime' not in modified_config:
                modified_config['runtime'] = {}
            if 'args' not in modified_config['runtime']:
                modified_config['runtime']['args'] = {}

            # Set standard runtime args if not already set
            runtime_args = modified_config['runtime']['args']
            if 'network' not in runtime_args:
                runtime_args['network'] = self.get_network_mode()
            if 'ipc' not in runtime_args:
                runtime_args['ipc'] = self.get_ipc_mode()
            if 'privileged' not in runtime_args:
                runtime_args['privileged'] = self.is_privileged()

            # Add InfiniBand device discovery via shell expansion (per-host)
            ib_device_expansion = '$(for dev in /dev/infiniband/*; do echo -n "--device $dev:$dev "; done)'

            return self.runtime.setup_containers(
                modified_config,
                container_name,
                volumes=volumes,
                devices=devices,
                capabilities=capabilities,
                security_opts=security_opts,
                environment=environment,
                groups=groups,
                ulimits=ulimits,
                device_expansion=ib_device_expansion,
            )

        # Assume containers are running
        image = self.container_config.get('image')
        if not image:
            self.log.error("Container image not specified in config")
            return False

        container_name = self.get_container_name(self.container_config, image)
        return self.verify_containers_running(container_name)

    def setup_sshd(self):
        """
        Setup SSH daemon in containers for passwordless communication.

        This method:
        - Copies SSH keys from /host_ssh mounted volume to /root/.ssh
        - Fixes permissions on SSH directory
        - Starts sshd on port 2224
        - Validates SSH daemon started successfully

        Returns:
            bool: True if SSH setup succeeded on all nodes, False otherwise

        Raises:
            RuntimeError: If no containers are currently running
        """
        if not self.container_id:
            raise RuntimeError("No containers running. Call setup_containers() first.")

        self.log.info(f"Setting up SSH daemon in containers: {self.container_id}")

        # Execute SSH setup commands
        # Note: Commands with shell operators must be wrapped in bash -c for proper execution inside container
        ssh_setup_commands = [
            "mkdir -p /root/.ssh",
            "bash -c 'cp -r /host_ssh/* /root/.ssh/'",  # bash -c ensures glob expands inside container
            "chown -R root:root /root/.ssh",
            "bash -c 'chmod 700 /root/.ssh && chmod 600 /root/.ssh/*'",
            "mkdir -p /run/sshd",  # Create privilege separation directory for sshd
            "/usr/sbin/sshd -p2224",
        ]

        for cmd in ssh_setup_commands:
            result = self.exec(cmd, timeout=10, detailed=True)
            # Check if command succeeded on all hosts
            for hostname, output in result.items():
                if output['exit_code'] != 0:
                    self.log.error(f"SSH setup command failed on {hostname}: {cmd}")
                    return False

        # Wait for sshd to start
        import time

        time.sleep(2)

        # Validate sshd is running by checking process
        check_cmd = "pgrep -f 'sshd.*2224' > /dev/null 2>&1"
        result = self.exec(check_cmd, timeout=10, detailed=True)

        # Check if all hosts have sshd running
        for hostname, output in result.items():
            if output['exit_code'] != 0:
                self.log.error(f"SSH daemon validation failed on {hostname}")
                return False
            else:
                self.log.info(f"SSH daemon started successfully on {hostname}")

        return True

    def verify_containers_running(self, container_name):
        """
        Verify that containers with the given name are running on all hosts.

        Args:
            container_name: Name of the container to check

        Returns:
            bool: True if container is running on all hosts, False otherwise
        """
        # Check if container is running on all hosts
        cmd = f"sudo docker ps --filter name=^{container_name}$ --filter status=running --format '{{{{.Names}}}}'"
        self.log.debug(f"Checking if container '{container_name}' is running on all hosts")
        result = self.all.exec(cmd, timeout=30)

        # Verify container is running on all hosts
        failed_hosts = []
        for host, output in result.items():
            if output.get('exit_code') != 0:
                failed_hosts.append(f"{host} (exit code {output.get('exit_code')})")
                continue
            running_name = output.get('stdout', '').strip()
            if running_name != container_name:
                failed_hosts.append(f"{host} (container not running, found: '{running_name}')")

        if failed_hosts:
            self.log.error(f"Container '{container_name}' not running on hosts: {failed_hosts}")
            return False

        self.container_id = container_name
        self.log.info(f"Verified container '{container_name}' is running on all hosts")
        return True

    def teardown_containers(self):
        """
        Tear down containers if they are running.

        This method should be called explicitly by tests for cleanup.
        It respects the 'enabled' flag and handles container lifecycle appropriately.
        If 'launch' is true, assumes containers should not be stopped (externally managed).

        Returns:
            bool: True if containers were torn down successfully or no teardown needed
        """
        if not self.container_config or not self.container_config.get('enabled', False):
            self.log.debug("Container mode not enabled, skipping teardown")
            return True

        if self.container_config.get('launch', False):
            self.log.debug("launch is true, not stopping externally managed containers")
            return True

        if not self.container_id:
            self.log.debug("No containers running")
            return True

        self.log.info("Tearing down containers...")
        self.log.info(f"Stopping containers: {self.container_id}")
        success = self.runtime.teardown_containers(self.container_id)
        self.container_id = None
        return success

    @staticmethod
    def sanitize_container_name(image_name):
        """
        Sanitize an image name to create a valid container name.

        Replaces any character that is not alphanumeric or underscore with underscore.

        Args:
            image_name: Docker image name (e.g., "rocm/cvs:latest")

        Returns:
            Sanitized string suitable for container name
        """
        return re.sub(r'[^a-zA-Z0-9_]', '_', image_name)

    @staticmethod
    def get_container_name(container_config, image):
        """
        Get container name from config or generate default.

        Args:
            container_config: Container configuration dict
            image: Docker image name

        Returns:
            Container name string
        """
        container_name = container_config.get('name')
        if not container_name:
            # Default: $(whoami)_<sanitize_image_name>
            username = getpass.getuser()
            sanitized_image = ContainerOrchestrator.sanitize_container_name(image)
            container_name = f"{username}_{sanitized_image}"
        return container_name

    def exec(self, cmd, hosts=None, timeout=None, detailed=False):
        """
        Execute command in running containers.

        Args:
            cmd: Command to execute inside container
            hosts: Target hosts (if None, uses all hosts)
            timeout: Command timeout
            detailed: If True, return detailed execution info including exit_code

        Returns:
            Dictionary mapping hosts to execution results

        Raises:
            RuntimeError: If no containers are currently running
        """
        if not self.container_id:
            raise RuntimeError("No containers running. Call setup_containers() first.")

        return self.runtime.exec(self.container_id, cmd, hosts, timeout, detailed)

    def exec_on_head(self, cmd, timeout=None):
        """
        Execute command directly on head node (baremetal).

        Args:
            cmd: Command to execute on head node
            timeout: Command timeout

        Returns:
            Dictionary mapping head node to execution result
        """
        return self.runtime.exec_on_head(self.container_id, cmd, timeout)

    def distribute_using_mpi(
        self,
        rank_cmd,
        mpi_hosts,
        ranks_per_host,
        env_vars,
        mpi_install_dir,
        mpi_extra_args=None,
        no_of_global_ranks=None,
    ):
        """
        Distribute MPI job across hosts using containers.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_hosts: List of host IPs/names for MPI hostfile
            ranks_per_host: Number of MPI ranks per host (uniform across hosts)
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory
            mpi_extra_args: List of additional mpirun arguments
            no_of_global_ranks: Total number of MPI ranks (optional, defaults to len(mpi_hosts) * ranks_per_host)

        Returns:
            Execution results
        """
        # Use parent class to build MPI command with wrapped rank command
        full_mpi_cmd = super().build_mpi_cmd(
            rank_cmd, mpi_hosts, ranks_per_host, env_vars, mpi_install_dir, mpi_extra_args, no_of_global_ranks
        )

        self.log.info("Launching MPI job in containers")
        self.log.debug(f"MPI command: {full_mpi_cmd}")

        # Execute on head node
        return self.exec_on_head(full_mpi_cmd, timeout=600)

    def cleanup(self, hosts):
        """
        Clean up container resources after test execution.

        Args:
            hosts: List of host addresses

        Returns:
            True if cleanup succeeded, False otherwise
        """
        if self.container_id:
            self.log.info("Tearing down containers")
            success = self.runtime.teardown_containers(self.container_id)
            if success:
                self.container_id = None
            return success
        return True
