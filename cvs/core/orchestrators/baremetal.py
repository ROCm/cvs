'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import shlex
import uuid

from cvs.core.orchestrators.base import Orchestrator
from cvs.core.scheduler import is_managed_compute
from cvs.lib.parallel.config import ParallelConfig
from cvs.lib.parallel.multiprocess_phandle import MultiProcessParallelHandle
from cvs.lib.utils_lib import get_passwordless_sudo_status


class BaremetalOrchestrator(Orchestrator):
    """
    Baremetal orchestrator implementation using Parallel-SSH.

    Executes commands directly on host systems via SSH without containers.
    Provides the foundation for containerized orchestrators.

    Integrates with OrchestratorConfig for standardized configuration.
    """

    def __init__(self, log, config, stop_on_errors=False):
        """
        Initialize baremetal orchestrator from OrchestratorConfig.

        Args:
            log: Logger instance
            config: OrchestratorConfig instance
            stop_on_errors: Whether to stop execution on first error
        """
        super().__init__(log, config, stop_on_errors)

        # Set orchestrator type for runtime identification
        self.orchestrator_type = "baremetal"

        # Cached result of the passwordless-sudo probe; None means not yet probed.
        self._needs_sudo = None

        # SSH port for MPI communication (overridable by subclasses)
        self.ssh_port = 22

        # Extract hosts from OrchestratorConfig node_dict
        self.hosts = (
            list(config.node_dict.keys())
            if isinstance(config.node_dict, dict)
            else [node.get('mgmt_ip') for node in config.node_dict]
        )

        self.head_node = self.hosts[0]  # First node is head

        self.user = config.get('username')
        self.pkey = config.get('priv_key_file')
        self.password = config.get('password')  # Optional

        # Initialize TWO ParallelSSH handles like original CVS pattern:
        # head - Single head node (for mpirun, result collection, etc.)
        # all - Parallel across all nodes (for setup, cleanup, verification)
        self.head = self._phandle([self.head_node])
        self.all = self._phandle(self.hosts)

    def _transport_kwargs(self):
        token_file = self.config.get('agent_token_file')
        node_dict = self.config.node_dict if isinstance(self.config.node_dict, dict) else {}
        return {
            'token_file': token_file,
            'agent_port_map': {host: node_dict.get(host, {}).get('agent_port') for host in self.hosts},
        }

    def _phandle(self, hosts):
        managed = is_managed_compute()
        overrides = self.config.get('parallel_handle') or {}
        config_overrides = overrides.get('config') or {}
        return MultiProcessParallelHandle(
            self.log,
            hosts,
            user=self.user,
            password=self.password,
            pkey=self.pkey,
            host_key_check=False,
            stop_on_errors=self.stop_on_errors,
            env_vars=self.config.get('env_vars'),
            config=ParallelConfig(**config_overrides) if config_overrides else None,
            transport='http' if managed else 'ssh',
            **(self._transport_kwargs() if managed else {}),
            **(overrides.get('transport_kwargs') or {}),
        )

    def close(self):
        """Destroy the two long-lived handles built in __init__.

        Subset handles are already destroyed at their call sites, but head/all live as
        long as the orchestrator does -- a whole test module under the ``orch`` fixture --
        and each one holds either SSH sessions or an HTTP connection pool.
        """
        for handle in (getattr(self, 'head', None), getattr(self, 'all', None)):
            if handle is None:
                continue
            try:
                handle.destroy_clients()
            except Exception as exc:
                self.log.debug("Error destroying parallel handle: %s", exc)

    def exec(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        """
        Execute command across hosts via SSH (baremetal execution).

        Args:
            cmd: Command to execute
            hosts: Target hosts (if None, uses all hosts)
            timeout: Command timeout
            detailed: If True, return detailed execution info including
                exit_code (mirrors ContainerOrchestrator.exec).
            print_console: If False, the command's output is returned but not
                logged. Use for bulk data the caller parses itself.

        Returns:
            Dictionary mapping hosts to execution results
        """
        if hosts is None:
            hosts = self.hosts

        # Use appropriate handle based on target hosts
        if set(hosts) == set(self.hosts):
            return self.all.exec(cmd, timeout=timeout, detailed=detailed, print_console=print_console)
        else:
            # For arbitrary subset (including head node), create temporary handle
            phandle = self._phandle(hosts)
            try:
                return phandle.exec(cmd, timeout=timeout, detailed=detailed, print_console=print_console)
            finally:
                phandle.destroy_clients()

    def exec_on_host(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        """Execute directly on the host OS."""
        return self.exec(
            cmd,
            hosts=hosts,
            timeout=timeout,
            detailed=detailed,
            print_console=print_console,
        )

    def sudo_prefix(self):
        """
        Return the command prefix needed for privileged commands, probing
        passwordless-sudo availability at most once per orchestrator instance.

        CVS's sudo model is passwordless-or-none, so a single boolean answer
        (rather than per-command retry) is sufficient. The fleet-wide answer
        is taken from the head node's result specifically (not an arbitrary
        dict-iteration-order pick) since exec_on_head is the dominant caller
        of privileged commands; if hosts disagree, a warning is logged but the
        head-node answer is still used for every command on every host.

        Returns:
            str: 'sudo -n ' if passwordless sudo is available, else ''.
        """
        if self._needs_sudo is None:
            sudo_status = get_passwordless_sudo_status(self.all)
            if len(set(sudo_status.values())) > 1:
                self.log.warning(f"Hosts disagree on passwordless sudo availability: {sudo_status}")
            self._needs_sudo = sudo_status.get(self.head_node, False)
        return 'sudo -n ' if self._needs_sudo else ''

    def exec_on_head(self, cmd, timeout=None, detailed=False, print_console=True):
        """
        Execute command on head node only via SSH.

        Args:
            cmd: Command to execute
            timeout: Command timeout
            detailed: See exec().
            print_console: See exec().

        Returns:
            Dictionary mapping head node to execution result
        """
        return self.head.exec(cmd, timeout=timeout, detailed=detailed, print_console=print_console)

    def upload_to_head(self, local_file, remote_file):
        """Upload a local file to the head node's host filesystem."""
        return self.head.upload_file(local_file, remote_file)

    def download_from_head(self, remote_file, local_file):
        """Download a file from the head node's host filesystem."""
        return self.head.download_file(remote_file, local_file)

    def download_file(self, remote_file, local_file, hosts=None):
        """Download a file through the orchestrator's host transport."""
        return self.all.download_file(remote_file, local_file, hosts=hosts)

    def setup_env(self, hosts, env_script=None):
        """Set up environment on hosts."""
        if not env_script:
            self.log.info("No environment script specified, skipping setup")
            return True

        self.log.info(f"Setting up environment on {len(hosts)} hosts")

        # Use appropriate handle
        if set(hosts) == set(self.hosts):
            result = self.all.exec(f"bash {env_script}", timeout=60, detailed=True)
        elif len(hosts) == 1 and hosts[0] == self.head_node:
            result = self.exec_on_head(f"bash {env_script}", timeout=60, detailed=True)
        else:
            phandle = self._phandle(hosts)
            try:
                result = phandle.exec(f"bash {env_script}", timeout=60, detailed=True)
            finally:
                phandle.destroy_clients()

        # Check if all hosts succeeded
        success = all(output['exit_code'] == 0 for output in result.values())

        if not success:
            failed = [host for host, output in result.items() if output['exit_code'] != 0]
            self.log.error(f"Environment setup failed on hosts: {failed}")

        return success

    def cleanup(self, hosts):
        """Clean up resources after test execution."""
        self.log.info(f"Cleaning up on {len(hosts)} hosts")
        # Basic cleanup - no specific resources to clean for SSH-only orchestrator
        return True

    def build_mpi_cmd(
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
        Build MPI command string for distributed execution.

        Writes the hostfile to a new private temp file on the head node. The
        returned command removes that file after mpirun exits, so run it once.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_hosts: List of host IPs/names for MPI hostfile
            ranks_per_host: Number of MPI ranks per host (uniform across hosts)
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory
            mpi_extra_args: List of additional mpirun arguments
            no_of_global_ranks: Total number of MPI ranks (optional, defaults to len(mpi_hosts) * ranks_per_host)

        Returns:
            Full MPI command string, exiting with mpirun's status
        """
        # Create MPI hostfile
        host_file_params = ''
        for host in mpi_hosts:
            host_file_params += f'{host} slots={ranks_per_host}\n'

        # mktemp creates a new mode-0600 file owned by the same user that runs
        # mpirun, without sudo, so concurrent runs never share a hostfile. The
        # template is a plain path because mktemp's options differ between GNU,
        # BusyBox, and BSD. The path is printed after a marker unique to this
        # call, so no other line of the head node's output can be taken for it.
        marker = f'cvs-mpi-hostfile-{uuid.uuid4().hex}:'
        cmd = (
            'hf=$(mktemp "${TMPDIR:-/tmp}/cvs_mpi_hosts.XXXXXXXX") || exit 1; '
            f'printf %s {shlex.quote(host_file_params)} > "$hf" || {{ rm -f "$hf"; exit 1; }}; '
            f'printf "%s%s\\n" {marker} "$hf"'
        )
        result = self.exec_on_head(cmd, detailed=True)
        failed = [
            f"{host} (exit {res.get('exit_code')}): {res.get('output', '').strip()}"
            for host, res in result.items()
            if res.get('exit_code') != 0
        ]
        if failed:
            raise RuntimeError(f"Failed to create MPI hostfile: {'; '.join(failed)}")

        host_files = [
            line[len(marker) :]
            for res in result.values()
            for line in res.get('output', '').splitlines()
            if line.startswith(marker)
        ]
        if not host_files:
            raise RuntimeError(f"Head node did not report the MPI hostfile path: {result}")
        quoted_host_file = shlex.quote(host_files[0])

        # Build MPI runner arguments
        if no_of_global_ranks is None:
            no_of_global_ranks = len(mpi_hosts) * ranks_per_host
        mpi_runner_args = [
            '--np',
            str(no_of_global_ranks),
            '--allow-run-as-root',
            '--hostfile',
            quoted_host_file,
        ]

        if mpi_extra_args:
            mpi_runner_args.extend(mpi_extra_args)

        full_mpi_cmd = self.get_mpi_command(rank_cmd, mpi_runner_args, env_vars, mpi_install_dir)

        # The cleanup travels in the returned string so it also runs inside the
        # container when ContainerOrchestrator executes it. The subshell keeps the
        # string a single command with mpirun's status, so a caller can still
        # append a pipe or `&&`.
        return f'({full_mpi_cmd}; rc=$?; rm -f {quoted_host_file}; exit $rc)'

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
        Distribute MPI job across hosts using the provided arguments.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_hosts: List of host IPs/names for MPI hostfile
            ranks_per_host: Number of MPI ranks per host (uniform across hosts)
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory
            mpi_extra_args: List of additional mpirun arguments
            no_of_global_ranks: Total number of MPI ranks (optional, defaults to len(mpi_hosts) * ranks_per_host)

        Returns:
            Execution results from head node
        """
        full_mpi_cmd = self.build_mpi_cmd(
            rank_cmd, mpi_hosts, ranks_per_host, env_vars, mpi_install_dir, mpi_extra_args, no_of_global_ranks
        )

        self.log.info("Launching MPI job")
        self.log.debug(f"MPI command: {full_mpi_cmd}")

        # Execute on head node
        result = self.exec_on_head(full_mpi_cmd, timeout=500)  # Default timeout

        return result

    def get_mpi_command(self, rank_cmd, mpi_runner_args, env_vars, mpi_install_dir):
        """
        Get the full MPI command string without executing it.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_runner_args: List of mpirun arguments
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory

        Returns:
            Full MPI command string
        """
        # Build environment variable arguments for mpirun
        env_args = []
        for key, value in env_vars.items():
            env_args.append(f'-x {key}={value}')

        # Build MPI runner arguments string
        mpi_runner_args_str = ' '.join(mpi_runner_args)

        # Add SSH options to auto-resolve host key issues in test environments
        ssh_options = f'--mca plm_rsh_agent ssh --mca plm_rsh_args "-p {self.ssh_port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"'

        # Construct full MPI command
        full_mpi_cmd = f'{mpi_install_dir}/mpirun {ssh_options} {mpi_runner_args_str} {" ".join(env_args)} {rank_cmd}'

        return full_mpi_cmd
