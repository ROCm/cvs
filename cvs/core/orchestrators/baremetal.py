'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import posixpath
import shlex

from cvs.core.orchestrators.base import Orchestrator
from cvs.core.scheduler import is_managed_compute
from cvs.lib.env_lib import redact_secrets
from cvs.lib.parallel.config import ParallelConfig
from cvs.lib.parallel.multiprocess_phandle import MultiProcessParallelHandle
from cvs.lib.utils_lib import get_passwordless_sudo_status


_MPI_HOSTFILE_PREFIX = 'mpi_hosts_'


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

        # Keep private hostfiles tracked until removal succeeds, so failures can be retried.
        self._mpi_hostfiles = []

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

    def exec(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True, inactivity_timeout=None):
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
            inactivity_timeout: Abort a host only after this many seconds with no
                new output (no total wall-clock cap). None keeps the plain
                ``timeout`` budget.

        Returns:
            Dictionary mapping hosts to execution results
        """
        if hosts is None:
            hosts = self.hosts

        # Use appropriate handle based on target hosts
        if set(hosts) == set(self.hosts):
            return self.all.exec(
                cmd,
                timeout=timeout,
                detailed=detailed,
                print_console=print_console,
                inactivity_timeout=inactivity_timeout,
            )
        else:
            # For arbitrary subset (including head node), create temporary handle
            phandle = self._phandle(hosts)
            try:
                return phandle.exec(
                    cmd,
                    timeout=timeout,
                    detailed=detailed,
                    print_console=print_console,
                    inactivity_timeout=inactivity_timeout,
                )
            finally:
                phandle.destroy_clients()

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """
        Execute different commands on different hosts via SSH.

        Each cmd_list[i] runs on hosts[i] — the parallel per-host equivalent of
        exec(). Positional correspondence between cmd_list and the
        orchestrator's host list is the caller's responsibility.

        Args:
            cmd_list: List of commands, one per host in host order
            timeout: Command timeout
            print_console: See exec().

        Returns:
            Dictionary mapping hosts to execution results
        """
        return self.all.exec_cmd_list(cmd_list, timeout=timeout, print_console=print_console)

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

    def _create_mpi_hostfile(self, mpi_hosts, ranks_per_host):
        """Create an exclusive, private hostfile as the user who runs mpirun.

        The random 0600 file cannot be pre-placed or redirected by another user.
        """
        host_file = ''.join(f'{host} slots={ranks_per_host}\n' for host in mpi_hosts)
        # shlex.quote protects the host list in the remote login shell.
        cmd = (
            '(umask 077 && f=$(mktemp -p "${TMPDIR:-/tmp}" ' + _MPI_HOSTFILE_PREFIX + 'XXXXXXXXXX)'
            f' && printf %s {shlex.quote(host_file)} > "$f" && printf %s "$f")'
        )
        result = self.exec_on_head(cmd, timeout=30, detailed=True)
        failed = [
            host for host, res in (result or {}).items() if not isinstance(res, dict) or res.get('exit_code') != 0
        ]
        if not result or failed:
            raise RuntimeError(f"Failed to create MPI hostfile on hosts: {failed or [self.head_node]}")

        head = result.get(self.head_node)
        if head is None:
            raise RuntimeError(f"Missing head node {self.head_node} in MPI hostfile result: {list(result)}")

        path = (head.get('output') or '').strip()
        if '\n' in path or not posixpath.isabs(path) or not posixpath.basename(path).startswith(_MPI_HOSTFILE_PREFIX):
            raise RuntimeError(f"MPI hostfile path from {self.head_node} failed validation: {path!r}")
        return path

    def release_mpi_hostfiles(self):
        """Remove tracked hostfiles after a direct build_mpi_cmd call or MPI run.

        Failed removals stay tracked for retry. Cleanup never masks a run's result.
        """
        remaining = []
        for path in self._mpi_hostfiles:
            try:
                result = self.exec_on_head(f'rm -f -- {shlex.quote(path)}', timeout=30, detailed=True)
            except Exception as exc:
                self.log.warning('Could not remove MPI hostfile %s on %s: %r', path, self.head_node, exc)
                remaining.append(path)
                continue
            failed = [
                host for host, res in (result or {}).items() if not isinstance(res, dict) or res.get('exit_code') != 0
            ]
            if not result or failed:
                self.log.warning('Could not remove MPI hostfile %s on %s: %r', path, self.head_node, result)
                remaining.append(path)
        self._mpi_hostfiles = remaining
        return not remaining

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
        Build an MPI command with a private hostfile tracked on the head node.

        distribute_using_mpi removes the hostfile. Direct callers should call
        release_mpi_hostfiles after running the command.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_hosts: List of host IPs/names for MPI hostfile
            ranks_per_host: Number of MPI ranks per host (uniform across hosts)
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory
            mpi_extra_args: List of additional mpirun arguments
            no_of_global_ranks: Total number of MPI ranks (optional, defaults to len(mpi_hosts) * ranks_per_host)

        Returns:
            Full MPI command string

        Raises:
            ValueError: If mpi_hosts is empty
            RuntimeError: If the hostfile cannot be created or validated
        """
        if not mpi_hosts:
            raise ValueError("build_mpi_cmd needs at least one host in mpi_hosts")

        self.release_mpi_hostfiles()
        hosts_file_path = self._create_mpi_hostfile(mpi_hosts, ranks_per_host)
        self._mpi_hostfiles.append(hosts_file_path)

        # Build MPI runner arguments
        if no_of_global_ranks is None:
            no_of_global_ranks = len(mpi_hosts) * ranks_per_host
        mpi_runner_args = [
            '--np',
            str(no_of_global_ranks),
            '--allow-run-as-root',
            '--hostfile',
            shlex.quote(hosts_file_path),
        ]

        if mpi_extra_args:
            mpi_runner_args.extend(mpi_extra_args)

        full_mpi_cmd = self.get_mpi_command(rank_cmd, mpi_runner_args, env_vars, mpi_install_dir)

        return full_mpi_cmd

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
        try:
            full_mpi_cmd = self.build_mpi_cmd(
                rank_cmd, mpi_hosts, ranks_per_host, env_vars, mpi_install_dir, mpi_extra_args, no_of_global_ranks
            )

            self.log.info("Launching MPI job")
            self.log.debug(f"MPI command: {redact_secrets(full_mpi_cmd)}")

            return self.exec_on_head(full_mpi_cmd, timeout=500)
        finally:
            self.release_mpi_hostfiles()

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
