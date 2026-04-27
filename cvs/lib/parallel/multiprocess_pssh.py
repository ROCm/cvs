'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.lib.env_lib import build_env_prefix
from cvs.lib.parallel.pssh import Pssh
from cvs.lib.parallel.config import ParallelConfig
from cvs.lib.parallel.pssh_sharder import PsshSharder
from cvs.lib import globals

global_log = globals.log


class MultiProcessPssh(Pssh):
    """
    Multi-process parallel SSH with automatic host sharding for large host lists.

    When ``len(host_list) > hosts_per_shard`` (default 32), operations like
    ``exec`` / ``exec_cmd_list`` / ``scp_file`` / ``reboot_connections`` run in child processes (spawn),
    each running a ``Pssh`` instance over a slice of hosts.

    Pass ``hosts_per_shard=0`` to disable process sharding (always one ``Pssh`` in the parent process).
    """

    def __init__(
        self,
        log,
        host_list,
        user=None,
        password=None,
        pkey='id_rsa',
        host_key_check=False,
        stop_on_errors=True,
        env_vars=None,
        config=None,
    ):
        # Initialize configuration
        self.config = config or ParallelConfig.from_env()
        hosts_per_shard = self.config.hosts_per_shard

        # Always ensure self.log is set for backward compatibility
        self.log = global_log

        n = len(host_list) if host_list is not None else 0
        use_mp = hosts_per_shard > 0 and n > hosts_per_shard

        if not use_mp:
            # Use single-process base class
            super().__init__(
                log,  # Pass through log parameter to parent
                host_list,
                user,
                password,
                pkey,
                host_key_check,
                stop_on_errors,
                env_vars,
                process_output=True,  # Default to True for compatibility
            )
            self._use_process_sharding = False
            # Ensure attributes needed by _shard_init_kwargs are available
            self.env_vars = env_vars
        else:
            # Initialize for multi-process sharding
            self._init_sharded(log, host_list, user, password, pkey, host_key_check, stop_on_errors, env_vars)

            # Create the sharder - no registration needed with direct operations!
            self.sharder = PsshSharder(self.config)

    def _init_sharded(
        self,
        log,
        host_list,
        user,
        password,
        pkey,
        host_key_check,
        stop_on_errors,
        env_vars,
    ):
        """Initialize for sharded multi-process execution."""
        # Always use global logger but maintain self.log for backward compatibility
        self.log = global_log
        self.host_list = host_list
        self.reachable_hosts = list(host_list)
        self.user = user
        self.pkey = pkey
        self.password = password
        self.host_key_check = host_key_check
        self.stop_on_errors = stop_on_errors
        self.unreachable_hosts = []
        self.env_vars = env_vars
        self.env_prefix = build_env_prefix(env_vars)
        self.process_output = True  # Default to True for compatibility
        self.client = None
        self._use_process_sharding = True
        self.log.debug(f"Environ vars: {self.env_prefix}")

    def _shard_init_kwargs(self):
        """Create initialization kwargs for shard workers (host_list will be added by create_payloads).

        Note: No logger parameter needed - child processes use module-level globals.log automatically.
        """
        return {
            'log': None,  # Backward compatibility - child processes use module-level log
            'user': self.user,
            'password': self.password,
            'pkey': self.pkey,
            'host_key_check': self.host_key_check,
            'stop_on_errors': self.stop_on_errors,
            'env_vars': self.env_vars,
        }

    def _merge_shard_returns(self, shard_returns, merge_unreachable=True):
        """
        Update reachable/unreachable host lists from shard workers and convert results to cmd_output dict.

        Handles both state updates and SimpleHostOutput conversion in one place for cleaner architecture.
        Returns cmd_output dict ready for _print_merged_outputs.
        """
        # 1. Update host lists (existing logic)
        reachable_set = set()
        for r in shard_returns:
            reachable_set.update(r['reachable_hosts'])
        self.reachable_hosts = [h for h in self.host_list if h in reachable_set]
        if merge_unreachable:
            for r in shard_returns:
                for u in r['unreachable_hosts']:
                    if u not in self.unreachable_hosts:
                        self.unreachable_hosts.append(u)

        # 2. Convert SimpleHostOutput objects to cmd_output dict
        merged = {}
        for shard in shard_returns:
            result = shard.get('result') or {}  # treat None as {} (for scp/reboot)

            if isinstance(result, list):
                # Handle SimpleHostOutput objects - convert using _process_output
                temp_dict = self._process_output(result, print_console=False)
                merged.update(temp_dict)
            else:
                # Handle old dict format (for scp/reboot)
                merged.update(result)

        # 3. Preserve original host order
        cmd_output = {}
        for host in self.host_list:
            if host in merged:
                cmd_output[host] = merged[host]

        return cmd_output

    def _print_merged_outputs(self, cmd_output, cmd=None, cmd_list=None, print_console=True):
        """Print command outputs in original host order."""
        if not print_console:
            return
        idx = {h: i for i, h in enumerate(self.host_list)}
        for host in self.host_list:
            if host not in cmd_output:
                continue
            self.log.info('#----------------------------------------------------------#')
            self.log.info(f'Host == {host} ==')
            self.log.info('#----------------------------------------------------------#')
            if cmd_list is not None:
                self.log.debug("%s", cmd_list[idx[host]])
            else:
                self.log.debug("%s", cmd)
            for line in cmd_output[host].splitlines():
                self.log.info("%s", line)

    def exec(self, cmd, timeout=None, print_console=True):
        """Execute command with automatic sharding if needed."""
        if not getattr(self, '_use_process_sharding', False):
            return super().exec(cmd, timeout=timeout, print_console=print_console)

        if self.env_prefix:
            full_cmd = f"{self.env_prefix} ; {cmd}"
        else:
            full_cmd = cmd

        self.log.info(f'cmd = {full_cmd}')

        # Log command execution
        if self.log:
            if timeout is not None:
                self.log.debug(
                    f"Executing command on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]: {full_cmd}"
                )
            else:
                self.log.debug(f"Executing command on {len(self.reachable_hosts)} host(s): {full_cmd}")

        # Use sharder for sharded execution
        host_chunks = list(self.sharder.chunk_hosts(self.reachable_hosts))
        payloads = self.sharder.create_payloads(
            'exec', host_chunks, self._shard_init_kwargs(), cmd=cmd, timeout=timeout
        )
        shard_returns = self.sharder.execute_sharded(payloads)

        # Merge results, update state, and convert to cmd_output dict
        cmd_output = self._merge_shard_returns(shard_returns)

        self._print_merged_outputs(cmd_output, cmd=full_cmd, cmd_list=None, print_console=print_console)

        # Log per-host execution completion
        if self.log:
            for host in cmd_output.keys():
                self.log.debug(f"Command completed on {host}: {cmd}")

        return cmd_output

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """Execute command list with automatic sharding if needed."""
        if not getattr(self, '_use_process_sharding', False):
            return super().exec_cmd_list(cmd_list, timeout=timeout, print_console=print_console)

        if len(cmd_list) != len(self.host_list):
            raise ValueError('cmd_list length must match host_list length')

        # Apply env_prefix to all commands (for logging compatibility)
        if self.env_prefix:
            expanded = [f"{self.env_prefix} ; {cmd}" for cmd in cmd_list]
        else:
            expanded = list(cmd_list)

        # Only filter commands if some hosts are unreachable
        if len(self.reachable_hosts) < len(self.host_list):
            # Filter commands to match reachable_hosts order
            filtered_commands = []
            for i, host in enumerate(self.host_list):
                if host in self.reachable_hosts:
                    filtered_commands.append(expanded[i])
        else:
            # All hosts are reachable, use expanded directly
            filtered_commands = expanded

        self.log.info("%s", filtered_commands)

        # Log command list execution
        if self.log:
            if timeout is not None:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]")
            else:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s)")

        # Prepare payloads for sharded execution
        payloads = []
        offset = 0
        chunk_size = self.config.hosts_per_shard
        for i in range(0, len(self.reachable_hosts), chunk_size):
            shard_hosts = self.reachable_hosts[i : i + chunk_size]
            k = len(shard_hosts)
            shard_commands = filtered_commands[offset : offset + k]
            offset += k
            payloads.append(
                self.sharder.create_payloads(
                    'cmd_list', [shard_hosts], self._shard_init_kwargs(), cmd_list=shard_commands, timeout=timeout
                )[0]
            )

        shard_returns = self.sharder.execute_sharded(payloads)

        # Merge results, update state, and convert to cmd_output dict
        cmd_output = self._merge_shard_returns(shard_returns)

        self._print_merged_outputs(cmd_output, cmd=None, cmd_list=expanded, print_console=print_console)

        # Log per-host command execution
        if self.log:
            for host, cmd in zip(self.host_list, expanded):
                self.log.debug(f"Command on {host}: {cmd}")

        return cmd_output

    def scp_file(self, local_file, remote_file, recurse=False):
        """Copy file with automatic sharding if needed."""
        if not getattr(self, '_use_process_sharding', False):
            return super().scp_file(local_file, remote_file, recurse=recurse)

        self.log.info('About to copy local file {} to remote {} on all Hosts'.format(local_file, remote_file))

        host_chunks = list(self.sharder.chunk_hosts(self.reachable_hosts))
        payloads = self.sharder.create_payloads(
            'scp',
            host_chunks,
            self._shard_init_kwargs(),
            local_file=local_file,
            remote_file=remote_file,
            recurse=recurse,
        )
        shard_returns = self.sharder.execute_sharded(payloads)
        return self._merge_shard_returns(shard_returns, merge_unreachable=False)

    def reboot_connections(self):
        """Reboot connections with automatic sharding if needed."""
        if not getattr(self, '_use_process_sharding', False):
            return super().reboot_connections()

        self.log.info('Rebooting Connections')

        host_chunks = list(self.sharder.chunk_hosts(self.reachable_hosts))
        payloads = self.sharder.create_payloads('reboot', host_chunks, self._shard_init_kwargs())
        shard_returns = self.sharder.execute_sharded(payloads)
        return self._merge_shard_returns(shard_returns, merge_unreachable=False)

    def destroy_clients(self):
        """Destroy clients - handle both sharded and non-sharded modes."""
        if getattr(self, '_use_process_sharding', False):
            self.log.info('Destroying Current phdl connections ..')
            self.client = None
            return
        super().destroy_clients()
