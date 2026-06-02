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
from cvs.lib.parallel.persistent_pssh_sharder import PersistentPsshSharder
from cvs.lib.parallel.interfaces import ShardableSshInterface
from cvs.lib import globals

global_log = globals.log


class MultiProcessPssh(ShardableSshInterface):
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
        **ssh_client_kwargs,
    ):
        # Initialize configuration
        self.config = config or ParallelConfig.from_env()
        hosts_per_shard = self.config.hosts_per_shard

        # Always ensure self.log is set for backward compatibility
        self.log = global_log

        # Store ssh_client_kwargs for forwarding to shard workers
        self.ssh_client_kwargs = ssh_client_kwargs

        n = len(host_list) if host_list is not None else 0
        use_mp = hosts_per_shard > 0 and n > hosts_per_shard

        if use_mp:
            # Initialize for multi-process sharding - no Pssh instance needed
            self._init_sharded(
                log, host_list, user, password, pkey, host_key_check, stop_on_errors, env_vars, **ssh_client_kwargs
            )

            if self.config.persistent_shards:
                self.sharder = PersistentPsshSharder(self.config)
            else:
                self.sharder = PsshSharder(self.config)
            self.sharder.initialize_workers(self.reachable_hosts, self._shard_init_kwargs())
            self.pssh = None  # No Pssh instance needed for sharded mode
        else:
            # No sharding - create Pssh instance for delegation
            jump_host_kwargs = {}
            if self.config.uses_jump_host:
                jump_host_kwargs.update(
                    {
                        'jump_host': self.config.jump_host,
                        'jump_user': self.config.jump_user,
                        'jump_password': self.config.jump_password,
                        'jump_pkey': self.config.jump_pkey,
                        'jump_port': self.config.jump_port,
                    }
                )

            self.pssh = Pssh(
                log,
                host_list,
                user,
                password,
                pkey,
                host_key_check,
                stop_on_errors,
                env_vars,
                process_output=True,  # Default to True for compatibility
                **jump_host_kwargs,
                **ssh_client_kwargs,
            )
            # Ensure attributes needed by _shard_init_kwargs are available
            self.host_list = host_list
            self.reachable_hosts = list(host_list)  # Initialize with all hosts
            self.unreachable_hosts = []
            self.env_vars = env_vars

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
        **ssh_client_kwargs,
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
        self.ssh_client_kwargs = ssh_client_kwargs
        self.env_prefix = build_env_prefix(env_vars)
        self.process_output = True  # Default to True for compatibility
        self.client = None
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
            **self.ssh_client_kwargs,  # Forward all ssh client kwargs to shard workers
        }

    def _sync_pssh_state(self):
        """Sync wrapper reachability state from Single-process Pssh (non-sharded mode)."""
        if self.pssh is None:
            return
        self.reachable_hosts = list(self.pssh.reachable_hosts)
        self.unreachable_hosts = list(self.pssh.unreachable_hosts)

    def sync_reachability_from_workers(self):
        """Sync wrapper reachability from sharder worker table."""
        if self.pssh is not None:
            return

        worker_table = self.sharder.get_worker_table()
        reachable_set = set()
        unreachable_set = set()
        for worker in worker_table:
            reachable_set.update(worker.reachable_hosts)
            unreachable_set.update(worker.unreachable_hosts)

        self.reachable_hosts = [h for h in self.host_list if h in reachable_set]
        self.unreachable_hosts = [h for h in self.host_list if h in unreachable_set and h not in reachable_set]

    def _merge_shard_returns(self, shard_returns, merge_unreachable=True):
        """
        Update reachable/unreachable host lists from shard workers and merge results to cmd_output dict.

        Returns cmd_output dict ready for _print_merged_outputs.
        """
        # 1. Update sharder worker table and sync wrapper reachability.
        self.sharder.update_worker_table(shard_returns, merge_unreachable=merge_unreachable)
        self.sync_reachability_from_workers()

        # 2. Merge results from shard workers (always dict format)
        merged = {}
        for shard in shard_returns:
            result = shard.get('result') or {}  # treat None as {} (for scp/reboot)
            # Shard workers always return processed dicts from _process_output
            merged.update(result)

        # 3. Preserve original host order
        cmd_output = {}
        for host in self.host_list:
            if host in merged:
                cmd_output[host] = merged[host]

        return cmd_output

    def _print_merged_outputs(self, cmd_output, cmd=None, cmd_list=None, cmd_hosts=None, print_console=True):
        """Print command outputs in original host order."""
        if not print_console:
            return
        idx = {h: i for i, h in enumerate(cmd_hosts)} if cmd_hosts is not None else {}
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

    def exec(self, cmd, timeout=None, print_console=True, detailed=False):
        """Execute command with automatic sharding if needed."""
        if self.pssh is not None:
            result = self.pssh.exec(cmd, timeout=timeout, print_console=print_console, detailed=detailed)
            self._sync_pssh_state()
            return result

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

        # Use sharder for sharded execution with worker routing
        routing_map = self.sharder.create_payloads(
            'exec', self._shard_init_kwargs(), cmd=cmd, timeout=timeout, detailed=detailed
        )
        shard_returns = self.sharder.execute_sharded(routing_map)

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
        if self.pssh is not None:
            result = self.pssh.exec_cmd_list(cmd_list, timeout=timeout, print_console=print_console)
            self._sync_pssh_state()
            return result

        # Keep host_list as full inventory; accept command lists aligned either
        # with full host_list or current reachable_hosts.
        if len(cmd_list) == len(self.host_list):
            cmd_hosts = list(self.host_list)
        elif len(cmd_list) == len(self.reachable_hosts):
            cmd_hosts = list(self.reachable_hosts)
        else:
            raise ValueError(
                "cmd_list length must match host_list length or reachable_hosts length "
                f"(cmd_list={len(cmd_list)}, host_list={len(self.host_list)}, "
                f"reachable_hosts={len(self.reachable_hosts)})"
            )

        # Keep raw commands for shard payloads (worker Pssh applies env_prefix once).
        # Build expanded commands separately for logging compatibility.
        raw_commands = list(cmd_list)
        if self.env_prefix:
            expanded = [f"{self.env_prefix} ; {cmd}" for cmd in raw_commands]
        else:
            expanded = list(raw_commands)

        # Build host->command mapping used by worker-table routing.
        command_by_host = dict(zip(cmd_hosts, raw_commands))
        self.log.info("%s", [command_by_host[h] for h in self.reachable_hosts if h in command_by_host])

        # Log command list execution
        if self.log:
            if timeout is not None:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]")
            else:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s)")

        # Prepare payloads for exec_cmd_list (manual routing due to per-host commands)
        routing_map = {}
        for worker in self.sharder.get_worker_table():
            if not worker.reachable_hosts:
                continue

            shard_hosts = list(worker.reachable_hosts)
            shard_commands = [command_by_host[h] for h in shard_hosts if h in command_by_host]

            # Create payload manually for this worker
            payload = {
                'operation': 'exec_cmd_list',
                'init': {**self._shard_init_kwargs(), 'host_list': shard_hosts},
                'cmd_list': shard_commands,
                'timeout': timeout,
            }

            routing_map[worker.worker_id] = payload

        shard_returns = self.sharder.execute_sharded(routing_map)

        # Merge results, update state, and convert to cmd_output dict
        cmd_output = self._merge_shard_returns(shard_returns)

        self._print_merged_outputs(
            cmd_output, cmd=None, cmd_list=expanded, cmd_hosts=cmd_hosts, print_console=print_console
        )

        # Log per-host command execution
        if self.log:
            for host, cmd in zip(cmd_hosts, expanded):
                self.log.debug(f"Command on {host}: {cmd}")

        return cmd_output

    def scp_file(self, local_file, remote_file, recurse=False):
        """
        Backward-compatible alias for upload_file.

        Kept so existing callers (and log-grep tooling looking for the legacy
        "About to copy local file..." line) keep working. New code should call
        upload_file directly.
        """
        self.log.info('About to copy local file {} to remote {} on all Hosts'.format(local_file, remote_file))
        return self.upload_file(local_file, remote_file, recurse=recurse)

    def upload_file(self, local_file, remote_file, recurse=False):
        """Upload file with automatic sharding if needed."""
        if self.pssh is not None:
            return self.pssh.upload_file(local_file, remote_file, recurse=recurse)

        self.log.info('SFTP upload %s -> %s on %s', local_file, remote_file, self.reachable_hosts)

        routing_map = self.sharder.create_payloads(
            'upload_file',
            self._shard_init_kwargs(),
            local_file=local_file,
            remote_file=remote_file,
            recurse=recurse,
        )
        shard_returns = self.sharder.execute_sharded(routing_map)
        return self._merge_shard_returns(shard_returns)

    def download_file(self, remote_file, local_file, recurse=False, suffix_separator='_'):
        """Download file with automatic sharding if needed."""
        if self.pssh is not None:
            return self.pssh.download_file(remote_file, local_file, recurse=recurse, suffix_separator=suffix_separator)

        self.log.info('SFTP download %s -> %s from %s', remote_file, local_file, self.reachable_hosts)

        routing_map = self.sharder.create_payloads(
            'download_file',
            self._shard_init_kwargs(),
            remote_file=remote_file,
            local_file=local_file,
            recurse=recurse,
            suffix_separator=suffix_separator,
        )
        shard_returns = self.sharder.execute_sharded(routing_map)
        return self._merge_shard_returns(shard_returns)

    def reboot_connections(self):
        """Reboot connections with automatic sharding if needed."""
        if self.pssh is not None:
            return self.pssh.reboot_connections()

        self.log.info('Rebooting Connections')

        routing_map = self.sharder.create_payloads('reboot_connections', self._shard_init_kwargs())
        shard_returns = self.sharder.execute_sharded(routing_map)
        return self._merge_shard_returns(shard_returns, merge_unreachable=False)

    def upload_file_list(self, node_path_map):
        """Upload different files to different hosts with automatic sharding if needed."""
        if self.pssh is not None:
            return self.pssh.upload_file_list(node_path_map)

        if not node_path_map:
            return {}

        self.log.info(f"Uploading files to hosts with sharding: {len(node_path_map)} file mappings")

        # Prepare routing map for upload_file_list (manual routing due to per-host file mappings)
        routing_map = {}
        for worker in self.sharder.get_worker_table():
            if not worker.reachable_hosts:
                continue

            shard_hosts = list(worker.reachable_hosts)
            # Create subset of node_path_map for this worker
            shard_node_path_map = {h: node_path_map[h] for h in shard_hosts if h in node_path_map}

            if shard_node_path_map:
                # Worker host_list must match node_path_map keys for upload_file_list.
                # Pssh.upload_file_list delegates to ParallelSSHClient.copy_file(copy_args=...),
                # which requires one copy_args entry per worker host in order.
                target_hosts = [h for h in shard_hosts if h in shard_node_path_map]

                # Create payload manually for this worker
                payload = {
                    'operation': 'upload_file_list',
                    'init': {**self._shard_init_kwargs(), 'host_list': target_hosts},
                    'node_path_map': shard_node_path_map,
                }

                routing_map[worker.worker_id] = payload

        if not routing_map:
            return {}

        shard_returns = self.sharder.execute_sharded(routing_map)

        # Merge results from all shards
        merged_results = {}
        for shard in shard_returns:
            result = shard.get('result', {})
            merged_results.update(result)

        return merged_results

    def _shard_init_kwargs(self):
        """Prepare kwargs for shard worker initialization."""
        kwargs = {
            'log': None,  # Backward compatibility - child processes use module-level log
            'user': self.user,
            'password': self.password,
            'pkey': self.pkey,
            'host_key_check': self.host_key_check,
            'stop_on_errors': self.stop_on_errors,
            'env_vars': self.env_vars,
        }

        # Add jump host parameters to shard init
        if self.config.uses_jump_host:
            kwargs.update(
                {
                    'jump_host': self.config.jump_host,
                    'jump_user': self.config.jump_user,
                    'jump_password': self.config.jump_password,
                    'jump_pkey': self.config.jump_pkey,
                    'jump_port': self.config.jump_port,
                }
            )

        return kwargs

    def prune_nodes(self, nodes_to_remove):
        """
        Explicitly prune nodes from future operations.

        Works in both modes:
        - sharded mode: updates wrapper host state
        - non-sharded mode: delegates to Single-process Pssh (which rebuilds its client)
        """
        if not nodes_to_remove:
            return []

        remove_set = {h for h in nodes_to_remove if h}
        removed = [h for h in self.reachable_hosts if h in remove_set]
        if not removed:
            return []

        if self.pssh is not None:
            # Non-sharded mode: let single-process Pssh own prune + client rebuild.
            removed = self.pssh.prune_nodes(removed)
            self._sync_pssh_state()
            return removed

        # Sharded mode: prune sharder source-of-truth, then sync wrapper cache.
        self.sharder.prune_worker_nodes(set(removed))
        self.sync_reachability_from_workers()
        return removed

    def destroy_clients(self):
        """Destroy clients - handle both sharded and non-sharded modes."""
        if self.pssh is None:
            self.log.info('Cleaning up sharded mode state ..')
            if hasattr(self, 'sharder') and self.sharder is not None:
                self.sharder.destroy_clients()
            self.client = None
        else:
            self.log.info('Destroying Current phdl connections ..')
            # In non-sharded mode, properly destroy the Pssh instance connections
            if self.pssh is not None:
                self.pssh.destroy_clients()
