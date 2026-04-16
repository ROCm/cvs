'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import print_function

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from cvs.lib.env_lib import build_env_prefix
from cvs.lib.parallel.base import PsshShard, _chunk_hosts, _pssh_shard_worker


class Pssh(PsshShard):
    """
    Parallel SSH with optional process sharding for large host lists.

    Same constructor and public API as PsshShard; when ``len(host_list) > hosts_per_shard`` (default 32),
    ``exec`` / ``exec_cmd_list`` / ``scp_file`` / ``reboot_connections`` run in child processes (spawn),
    each running a ``PsshShard`` over a slice of hosts.

    Pass ``hosts_per_shard=0`` to disable process sharding (always one ``PsshShard`` in the parent process).
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
        hosts_per_shard=32,
        **ssh_client_kwargs,
    ):
        self.hosts_per_shard = hosts_per_shard
        n = len(host_list) if host_list is not None else 0
        use_mp = hosts_per_shard > 0 and n > hosts_per_shard

        if not use_mp:
            PsshShard.__init__(
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
            )
            self._use_process_sharding = False
        else:
            self._init_sharded(
                log, host_list, user, password, pkey, host_key_check, stop_on_errors, env_vars, ssh_client_kwargs
            )

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
        ssh_client_kwargs,
    ):
        self.log = log
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
        self.client = None
        self._use_process_sharding = True
        if self.log:
            self.log.debug(f"Environ vars: {self.env_prefix}")

        if self.password is None:
            print(self.reachable_hosts)
            print(self.user)
            print(self.pkey)

    def _shard_init_kwargs(self, shard_hosts):
        return {
            'log': None,
            'host_list': shard_hosts,
            'user': self.user,
            'password': self.password,
            'pkey': self.pkey,
            'host_key_check': self.host_key_check,
            'stop_on_errors': self.stop_on_errors,
            'env_vars': self.env_vars,
            **self.ssh_client_kwargs,
        }

    def _merge_shard_returns(self, shard_returns, merge_unreachable=True):
        """Update reachable / unreachable host lists from shard workers."""
        reachable_set = set()
        for r in shard_returns:
            reachable_set.update(r['reachable_hosts'])
        self.reachable_hosts = [h for h in self.host_list if h in reachable_set]
        if merge_unreachable:
            for r in shard_returns:
                for u in r['unreachable_hosts']:
                    if u not in self.unreachable_hosts:
                        self.unreachable_hosts.append(u)

    def _print_merged_outputs(self, cmd_output, cmd=None, cmd_list=None, print_console=True):
        if not print_console:
            return
        idx = {h: i for i, h in enumerate(self.host_list)}
        for host in self.host_list:
            if host not in cmd_output:
                continue
            print('#----------------------------------------------------------#')
            print(f'Host == {host} ==')
            print('#----------------------------------------------------------#')
            if cmd_list is not None:
                print(cmd_list[idx[host]])
            else:
                print(cmd)
            for line in cmd_output[host].splitlines():
                print(line)

    def _run_sharded_pool(self, payloads):
        """Run one worker payload per shard; preserve exception behavior (first failure propagates)."""
        ctx = mp.get_context('spawn')
        max_workers = min(len(payloads), max(32, (os.cpu_count() or 4) * 4))
        results = [None] * len(payloads)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = {ex.submit(_pssh_shard_worker, p): i for i, p in enumerate(payloads)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
        return results

    def exec(self, cmd, timeout=None, print_console=True):
        if not getattr(self, '_use_process_sharding', False):
            return PsshShard.exec(self, cmd, timeout, print_console)

        if self.env_prefix:
            full_cmd = f"{self.env_prefix} ; {cmd}"
        else:
            full_cmd = cmd

        print(f'cmd = {full_cmd}')
        if self.log:
            if timeout is not None:
                self.log.debug(
                    f"Executing command on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]: {full_cmd}"
                )
            else:
                self.log.debug(f"Executing command on {len(self.reachable_hosts)} host(s): {full_cmd}")

        payloads = []
        for shard_hosts in _chunk_hosts(self.reachable_hosts, self.hosts_per_shard):
            payloads.append(
                {
                    'mode': 'exec',
                    'init': self._shard_init_kwargs(shard_hosts),
                    'cmd': cmd,
                    'timeout': timeout,
                }
            )

        shard_returns = self._run_sharded_pool(payloads)
        self._merge_shard_returns(shard_returns)
        cmd_output = {}
        for r in shard_returns:
            cmd_output.update(r['result'])

        self._print_merged_outputs(cmd_output, cmd=full_cmd, cmd_list=None, print_console=print_console)

        if self.log:
            for host in cmd_output.keys():
                self.log.debug(f"Command completed on {host}: {cmd}")

        return cmd_output

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        if not getattr(self, '_use_process_sharding', False):
            return PsshShard.exec_cmd_list(self, cmd_list, timeout, print_console)

        if len(cmd_list) != len(self.host_list):
            raise ValueError('cmd_list length must match host_list length')

        if self.env_prefix:
            expanded = [f"{self.env_prefix} ; {cmd}" for cmd in cmd_list]
        else:
            expanded = list(cmd_list)

        print(expanded)
        if self.log:
            if timeout is not None:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]")
            else:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s)")

        payloads = []
        offset = 0
        for shard_hosts in _chunk_hosts(self.reachable_hosts, self.hosts_per_shard):
            k = len(shard_hosts)
            sub_list = expanded[offset : offset + k]
            offset += k
            payloads.append(
                {
                    'mode': 'cmd_list',
                    'init': self._shard_init_kwargs(shard_hosts),
                    'cmd_list': sub_list,
                    'timeout': timeout,
                }
            )

        shard_returns = self._run_sharded_pool(payloads)
        self._merge_shard_returns(shard_returns)
        cmd_output = {}
        for r in shard_returns:
            cmd_output.update(r['result'])

        self._print_merged_outputs(cmd_output, cmd=None, cmd_list=expanded, print_console=print_console)

        if self.log:
            for host, cmd in zip(self.host_list, expanded):
                self.log.debug(f"Command on {host}: {cmd}")

        return cmd_output

    def scp_file(self, local_file, remote_file, recurse=False):
        if not getattr(self, '_use_process_sharding', False):
            return PsshShard.scp_file(self, local_file, remote_file, recurse=recurse)

        print('About to copy local file {} to remote {} on all Hosts'.format(local_file, remote_file))
        payloads = []
        for shard_hosts in _chunk_hosts(self.reachable_hosts, self.hosts_per_shard):
            payloads.append(
                {
                    'mode': 'scp',
                    'init': self._shard_init_kwargs(shard_hosts),
                    'local_file': local_file,
                    'remote_file': remote_file,
                    'recurse': recurse,
                }
            )
        shard_returns = self._run_sharded_pool(payloads)
        self._merge_shard_returns(shard_returns, merge_unreachable=False)

    def reboot_connections(self):
        if not getattr(self, '_use_process_sharding', False):
            return PsshShard.reboot_connections(self)

        print('Rebooting Connections')
        payloads = []
        for shard_hosts in _chunk_hosts(self.reachable_hosts, self.hosts_per_shard):
            payloads.append({'mode': 'reboot', 'init': self._shard_init_kwargs(shard_hosts)})
        shard_returns = self._run_sharded_pool(payloads)
        self._merge_shard_returns(shard_returns, merge_unreachable=False)

    def destroy_clients(self):
        if getattr(self, '_use_process_sharding', False):
            print('Destroying Current phdl connections ..')
            self.client = None
            return
        PsshShard.destroy_clients(self)

    def copy_script_list(self, node_script_map, remote_path="/tmp/script.sh"):
        if not getattr(self, '_use_process_sharding', False):
            return PsshShard.copy_script_list(self, node_script_map, remote_path)

        # For sharded mode, delegate to base class as it already handles concurrent operations well
        return PsshShard.copy_script_list(self, node_script_map, remote_path)
