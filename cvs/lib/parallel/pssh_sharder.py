'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import inspect
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from cvs.lib.parallel.pssh import Pssh
from cvs.lib.parallel.interfaces import ShardableSshInterface


# Dynamically discover supported operations from ABC (computed once at import time)

SUPPORTED_OPERATIONS = {
    name for name, method in inspect.getmembers(ShardableSshInterface) if getattr(method, '__isabstractmethod__', False)
}


class PsshSharder:
    """Shards SSH operations across multiple processes for large host lists."""

    def __init__(self, config):
        self.config = config

    def chunk_hosts(self, host_list):
        """Divide hosts into processing shards."""
        chunk_size = self.config.hosts_per_shard
        for i in range(0, len(host_list), chunk_size):
            yield host_list[i : i + chunk_size]

    def create_payloads(self, operation, host_chunks, shard_init_kwargs, **operation_args):
        """Build payloads for worker processes."""
        payloads = []
        for chunk in host_chunks:
            payloads.append(
                {
                    'operation': operation,
                    'init': {**shard_init_kwargs, 'host_list': chunk},
                    **operation_args,
                }
            )
        return payloads

    def execute_sharded(self, payloads):
        """Execute payloads across worker processes."""
        if not payloads:
            return []

        ctx = mp.get_context('spawn')
        max_workers = min(len(payloads), self.config.max_workers)
        results = [None] * len(payloads)

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = {ex.submit(PsshSharder.run_shard, p): i for i, p in enumerate(payloads)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()

        return results

    def merge_results(self, shard_returns, original_host_list):
        """Merge results from all shards maintaining original host order."""
        # Step 1: Build a single lookup map
        merged = {}
        for shard in shard_returns:
            result = shard.get('result') or {}  # treat None as {}
            merged.update(result)

        # Step 2: Preserve original order
        cmd_output = {}
        for host in original_host_list:
            if host in merged:
                cmd_output[host] = merged[host]

        return cmd_output

    @staticmethod
    def run_shard(payload):
        """
        Run an SSH operation on a shard of hosts (must be picklable for multiprocessing).

        Dynamically supports all abstract methods defined in ShardableSshInterface.

        Args:
            payload: Dict with 'operation' (operation type), 'init' (SSH setup), and operation args

        Returns:
            Dict with operation result and host reachability status
        """
        operation = payload['operation']
        if not isinstance(operation, str):
            raise TypeError('payload["operation"] must be str, got %r' % (type(operation),))

        # Validate operation is supported by ABC
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f'Unknown operation: {operation}. Supported: {sorted(SUPPORTED_OPERATIONS)}')

        # Create SSH client for this shard of hosts
        init_kwargs = payload['init']
        init_kwargs['process_output'] = False  # Force raw output mode for sharding
        shard = Pssh(**init_kwargs)

        try:
            # Ensure method exists in Pssh
            if not hasattr(shard, operation):
                raise RuntimeError(f'Method {operation} not found in Pssh class')

            shard_method = getattr(shard, operation)

            # Extract operation arguments from payload (excluding 'operation' and 'init')
            args = {k: v for k, v in payload.items() if k not in ['operation', 'init']}

            # Add common parameters for operations that support them
            if operation in ['exec', 'exec_cmd_list']:
                args['print_console'] = False  # Always False for sharded operations

            # Call the method dynamically
            result = shard_method(**args)

            # Handle operations that should return None (void operations)
            if operation in ['upload_file', 'reboot_connections']:
                result = None

            return {
                'result': result,
                'reachable_hosts': list(shard.reachable_hosts),
                'unreachable_hosts': list(shard.unreachable_hosts),
            }
        finally:
            try:
                shard.destroy_clients()
            except Exception:
                pass
