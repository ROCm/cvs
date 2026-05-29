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
from cvs.lib.parallel.interfaces import ShardableSshInterface, SharderInterface, WorkerTable


# Dynamically discover supported operations from ABC (computed once at import time)

SUPPORTED_OPERATIONS = {
    name for name, method in inspect.getmembers(ShardableSshInterface) if getattr(method, '__isabstractmethod__', False)
}


class PsshSharder(SharderInterface):
    """Shards SSH operations across multiple processes for large host lists."""

    def __init__(self, config):
        self.config = config
        self._worker_state_table = WorkerTable()
        self._all_hosts = []

    def chunk_hosts(self, host_list):
        """Divide hosts into processing shards."""
        chunk_size = self.config.hosts_per_shard
        for i in range(0, len(host_list), chunk_size):
            yield host_list[i : i + chunk_size]

    def create_payloads(self, operation, shard_init_kwargs, **operation_args):
        """Create payloads with worker routing for transient workers."""
        routing_map = {}

        # Internal logic: get active workers from worker table
        for worker_state in self._worker_state_table:
            if not worker_state.reachable_hosts:
                continue

            payload = {
                'operation': operation,
                'init': {**shard_init_kwargs, 'host_list': list(worker_state.reachable_hosts)},
                **operation_args,
            }

            routing_map[worker_state.worker_id] = payload

        return routing_map

    def execute_sharded(self, routing_map):
        """Execute payloads with explicit worker_id → payload routing."""
        if not routing_map:
            return []

        ctx = mp.get_context('spawn')
        payloads = list(routing_map.values())
        max_workers = min(len(payloads), self.config.max_workers)
        results = [None] * len(payloads)

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = {ex.submit(PsshSharder.run_shard, p): i for i, p in enumerate(payloads)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()

        return results

    def initialize_workers(self, hosts, shard_init_kwargs=None):
        """Build per-call transient worker table from current reachable hosts."""
        self._all_hosts = list(hosts)
        self._worker_state_table.clear()
        for idx, chunk in enumerate(self.chunk_hosts(hosts)):
            self._worker_state_table.append(idx, chunk, chunk, ())

    def get_worker_table(self):
        """Return transient worker-state iterator for current call."""
        return iter(self._worker_state_table)

    def update_worker_table(self, shard_returns, merge_unreachable=True):
        """
        Update transient worker-state snapshot and rebalance worker layout.

        Transient mode has per-call workers; after each operation we rebuild worker
        partitions from the latest reachable host set for the next operation.
        """
        reachable_set = set()
        for shard_ret in shard_returns:
            reachable_set.update(shard_ret.get('reachable_hosts', []))

        # Preserve deterministic host order based on initial inventory.
        reachable_hosts = [h for h in self._all_hosts if h in reachable_set]
        unreachable_hosts = [h for h in self._all_hosts if h not in reachable_set] if merge_unreachable else []

        self._worker_state_table.clear()
        for idx, chunk in enumerate(self.chunk_hosts(reachable_hosts)):
            # Keep explicit unreachable information in the worker table so
            # wrapper aggregation does not need a fallback path.
            self._worker_state_table.append(idx, chunk, chunk, unreachable_hosts)

    def prune_worker_nodes(self, remove_set):
        """Prune hosts from transient worker-state table."""
        for idx, row in enumerate(self._worker_state_table.list()):
            pruned_reachable = [h for h in row.reachable_hosts if h not in remove_set]
            pruned_unreachable = [h for h in row.unreachable_hosts if h not in remove_set]
            # In transient mode, host_list mirrors currently reachable hosts.
            self._worker_state_table.update(
                idx,
                row.worker_id,
                pruned_reachable,
                pruned_reachable,
                pruned_unreachable,
            )

    def destroy_clients(self):
        """Transient sharder has no persistent clients to clean up."""
        return None

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
