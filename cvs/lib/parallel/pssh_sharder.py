'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from cvs.lib.parallel.pssh import Pssh


# Operation registry - single source of truth for supported operations
# Operation names now match method names for consistency and simplicity
SUPPORTED_OPERATIONS = {
    'exec': {
        'required_params': ['cmd'],
        'optional_params': ['timeout', 'detailed'],
        'returns_result': True,
    },
    'exec_cmd_list': {
        'required_params': ['cmd_list'],
        'optional_params': ['timeout'],
        'returns_result': True,
    },
    'scp_file': {
        'required_params': ['local_file', 'remote_file'],
        'optional_params': ['recurse'],
        'returns_result': False,
    },
    'upload_file': {
        'required_params': ['local_file', 'remote_file'],
        'optional_params': ['recurse'],
        'returns_result': False,
    },
    'download_file': {
        'required_params': ['remote_file', 'local_file'],
        'optional_params': ['recurse', 'suffix_separator'],
        'returns_result': True,
    },
    'reboot_connections': {
        'required_params': [],
        'optional_params': [],
        'returns_result': False,
    },
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

        Args:
            payload: Dict with 'operation' (operation type), 'init' (SSH setup), and operation args

        Returns:
            Dict with operation result and host reachability status
        """
        operation = payload['operation']
        if not isinstance(operation, str):
            raise TypeError('payload["operation"] must be str, got %r' % (type(operation),))

        # Create SSH client for this shard of hosts
        init_kwargs = payload['init']
        init_kwargs['process_output'] = False  # Force raw output mode for sharding
        shard = Pssh(**init_kwargs)

        try:
            # Use operation registry for parameter validation and documentation
            op_config = SUPPORTED_OPERATIONS.get(operation)
            if not op_config:
                # Let the else clause in the operation handler catch unknown operations
                op_config = {'required_params': [], 'optional_params': [], 'returns_result': False}
            method_name = operation  # Operation name IS the method name!
            shard_method = getattr(shard, method_name)

            # Build arguments from payload based on operation configuration
            args = {}

            # Add required parameters
            for param in op_config['required_params']:
                if param not in payload:
                    raise ValueError(f"Missing required parameter '{param}' for operation '{operation}'")
                args[param] = payload[param]

            # Add optional parameters if present
            for param in op_config['optional_params']:
                if param in payload:
                    args[param] = payload[param]

            # Add common parameters for operations that support them
            if method_name in ['exec', 'exec_cmd_list']:
                args['print_console'] = False  # Always False for sharded operations

            # Execute the operation - handle special cases for method signatures
            if operation == 'exec':
                # Maintain existing call signature for backward compatibility
                kwargs = {
                    'timeout': args.get('timeout'),
                    'print_console': False,
                }
                if 'detailed' in args:
                    kwargs['detailed'] = args['detailed']
                result = shard_method(args['cmd'], **kwargs)
            elif operation == 'exec_cmd_list':
                # Maintain existing call signature for backward compatibility
                result = shard_method(args['cmd_list'], timeout=args.get('timeout'), print_console=False)
            elif operation in ['scp_file', 'upload_file']:
                # File upload operations
                shard_method(args['local_file'], args['remote_file'], recurse=args.get('recurse', False))
                result = None
            elif operation == 'download_file':
                # File download operation
                result = shard_method(
                    args['remote_file'],
                    args['local_file'],
                    recurse=args.get('recurse', False),
                    suffix_separator=args.get('suffix_separator', '_'),
                )
            elif operation == 'reboot_connections':
                # Reboot operation
                shard_method()
                result = None
            else:
                # Enhanced error handling to distinguish user vs implementation errors
                if operation in SUPPORTED_OPERATIONS:
                    # Case #2: Implementation Error - Operation in registry but no handler
                    raise RuntimeError(
                        f'Implementation bug: Operation \'{operation}\' is supported (in registry) '
                        f'but handler is missing in run_shard().'
                    )
                else:
                    # Case #1: User Error - Operation not supported at all
                    raise ValueError(
                        f'Unknown operation: {operation}. Supported operations: {list(SUPPORTED_OPERATIONS.keys())}'
                    )

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
