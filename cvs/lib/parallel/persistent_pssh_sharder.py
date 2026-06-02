'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import inspect
import math
import multiprocessing as mp
import time
import uuid
from queue import Empty

from cvs.lib.parallel.pssh import Pssh
from cvs.lib.parallel.interfaces import SharderInterface, ShardableSshInterface, WorkerTable


SUPPORTED_OPERATIONS = {
    name for name, method in inspect.getmembers(ShardableSshInterface) if getattr(method, '__isabstractmethod__', False)
}


def _build_pssh(init_kwargs):
    """Create a shard-local Pssh instance with raw output mode."""
    kwargs = dict(init_kwargs)
    kwargs['process_output'] = False
    return Pssh(**kwargs)


def _persistent_worker_main(init_payload, req_q, resp_q):
    """
    Worker process loop.

    Keeps a long-lived Pssh instance alive across requests to preserve SSH session reuse.
    """
    try:
        shard = _build_pssh(init_payload)
        resp_q.put({'type': 'init', 'ok': True, 'error': None})
    except Exception as exc:
        resp_q.put({'type': 'init', 'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
        return
    running = True

    while running:
        message = req_q.get()
        msg_type = message.get('type')

        if msg_type == 'shutdown':
            running = False
            continue

        if msg_type != 'request':
            continue

        request_id = message['request_id']
        operation = message['operation']
        args = dict(message.get('args', {}))

        try:
            if operation not in SUPPORTED_OPERATIONS:
                raise ValueError(f'Unknown operation: {operation}. Supported: {sorted(SUPPORTED_OPERATIONS)}')

            shard_method = getattr(shard, operation)
            if operation in ['exec', 'exec_cmd_list']:
                args['print_console'] = False

            result = shard_method(**args)
            if operation in ['upload_file', 'reboot_connections']:
                result = None

            resp_q.put(
                {
                    'request_id': request_id,
                    'ok': True,
                    'result': result,
                    'reachable_hosts': list(shard.reachable_hosts),
                    'unreachable_hosts': list(shard.unreachable_hosts),
                    'error': None,
                }
            )
        except Exception as exc:
            resp_q.put(
                {
                    'request_id': request_id,
                    'ok': False,
                    'result': {},
                    'reachable_hosts': list(getattr(shard, 'reachable_hosts', [])),
                    'unreachable_hosts': list(getattr(shard, 'unreachable_hosts', [])),
                    'error': f'{type(exc).__name__}: {exc}',
                }
            )

    try:
        shard.destroy_clients()
    except Exception:
        pass


class PersistentPsshSharder(SharderInterface):
    """Persistent-process sharder with long-lived Pssh per shard worker."""

    _RESPONSE_TIMEOUT_SEC = 60  # Default timeout when no operation timeout provided
    _RESPONSE_TIMEOUT_BUFFER_SEC = 30  # Buffer added to operation timeouts for IPC/network overhead
    _INIT_TIMEOUT_SEC = 10

    def __init__(self, config):
        self.config = config
        self.ctx = mp.get_context('spawn')
        self._workers = {}  # worker_id → physical_worker (changed from list to dict)
        self._worker_state_table = WorkerTable()

    def initialize_workers(self, hosts, shard_init_kwargs=None):
        """
        Initialize persistent workers once from the provided hosts list.

        Args:
            hosts: Initial host inventory for persistent workers
            shard_init_kwargs: Base kwargs for worker Pssh initialization
        """
        if self._workers:
            return
        shard_init_kwargs = shard_init_kwargs or {}
        host_chunks = list(self.chunk_hosts(hosts))
        for idx, chunk in enumerate(host_chunks):
            worker_id = idx  # Use index as worker_id for initial assignment
            init_payload = {**shard_init_kwargs, 'host_list': list(chunk)}
            physical_worker = self._start_worker(init_payload)
            physical_worker['worker_id'] = worker_id  # Tag with ID
            self._workers[worker_id] = physical_worker
            self._worker_state_table.append(worker_id, chunk, chunk, ())

    def chunk_hosts(self, host_list):
        """Divide hosts into processing shards, respecting max_workers limit."""
        total_hosts = len(host_list)
        base_chunk_size = self.config.hosts_per_shard

        # Calculate natural number of workers needed
        natural_workers = math.ceil(total_hosts / base_chunk_size)

        if natural_workers <= self.config.max_workers:
            # Use normal chunking - we're within limits
            chunk_size = base_chunk_size
        else:
            # Too many workers - increase chunk size to fit within max_workers
            chunk_size = math.ceil(total_hosts / self.config.max_workers)

        # Generate chunks with the calculated size
        for i in range(0, total_hosts, chunk_size):
            yield host_list[i : i + chunk_size]

    def create_payloads(self, operation, shard_init_kwargs, **operation_args):
        """Create payloads with worker routing based on current active workers.

        Returns:
            Dict[worker_id, payload] - Maps each active worker_id to its payload
        """
        routing_map = {}

        # Internal logic: get active workers from worker table
        for worker_state in self._worker_state_table:
            # Only create payloads for workers with reachable hosts
            if not worker_state.reachable_hosts:
                continue

            # Create payload for this specific worker
            payload = {
                'operation': operation,
                'init': {**shard_init_kwargs, 'host_list': list(worker_state.reachable_hosts)},
                **operation_args,
            }

            # Map worker_id to its payload
            routing_map[worker_state.worker_id] = payload

        return routing_map

    def execute_sharded(self, routing_map):
        """Execute payloads with explicit worker_id → payload routing.

        Args:
            routing_map: Dict[worker_id, payload] mapping worker IDs to their payloads

        Returns:
            List of shard results in same order as worker_ids
        """
        if not routing_map:
            return []

        # Ensure all target workers exist and are healthy
        self._ensure_workers(routing_map)

        # Send requests to workers
        request_tracking = []  # List of (request_id, worker_id) for result collection

        for worker_id, payload in routing_map.items():
            # Validate worker exists
            if worker_id not in self._workers:
                raise RuntimeError(f"Worker {worker_id} not found in active workers")

            physical_worker = self._workers[worker_id]
            request_id = uuid.uuid4().hex
            request_tracking.append((request_id, worker_id))

            # Extract operation arguments
            args = {k: v for k, v in payload.items() if k not in ['operation', 'init']}

            # Send request to specific worker
            physical_worker['req_q'].put(
                {
                    'type': 'request',
                    'request_id': request_id,
                    'operation': payload['operation'],
                    'args': args,
                }
            )

        # Collect results in consistent order (by worker_id)
        results = []
        for request_id, worker_id in request_tracking:
            physical_worker = self._workers[worker_id]
            # Extract operation timeout from routing_map for this worker
            operation_timeout = routing_map[worker_id].get('timeout')
            response = self._wait_for_response(physical_worker['resp_q'], request_id, operation_timeout)

            if response['ok']:
                results.append(
                    {
                        'result': response.get('result') or {},
                        'reachable_hosts': response.get('reachable_hosts', []),
                        'unreachable_hosts': response.get('unreachable_hosts', []),
                    }
                )
            else:
                # Handle error case
                host_list = self._get_hosts_for_worker_id(worker_id)
                error_msg = response.get('error', 'Unknown worker error')
                results.append(
                    {
                        'result': {host: f'ERROR: {error_msg}' for host in host_list},
                        'reachable_hosts': response.get('reachable_hosts', []),
                        'unreachable_hosts': response.get('unreachable_hosts', host_list),
                    }
                )

        return results

    def get_worker_table(self):
        """Return persistent worker-state iterator."""
        return iter(self._worker_state_table)

    def update_worker_table(self, shard_returns, merge_unreachable=True):
        """Update persistent worker-state from shard execution results."""
        active_workers = WorkerTable()
        for idx, previous in enumerate(self._worker_state_table.list()):
            if idx >= len(shard_returns):
                continue
            shard_ret = shard_returns[idx]
            reachable_hosts = list(shard_ret.get('reachable_hosts', []))
            # Active workers are workers with at least one reachable host.
            if not reachable_hosts:
                continue
            active_workers.append(
                previous.worker_id,
                previous.host_list,
                reachable_hosts,
                shard_ret.get('unreachable_hosts', []) if merge_unreachable else (),
            )
        self._worker_state_table = active_workers

    def prune_worker_nodes(self, remove_set):
        """Prune hosts from persistent worker-state table."""
        active_workers = WorkerTable()
        workers_to_remove = []

        for row in self._worker_state_table.list():
            pruned_reachable = [h for h in row.reachable_hosts if h not in remove_set]

            # If no reachable hosts, mark worker for removal
            if not pruned_reachable:
                workers_to_remove.append(row.worker_id)
                continue

            # Keep worker with updated host list
            active_workers.append(
                row.worker_id,  # Keep same worker_id!
                [h for h in row.host_list if h not in remove_set],
                pruned_reachable,
                [h for h in row.unreachable_hosts if h not in remove_set],
            )

        # Remove dead workers from physical storage
        for worker_id in workers_to_remove:
            if worker_id in self._workers:
                # Shutdown the worker
                worker = self._workers[worker_id]
                try:
                    worker['req_q'].put({'type': 'shutdown'})
                    worker['process'].terminate()
                except Exception:
                    pass
                del self._workers[worker_id]

        self._worker_state_table = active_workers

    def _ensure_workers(self, routing_map):
        """Ensure all workers exist and are healthy - restart/create as needed."""
        for worker_id, payload in routing_map.items():
            host_list = payload['init']['host_list']

            # Create worker if missing or restart if dead
            if worker_id not in self._workers or not self._workers[worker_id]['process'].is_alive():
                # Cleanup old worker if exists
                if worker_id in self._workers:
                    try:
                        self._workers[worker_id]['process'].terminate()
                    except Exception:
                        pass

                # Start new worker with current host list
                physical_worker = self._start_worker({'host_list': host_list})
                physical_worker['worker_id'] = worker_id
                self._workers[worker_id] = physical_worker

    def _find_worker_state_by_id(self, worker_id):
        """Find WorkerState by worker_id."""
        for worker_state in self._worker_state_table:
            if worker_state.worker_id == worker_id:
                return worker_state
        return None

    def _get_hosts_for_worker_id(self, worker_id):
        """Get host list for a specific worker_id."""
        worker_state = self._find_worker_state_by_id(worker_id)
        return list(worker_state.host_list) if worker_state else []

    def _start_worker(self, init_payload):
        req_q = self.ctx.Queue()
        resp_q = self.ctx.Queue()
        process = self.ctx.Process(
            target=_persistent_worker_main,
            args=(init_payload, req_q, resp_q),
            daemon=True,
        )
        process.start()

        try:
            init_status = resp_q.get(timeout=self._INIT_TIMEOUT_SEC)
        except Empty:
            init_status = {'type': 'init', 'ok': False, 'error': 'Timed out waiting for worker init'}

        if init_status.get('type') != 'init' or not init_status.get('ok'):
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            error = init_status.get('error', 'Unknown worker init error')
            raise RuntimeError(f'Persistent shard worker init failed: {error}')

        return {
            'process': process,
            'req_q': req_q,
            'resp_q': resp_q,
            'init': dict(init_payload),
        }

    def _wait_for_response(self, resp_q, request_id, operation_timeout=None):
        """Wait for a specific response id from a worker response queue."""
        if operation_timeout is not None:
            # Add buffer to operation timeout to account for IPC and network overhead
            timeout_seconds = operation_timeout + self._RESPONSE_TIMEOUT_BUFFER_SEC
        else:
            # Use default pssh timeout + buffer when no operation timeout provided
            # pssh defaults to ~60s, so we need 60 + buffer for the same reason
            timeout_seconds = self._RESPONSE_TIMEOUT_SEC + self._RESPONSE_TIMEOUT_BUFFER_SEC
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                response = resp_q.get(timeout=0.5)
            except Empty:
                continue
            if response.get('request_id') == request_id:
                return response
        return {
            'request_id': request_id,
            'ok': False,
            'result': {},
            'reachable_hosts': [],
            'unreachable_hosts': [],
            'error': f'Timeout waiting for shard response (request_id={request_id})',
        }

    def destroy_clients(self):
        """Shutdown persistent shard workers."""
        for worker_id, worker in self._workers.items():
            try:
                worker['req_q'].put({'type': 'shutdown'})
            except Exception:
                pass

        for worker_id, worker in self._workers.items():
            proc = worker['process']
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)

        self._workers = {}
        self._worker_state_table.clear()


def _persistent_worker_main(init_payload, req_queue, resp_queue):
    """Main function for persistent shard worker process."""
    from cvs.lib.parallel.pssh import Pssh
    from cvs.lib import globals
    import uuid

    logger = globals.log
    pssh = None

    try:
        # Extract jump host parameters from init_payload
        jump_host_kwargs = {}
        for key in ['jump_host', 'jump_user', 'jump_password', 'jump_pkey', 'jump_port']:
            if key in init_payload:
                jump_host_kwargs[key] = init_payload[key]

        # Create Pssh instance with jump host support
        pssh = Pssh(
            log=None,  # Use global logger in worker
            host_list=init_payload['host_list'],
            user=init_payload['user'],
            password=init_payload.get('password'),
            pkey=init_payload.get('pkey', 'id_rsa'),
            host_key_check=init_payload.get('host_key_check', False),
            stop_on_errors=init_payload.get('stop_on_errors', True),
            env_vars=init_payload.get('env_vars'),
            **jump_host_kwargs,  # Jump host support
        )

        # Send initialization success
        resp_queue.put({'type': 'init', 'ok': True})
        logger.info(f"Persistent worker initialized with {len(init_payload['host_list'])} hosts")

        # Main worker loop
        while True:
            try:
                request = req_queue.get(timeout=1)
            except:
                continue

            if request.get('type') == 'shutdown':
                logger.info("Persistent worker shutting down")
                break

            # Handle operation requests
            request_id = request.get('request_id', str(uuid.uuid4()))
            operation = request.get('operation')

            try:
                if operation == 'exec':
                    result = pssh.exec(
                        request['cmd'],
                        timeout=request.get('timeout'),
                        print_console=request.get('print_console', True),
                        detailed=request.get('detailed', False),
                    )
                elif operation == 'exec_cmd_list':
                    result = pssh.exec_cmd_list(
                        request['cmd_list'],
                        timeout=request.get('timeout'),
                        print_console=request.get('print_console', True),
                    )
                elif operation == 'upload_file':
                    result = pssh.upload_file(
                        request['local_file'], request['remote_file'], recurse=request.get('recurse', False)
                    )
                elif operation == 'upload_file_list':
                    result = pssh.upload_file_list(request['node_path_map'])
                elif operation == 'reboot_connections':
                    result = pssh.reboot_connections()
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                # Send successful response
                resp_queue.put(
                    {
                        'request_id': request_id,
                        'ok': True,
                        'result': result,
                        'reachable_hosts': pssh.reachable_hosts,
                        'unreachable_hosts': pssh.unreachable_hosts,
                    }
                )

            except Exception as e:
                logger.error(f"Persistent worker operation failed: {e}")
                resp_queue.put(
                    {
                        'request_id': request_id,
                        'ok': False,
                        'result': {},
                        'reachable_hosts': getattr(pssh, 'reachable_hosts', []),
                        'unreachable_hosts': getattr(pssh, 'unreachable_hosts', []),
                        'error': str(e),
                    }
                )

    except Exception as e:
        logger.error(f"Persistent worker init failed: {e}")
        resp_queue.put({'type': 'init', 'ok': False, 'error': str(e)})
    finally:
        if pssh:
            try:
                pssh.destroy_clients()
            except:
                pass
