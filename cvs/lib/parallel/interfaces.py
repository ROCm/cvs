'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerState:
    """Immutable worker row used by sharder worker tables."""

    worker_id: int
    host_list: tuple
    reachable_hosts: tuple
    unreachable_hosts: tuple


class WorkerTable:
    """Minimal ordered table of WorkerState rows."""

    def __init__(self):
        self._rows = []

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def list(self):
        return list(self._rows)

    def clear(self):
        self._rows.clear()

    def _build_row(self, worker_id, host_list=None, reachable_hosts=None, unreachable_hosts=None):
        if host_list is None:
            raise ValueError('host_list is required')
        if reachable_hosts is None:
            reachable_hosts = host_list
        if unreachable_hosts is None:
            unreachable_hosts = ()
        return WorkerState(
            worker_id=worker_id,
            host_list=tuple(host_list),
            reachable_hosts=tuple(reachable_hosts),
            unreachable_hosts=tuple(unreachable_hosts),
        )

    def append(self, worker_id, host_list=None, reachable_hosts=None, unreachable_hosts=None):
        row = self._build_row(worker_id, host_list, reachable_hosts, unreachable_hosts)
        self._rows.append(row)

    def update(self, index, worker_id, host_list=None, reachable_hosts=None, unreachable_hosts=None):
        row = self._build_row(worker_id, host_list, reachable_hosts, unreachable_hosts)
        self._rows[index] = row


class SharderInterface(ABC):
    """Abstract contract for sharder strategy implementations.

    This interface allows MultiProcessPssh to compose either transient
    (one-shot worker processes) or persistent (long-lived worker processes)
    sharding implementations behind a common API.
    """

    @abstractmethod
    def chunk_hosts(self, host_list):
        """Split host list into shard-sized chunks."""
        pass

    @abstractmethod
    def create_payloads(self, operation, shard_init_kwargs, **operation_args):
        """Create shard payloads for a given operation.

        Returns:
            Dict[worker_id, payload] - Maps worker IDs to their payloads

        This method examines the current worker state table, creates payloads
        for active workers (those with reachable hosts), and returns an
        explicit routing map.
        """
        pass

    @abstractmethod
    def execute_sharded(self, routing_map):
        """Execute payloads with explicit worker_id → payload routing.

        Args:
            routing_map: Dict[worker_id, payload] mapping worker IDs to their payloads

        Returns:
            List of shard results
        """
        pass

    def initialize_workers(self, hosts, shard_init_kwargs=None):
        """Optional initialization hook for sharder worker table."""
        return None

    def get_worker_table(self):
        """Return normalized worker-table iterator."""
        return iter(())

    def update_worker_table(self, shard_returns, merge_unreachable=True):
        """Update worker table from execution results."""
        return None

    def prune_worker_nodes(self, remove_set):
        """Optional hook to prune nodes from worker state."""
        return None

    def destroy_clients(self):
        """Optional lifecycle cleanup hook (no-op by default)."""
        return None


class ShardableSshInterface(ABC):
    """Abstract base class defining operations that MUST support sharding.

    Any class implementing this interface must provide sharded implementations
    for all these methods. This prevents silent performance degradation where
    operations fall back to non-sharded implementations.
    """

    @abstractmethod
    def exec(self, cmd, timeout=None, print_console=True, detailed=False):
        """Execute command - must support sharding for performance."""
        pass

    @abstractmethod
    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """Execute command list - must support sharding for performance."""
        pass

    @abstractmethod
    def upload_file(self, local_file, remote_file, recurse=False):
        """Upload file via SFTP - must support sharding for performance."""
        pass

    @abstractmethod
    def upload_file_list(self, node_path_map):
        """Upload different files to different hosts - must support sharding for performance."""
        pass

    @abstractmethod
    def download_file(self, remote_file, local_file, recurse=False, suffix_separator='_'):
        """Download file via SFTP - must support sharding for performance."""
        pass

    @abstractmethod
    def reboot_connections(self):
        """Reboot connections - must support sharding for performance."""
        pass
