"""
Base classes and utilities for preflight testing modules.
"""

from abc import ABC, abstractmethod
from cvs.lib import globals

log = globals.log


class PreflightCheck(ABC):
    """
    Abstract base class for all preflight checks.

    Each preflight check should inherit from this class and implement the run() method.
    This provides a consistent interface for all preflight operations.
    """

    def __init__(self, phdl, config_dict=None):
        """
        Initialize the preflight check.

        Args:
            phdl: Parallel SSH handle for cluster nodes
            config_dict: Optional configuration dictionary
        """
        self.phdl = phdl
        self.config_dict = config_dict or {}
        self.results = {}

    @abstractmethod
    def run(self):
        """
        Execute the preflight check.

        Returns:
            dict: Results of the preflight check
        """
        pass

    def get_results(self):
        """Get the results of the last run."""
        return self.results

    def log_info(self, message):
        """Log an info message."""
        log.info(f"[{self.__class__.__name__}] {message}")

    def log_error(self, message):
        """Log an error message."""
        log.error(f"[{self.__class__.__name__}] {message}")

    def log_warning(self, message):
        """Log a warning message."""
        log.warning(f"[{self.__class__.__name__}] {message}")

    @staticmethod
    def partition_nodes_into_groups(node_list, group_size):
        """
        Partition nodes into groups based on configurable group_size.

        Args:
            node_list: List of all cluster nodes
            group_size: Configured group size from preflight config

        Returns:
            dict: {group_id: [node_list]} mapping
        """
        import math

        groups = {}
        num_groups = math.ceil(len(node_list) / group_size)

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min(start_idx + group_size, len(node_list))
            groups[f"group_{i + 1}"] = node_list[start_idx:end_idx]

        return groups

    @staticmethod
    def calculate_resource_requirements(group_size, num_interfaces=8):
        """
        Calculate resource requirements for connectivity testing.

        Args:
            group_size: Number of nodes per group
            num_interfaces: Number of interfaces per node (default: 8)

        Returns:
            dict: Resource requirement calculations
        """
        intra_group_fds = (group_size - 1) * (num_interfaces**2)
        inter_group_fds = group_size * (num_interfaces**2)

        return {
            'intra_group_fds_per_node': intra_group_fds,
            'inter_group_fds_per_node': inter_group_fds,
            'max_fds_per_node': inter_group_fds,
            'script_size_kb': (inter_group_fds * 85) // 1024,  # ~85 chars per command
        }

    @staticmethod
    def find_host_group(host, groups):
        """
        Find which group a host belongs to.

        Args:
            host: Hostname to find
            groups: Dictionary of group_id -> [nodes]

        Returns:
            str: Group ID that contains the host, or None if not found
        """
        for group_id, group_nodes in groups.items():
            if host in group_nodes:
                return group_id
        return None


# Module-level aliases for imports: cvs.lib.preflight.base.partition_nodes_into_groups
partition_nodes_into_groups = PreflightCheck.partition_nodes_into_groups
calculate_resource_requirements = PreflightCheck.calculate_resource_requirements
find_host_group = PreflightCheck.find_host_group
