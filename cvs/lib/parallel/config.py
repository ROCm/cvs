'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os


class ParallelConfig:
    """Configuration for parallel SSH operations."""

    def __init__(self, hosts_per_shard=32, max_workers_per_cpu=4):
        self.hosts_per_shard = hosts_per_shard
        self.max_workers_per_cpu = max_workers_per_cpu

    @property
    def max_workers(self):
        """Calculate maximum worker processes based on CPU count."""
        cpu_count = os.cpu_count() or 4
        return max(self.hosts_per_shard, cpu_count * self.max_workers_per_cpu)

    @classmethod
    def from_env(cls):
        """Create configuration from environment variables."""
        return cls(
            hosts_per_shard=int(os.environ.get('CVS_HOSTS_PER_SHARD', 32)),
            max_workers_per_cpu=int(os.environ.get('CVS_WORKERS_PER_CPU', 4)),
        )
