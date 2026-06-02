'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os

from cvs.lib.env_vars import get


class ParallelConfig:
    """Configuration for parallel SSH operations."""

    def __init__(
        self,
        hosts_per_shard=32,
        max_workers_per_cpu=4,
        persistent_shards=False,
        # Jump host parameters
        jump_host=None,
        jump_user=None,
        jump_password=None,
        jump_pkey=None,
        jump_port=22,
    ):
        self.hosts_per_shard = hosts_per_shard
        self.max_workers_per_cpu = max_workers_per_cpu
        self.persistent_shards = persistent_shards

        # Jump host configuration
        self.jump_host = jump_host
        self.jump_user = jump_user
        self.jump_password = jump_password
        self.jump_pkey = jump_pkey
        self.jump_port = jump_port

    @property
    def max_workers(self):
        """Calculate maximum worker processes based on CPU count."""
        cpu_count = os.cpu_count() or 4
        return max(self.hosts_per_shard, cpu_count * self.max_workers_per_cpu)

    @property
    def uses_jump_host(self):
        """True if jump host is configured."""
        return self.jump_host is not None

    @classmethod
    def from_env(cls):
        """Create configuration from environment variables (see cvs.lib.env_vars)."""
        return cls(
            hosts_per_shard=get('CVS_HOSTS_PER_SHARD'),
            max_workers_per_cpu=get('CVS_WORKERS_PER_CPU'),
            persistent_shards=get('CVS_PERSISTENT_SHARDS'),
            # Jump host configuration from environment
            jump_host=get('CVS_JUMP_HOST'),
            jump_user=get('CVS_JUMP_USER'),
            jump_password=get('CVS_JUMP_PASSWORD'),
            jump_pkey=get('CVS_JUMP_PKEY'),
            jump_port=get('CVS_JUMP_PORT'),
        )
