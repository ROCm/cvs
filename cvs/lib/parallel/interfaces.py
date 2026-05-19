'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from abc import ABC, abstractmethod


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
