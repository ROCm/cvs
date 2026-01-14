'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from typing import Protocol


class ContainerRuntime(Protocol):
    """Protocol for container runtime implementations."""

    def setup_containers(self, container_config, container_name):
        """Set up containers on all nodes."""
        ...

    def teardown_containers(self, container_name):
        """Tear down containers on all nodes."""
        ...

    def exec(self, container_name, cmd, hosts=None, timeout=None):
        """Execute command in running containers."""
        ...

    def exec_on_head(self, container_name, cmd, timeout=None):
        """Execute command directly on head node (baremetal)."""
        ...

    def load_image(self, tar_path, timeout=None):
        """Load container image from tar file on all hosts."""
        ...
