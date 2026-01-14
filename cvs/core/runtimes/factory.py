'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from .docker import DockerRuntime
from .enroot import EnrootRuntime


class RuntimeFactory:
    """Factory for creating container runtime instances."""

    @staticmethod
    def create(runtime_name, log, orchestrator):
        """Create a container runtime instance."""
        runtime_name = runtime_name.lower()

        if runtime_name == 'docker':
            return DockerRuntime(log, orchestrator)
        elif runtime_name == 'enroot':
            return EnrootRuntime(log, orchestrator)
        else:
            raise ValueError(f"Unsupported container runtime: {runtime_name}")
