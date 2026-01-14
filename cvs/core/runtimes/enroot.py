'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''


class EnrootRuntime:
    """Enroot container runtime implementation."""

    def __init__(self, log, orchestrator):
        self.log = log
        self.orchestrator = orchestrator

    def setup_containers(self, container_config, container_name):
        """Set up Enroot containers - not yet implemented."""
        self.log.warning("Enroot runtime not yet implemented")
        return False

    def teardown_containers(self, container_name):
        """Tear down Enroot containers - not yet implemented."""
        self.log.warning("Enroot runtime not yet implemented")
        return True

    def exec(self, container_name, cmd, hosts=None, timeout=None):
        """Execute in Enroot containers - not yet implemented."""
        self.log.error("Enroot runtime not yet implemented")
        return {}

    def exec_on_head(self, container_name, cmd, timeout=None):
        """Execute on head in Enroot containers - not yet implemented."""
        self.log.error("Enroot runtime not yet implemented")
        return {}
