'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.orchestrators.pssh import PsshOrchestrator
from cvs.core.orchestrators.docker import DockerOrchestrator


class OrchestratorFactory:
    """
    Factory for creating orchestrator instances based on configuration.

    Provides pluggable architecture for adding new backends without
    modifying existing test code.
    """

    @staticmethod
    def create_orchestrator(log, cluster_dict, stop_on_errors=False):
        """
        Create orchestrator instance from CVS cluster_dict.

        Args:
            log: Logger instance
            cluster_dict: CVS cluster configuration from cluster.json
            stop_on_errors: Whether to stop execution on first error

        Returns:
            Orchestrator instance

        Raises:
            ValueError: If orchestrator type is unsupported
        """
        orchestrator_type = cluster_dict.get('orchestrator', 'pssh').lower()

        if orchestrator_type == 'pssh':
            return PsshOrchestrator(log, cluster_dict, stop_on_errors)
        elif orchestrator_type == 'docker':
            return DockerOrchestrator(log, cluster_dict, stop_on_errors)
        elif orchestrator_type == 'slurm':
            raise NotImplementedError("SlurmOrchestrator not yet implemented")
        elif orchestrator_type == 'k8s':
            raise NotImplementedError("K8sOrchestrator not yet implemented")
        else:
            raise ValueError(f"Unsupported orchestrator type: {orchestrator_type}")

    @staticmethod
    def get_supported_backends():
        """
        Get list of supported orchestrator backends.

        Returns:
            List of backend names
        """
        return ['pssh', 'docker', 'slurm', 'k8s']
