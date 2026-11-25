'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os

from cvs.core.schedulers.base import SchedulerBackend, env_int

JOB_STEP_VARS = ("CVS_K8S_JOB_ID", "JOB_COMPLETION_INDEX")


class KubernetesBackend(SchedulerBackend):
    """A pod of a CVS-generated Indexed Job.

    The Job controller sets JOB_COMPLETION_INDEX; the CVS manifest sets the CVS_K8S_*
    variables and CVS_WORLD_SIZE. Every value comes from the environment, so a pod
    needs no kubectl or API access.
    """

    name = 'kubernetes'

    def in_managed_step(self):
        return all(os.environ.get(var) is not None for var in JOB_STEP_VARS)

    def job_id(self):
        job_id = os.environ.get("CVS_K8S_JOB_ID")
        if job_id is None:
            raise RuntimeError("managed CVS run requires CVS_K8S_JOB_ID")
        return job_id

    def rank(self):
        rank = env_int("JOB_COMPLETION_INDEX")
        world_size = self.world_size()
        if not 0 <= rank < world_size:
            raise RuntimeError(f"invalid managed rank {rank} for world size {world_size}")
        return rank

    def world_size(self):
        return env_int("CVS_WORLD_SIZE")

    def hosts(self):
        """CVS_K8S_HOSTS: comma-separated pod hostnames in rank order."""
        hosts = [host for host in os.environ.get("CVS_K8S_HOSTS", "").split(",") if host]
        if not hosts:
            raise RuntimeError("managed Kubernetes run requires CVS_K8S_HOSTS")
        return hosts

    def mpi_launcher(self):
        return 'mpirun-ssh'

    def node_name(self):
        return os.environ.get("CVS_K8S_NODE_NAME") or None
