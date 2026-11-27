'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.schedulers.base import SchedulerBackend


def _no_scheduler(value):
    return RuntimeError(f"a bare-metal run has no scheduler {value}; it is only defined inside a managed run")


class BareMetalBackend(SchedulerBackend):
    """Nodes reached over SSH from the cluster file. Never managed, whatever SLURM_* says."""

    name = 'bare_metal'

    def in_managed_step(self):
        return False

    def job_id(self):
        raise _no_scheduler("job id")

    def rank(self):
        raise _no_scheduler("rank")

    def world_size(self):
        raise _no_scheduler("world size")

    def hosts(self):
        raise _no_scheduler("host list")

    def mpi_launcher(self):
        return 'mpirun-ssh'

    def node_name(self):
        return None
