'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
from abc import ABC, abstractmethod


def env_int(name):
    """The integer value of environment variable name; RuntimeError naming it when unset or not an integer."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"managed CVS run requires {name} to be set to an integer") from exc


class SchedulerBackend(ABC):
    """What CVS asks the platform it runs on: managed or not, job id, rank, hosts, launcher.

    cvs.core.scheduler.get_backend() returns the backend that matches detection.
    Job id, rank, world size and hosts are only meaningful inside a managed step,
    so callers check in_managed_step() first: a scheduler backend reads whatever
    its environment holds, and a backend that is never managed raises RuntimeError.
    """

    name = None

    @abstractmethod
    def in_managed_step(self):
        """True if the scheduler itself launched this process as one rank of a job step."""

    @abstractmethod
    def job_id(self):
        """The scheduler's id for this job, shared by every rank."""

    @abstractmethod
    def rank(self):
        """This process's rank, from 0 to world_size() - 1."""

    @abstractmethod
    def world_size(self):
        """The number of ranks in this job step."""

    @abstractmethod
    def hosts(self):
        """Every host in the job, in rank order."""

    @abstractmethod
    def mpi_launcher(self):
        """How MPI ranks are launched: 'srun' for a scheduler job step, 'mpirun-ssh' for mpirun over SSH."""

    @abstractmethod
    def node_name(self):
        """The scheduler's name for this node, or None when it gives none."""
