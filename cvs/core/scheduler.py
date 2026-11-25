'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import shutil
import subprocess
from enum import Enum
import os

from cvs.core.schedulers.bare_metal import BareMetalBackend
from cvs.core.schedulers.kubernetes import KubernetesBackend
from cvs.core.schedulers.slurm import SlurmBackend
from cvs.core.schedulers.spur import SpurBackend

CHECK_TIMEOUT = 5
SCHEDULER_ENV_VAR = "CVS_SCHEDULER"


class Scheduler(Enum):
    SPUR = 'spur'
    SLURM = 'slurm'
    BARE_METAL = 'bare_metal'
    KUBERNETES = 'kubernetes'


# SPUR must be checked before SLURM or a spur cluster gets misclassified.
SCHEDULER_CHECK_COMMANDS = {
    Scheduler.SPUR: [["spur", "version"]],
    Scheduler.SLURM: [["scontrol", "version"]],
}

_BACKENDS = {
    Scheduler.SPUR: SpurBackend,
    Scheduler.SLURM: SlurmBackend,
    Scheduler.BARE_METAL: BareMetalBackend,
    Scheduler.KUBERNETES: KubernetesBackend,
}


def _command_succeeds(cmd):
    if shutil.which(cmd[0]) is None:
        return False
    try:
        result = subprocess.run(cmd, timeout=CHECK_TIMEOUT, capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _scheduler_from_job_env():
    """Classify from job-id env vars, which are set inside a step even when
    spur/scontrol are not on PATH (compute nodes often have neither).

    Only CVS-generated pods set CVS_K8S_JOB_ID, so it goes first: Slurm running
    inside Kubernetes pods sets no CVS_K8S_JOB_ID and is still classified as Slurm.
    SPUR exports SPUR_* alongside SLURM_* twins; real Slurm sets no SPUR_*.
    Check SPUR_JOB_ID next so a spur step is not labeled SLURM.
    """
    if os.environ.get("CVS_K8S_JOB_ID"):
        return Scheduler.KUBERNETES
    if os.environ.get("SPUR_JOB_ID"):
        return Scheduler.SPUR
    if os.environ.get("SLURM_JOB_ID"):
        return Scheduler.SLURM
    return None


def detect_scheduler():
    """Detect which scheduler, if any, manages this cluster's compute nodes.

    Order: CVS_SCHEDULER override, then job-id env (inside a step), then the
    spur/scontrol binary probe (head node before submission).
    """
    scheduler = os.environ.get(SCHEDULER_ENV_VAR)
    if scheduler is not None:
        normalized = scheduler.strip().lower()
        try:
            return Scheduler(normalized)
        except ValueError as exc:
            valid = [s.value for s in Scheduler]
            raise ValueError(
                f"Unknown scheduler type {scheduler!r} in {SCHEDULER_ENV_VAR}, expected one of: {valid}"
            ) from exc
    from_job = _scheduler_from_job_env()
    if from_job is not None:
        return from_job
    for scheduler, cmds in SCHEDULER_CHECK_COMMANDS.items():
        if all(_command_succeeds(cmd) for cmd in cmds):
            return scheduler
    return Scheduler.BARE_METAL


def get_backend():
    """The backend for the scheduler detect_scheduler() finds, built fresh on each call."""
    return _BACKENDS[detect_scheduler()]()


def _running_in_job_step():
    """True if srun launched this process as part of a job step, or the Kubernetes Job
    controller started it as one indexed pod, whichever scheduler is detected.

    is_managed_compute() is the check callers use: it asks only the detected backend.
    """
    return SlurmBackend().in_managed_step() or KubernetesBackend().in_managed_step()


def is_managed_compute():
    """True if this process is running inside a scheduler-launched job step, False otherwise."""
    return get_backend().in_managed_step()


def scheduler_hosts():
    """Every host in the current managed job, in scheduler order."""
    return get_backend().hosts()


def scheduler_rank():
    """This process's rank and the job's world size."""
    backend = get_backend()
    return backend.rank(), backend.world_size()
