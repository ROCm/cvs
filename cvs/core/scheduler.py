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

CHECK_TIMEOUT = 5
SCHEDULER_ENV_VAR = "CVS_SCHEDULER"


class Scheduler(Enum):
    SPUR = 'spur'
    SLURM = 'slurm'
    BARE_METAL = 'bare_metal'


# SPUR must be checked before SLURM or a spur cluster gets misclassified.
SCHEDULER_CHECK_COMMANDS = {
    Scheduler.SPUR: [["spur", "version"]],
    Scheduler.SLURM: [["scontrol", "version"]],
}


def _command_succeeds(cmd: list[str]) -> bool:
    if shutil.which(cmd[0]) is None:
        return False
    try:
        result = subprocess.run(cmd, timeout=CHECK_TIMEOUT, capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def detect_scheduler() -> Scheduler:
    """Detect which scheduler, if any, manages this cluster's compute nodes."""
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
    for scheduler, cmds in SCHEDULER_CHECK_COMMANDS.items():
        if all(_command_succeeds(cmd) for cmd in cmds):
            return scheduler
    return Scheduler.BARE_METAL


def _running_in_job_step() -> bool:
    """True if this process was itself launched by srun as part of a job step.

    scontrol/spur version succeeding only means scheduler tooling is installed
    and the controller is reachable from wherever this process happens to run
    (e.g. a plain SSH login to a scheduler-managed head node) - it says nothing
    about whether this invocation is actually a step srun launched. SLURM_JOB_ID
    alone is also insufficient: salloc sets it for a bare allocation with no
    step. SLURM_STEP_ID/SLURM_PROCID are only set once srun has launched a step,
    which is what "managed compute" actually needs to gate on. Verified
    identical on SPUR (SLURM_JOB_ID/SLURM_STEP_ID/SLURM_PROCID) and real SLURM.
    """
    return (
        os.environ.get("SLURM_JOB_ID") is not None
        and os.environ.get("SLURM_STEP_ID") is not None
        and os.environ.get("SLURM_PROCID") is not None
    )


def is_managed_compute() -> bool:
    """True if this process is running inside a scheduler-launched job step, False otherwise."""
    return detect_scheduler() != Scheduler.BARE_METAL and _running_in_job_step()
