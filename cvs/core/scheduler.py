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


def is_managed_compute() -> bool:
    """True if compute nodes are managed by a scheduler (SPUR/SLURM), False for baremetal SSH."""
    return detect_scheduler() != Scheduler.BARE_METAL
