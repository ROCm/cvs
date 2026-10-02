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

    SPUR exports SPUR_* alongside SLURM_* twins; real Slurm sets no SPUR_*.
    Check SPUR_JOB_ID first so a spur step is not labeled SLURM.
    """
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


def _running_in_job_step():
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


def is_managed_compute():
    """True if this process is running inside a scheduler-launched job step, False otherwise."""
    return detect_scheduler() != Scheduler.BARE_METAL and _running_in_job_step()


def _split_comma_outside_brackets(expression):
    """Split on commas that are not inside [ ].

    Examples:
        "a,b" → ["a", "b"]
        "node[01-02],gpu[1-2]" → ["node[01-02]", "gpu[1-2]"]
        "prefix[a,b,c]" → ["prefix[a,b,c]"]
    """
    parts = []
    buf = []
    depth = 0
    for char in expression:
        if char == "[":
            depth += 1
            buf.append(char)
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced ']'")
            buf.append(char)
        elif char == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(char)
    if depth != 0:
        raise ValueError("unbalanced '['")
    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts


def _expand_range(spec):
    """Expand a bracket entry such as 006-007 or 020; pad to the width of the left number.

    Examples:
        "006-007" → ["006", "007"]
        "020" → ["020"]
        "1-3" → ["1", "2", "3"]
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty hostlist range")
    if "-" in spec:
        low, high = spec.split("-", 1)
        if not low.isdigit() or not high.isdigit():
            raise ValueError(f"invalid hostlist range {spec!r}")
        start, end = int(low), int(high)
        if start > end:
            raise ValueError(f"invalid hostlist range {spec!r}")
        width = len(low)
        return [f"{index:0{width}d}" for index in range(start, end + 1)]
    if not spec.isdigit():
        raise ValueError(f"invalid hostlist range {spec!r}")
    return [spec]


def _expand_hostlist_token(token):
    """Expand one hostlist token; later [ ] groups are a cartesian product.

    Examples:
        "node01" → ["node01"]
        "node[01-02]" → ["node01", "node02"]
        "node[01-02][a-b]" → ["node01a", "node01b", "node02a", "node02b"]
    """
    start = token.find("[")
    if start == -1:
        if "]" in token:
            raise ValueError("unbalanced ']'")
        return [token]
    end = token.find("]", start)
    if end == -1:
        raise ValueError("unbalanced '['")
    prefix = token[:start]
    suffix = token[end + 1 :]
    hosts = []
    for spec in _split_comma_outside_brackets(token[start + 1 : end]):
        for number in _expand_range(spec):
            hosts.extend(_expand_hostlist_token(f"{prefix}{number}{suffix}"))
    return hosts


def _expand_hostlist(node_list):
    """Expand Slurm/SPUR hostlist syntax into hostnames in list order.

    Examples:
        "node[01-02]" → ["node01", "node02"]
        "crsuse2-m2m-[006-007,020]" → ["crsuse2-m2m-006", "crsuse2-m2m-007", "crsuse2-m2m-020"]
        "node[01-02],gpu[1-2]" → ["node01", "node02", "gpu1", "gpu2"]
    """
    hosts = []
    for token in _split_comma_outside_brackets(node_list.strip()):
        hosts.extend(_expand_hostlist_token(token))
    return hosts


def scheduler_hosts():
    """Expand the current Slurm/SPUR job's node list in scheduler order."""
    node_list = os.environ.get("SPUR_NODES") or os.environ.get("SLURM_NODELIST")
    if not node_list:
        raise RuntimeError("managed CVS run requires SPUR_NODES or SLURM_NODELIST")
    try:
        hosts = _expand_hostlist(node_list)
    except ValueError as exc:
        raise RuntimeError(f"could not expand scheduler node list {node_list!r}: {exc}") from exc
    if not hosts:
        raise RuntimeError(f"scheduler node list {node_list!r} expanded to no hosts")
    return hosts


def scheduler_rank():
    """This process's task index and the job's task count (SLURM_PROCID, SLURM_NTASKS)."""
    try:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("managed CVS run requires SLURM_PROCID and SLURM_NTASKS") from exc
    if not 0 <= rank < world_size:
        raise RuntimeError(f"invalid managed rank {rank} for world size {world_size}")
    return rank, world_size
