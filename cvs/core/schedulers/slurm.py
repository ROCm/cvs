'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os

from cvs.core.schedulers.base import SchedulerBackend, env_int

JOB_STEP_VARS = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_PROCID")


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


class SlurmBackend(SchedulerBackend):
    """A Slurm job step launched by srun."""

    name = 'slurm'

    def in_managed_step(self):
        """True if srun launched this process as part of a job step.

        scontrol/spur version succeeding only means scheduler tooling is installed
        and the controller is reachable from wherever this process happens to run
        (e.g. a plain SSH login to a scheduler-managed head node) - it says nothing
        about whether this invocation is actually a step srun launched. SLURM_JOB_ID
        alone is also insufficient: salloc sets it for a bare allocation with no
        step. SLURM_STEP_ID/SLURM_PROCID are only set once srun has launched a step,
        which is what "managed compute" actually needs to gate on. Verified
        identical on SPUR (SLURM_JOB_ID/SLURM_STEP_ID/SLURM_PROCID) and real SLURM.
        """
        return all(os.environ.get(var) is not None for var in JOB_STEP_VARS)

    def job_id(self):
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is None:
            raise RuntimeError("managed CVS run requires SLURM_JOB_ID")
        return job_id

    def rank(self):
        rank = env_int("SLURM_PROCID")
        world_size = self.world_size()
        if not 0 <= rank < world_size:
            raise RuntimeError(f"invalid managed rank {rank} for world size {world_size}")
        return rank

    def world_size(self):
        return env_int("SLURM_NTASKS")

    def hosts(self):
        """Expand the job's node list in scheduler order."""
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

    def mpi_launcher(self):
        return 'srun'

    def node_name(self):
        return os.environ.get("SLURM_NODENAME") or os.environ.get("SLURMD_NODENAME")
