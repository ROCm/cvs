"""PMIx-child execution for managed launches."""

import asyncio
import os
import signal
import socket
from pathlib import Path

from . import messages


def _launch_env(overrides):
    """Preserve scheduler PMIx state and add the proven Open MPI compatibility settings."""
    protected = [
        key
        for key in overrides
        if key in ("PMIX_RANK", "PMIX_NAMESPACE", "PMIX_SIZE")
        or key.startswith("PMIX_SERVER_")
        or key.startswith(("SLURM_", "SPUR_"))
        or key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
    ]
    if protected:
        raise ValueError(f"launch env may not override scheduler-owned variables: {sorted(protected)}")
    env = {**os.environ, **overrides}
    gds = env.get("PMIX_GDS_MODULE", "")
    if "shmem" in gds:
        env["PMIX_GDS_MODULE"] = "hash"
    if env.get("ROCR_VISIBLE_DEVICES"):
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)

    job_id = env.get("SPUR_JOB_ID") or env.get("SLURM_JOB_ID") or "nojob"
    tmpdir_base = Path(f"/tmp/ompi.{os.getuid()}.{job_id}")
    tmpdir_base.mkdir(mode=0o700, parents=True, exist_ok=True)
    env.setdefault("OMPI_MCA_orte_tmpdir_base", str(tmpdir_base))
    return env


async def _terminate(process):
    """SIGTERM the child's process group, then SIGKILL if it does not exit."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=messages.TERMINATE_GRACE_PERIOD_SECONDS)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


class ActiveLaunches:
    """Track launch children so HTTP shutdown can terminate them."""

    def __init__(self):
        """Start with no tracked children."""
        self._processes = set()

    def add(self, process):
        """Remember a child so shutdown/cancel can signal it."""
        self._processes.add(process)

    def discard(self, process):
        """Drop a child after it has exited."""
        self._processes.discard(process)

    async def terminate_all(self):
        """Signal every tracked child process group."""
        await asyncio.gather(*(_terminate(process) for process in list(self._processes)))


async def run_launch_child(
    request,
    rank,
    active=None,
    on_spawn=None,
):
    """Start one child, write rank-scoped logs, and return without exiting the agent.

    on_spawn is called once the child exists, so the node coordinator can tell a slow
    MPI startup apart from a rank whose child never started.
    """
    hostname = socket.gethostname()
    stdout_path = request.out_path / f"rank-{rank:04d}.stdout"
    stderr_path = request.out_path / f"rank-{rank:04d}.stderr"
    try:
        await asyncio.to_thread(request.out_path.mkdir, parents=True, exist_ok=True)
        env = _launch_env(request.env)
        stdout_file = await asyncio.to_thread(open, stdout_path, "wb")
        stderr_file = await asyncio.to_thread(open, stderr_path, "wb")
        try:
            process = await asyncio.create_subprocess_exec(
                *request.argv,
                cwd=request.cwd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        finally:
            stdout_file.close()
            stderr_file.close()
        if on_spawn is not None:
            on_spawn()

        timed_out = False
        if active is not None:
            active.add(process)
        try:
            await asyncio.wait_for(process.wait(), timeout=request.timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await _terminate(process)
        finally:
            if active is not None:
                active.discard(process)
        return messages.LaunchRankResult(
            rank=rank,
            hostname=hostname,
            exit_code=process.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timed_out=timed_out,
        )
    except Exception as exc:  # returned in-band so one rank's setup failure is visible at rank 0
        return messages.LaunchRankResult(
            rank=rank,
            hostname=hostname,
            exit_code=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timed_out=False,
            error=str(exc),
        )
