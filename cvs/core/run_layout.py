'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Shared-filesystem run layout, resolved once before pytest or agent bootstrap.

Every rank in a scheduler-managed job derives workspace/run_dir/agent_dir
independently and must arrive at the same answer, because those paths are the
rendezvous: worker ranks never enter pytest, they read rank0's port out of
agent_dir. Resolution therefore depends only on inputs every rank shares -- the
job id, the environment, and the venv location -- and is cached on the class so
a run_id containing a timestamp cannot drift between callers.

Consumers hold the object: `RunLayout.instance().agent_dir`. The one exception
is cvs/lib/utils_lib.py, which cannot import this module at module level
(cvs/core/__init__.py imports the orchestrator factory, which imports utils_lib
back) and reads CVS_RUN_DIR from the environment instead.
'''

import os
import sys
from datetime import datetime
from pathlib import Path

from cvs.core.scheduler import _running_in_job_step

WORKSPACE_ENV_VAR = "CVS_WORKSPACE"
RUN_DIR_ENV_VAR = "CVS_RUN_DIR"
DEFAULT_WORKSPACE_DIR_NAME = "cvs_runs"
# SPUR exports the SLURM_* variables verbatim, so one set of names covers both.
JOB_ID_ENV_VAR = "SLURM_JOB_ID"
STEP_ID_ENV_VAR = "SLURM_STEP_ID"
PROC_ID_ENV_VAR = "SLURM_PROCID"
RESTART_COUNT_ENV_VAR = "SLURM_RESTART_COUNT"


def _new_run_timestamp():
    '''Filesystem-safe local timestamp for an unmanaged run id.'''
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_workspace():
    '''Fallback workspace: the venv's parent directory.

    CVS is installed into a venv (Makefile builds .cvs_venv/.test_venv, the image
    builds /opt/cvs-venv), so sys.prefix is the venv root. On a cluster whose home
    is a shared mount this lands on shared storage, which is what the agent
    rendezvous needs; inside the container image it does not, so managed runs
    there must pass --workspace.
    '''
    if sys.prefix == sys.base_prefix:
        # A system-interpreter install would derive "/cvs_runs" here, which fails
        # as a permission error much later and for no obvious reason.
        raise RuntimeError(
            "CVS is not running from a virtualenv, so a default workspace cannot be derived. "
            f"Pass --workspace or set {WORKSPACE_ENV_VAR} to a shared-filesystem path."
        )
    return Path(sys.prefix).parent / DEFAULT_WORKSPACE_DIR_NAME


def _resolve_workspace(workspace=None):
    '''Priority: explicit argument, then CVS_WORKSPACE, then the venv parent.'''
    candidate = workspace or os.environ.get(WORKSPACE_ENV_VAR)
    if not candidate:
        return _default_workspace()
    # Anchored to the cwd at resolution time so that nothing which chdirs later
    # can move the run directory out from under an already-published path.
    return Path(candidate).expanduser().absolute()


def _resolve_run_id(managed):
    '''Identifies this run, and must come out identical on every rank of a step.

    The job id alone does not identify a run: concurrent steps in one allocation
    share it (and would then share an agent_dir, letting a worker read the wrong
    rank0 port), and a requeued job repeats it.
    '''
    if not managed:
        return f"local-{_new_run_timestamp()}"
    # _running_in_job_step() guarantees both of these are set.
    run_id = f"{os.environ[JOB_ID_ENV_VAR]}.{os.environ[STEP_ID_ENV_VAR]}"
    restart_count = os.environ.get(RESTART_COUNT_ENV_VAR)
    if restart_count and restart_count != "0":
        run_id = f"{run_id}.r{restart_count}"
    return run_id


class RunLayout:
    '''The run's directory layout, resolved once per process.'''

    _instance = None

    def __init__(self, workspace, run_id, managed):
        self.workspace = workspace
        self.run_id = run_id
        self.managed = managed
        self.run_dir = workspace / "cvs" / "runs" / run_id
        self.agent_dir = self.run_dir / "agent"

    @classmethod
    def initialize(cls, workspace=None):
        '''Resolve the layout, create the directories, publish CVS_RUN_DIR.

        Idempotent: later calls return the existing layout so the run_id stays
        put. Passing a different explicit workspace is a programming error
        rather than a reconfigure, since paths already handed out would go stale.
        '''
        if cls._instance is not None:
            if workspace is not None and _resolve_workspace(workspace) != cls._instance.workspace:
                raise RuntimeError(
                    f"RunLayout already initialized with workspace {cls._instance.workspace}, "
                    f"cannot re-initialize with {workspace}"
                )
            return cls._instance

        # Whether a scheduler launched this, keyed on the job-step environment
        # rather than on is_managed_compute(): that predicate also probes for the
        # scontrol/spur binaries, which are absent from the CVS container image
        # even when srun launched the step and exported the SLURM_* variables.
        # Ranks would then each fall back to their own wall clock and land on
        # different run_dirs -- a silent hang at the rendezvous rather than an
        # error. The env vars are set by the launch itself, so ranks agree.
        managed = _running_in_job_step()
        layout = cls(_resolve_workspace(workspace), _resolve_run_id(managed), managed)
        try:
            # parents=True also creates run_dir; exist_ok because every rank in a
            # job step races to create the same shared-FS directories.
            layout.agent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Unmounted, read-only or full shared storage is routine on a cluster,
            # and a bare pathlib traceback names none of the things a user can act on.
            raise RuntimeError(
                f"Could not create the run directory {layout.agent_dir}: {exc}. "
                f"Check that the workspace is a writable shared-filesystem path "
                f"(--workspace or {WORKSPACE_ENV_VAR})."
            ) from exc
        os.environ[RUN_DIR_ENV_VAR] = str(layout.run_dir)
        cls._instance = layout
        return layout

    @classmethod
    def instance(cls):
        '''The layout for this run. Raises if nothing initialized it yet.'''
        if cls._instance is None:
            raise RuntimeError("RunLayout.initialize() must be called before RunLayout.instance()")
        return cls._instance

    @classmethod
    def instance_or_none(cls):
        '''The layout if one was resolved, None otherwise.

        For consumers that must also work when the suite was launched with bare
        pytest instead of `cvs run` (see cvs/tests/health/README.md), which never
        initializes a layout. Callers that genuinely require one use instance().
        '''
        return cls._instance

    @classmethod
    def _reset(cls):
        '''Drop the cached layout. For unit-test isolation only.'''
        cls._instance = None
