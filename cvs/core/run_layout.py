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
job id, the environment, and the venv location.

Consumers hold the object: `RunLayout.instance().agent_dir`. cvs/lib/utils_lib.py
imports it inside the function instead of at module level, because this module
pulls in cvs/core/__init__.py, whose orchestrator factory reaches
cvs/core/orchestrators/baremetal.py, which imports utils_lib back.
'''

import os
import sys
from datetime import datetime
from pathlib import Path

from cvs.core.scheduler import is_managed_compute


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
            "Pass --workspace or set CVS_WORKSPACE to a shared-filesystem path."
        )
    return Path(sys.prefix).parent / "cvs_runs"


def _resolve_workspace(workspace=None):
    '''Priority: explicit argument, then CVS_WORKSPACE, then the venv parent.'''
    candidate = workspace or os.environ.get("CVS_WORKSPACE")
    if not candidate:
        return _default_workspace()
    # Anchored to the cwd at resolution time so that nothing which chdirs later
    # can move the run directory out from under an already-published path.
    return Path(candidate).expanduser().absolute()


def _resolve_run_id():
    '''The scheduler's job id: the one name every rank of a step already agrees on.

    SPUR mirrors each SPUR_* variable it sets to a SLURM_* twin, so the SLURM name
    resolves under either scheduler. is_managed_compute() is true only inside a job
    step, which is what sets it.
    '''
    if not is_managed_compute():
        return f"local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return os.environ["SLURM_JOB_ID"]


class RunLayout:
    '''The run's directory layout, resolved once per process.'''

    _instance = None

    def __init__(self, workspace, run_id):
        self.workspace = workspace
        self.run_id = run_id
        self.run_dir = workspace / "cvs" / "runs" / run_id
        self.agent_dir = self.run_dir / "agent"

    @classmethod
    def initialize(cls, workspace=None):
        '''Resolve the layout and create the directories.

        Idempotent: later calls return the existing layout, so a run_id carrying a
        timestamp cannot drift between callers.
        '''
        if cls._instance is not None:
            return cls._instance

        layout = cls(_resolve_workspace(workspace), _resolve_run_id())
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
                f"(--workspace or CVS_WORKSPACE)."
            ) from exc
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

        For cvs/lib/utils_lib.py, which resolves config placeholders for every
        suite: most configs never mention {run_dir}, so needing a layout to be
        resolved is the exception and cannot be made a precondition of the call.
        Callers that genuinely require one use instance().
        '''
        return cls._instance

    @classmethod
    def _reset(cls):
        '''Drop the cached layout. For unit-test isolation only.'''
        cls._instance = None
