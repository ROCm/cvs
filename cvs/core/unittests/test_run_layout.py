'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/run_layout.py: the shared-filesystem run layout every
# rank derives before agent bootstrap. What makes a run "managed" is
# cvs.core.scheduler's contract and is tested there; these tests set the real
# SLURM_* variables and only pin what the layout itself does with the answer.
# sys.prefix and the workspace are both redirected into a tmpdir so no test can
# write to the real venv parent.

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvs.core.run_layout import RunLayout
from cvs.core.scheduler import SCHEDULER_ENV_VAR

LOCAL_RUN_ID = re.compile(r"^local-\d{8}-\d{6}$")


class _RunLayoutTestCase(unittest.TestCase):
    '''Shared isolation: the layout is cached on the class and run_all_unittests.py
    runs every suite in one process, so a layout built by one test would otherwise
    leak into the next and make results depend on discovery order.

    sys.prefix is redirected into the tmpdir as well. Several tests initialize with
    no explicit workspace and rely on the venv-parent default; left unpatched, a
    regression in workspace precedence would create a real cvs_runs tree in the
    repo instead of failing cleanly.'''

    def setUp(self):
        RunLayout._reset()
        self.addCleanup(RunLayout._reset)
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = self.tmp.name
        self.fake_prefix = str(Path(self.workspace) / ".cvs_venv")
        self._patch_prefix(self.fake_prefix)

    def _patch_prefix(self, prefix, base_prefix="/usr"):
        '''Point the venv-parent default at the tmpdir. A base_prefix that differs
        from prefix is what makes CVS look venv-installed.'''
        for attr, value in (("prefix", prefix), ("base_prefix", base_prefix)):
            patcher = patch(f"cvs.core.run_layout.sys.{attr}", value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _enter_job_step(self, job_id="424242", proc_id="0"):
        '''Make this process look like a rank srun launched.

        Step id and proc id are what cvs.core.scheduler keys its job-step check on;
        the layout itself only reads the job id. CVS_SCHEDULER is set because the
        other half of is_managed_compute() shells out to `spur version` / `scontrol
        version`, which would otherwise make the result depend on what happens to be
        installed on the machine running the tests.
        '''
        os.environ["SLURM_JOB_ID"] = job_id
        os.environ["SLURM_STEP_ID"] = "0"
        os.environ["SLURM_PROCID"] = proc_id
        os.environ[SCHEDULER_ENV_VAR] = "slurm"


class TestWorkspaceResolution(_RunLayoutTestCase):
    def test_explicit_workspace_wins_over_env(self):
        os.environ["CVS_WORKSPACE"] = str(Path(self.workspace) / "from_env")
        layout = RunLayout.get(self.workspace)
        self.assertEqual(layout.workspace, Path(self.workspace))

    def test_env_wins_over_venv_parent_default(self):
        env_workspace = Path(self.workspace) / "from_env"
        os.environ["CVS_WORKSPACE"] = str(env_workspace)
        layout = RunLayout.get()
        self.assertEqual(layout.workspace, env_workspace)

    def test_defaults_to_venv_parent_cvs_runs(self):
        layout = RunLayout.get()
        self.assertEqual(layout.workspace, Path(self.workspace) / "cvs_runs")

    def test_empty_env_var_falls_through_to_default(self):
        # An exported-but-empty CVS_WORKSPACE is an unset one, not a request to
        # use the filesystem root.
        os.environ["CVS_WORKSPACE"] = ""
        layout = RunLayout.get()
        self.assertEqual(layout.workspace, Path(self.workspace) / "cvs_runs")

    def test_workspace_is_absolute(self):
        # Anchored to the cwd at resolution time so nothing that chdirs later can
        # move the run directory out from under an already-published path.
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.workspace)
        layout = RunLayout.get("relative_ws")
        self.assertTrue(layout.workspace.is_absolute())

    def test_tilde_in_workspace_is_expanded(self):
        # Without expanduser this becomes a literal "~" directory under the cwd,
        # which differs per rank -- the same silent rendezvous break an absolute
        # path is supposed to prevent.
        os.environ["HOME"] = self.workspace
        layout = RunLayout.get("~/shared_ws")
        self.assertEqual(layout.workspace, Path(self.workspace) / "shared_ws")

    def test_not_installed_in_venv_is_a_clean_error(self):
        # sys.prefix == sys.base_prefix means a system-interpreter install, where
        # the venv-parent default would resolve to "/cvs_runs".
        self._patch_prefix("/usr", base_prefix="/usr")
        with self.assertRaisesRegex(RuntimeError, "--workspace"):
            RunLayout.get()

    def test_explicit_workspace_works_outside_a_venv(self):
        # The venv check guards only the derived default; an explicit workspace
        # must still work for a system-interpreter install.
        self._patch_prefix("/usr", base_prefix="/usr")
        layout = RunLayout.get(self.workspace)
        self.assertEqual(layout.workspace, Path(self.workspace))


class TestRunIdResolution(_RunLayoutTestCase):
    def test_job_step_uses_the_job_id(self):
        # The scheduler's own identity for the run, taken as-is. CVS does not
        # subdivide it further; anything needing a finer identity brings its own.
        self._enter_job_step(job_id="424242")
        layout = RunLayout.get(self.workspace)
        self.assertEqual(layout.run_id, "424242")

    def test_every_rank_of_a_step_agrees(self):
        # The whole point: ranks resolve independently and must land on one run_dir.
        self._enter_job_step(job_id="424242", proc_id="0")
        rank0 = RunLayout.get(self.workspace).run_dir
        RunLayout._reset()
        self._enter_job_step(job_id="424242", proc_id="7")
        rank7 = RunLayout.get(self.workspace).run_dir
        self.assertEqual(rank0, rank7)

    def test_unmanaged_uses_local_timestamp(self):
        layout = RunLayout.get(self.workspace)
        self.assertRegex(layout.run_id, LOCAL_RUN_ID)


class TestPathComposition(_RunLayoutTestCase):
    def test_run_dir_and_agent_dir_are_composed_and_created(self):
        self._enter_job_step(job_id="99")
        expected_run_dir = Path(self.workspace) / "cvs_runs" / "99"
        layout = RunLayout.get(self.workspace)
        self.assertEqual(layout.run_dir, expected_run_dir)
        self.assertEqual(layout.agent_dir, expected_run_dir / "agent")
        self.assertTrue(layout.agent_dir.is_dir())

    def test_get_tolerates_preexisting_directories(self):
        # Every rank in a job step initializes against the same shared-FS paths,
        # so all but the first always find the directories already there.
        self._enter_job_step(job_id="99")
        expected_agent_dir = Path(self.workspace) / "cvs_runs" / "99" / "agent"
        expected_agent_dir.mkdir(parents=True)
        layout = RunLayout.get(self.workspace)
        self.assertEqual(layout.agent_dir, expected_agent_dir)
        self.assertTrue(layout.agent_dir.is_dir())

    def test_unwritable_workspace_reports_the_path(self):
        # Shared storage that is unmounted, full, or read-only is routine on a
        # cluster; a raw pathlib traceback out of the CLI names none of the things
        # the user can act on.
        unwritable = Path(self.workspace) / "ro"
        unwritable.mkdir()
        unwritable.chmod(0o500)
        self.addCleanup(unwritable.chmod, 0o700)
        with self.assertRaises(RuntimeError) as ctx:
            RunLayout.get(str(unwritable / "ws"))
        message = str(ctx.exception)
        self.assertIn(str(unwritable / "ws"), message)
        self.assertIn("CVS_WORKSPACE", message)


class TestSingletonSemantics(_RunLayoutTestCase):
    def test_repeated_get_returns_same_object(self):
        # run_id must not drift between callers, which is what makes the paths a
        # rendezvous rather than a guess.
        first = RunLayout.get(self.workspace)
        self.assertIs(RunLayout.get(), first)

    def test_later_get_ignores_a_different_workspace(self):
        # One layout per process. A second caller asking for somewhere else gets the
        # resolved one, so consumers cannot split the run across two directories.
        first = RunLayout.get(self.workspace)
        elsewhere = Path(self.workspace) / "other"
        self.assertIs(RunLayout.get(str(elsewhere)), first)
        self.assertFalse(elsewhere.exists())

    def test_get_resolves_a_layout_when_none_exists_yet(self):
        # No separate initialize step: the first caller resolves, later ones read.
        layout = RunLayout.get(self.workspace)
        self.assertEqual(layout.workspace, Path(self.workspace))
        self.assertTrue(layout.agent_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
