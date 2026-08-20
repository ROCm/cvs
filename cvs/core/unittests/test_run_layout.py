'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/run_layout.py: the shared-filesystem run layout every
# rank derives before agent bootstrap. Run-id resolution is driven by setting the
# real SLURM_* variables rather than by mocking a predicate, because the bug this
# module exists to prevent is a rank misreading its own environment. sys.prefix
# and the workspace are both redirected into a tmpdir so no test can write to the
# real venv parent.

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvs.core.run_layout import (
    DEFAULT_WORKSPACE_DIR_NAME,
    JOB_ID_ENV_VAR,
    RUN_DIR_ENV_VAR,
    WORKSPACE_ENV_VAR,
    RunLayout,
    _default_workspace,
)

FAKE_STAMP = "20260819-120000"


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

    def _enter_job_step(self, job_id="424242", step_id="0", proc_id="0"):
        '''Make this process look like a rank srun launched.

        Step id and proc id are what cvs.core.scheduler keys its job-step check on;
        the layout itself only reads the job id.
        '''
        os.environ[JOB_ID_ENV_VAR] = job_id
        os.environ["SLURM_STEP_ID"] = step_id
        os.environ["SLURM_PROCID"] = proc_id


class TestEnvVarContract(_RunLayoutTestCase):
    '''The env var names are a user-facing contract: CVS_WORKSPACE is what a job
    script exports and cvs_runs is what --workspace help promises. Every other test
    reads these constants back out of the module under test, so a rename would move
    both sides of the comparison together and go unnoticed.'''

    def test_env_var_names(self):
        self.assertEqual(WORKSPACE_ENV_VAR, "CVS_WORKSPACE")
        self.assertEqual(RUN_DIR_ENV_VAR, "CVS_RUN_DIR")
        self.assertEqual(DEFAULT_WORKSPACE_DIR_NAME, "cvs_runs")

    def test_slurm_env_var_names(self):
        # SPUR mirrors each SPUR_* variable it sets to a SLURM_* twin, so this name
        # covers both schedulers.
        self.assertEqual(JOB_ID_ENV_VAR, "SLURM_JOB_ID")


class TestWorkspaceResolution(_RunLayoutTestCase):
    def test_explicit_workspace_wins_over_env(self):
        os.environ[WORKSPACE_ENV_VAR] = str(Path(self.workspace) / "from_env")
        layout = RunLayout.initialize(self.workspace)
        self.assertEqual(layout.workspace, Path(self.workspace))

    def test_env_wins_over_venv_parent_default(self):
        env_workspace = Path(self.workspace) / "from_env"
        os.environ["CVS_WORKSPACE"] = str(env_workspace)
        layout = RunLayout.initialize()
        self.assertEqual(layout.workspace, env_workspace)

    def test_defaults_to_venv_parent_cvs_runs(self):
        layout = RunLayout.initialize()
        self.assertEqual(layout.workspace, Path(self.workspace) / "cvs_runs")

    def test_empty_env_var_falls_through_to_default(self):
        # An exported-but-empty CVS_WORKSPACE is an unset one, not a request to
        # use the filesystem root.
        os.environ[WORKSPACE_ENV_VAR] = ""
        layout = RunLayout.initialize()
        self.assertEqual(layout.workspace, Path(self.workspace) / "cvs_runs")

    def test_workspace_is_absolute(self):
        # Anchored to the cwd at resolution time so nothing that chdirs later can
        # move the run directory out from under an already-published path.
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.workspace)
        layout = RunLayout.initialize("relative_ws")
        self.assertTrue(layout.workspace.is_absolute())

    def test_tilde_in_workspace_is_expanded(self):
        # Without expanduser this becomes a literal "~" directory under the cwd,
        # which differs per rank -- the same silent rendezvous break an absolute
        # path is supposed to prevent.
        os.environ["HOME"] = self.workspace
        layout = RunLayout.initialize("~/shared_ws")
        self.assertEqual(layout.workspace, Path(self.workspace) / "shared_ws")

    def test_not_installed_in_venv_is_a_clean_error(self):
        # sys.prefix == sys.base_prefix means a system-interpreter install, where
        # the venv-parent default would resolve to "/cvs_runs".
        self._patch_prefix("/usr", base_prefix="/usr")
        with self.assertRaisesRegex(RuntimeError, "--workspace"):
            RunLayout.initialize()

    def test_explicit_workspace_works_outside_a_venv(self):
        # The venv check guards only the derived default; an explicit workspace
        # must still work for a system-interpreter install.
        self._patch_prefix("/usr", base_prefix="/usr")
        layout = RunLayout.initialize(self.workspace)
        self.assertEqual(layout.workspace, Path(self.workspace))


class TestRunIdResolution(_RunLayoutTestCase):
    def test_job_step_uses_the_job_id(self):
        # The scheduler's own identity for the run, taken as-is. CVS does not
        # subdivide it further; anything needing a finer identity brings its own.
        self._enter_job_step(job_id="424242")
        layout = RunLayout.initialize(self.workspace)
        self.assertEqual(layout.run_id, "424242")

    def test_every_rank_of_a_step_agrees(self):
        # The whole point: ranks resolve independently and must land on one run_dir.
        self._enter_job_step(job_id="424242", proc_id="0")
        rank0 = RunLayout.initialize(self.workspace).run_dir
        RunLayout._reset()
        self._enter_job_step(job_id="424242", proc_id="7")
        rank7 = RunLayout.initialize(self.workspace).run_dir
        self.assertEqual(rank0, rank7)

    def test_run_id_does_not_depend_on_scheduler_binaries(self):
        # The container case: srun exports the SLURM_* variables into the image but
        # scontrol/spur are not installed there. Resolution must not consult them --
        # if it does, every rank falls back to its own clock and they diverge.
        self._enter_job_step(job_id="424242")
        with patch("shutil.which", return_value=None):
            layout = RunLayout.initialize(self.workspace)
        self.assertEqual(layout.run_id, "424242")

    @patch("cvs.core.run_layout._new_run_timestamp", return_value=FAKE_STAMP)
    def test_unmanaged_uses_local_timestamp(self, _mock_stamp):
        layout = RunLayout.initialize(self.workspace)
        self.assertEqual(layout.run_id, f"local-{FAKE_STAMP}")

    @patch("cvs.core.run_layout._new_run_timestamp", return_value=FAKE_STAMP)
    def test_salloc_without_job_step_uses_timestamp_not_job_id(self, _mock_stamp):
        # salloc sets SLURM_JOB_ID for a bare allocation with no step. Keying the
        # run_id off it there would give two sequential runs in one allocation the
        # same run_dir, and the second would clobber the first.
        os.environ[JOB_ID_ENV_VAR] = "424242"
        layout = RunLayout.initialize(self.workspace)
        self.assertEqual(layout.run_id, f"local-{FAKE_STAMP}")


class TestPathComposition(_RunLayoutTestCase):
    def test_run_dir_and_agent_dir_layout(self):
        self._enter_job_step(job_id="99", step_id="0")
        layout = RunLayout.initialize(self.workspace)
        self.assertEqual(layout.run_dir, Path(self.workspace) / "cvs" / "runs" / "99")
        self.assertEqual(layout.agent_dir, layout.run_dir / "agent")

    def test_directories_are_created(self):
        self._enter_job_step(job_id="99", step_id="0")
        expected_agent_dir = Path(self.workspace) / "cvs" / "runs" / "99" / "agent"
        layout = RunLayout.initialize(self.workspace)
        # Asserted against the concrete expected path, not against layout.agent_dir,
        # so a layout that resolved to the wrong place cannot satisfy this.
        self.assertTrue(expected_agent_dir.is_dir())
        self.assertTrue(layout.run_dir.is_dir())
        self.assertTrue(layout.agent_dir.is_dir())

    def test_initialize_tolerates_preexisting_directories(self):
        # Every rank in a job step initializes against the same shared-FS paths,
        # so all but the first always find the directories already there.
        self._enter_job_step(job_id="99", step_id="0")
        expected_agent_dir = Path(self.workspace) / "cvs" / "runs" / "99" / "agent"
        expected_agent_dir.mkdir(parents=True)
        layout = RunLayout.initialize(self.workspace)
        self.assertEqual(layout.agent_dir, expected_agent_dir)
        self.assertTrue(layout.agent_dir.is_dir())

    def test_exports_run_dir_to_environment(self):
        self._enter_job_step(job_id="99", step_id="0")
        layout = RunLayout.initialize(self.workspace)
        self.assertEqual(os.environ[RUN_DIR_ENV_VAR], str(layout.run_dir))

    def test_unwritable_workspace_reports_the_path(self):
        # Shared storage that is unmounted, full, or read-only is routine on a
        # cluster; a raw pathlib traceback out of the CLI names none of the things
        # the user can act on.
        unwritable = Path(self.workspace) / "ro"
        unwritable.mkdir()
        unwritable.chmod(0o500)
        self.addCleanup(unwritable.chmod, 0o700)
        with self.assertRaises(RuntimeError) as ctx:
            RunLayout.initialize(str(unwritable / "ws"))
        message = str(ctx.exception)
        self.assertIn(str(unwritable / "ws"), message)
        self.assertIn(WORKSPACE_ENV_VAR, message)


class TestSingletonSemantics(_RunLayoutTestCase):
    def test_repeated_initialize_returns_same_object(self):
        first = RunLayout.initialize(self.workspace)
        self.assertIs(RunLayout.initialize(), first)
        self.assertIs(RunLayout.initialize(self.workspace), first)

    def test_timestamp_is_computed_once(self):
        # The point of the singleton: run_id must not drift between calls. A live
        # clock would make two calls in the same second pass by luck, so the
        # helper is stubbed to return a different value every time it is called.
        stamps = iter(["20260819-120000", "20260819-130000", "20260819-140000"])
        with patch("cvs.core.run_layout._new_run_timestamp", side_effect=lambda: next(stamps)):
            first = RunLayout.initialize(self.workspace)
            second = RunLayout.initialize()
            third = RunLayout.instance()
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.run_id, third.run_id)

    def test_instance_returns_initialized_layout(self):
        layout = RunLayout.initialize(self.workspace)
        self.assertIs(RunLayout.instance(), layout)

    def test_conflicting_workspace_raises(self):
        RunLayout.initialize(self.workspace)
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            RunLayout.initialize(f"{self.workspace}/somewhere-else")

    def test_instance_before_initialize_raises(self):
        with self.assertRaisesRegex(RuntimeError, "initialize"):
            RunLayout.instance()

    def test_reset_clears_cached_layout(self):
        RunLayout.initialize(self.workspace)
        RunLayout._reset()
        with self.assertRaises(RuntimeError):
            RunLayout.instance()


class TestVenvParentDefaultShape(_RunLayoutTestCase):
    def test_default_derives_from_sys_prefix(self):
        # CVS is always installed into a venv (Makefile .cvs_venv/.test_venv,
        # Dockerfile /opt/cvs-venv), so sys.prefix is the venv root and its
        # parent is the intended shared-FS location.
        self.assertEqual(_default_workspace(), Path(self.fake_prefix).parent / DEFAULT_WORKSPACE_DIR_NAME)


if __name__ == "__main__":
    unittest.main()
