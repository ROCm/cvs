'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/scheduler.py: scheduler-type detection used to decide
# managed (SPUR/SLURM) vs unmanaged (baremetal SSH) orchestration. Mocks shutil.which
# and subprocess.run so tests run with no scheduler binaries or live cluster access.

import os
import subprocess
import unittest
from unittest.mock import patch

from cvs.core.scheduler import Scheduler, _running_in_job_step, detect_scheduler, is_managed_compute

JOB_STEP_ENV = {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0", "SLURM_PROCID": "0"}


def _which_only(*present):
    def fake_which(cmd):
        return f"/usr/local/bin/{cmd}" if cmd in present else None

    return fake_which


class TestDetectScheduler(unittest.TestCase):
    def setUp(self):
        # CVS_SCHEDULER must not leak in from the ambient environment, since it
        # short-circuits detection entirely and several tests rely on the
        # command-probing path actually running.
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only("spur"))
    def test_spur_present_and_responding(self, _mock_which, mock_run):
        mock_run.return_value.returncode = 0
        self.assertEqual(detect_scheduler(), Scheduler.SPUR)
        mock_run.assert_called_once_with(["spur", "version"], timeout=5, capture_output=True, text=True)

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only("scontrol"))
    def test_slurm_present_and_responding(self, _mock_which, mock_run):
        mock_run.return_value.returncode = 0
        self.assertEqual(detect_scheduler(), Scheduler.SLURM)

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only("spur", "scontrol"))
    def test_spur_checked_before_slurm(self, _mock_which, mock_run):
        # spur ships its own scontrol/sinfo/squeue shims, so a spur cluster would also
        # pass a generic scontrol check. SPUR must win when both are present.
        mock_run.return_value.returncode = 0
        self.assertEqual(detect_scheduler(), Scheduler.SPUR)

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only())
    def test_no_scheduler_binaries_on_path(self, _mock_which, mock_run):
        self.assertEqual(detect_scheduler(), Scheduler.BARE_METAL)
        mock_run.assert_not_called()

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only("spur", "scontrol"))
    def test_falls_through_to_slurm_when_spur_command_fails(self, _mock_which, mock_run):
        # Regression test for the original bug: a failed spur check must not abort
        # detection outright, it must fall through and still check SLURM.
        def fake_run(cmd, **_kwargs):
            result = unittest.mock.Mock()
            result.returncode = 0 if cmd[0] == "scontrol" else 1
            return result

        mock_run.side_effect = fake_run
        self.assertEqual(detect_scheduler(), Scheduler.SLURM)

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only("spur", "scontrol"))
    def test_both_commands_fail_returns_bare_metal(self, _mock_which, mock_run):
        mock_run.return_value.returncode = 1
        self.assertEqual(detect_scheduler(), Scheduler.BARE_METAL)

    @patch("cvs.core.scheduler.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="spur", timeout=5))
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only("spur"))
    def test_hanging_command_treated_as_failure(self, _mock_which, _mock_run):
        self.assertEqual(detect_scheduler(), Scheduler.BARE_METAL)

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only())
    @patch.dict(os.environ, {"CVS_SCHEDULER": "slurm"})
    def test_env_override_takes_precedence_over_probing(self, _mock_which, mock_run):
        self.assertEqual(detect_scheduler(), Scheduler.SLURM)
        mock_run.assert_not_called()

    @patch.dict(os.environ, {"CVS_SCHEDULER": "  SPUR  "})
    def test_env_override_normalizes_case_and_whitespace(self):
        self.assertEqual(detect_scheduler(), Scheduler.SPUR)

    @patch.dict(os.environ, {"CVS_SCHEDULER": "kubernetes"})
    def test_env_override_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "Unknown scheduler type 'kubernetes'"):
            detect_scheduler()


class TestRunningInJobStep(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch.dict(os.environ, JOB_STEP_ENV)
    def test_true_when_all_three_vars_present(self):
        self.assertTrue(_running_in_job_step())

    def test_false_when_no_vars_present(self):
        self.assertFalse(_running_in_job_step())

    @patch.dict(os.environ, {"SLURM_STEP_ID": "0", "SLURM_PROCID": "0"})
    def test_false_when_job_id_missing(self):
        # e.g. plain SSH login to a scheduler-managed head node with scheduler
        # tooling installed, but no srun step actually launched this process.
        self.assertFalse(_running_in_job_step())

    @patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_PROCID": "0"})
    def test_false_when_step_id_missing(self):
        # e.g. salloc grants a bare allocation without launching a step.
        self.assertFalse(_running_in_job_step())

    @patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0"})
    def test_false_when_procid_missing(self):
        self.assertFalse(_running_in_job_step())


class TestIsManagedCompute(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("cvs.core.scheduler.detect_scheduler", return_value=Scheduler.SPUR)
    @patch.dict(os.environ, JOB_STEP_ENV)
    def test_true_for_spur_in_job_step(self, _mock_detect):
        self.assertTrue(is_managed_compute())

    @patch("cvs.core.scheduler.detect_scheduler", return_value=Scheduler.SLURM)
    @patch.dict(os.environ, JOB_STEP_ENV)
    def test_true_for_slurm_in_job_step(self, _mock_detect):
        self.assertTrue(is_managed_compute())

    @patch("cvs.core.scheduler.detect_scheduler", return_value=Scheduler.BARE_METAL)
    @patch.dict(os.environ, JOB_STEP_ENV)
    def test_false_for_bare_metal_even_in_job_step(self, _mock_detect):
        self.assertFalse(is_managed_compute())

    @patch("cvs.core.scheduler.detect_scheduler", return_value=Scheduler.SLURM)
    def test_false_when_scheduler_managed_but_not_in_job_step(self, _mock_detect):
        # The exact scenario from the PR review: scheduler tooling works (e.g. a
        # plain SSH login to a managed head node) but this process was never
        # launched via srun, so it must not be treated as managed compute.
        self.assertFalse(is_managed_compute())


if __name__ == "__main__":
    unittest.main()
