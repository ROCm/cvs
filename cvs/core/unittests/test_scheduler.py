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
import subprocess  # nosec B404 - only for subprocess.TimeoutExpired; the test runs no commands
import unittest
from unittest.mock import patch

from cvs.core.scheduler import (
    SCHEDULER_ENV_VAR,
    Scheduler,
    _running_in_job_step,
    detect_scheduler,
    get_backend,
    is_managed_compute,
    scheduler_hosts,
    scheduler_rank,
)
from cvs.core.schedulers.bare_metal import BareMetalBackend
from cvs.core.schedulers.kubernetes import KubernetesBackend
from cvs.core.schedulers.slurm import SlurmBackend
from cvs.core.schedulers.spur import SpurBackend

JOB_STEP_ENV = {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0", "SLURM_PROCID": "0"}
K8S_POD_ENV = {"CVS_K8S_JOB_ID": "cvs-health-7f3a", "JOB_COMPLETION_INDEX": "0"}


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

    @patch("cvs.core.scheduler.subprocess.run")
    @patch.dict(os.environ, {"CVS_SCHEDULER": " Kubernetes "})
    def test_env_override_accepts_kubernetes(self, mock_run):
        self.assertEqual(detect_scheduler(), Scheduler.KUBERNETES)
        mock_run.assert_not_called()

    @patch.dict(os.environ, {"CVS_SCHEDULER": "pbs"})
    def test_env_override_rejects_unknown_value(self):
        with self.assertRaises(ValueError) as ctx:
            detect_scheduler()
        message = str(ctx.exception)
        self.assertIn("Unknown scheduler type 'pbs'", message)
        for value in ("spur", "slurm", "bare_metal", "kubernetes"):
            self.assertIn(f"'{value}'", message)

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only("spur", "scontrol"))
    @patch.dict(os.environ, {"CVS_K8S_JOB_ID": "cvs-health-7f3a", "SPUR_JOB_ID": "1", "SLURM_JOB_ID": "1"})
    def test_k8s_job_id_wins_over_spur_and_slurm(self, _mock_which, mock_run):
        self.assertEqual(detect_scheduler(), Scheduler.KUBERNETES)
        mock_run.assert_not_called()

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only())
    def test_empty_k8s_job_id_falls_through(self, _mock_which, mock_run):
        # An exported-but-empty CVS_K8S_JOB_ID is unset, as an empty SPUR_JOB_ID is.
        with patch.dict(os.environ, {"CVS_K8S_JOB_ID": "", "SLURM_JOB_ID": "99"}):
            self.assertEqual(detect_scheduler(), Scheduler.SLURM)
        with patch.dict(os.environ, {"CVS_K8S_JOB_ID": ""}):
            self.assertEqual(detect_scheduler(), Scheduler.BARE_METAL)
        mock_run.assert_not_called()

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only())
    @patch.dict(os.environ, {"SLURM_JOB_ID": "99", "KUBERNETES_SERVICE_HOST": "10.0.0.1", "JOB_COMPLETION_INDEX": "0"})
    def test_slurm_inside_pod_is_slurm(self, _mock_which, mock_run):
        # Every pod carries KUBERNETES_SERVICE_HOST; only a CVS-generated pod sets
        # CVS_K8S_JOB_ID, so Slurm running in pods stays Slurm.
        self.assertEqual(detect_scheduler(), Scheduler.SLURM)
        mock_run.assert_not_called()

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only())
    @patch.dict(os.environ, {"SPUR_JOB_ID": "107252", "SLURM_JOB_ID": "107252"})
    def test_spur_job_id_wins_without_binaries(self, _mock_which, mock_run):
        # AIMVT-319: compute nodes have no spur/scontrol; SPUR still sets SPUR_JOB_ID
        # (and SLURM_* twins). Detection must not fall through to BARE_METAL.
        self.assertEqual(detect_scheduler(), Scheduler.SPUR)
        mock_run.assert_not_called()

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only())
    @patch.dict(os.environ, {"SLURM_JOB_ID": "99"})
    def test_slurm_job_id_without_spur_job_id(self, _mock_which, mock_run):
        self.assertEqual(detect_scheduler(), Scheduler.SLURM)
        mock_run.assert_not_called()

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only("scontrol"))
    @patch.dict(os.environ, {"SPUR_JOB_ID": "1", "SLURM_JOB_ID": "1"})
    def test_spur_job_id_checked_before_slurm_job_id(self, _mock_which, mock_run):
        self.assertEqual(detect_scheduler(), Scheduler.SPUR)
        mock_run.assert_not_called()

    @patch("cvs.core.scheduler.subprocess.run")
    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only())
    @patch.dict(os.environ, {"CVS_SCHEDULER": "slurm", "CVS_K8S_JOB_ID": "cvs-health-7f3a", "SPUR_JOB_ID": "1"})
    def test_env_override_takes_precedence_over_job_id(self, _mock_which, mock_run):
        self.assertEqual(detect_scheduler(), Scheduler.SLURM)
        mock_run.assert_not_called()


class TestRunningInJobStep(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch.dict(os.environ, JOB_STEP_ENV)
    def test_true_only_inside_a_slurm_step(self):
        # The variable-by-variable cases live in test_slurm.py; this pins the delegation.
        self.assertTrue(_running_in_job_step())
        del os.environ["SLURM_STEP_ID"]
        self.assertFalse(_running_in_job_step())

    @patch.dict(os.environ, K8S_POD_ENV)
    def test_true_in_an_indexed_cvs_pod(self):
        self.assertTrue(_running_in_job_step())


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

    @patch("cvs.core.scheduler.shutil.which", side_effect=_which_only())
    @patch.dict(
        os.environ,
        {
            "SPUR_JOB_ID": "107252",
            "SLURM_JOB_ID": "107252",
            "SLURM_STEP_ID": "0",
            "SLURM_PROCID": "0",
        },
    )
    def test_true_in_spur_step_without_scheduler_binaries(self, _mock_which):
        # End-to-end AIMVT-319: a spur srun step on a compute node with empty PATH.
        self.assertTrue(is_managed_compute())

    @patch.dict(os.environ, {"CVS_SCHEDULER": "kubernetes", "CVS_K8S_JOB_ID": "cvs-health-7f3a", **JOB_STEP_ENV})
    def test_kubernetes_ignores_a_stray_slurm_step(self):
        # The Kubernetes backend decides; a full set of SLURM_* step variables in
        # the pod must not make it managed.
        self.assertFalse(is_managed_compute())


class TestGetBackend(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_get_backend_matches_detection(self):
        # A swapped row in the backend table would hand a SPUR job Slurm's name, or
        # bare metal a managed backend.
        expected = {
            Scheduler.SPUR: SpurBackend,
            Scheduler.SLURM: SlurmBackend,
            Scheduler.BARE_METAL: BareMetalBackend,
            Scheduler.KUBERNETES: KubernetesBackend,
        }
        for scheduler in Scheduler:
            with self.subTest(scheduler=scheduler):
                with patch("cvs.core.scheduler.detect_scheduler", return_value=scheduler):
                    backend = get_backend()
                self.assertIs(type(backend), expected[scheduler])
                self.assertEqual(backend.name, scheduler.value)

    def test_backend_follows_the_environment_on_every_call(self):
        # Callers ask more than once per process; a cached backend would keep
        # answering for the scheduler that was detected first.
        with patch.dict(os.environ, {SCHEDULER_ENV_VAR: "slurm"}):
            self.assertIsInstance(get_backend(), SlurmBackend)
        with patch.dict(os.environ, {SCHEDULER_ENV_VAR: "bare_metal"}):
            self.assertIsInstance(get_backend(), BareMetalBackend)


class TestSchedulerWrappers(unittest.TestCase):
    """scheduler_hosts() and scheduler_rank() answer through the detected backend."""

    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch.dict(
        os.environ,
        {
            "SPUR_JOB_ID": "7",
            "SLURM_JOB_ID": "7",
            "SPUR_NODES": "node[01-02]",
            "SLURM_PROCID": "1",
            "SLURM_NTASKS": "2",
        },
    )
    def test_spur_step_hosts_and_rank(self):
        self.assertEqual(scheduler_hosts(), ["node01", "node02"])
        self.assertEqual(scheduler_rank(), (1, 2))

    @patch.dict(
        os.environ,
        {SCHEDULER_ENV_VAR: "bare_metal", "SLURM_NODELIST": "node[01-02]", "SLURM_PROCID": "0", "SLURM_NTASKS": "2"},
    )
    def test_bare_metal_ignores_slurm_variables(self):
        # Reading SLURM_* directly instead of asking the backend would return a
        # host list and rank for a run that has neither.
        with self.assertRaisesRegex(RuntimeError, "bare-metal"):
            scheduler_hosts()
        with self.assertRaisesRegex(RuntimeError, "bare-metal"):
            scheduler_rank()


if __name__ == "__main__":
    unittest.main()
