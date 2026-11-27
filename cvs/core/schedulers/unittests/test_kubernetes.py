'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/schedulers/kubernetes.py: a CVS-generated Indexed Job pod
# is managed only with both CVS_K8S_JOB_ID and JOB_COMPLETION_INDEX, its rank is the
# completion index bounded by CVS_WORLD_SIZE, its hosts keep CVS_K8S_HOSTS order, and
# it never falls back to the SLURM_* variables.

import os
import unittest
from unittest.mock import patch

from cvs.core.schedulers.kubernetes import KubernetesBackend

POD_ENV = {
    "CVS_K8S_JOB_ID": "cvs-health-7f3a",
    "JOB_COMPLETION_INDEX": "1",
    "CVS_WORLD_SIZE": "2",
    "CVS_K8S_HOSTS": "cvs-health-7f3a-0.cvs-health-7f3a,cvs-health-7f3a-1.cvs-health-7f3a",
    "CVS_K8S_NODE_NAME": "gpu-node-b",
}


class _CleanEnv(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.backend = KubernetesBackend()


class TestInManagedStep(_CleanEnv):
    def test_managed_with_job_id_and_index(self):
        # Set means present, as for the SLURM_* step variables; an empty job id
        # still counts here, and detection is what ignores it.
        for job_id in ("cvs-health-7f3a", ""):
            env = {"CVS_K8S_JOB_ID": job_id, "JOB_COMPLETION_INDEX": "0"}
            with self.subTest(job_id=job_id), patch.dict(os.environ, env, clear=True):
                self.assertTrue(self.backend.in_managed_step())

    def test_not_managed_with_only_one_variable(self):
        # Either variable alone is not a rank: a plain Job sets only the index, and
        # a CVS pod outside an Indexed Job sets only the job id.
        for env in ({"CVS_K8S_JOB_ID": "cvs-health-7f3a"}, {"JOB_COMPLETION_INDEX": "0"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertFalse(self.backend.in_managed_step())

    @patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURM_STEP_ID": "0", "SLURM_PROCID": "0"})
    def test_slurm_step_variables_do_not_count(self):
        self.assertFalse(self.backend.in_managed_step())


class TestRank(_CleanEnv):
    def test_rank_is_completion_index(self):
        # Index 0 is rank 0, the leader that writes the cluster file.
        for index in ("0", "1", "3"):
            with (
                self.subTest(index=index),
                patch.dict(os.environ, {"JOB_COMPLETION_INDEX": index, "CVS_WORLD_SIZE": "4"}),
            ):
                self.assertEqual(self.backend.rank(), int(index))
                self.assertEqual(self.backend.world_size(), 4)

    def test_rank_outside_world_size(self):
        for index in ("4", "-1"):
            with (
                self.subTest(index=index),
                patch.dict(os.environ, {"JOB_COMPLETION_INDEX": index, "CVS_WORLD_SIZE": "4"}),
            ):
                with self.assertRaisesRegex(RuntimeError, f"invalid managed rank {index} for world size 4"):
                    self.backend.rank()

    def test_missing_rank_variable_is_named(self):
        cases = [
            ({"CVS_WORLD_SIZE": "2"}, "JOB_COMPLETION_INDEX"),
            ({"JOB_COMPLETION_INDEX": "0"}, "CVS_WORLD_SIZE"),
        ]
        for env, variable in cases:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, variable):
                    self.backend.rank()

    @patch.dict(os.environ, {"SLURM_PROCID": "0", "SLURM_NTASKS": "2"})
    def test_slurm_rank_variables_are_not_read(self):
        with self.assertRaisesRegex(RuntimeError, "JOB_COMPLETION_INDEX"):
            self.backend.rank()


class TestHosts(_CleanEnv):
    def test_hosts_keep_rank_order(self):
        # Rank i runs on hosts[i]; sorting would hand ranks the wrong pod.
        for value, expected in (("pod-0", ["pod-0"]), ("pod-2,pod-0,pod-1", ["pod-2", "pod-0", "pod-1"])):
            with self.subTest(value=value), patch.dict(os.environ, {"CVS_K8S_HOSTS": value}):
                self.assertEqual(self.backend.hosts(), expected)

    def test_hosts_required(self):
        for env in ({}, {"CVS_K8S_HOSTS": ""}, {"CVS_K8S_HOSTS": ","}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, "CVS_K8S_HOSTS"):
                    self.backend.hosts()

    @patch.dict(os.environ, {"SLURM_NODELIST": "node[01-02]"})
    def test_slurm_nodelist_is_not_read(self):
        with self.assertRaisesRegex(RuntimeError, "CVS_K8S_HOSTS"):
            self.backend.hosts()


class TestJobValues(_CleanEnv):
    @patch.dict(os.environ, POD_ENV)
    def test_job_id_node_name_and_launcher(self):
        self.assertEqual(self.backend.name, "kubernetes")
        self.assertEqual(self.backend.job_id(), "cvs-health-7f3a")
        self.assertEqual(self.backend.node_name(), "gpu-node-b")
        self.assertEqual(self.backend.mpi_launcher(), "mpirun-ssh")

    @patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURM_NODENAME": "node01"})
    def test_unset_job_id_raises_and_node_name_is_none(self):
        with self.assertRaisesRegex(RuntimeError, "CVS_K8S_JOB_ID"):
            self.backend.job_id()
        self.assertIsNone(self.backend.node_name())

    @patch.dict(os.environ, {"CVS_K8S_NODE_NAME": ""})
    def test_empty_node_name_is_none(self):
        # An empty name would replace the OS hostname fallback with "".
        self.assertIsNone(self.backend.node_name())


if __name__ == "__main__":
    unittest.main()
