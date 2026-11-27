'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/schedulers/slurm.py: the Slurm backend reads only the
# SLURM_* (and SPUR_NODES) variables srun sets. Pins the three-variable step check,
# the rank bounds, hostlist expansion in scheduler order, and the node-name order.

import os
import unittest
from unittest.mock import patch

from cvs.core.schedulers.slurm import SlurmBackend, _expand_hostlist

JOB_STEP_ENV = {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0", "SLURM_PROCID": "0"}


class _CleanEnv(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.backend = SlurmBackend()


class TestInManagedStep(_CleanEnv):
    @patch.dict(os.environ, JOB_STEP_ENV)
    def test_true_when_all_three_vars_present(self):
        self.assertTrue(self.backend.in_managed_step())

    def test_false_when_any_one_var_is_missing(self):
        # salloc sets SLURM_JOB_ID with no step; a login shell may carry STEP_ID or
        # PROCID alone. Dropping any one of the three checks must turn this red.
        for missing in JOB_STEP_ENV:
            env = {k: v for k, v in JOB_STEP_ENV.items() if k != missing}
            with self.subTest(missing=missing), patch.dict(os.environ, env, clear=True):
                self.assertFalse(self.backend.in_managed_step())


class TestJobId(_CleanEnv):
    @patch.dict(os.environ, {"SLURM_JOB_ID": "424242", "SPUR_JOB_ID": "other"})
    def test_reads_slurm_job_id(self):
        self.assertEqual(self.backend.job_id(), "424242")

    def test_missing_job_id_names_the_variable(self):
        with self.assertRaisesRegex(RuntimeError, "SLURM_JOB_ID"):
            self.backend.job_id()


class TestRank(_CleanEnv):
    def test_reads_procid_and_ntasks(self):
        with patch.dict(os.environ, {"SLURM_PROCID": "1", "SLURM_NTASKS": "2"}):
            self.assertEqual((self.backend.rank(), self.backend.world_size()), (1, 2))

    def test_first_and_last_rank_are_valid(self):
        for rank in ("0", "3"):
            with self.subTest(rank=rank), patch.dict(os.environ, {"SLURM_PROCID": rank, "SLURM_NTASKS": "4"}):
                self.assertEqual(self.backend.rank(), int(rank))

    def test_rejects_rank_outside_world_size(self):
        # rank == world size is the off-by-one a `<=` would let through.
        for rank in ("4", "-1"):
            with self.subTest(rank=rank), patch.dict(os.environ, {"SLURM_PROCID": rank, "SLURM_NTASKS": "4"}):
                with self.assertRaisesRegex(RuntimeError, f"invalid managed rank {rank} for world size 4"):
                    self.backend.rank()

    def test_missing_or_non_integer_variable_is_named(self):
        cases = [
            ({"SLURM_NTASKS": "2"}, "SLURM_PROCID"),
            ({"SLURM_PROCID": "0"}, "SLURM_NTASKS"),
            ({"SLURM_PROCID": "zero", "SLURM_NTASKS": "2"}, "SLURM_PROCID"),
            ({"SLURM_PROCID": "0", "SLURM_NTASKS": ""}, "SLURM_NTASKS"),
        ]
        for env, variable in cases:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, variable):
                    self.backend.rank()


class TestHosts(_CleanEnv):
    @patch.dict(os.environ, {"SLURM_NODELIST": "node[01-02]"})
    def test_expands_slurm_nodelist(self):
        self.assertEqual(self.backend.hosts(), ["node01", "node02"])

    @patch.dict(os.environ, {"SPUR_NODES": "a,b", "SLURM_NODELIST": "ignored"})
    def test_prefers_spur_nodes(self):
        self.assertEqual(self.backend.hosts(), ["a", "b"])

    @patch.dict(os.environ, {"SPUR_NODES": "", "SLURM_NODELIST": "node1"})
    def test_empty_spur_nodes_falls_back_to_slurm_nodelist(self):
        self.assertEqual(self.backend.hosts(), ["node1"])

    @patch.dict(os.environ, {"SLURM_NODELIST": "node[01-02"})
    def test_invalid_hostlist_is_a_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "could not expand scheduler node list"):
            self.backend.hosts()

    def test_requires_a_nodelist(self):
        with self.assertRaisesRegex(RuntimeError, "SPUR_NODES or SLURM_NODELIST"):
            self.backend.hosts()

    @patch.dict(os.environ, {"SLURM_NODELIST": " , "})
    def test_nodelist_with_no_hosts_is_an_error(self):
        # A rank indexes hosts by its rank; an empty list must fail here, not as an
        # IndexError later.
        with self.assertRaisesRegex(RuntimeError, "expanded to no hosts"):
            self.backend.hosts()


class TestExpandHostlist(unittest.TestCase):
    def test_plain_names(self):
        self.assertEqual(_expand_hostlist("a,b"), ["a", "b"])

    def test_padded_range_and_singletons(self):
        self.assertEqual(
            _expand_hostlist("crsuse2-m2m-[006-007,020,093,191,289]"),
            [
                "crsuse2-m2m-006",
                "crsuse2-m2m-007",
                "crsuse2-m2m-020",
                "crsuse2-m2m-093",
                "crsuse2-m2m-191",
                "crsuse2-m2m-289",
            ],
        )

    def test_multiple_bracket_groups(self):
        self.assertEqual(_expand_hostlist("n[01-02],gpu[1-2]"), ["n01", "n02", "gpu1", "gpu2"])

    def test_later_bracket_groups_are_a_cartesian_product_in_order(self):
        self.assertEqual(_expand_hostlist("n[1-2]g[3-4]"), ["n1g3", "n1g4", "n2g3", "n2g4"])

    def test_rejects_malformed_lists(self):
        for node_list in ("node[01-02", "node01]", "node[02-01]", "node[a-b]", "node[x]"):
            with self.subTest(node_list=node_list):
                with self.assertRaises(ValueError):
                    _expand_hostlist(node_list)


class TestNodeName(_CleanEnv):
    @patch.dict(os.environ, {"SLURM_NODENAME": "step-node", "SLURMD_NODENAME": "daemon-node"})
    def test_slurm_nodename_wins_over_slurmd_nodename(self):
        self.assertEqual(self.backend.node_name(), "step-node")

    @patch.dict(os.environ, {"SLURMD_NODENAME": "daemon-node"})
    def test_falls_back_to_slurmd_nodename(self):
        self.assertEqual(self.backend.node_name(), "daemon-node")

    def test_none_when_scheduler_gives_no_name(self):
        self.assertIsNone(self.backend.node_name())


class TestLauncher(unittest.TestCase):
    def test_launcher(self):
        self.assertEqual(SlurmBackend().mpi_launcher(), "srun")


if __name__ == "__main__":
    unittest.main()
