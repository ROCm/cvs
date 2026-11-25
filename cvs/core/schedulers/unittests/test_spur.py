'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/schedulers/spur.py: a SPUR step exports SPUR_* variables
# with SLURM_* twins, and the SPUR backend must answer exactly as Slurm does from
# those twins while naming itself spur.

import os
import unittest
from unittest.mock import patch

from cvs.core.schedulers.spur import SpurBackend

SPUR_STEP_ENV = {
    "SPUR_JOB_ID": "107252",
    "SPUR_NODES": "node[06-07]",
    "SLURM_JOB_ID": "107252",
    "SLURM_STEP_ID": "0",
    "SLURM_PROCID": "1",
    "SLURM_NTASKS": "2",
    "SLURM_NODELIST": "node[06-07]",
    "SLURM_NODENAME": "node07",
}


class TestSpurBackend(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, SPUR_STEP_ENV, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.backend = SpurBackend()

    def test_spur_step_values(self):
        self.assertTrue(self.backend.in_managed_step())
        self.assertEqual(self.backend.job_id(), "107252")
        self.assertEqual(self.backend.rank(), 1)
        self.assertEqual(self.backend.world_size(), 2)
        self.assertEqual(self.backend.hosts(), ["node06", "node07"])
        self.assertEqual(self.backend.node_name(), "node07")

    def test_launcher(self):
        self.assertEqual(self.backend.mpi_launcher(), "srun")


if __name__ == "__main__":
    unittest.main()
