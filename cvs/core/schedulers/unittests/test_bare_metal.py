'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/schedulers/bare_metal.py: bare metal is never managed and
# has no job id, rank, world size or host list, even when stray SLURM_* variables
# are in the environment (CVS_SCHEDULER=bare_metal inside an allocation).

import os
import unittest
from unittest.mock import patch

from cvs.core.schedulers.bare_metal import BareMetalBackend

SLURM_STEP_ENV = {
    "SLURM_JOB_ID": "123",
    "SLURM_STEP_ID": "0",
    "SLURM_PROCID": "0",
    "SLURM_NTASKS": "2",
    "SLURM_NODELIST": "node[01-02]",
    "SLURM_NODENAME": "node01",
}


@patch.dict(os.environ, SLURM_STEP_ENV, clear=True)
class TestBareMetalBackend(unittest.TestCase):
    def test_never_managed_even_inside_a_slurm_step(self):
        self.assertFalse(BareMetalBackend().in_managed_step())

    def test_managed_only_values_raise(self):
        backend = BareMetalBackend()
        for method in (backend.job_id, backend.rank, backend.world_size, backend.hosts):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(RuntimeError, "bare-metal run has no scheduler"):
                    method()

    def test_no_node_name_from_slurm_variables(self):
        self.assertIsNone(BareMetalBackend().node_name())

    def test_name_and_launcher(self):
        self.assertEqual(BareMetalBackend.name, "bare_metal")
        self.assertEqual(BareMetalBackend().mpi_launcher(), "mpirun-ssh")


if __name__ == "__main__":
    unittest.main()
