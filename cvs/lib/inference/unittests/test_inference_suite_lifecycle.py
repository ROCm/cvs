'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for inference_suite_lifecycle helpers.
'''

import unittest

from cvs.lib.inference.utils.cache_probe import du_bytes
from cvs.lib.inference.unittests.fake_orch import FakeOrch


class TestDuBytes(unittest.TestCase):
    def test_sums_bytes_across_hosts(self):
        orch = FakeOrch(exec_return={"node0": "1000", "node1": "2000"})
        self.assertEqual(du_bytes(orch, "/models"), 3000)

    def test_missing_path_returns_zero(self):
        orch = FakeOrch(exec_return={"node0": "__MISSING__"})
        self.assertEqual(du_bytes(orch, "/models"), 0)

    def test_du_error_returns_none(self):
        orch = FakeOrch(exec_return={"node0": "__DU_ERROR__"})
        self.assertIsNone(du_bytes(orch, "/models"))


if __name__ == "__main__":
    unittest.main()
