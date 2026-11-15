'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for inference_suite_lifecycle helpers.
'''

import unittest

from cvs.lib.inference.utils.cache_probe import du_bytes
from cvs.lib.inference.utils import inference_suite_lifecycle as lifecycle_mod
from cvs.lib.inference.utils.inference_suite_lifecycle import _format_lifecycle_cell_value
from cvs.lib.inference.unittests.fake_orch import FakeOrch
from cvs.lib.utils import container_stages


class TestFormatLifecycleCellValue(unittest.TestCase):
    def test_formats_float_with_one_decimal(self):
        self.assertEqual(_format_lifecycle_cell_value(1.234), "1.2")

    def test_formats_int_and_str_without_float_spec(self):
        self.assertEqual(_format_lifecycle_cell_value(15736427790659), "15736427790659")
        self.assertEqual(
            _format_lifecycle_cell_value("/it-share-prj2-1/models/Qwen3.5-397B-A17B-FP8"),
            "/it-share-prj2-1/models/Qwen3.5-397B-A17B-FP8",
        )

    def test_none_renders_dash(self):
        self.assertEqual(_format_lifecycle_cell_value(None), "-")


class TestContainerStageReExports(unittest.TestCase):
    def test_stage_tests_are_the_container_stages_objects(self):
        # ATOM imports and collects these names from this module.
        for name in ("test_launch_container", "test_setup_sshd", "test_teardown"):
            with self.subTest(name=name):
                self.assertIs(getattr(lifecycle_mod, name), getattr(container_stages, name))


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
