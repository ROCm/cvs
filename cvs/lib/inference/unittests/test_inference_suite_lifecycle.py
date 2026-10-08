'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for inference_suite_lifecycle helpers.
'''

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cvs.lib.inference.utils.cache_probe import du_bytes
from cvs.lib.inference.utils import inference_suite_lifecycle as lifecycle_mod
from cvs.lib.inference.utils.inference_suite_lifecycle import (
    InferenceLifecycle,
    _format_lifecycle_cell_value,
)
from cvs.lib.inference.unittests.fake_orch import FakeOrch


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


class TestSetupSshd(unittest.TestCase):
    def _request(self):
        return SimpleNamespace(node=SimpleNamespace(nodeid="test_setup_sshd"))

    def test_managed_multinode_does_not_probe_listen_port(self):
        orch = FakeOrch(hosts=["node0", "node1"])
        orch.setup_sshd = lambda: True
        lifecycle = InferenceLifecycle()
        with patch("cvs.core.scheduler.is_managed_compute", return_value=True):
            lifecycle_mod.test_setup_sshd(orch, lifecycle, self._request())
        self.assertFalse(lifecycle.failed)
        self.assertEqual(orch.commands, [])

    def test_baremetal_multinode_still_requires_listen_port(self):
        orch = FakeOrch(exec_return={"node0": "NO", "node1": "NO"}, hosts=["node0", "node1"])
        orch.setup_sshd = lambda: True
        lifecycle = InferenceLifecycle()
        with patch("cvs.core.scheduler.is_managed_compute", return_value=False):
            with self.assertRaises(pytest.fail.Exception):
                lifecycle_mod.test_setup_sshd(orch, lifecycle, self._request())
        self.assertTrue(lifecycle.failed)
        self.assertTrue(orch.commands)


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
