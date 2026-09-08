'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest
from unittest import mock

import pytest

from cvs.tests.inference.vllm import _shared


class TestPrintResultsTable(unittest.TestCase):
    def test_blank_gpu_report_slot_preserves_six_field_key(self):
        results = {("model", "", "1024", "1024", "default", 16): {"host": {"client.total_token_throughput": 1.0}}}

        with mock.patch.object(_shared.log, "info") as log_info:
            _shared.test_print_results_table(results)

        log_info.assert_called_once()


class TestExecutionMode(unittest.TestCase):
    class _Config:
        def __init__(self, **options):
            self.options = options

        def getoption(self, name, default=None):
            return self.options.get(name, default)

    def test_serial_single_run_is_allowed(self):
        _shared.validate_vllm_execution_mode(self._Config(numprocesses=0, count=1))

    def test_xdist_is_rejected(self):
        with self.assertRaisesRegex(pytest.UsageError, "xdist"):
            _shared.validate_vllm_execution_mode(self._Config(numprocesses=2, count=1))

    def test_repeat_is_rejected(self):
        with self.assertRaisesRegex(pytest.UsageError, "pytest-repeat"):
            _shared.validate_vllm_execution_mode(self._Config(numprocesses=0, count=2))


if __name__ == "__main__":
    unittest.main()
