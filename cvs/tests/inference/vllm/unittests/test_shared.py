'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest
from unittest import mock

from cvs.tests.inference.vllm import _shared


class TestPrintResultsTable(unittest.TestCase):
    def test_blank_gpu_report_slot_preserves_six_field_key(self):
        results = {("model", "", "1024", "1024", "default", 16): {"host": {"client.total_token_throughput": 1.0}}}

        with mock.patch.object(_shared.log, "info") as log_info:
            _shared.test_print_results_table(results)

        log_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
