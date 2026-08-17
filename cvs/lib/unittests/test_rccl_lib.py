# cvs/lib/unittests/test_rccl_lib.py
import unittest
from unittest.mock import patch
import cvs.lib.rccl_lib as rccl_lib


class TestRcclLib(unittest.TestCase):
    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_avg_bus_bw_success(self, mock_fail_test):
        output = "# Avg bus bandwidth : 100.5"
        exp_res_dict = {'avg_bus_bw': 100.0}
        rccl_lib.check_avg_bus_bw(output, exp_res_dict)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_avg_bus_bw_failure(self, mock_fail_test):
        output = "# Avg bus bandwidth : 90.0"
        exp_res_dict = {'avg_bus_bw': 100.0}
        rccl_lib.check_avg_bus_bw(output, exp_res_dict)
        mock_fail_test.assert_called_once()

    def test_check_avg_bus_bw_no_match(self):
        output = "No bandwidth info"
        exp_res_dict = {'avg_bus_bw': 100.0}
        # Should not raise or fail
        rccl_lib.check_avg_bus_bw(output, exp_res_dict)

    def test_convert_to_graph_dict(self):
        # Test with sample data
        result_dict = {
            'allreduce': [{'size': 1024, 'name': 'allreduce', 'inPlace': 0, 'busBw': 100.0, 'algBw': 90.0, 'time': 1.0}]
        }
        result = rccl_lib.convert_to_graph_dict(result_dict)
        self.assertIsInstance(result, dict)

    # Tests for new verification functions

    def test_is_severe_wrong_corruption_error(self):
        """Test severe corruption error detection"""

        # Test with actual ValidationError-like object for structured errors
        class MockValidationError:
            def errors(self):
                return [{'msg': 'SEVERE DATA CORRUPTION detected'}]

            def __str__(self):
                return "ValidationError with SEVERE DATA CORRUPTION"

        mock_error = MockValidationError()
        self.assertTrue(rccl_lib._is_severe_wrong_corruption_error(mock_error))

        # Test with '#wrong' pattern
        class MockWrongError:
            def errors(self):
                return [{'msg': "Field validation failed: '#wrong' > 0"}]

            def __str__(self):
                return "ValidationError with wrong"

        mock_error = MockWrongError()
        self.assertTrue(rccl_lib._is_severe_wrong_corruption_error(mock_error))

        # Test fallback to string search with '#wrong' pattern
        class MockStringError:
            def errors(self):
                raise Exception("No structured errors")

            def __str__(self):
                return "ValidationError contains '#wrong' > 0"

        mock_error = MockStringError()
        self.assertTrue(rccl_lib._is_severe_wrong_corruption_error(mock_error))

        # Test normal error (should not be severe)
        class MockNormalError:
            def errors(self):
                raise Exception("No structured errors")

            def __str__(self):
                return "Normal validation error"

        mock_error = MockNormalError()
        self.assertFalse(rccl_lib._is_severe_wrong_corruption_error(mock_error))

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_scan_rccl_logs_success(self, mock_fail_test):
        """Test successful log scanning"""
        output = """
        INFO: Test starting
        NCCL WARN: Performance warning
        # Avg bus bandwidth    :   85.5
        Test completed successfully
        """

        rccl_lib.scan_rccl_logs(output)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_scan_rccl_logs_orte_error(self, mock_fail_test):
        """Test log scanning with ORTE error"""
        output = """
        INFO: Test starting
        ORTE does not know how to route to destination
        """

        rccl_lib.scan_rccl_logs(output)
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_scan_rccl_logs_nccl_error(self, mock_fail_test):
        """Test log scanning with NCCL error"""
        output = """
        INFO: Test starting
        NCCL ERROR: Something went wrong
        """

        rccl_lib.scan_rccl_logs(output)
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_scan_rccl_logs_missing_bandwidth(self, mock_fail_test):
        """Test log scanning without bandwidth marker"""
        output = """
        INFO: Test starting
        Test completed but no bandwidth printed
        """

        rccl_lib.scan_rccl_logs(output)
        mock_fail_test.assert_called_with(
            'RCCL test did not complete successfully, no bandwidth numbers printed - pls check'
        )

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bus_bw_success(self, mock_fail_test):
        """Test successful bus bandwidth validation"""
        test_name = "all_reduce_perf"
        output = [
            {
                "name": "all_reduce_perf",
                "size": 1024,
                "type": "float",
                "inPlace": 1,
                "busBw": 90.0,
                "algBw": 45.0,
                "time": 12.3,
            }
        ]
        exp_res_dict = {"1024": {"bus_bw": 80.0}}

        rccl_lib.check_bus_bw(test_name, output, exp_res_dict)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bus_bw_failure(self, mock_fail_test):
        """Test bus bandwidth validation failure"""
        test_name = "all_reduce_perf"
        output = [
            {
                "name": "all_reduce_perf",
                "size": 1024,
                "type": "float",
                "inPlace": 1,
                "busBw": 70.0,  # Below threshold
                "algBw": 35.0,
                "time": 12.3,
            }
        ]
        exp_res_dict = {
            "1024": {"bus_bw": 80.0}  # 95% threshold would be 76.0
        }

        rccl_lib.check_bus_bw(test_name, output, exp_res_dict)
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bus_bw_alltoall(self, mock_fail_test):
        """Test bus bandwidth validation for alltoall (out-of-place)"""
        test_name = "alltoall"
        output = [
            {
                "name": "alltoall",
                "size": 1024,
                "type": "float",
                "inPlace": 0,  # Out-of-place for alltoall
                "busBw": 90.0,
                "algBw": 45.0,
                "time": 12.3,
            }
        ]
        exp_res_dict = {"1024": {"bus_bw": 80.0}}

        rccl_lib.check_bus_bw(test_name, output, exp_res_dict)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bw_dip_success(self, mock_fail_test):
        """Test successful bandwidth dip validation"""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "inPlace": 1, "busBw": 80.0},
            {"size": 2048, "inPlace": 1, "busBw": 85.0},  # Increasing BW
            {"size": 4096, "inPlace": 1, "busBw": 90.0},  # Still increasing
        ]
        exp_res_dict = {"1024": {"bus_bw": 75.0}, "2048": {"bus_bw": 80.0}, "4096": {"bus_bw": 85.0}}

        rccl_lib.check_bw_dip(test_name, output, exp_res_dict)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bw_dip_failure(self, mock_fail_test):
        """Test bandwidth dip detection failure"""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "inPlace": 1, "busBw": 100.0},
            {"size": 2048, "inPlace": 1, "busBw": 90.0},  # Significant drop
        ]
        exp_res_dict = {"1024": {"bus_bw": 95.0}, "2048": {"bus_bw": 85.0}}

        rccl_lib.check_bw_dip(test_name, output, exp_res_dict)
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bw_dip_no_reference(self, mock_fail_test):
        """Test bandwidth dip check without reference data"""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "inPlace": 1, "busBw": 100.0},
            {"size": 2048, "inPlace": 1, "busBw": 50.0},  # Big drop but no reference
        ]

        rccl_lib.check_bw_dip(test_name, output, None)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_lat_dip_success(self, mock_fail_test):
        """Test successful latency dip validation"""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "inPlace": 1, "time": 10.0},
            {"size": 2048, "inPlace": 1, "time": 15.0},  # Increasing latency (normal)
            {"size": 4096, "inPlace": 1, "time": 20.0},  # Still increasing
        ]
        exp_res_dict = {"1024": {"bus_bw": 75.0}, "2048": {"bus_bw": 80.0}, "4096": {"bus_bw": 85.0}}

        rccl_lib.check_lat_dip(test_name, output, exp_res_dict)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_lat_dip_failure(self, mock_fail_test):
        """Test latency dip detection failure"""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "inPlace": 1, "time": 20.0},
            {"size": 2048, "inPlace": 1, "time": 15.0},  # Unexpected latency decrease
        ]
        exp_res_dict = {"1024": {"bus_bw": 95.0}, "2048": {"bus_bw": 85.0}}

        rccl_lib.check_lat_dip(test_name, output, exp_res_dict)
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_lat_dip_no_reference(self, mock_fail_test):
        """Test latency dip check without reference data"""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "inPlace": 1, "time": 20.0},
            {"size": 2048, "inPlace": 1, "time": 5.0},  # Big drop but no reference
        ]

        rccl_lib.check_lat_dip(test_name, output, None)
        mock_fail_test.assert_not_called()

    # ---------------------------------------------------------------------
    # Group-by / key-correctness regression tests (AIMVT-196)
    # ---------------------------------------------------------------------

    def test_group_rccl_results_groups_and_sorts(self):
        """Rows are grouped by (type, inPlace) and sorted ascending by size."""
        rows = [
            {"size": 2048, "type": "float", "inPlace": 1, "busBw": 20.0},
            {"size": 1024, "type": "float", "inPlace": 1, "busBw": 10.0},
            {"size": 1024, "type": "float", "inPlace": 0, "busBw": 9.0},
            {"size": 1024, "type": "bfloat16", "inPlace": 1, "busBw": 11.0},
        ]
        groups = rccl_lib.group_rccl_results(rows)
        # Three distinct (type, inPlace) groups
        self.assertEqual(
            set(groups.keys()),
            {("float", 1), ("float", 0), ("bfloat16", 1)},
        )
        # float/in-place group sorted ascending by size
        sizes = [r["size"] for r in groups[("float", 1)]]
        self.assertEqual(sizes, [1024, 2048])

    def test_convert_to_graph_dict_preserves_inplace(self):
        """In-place and out-of-place rows for the same size must NOT collapse."""
        result_dict = {
            "all_reduce_perf-NCCL_ALGO=Ring": [
                {"size": 1024, "name": "AllReduce", "type": "float", "inPlace": 0,
                 "busBw": 10.0, "algBw": 5.0, "time": 1.0},
                {"size": 1024, "name": "AllReduce", "type": "float", "inPlace": 1,
                 "busBw": 99.0, "algBw": 50.0, "time": 2.0},
            ]
        }
        graph = rccl_lib.convert_to_graph_dict(result_dict)
        # Two separate series, one per inPlace orientation
        self.assertEqual(len(graph), 2)
        in_series = next(k for k in graph if "in_place" in k)
        out_series = next(k for k in graph if "out_of_place" in k)
        self.assertEqual(graph[out_series][1024]["bus_bw"], 10.0)
        self.assertEqual(graph[in_series][1024]["bus_bw"], 99.0)

    def test_convert_to_graph_dict_preserves_dtype(self):
        """Different data types for the same size must NOT collapse."""
        result_dict = {
            "all_reduce_perf-NCCL_ALGO=Ring": [
                {"size": 1024, "name": "AllReduce", "type": "float", "inPlace": 1,
                 "busBw": 10.0, "algBw": 5.0, "time": 1.0},
                {"size": 1024, "name": "AllReduce", "type": "bfloat16", "inPlace": 1,
                 "busBw": 20.0, "algBw": 10.0, "time": 1.0},
            ]
        }
        graph = rccl_lib.convert_to_graph_dict(result_dict)
        self.assertEqual(len(graph), 2)
        float_series = next(k for k in graph if "type=float" in k)
        bf16_series = next(k for k in graph if "type=bfloat16" in k)
        self.assertEqual(graph[float_series][1024]["bus_bw"], 10.0)
        self.assertEqual(graph[bf16_series][1024]["bus_bw"], 20.0)

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bw_dip_multi_dtype_no_false_positive(self, mock_fail_test):
        """A data-type boundary must not be mistaken for a bandwidth dip."""
        test_name = "all_reduce_perf"
        # float ascending then bfloat16 restarting at the smallest size.
        output = [
            {"size": 1024, "type": "float", "inPlace": 1, "busBw": 10.0, "time": 1.0},
            {"size": 2048, "type": "float", "inPlace": 1, "busBw": 20.0, "time": 2.0},
            {"size": 1024, "type": "bfloat16", "inPlace": 1, "busBw": 10.0, "time": 1.0},
            {"size": 2048, "type": "bfloat16", "inPlace": 1, "busBw": 20.0, "time": 2.0},
        ]
        ref = {"1024": {"bus_bw": 1}, "2048": {"bus_bw": 1}}
        rccl_lib.check_bw_dip(test_name, output, ref)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bw_dip_detects_real_dip_within_dtype(self, mock_fail_test):
        """A genuine within-data-type bandwidth dip is still flagged."""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "type": "float", "inPlace": 1, "busBw": 100.0, "time": 1.0},
            {"size": 2048, "type": "float", "inPlace": 1, "busBw": 50.0, "time": 2.0},
        ]
        ref = {"1024": {"bus_bw": 1}, "2048": {"bus_bw": 1}}
        rccl_lib.check_bw_dip(test_name, output, ref)
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_lat_dip_multi_dtype_no_false_positive(self, mock_fail_test):
        """A data-type boundary must not be mistaken for a latency dip."""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "type": "float", "inPlace": 1, "busBw": 10.0, "time": 10.0},
            {"size": 2048, "type": "float", "inPlace": 1, "busBw": 20.0, "time": 20.0},
            {"size": 1024, "type": "bfloat16", "inPlace": 1, "busBw": 10.0, "time": 10.0},
            {"size": 2048, "type": "bfloat16", "inPlace": 1, "busBw": 20.0, "time": 20.0},
        ]
        ref = {"1024": {"bus_bw": 1}, "2048": {"bus_bw": 1}}
        rccl_lib.check_lat_dip(test_name, output, ref)
        mock_fail_test.assert_not_called()

    def test_format_run_command_log_entry(self):
        """Clean run record: command collapsed to one line; output + headers present."""
        command = (
            "/opt/ompi/bin/mpirun \\\n"
            "    --allow-run-as-root \\\n"
            "    -np 32 \\\n"
            "    all_reduce_perf -b 1K -e 4G"
        )
        output = "#  size  busbw\n  1024  0.02\n# Avg bus bandwidth : 100.0\n"
        entry = rccl_lib.format_run_command_log_entry("all_reduce_perf [NCCL_ALGO=Ring] -> ref_float_r0.json",
                                                      command, output)
        # Command is collapsed to a single copy/paste-able line (no backslashes/newlines).
        self.assertIn("mpirun --allow-run-as-root -np 32 all_reduce_perf -b 1K -e 4G", entry)
        self.assertNotIn("\\\n", entry)
        # Labels + raw output preserved.
        self.assertIn("# RUN : all_reduce_perf [NCCL_ALGO=Ring] -> ref_float_r0.json", entry)
        self.assertIn("MPI launch command", entry)
        self.assertIn("# Avg bus bandwidth : 100.0", entry)

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bus_bw_multi_dtype_each_compared(self, mock_fail_test):
        """Both data types are compared against the size-keyed reference."""
        test_name = "all_reduce_perf"
        output = [
            {"name": "all_reduce_perf", "size": 1024, "type": "float", "inPlace": 1,
             "busBw": 90.0, "algBw": 45.0, "time": 12.3},
            {"name": "all_reduce_perf", "size": 1024, "type": "bfloat16", "inPlace": 1,
             "busBw": 70.0, "algBw": 35.0, "time": 12.3},  # below threshold -> must fail
        ]
        exp_res_dict = {"1024": {"bus_bw": 80.0}}  # 95% threshold = 76.0
        rccl_lib.check_bus_bw(test_name, output, exp_res_dict)
        mock_fail_test.assert_called()


if __name__ == '__main__':
    unittest.main()
