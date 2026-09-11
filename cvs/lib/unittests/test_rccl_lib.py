# cvs/lib/unittests/test_rccl_lib.py
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import cvs.lib.rccl_lib as rccl_lib


class TestRcclLib(unittest.TestCase):
    @patch(
        'cvs.lib.rccl_lib.JobStep',
        new=SimpleNamespace(kind=rccl_lib.Scheduler.SPUR),
    )
    @patch('cvs.lib.rccl_lib.os.path.isdir', return_value=False)
    def test_managed_rccl_env_contains_proven_spur_defaults(self, _isdir):
        env = rccl_lib._managed_rccl_env('/opt/openmpi', {'NCCL_DEBUG': 'INFO'})
        self.assertEqual(env['PMIX_GDS_MODULE'], 'hash')
        self.assertEqual(env['OMPI_MCA_pml'], 'ucx')
        self.assertEqual(env['NCCL_SOCKET_IFNAME'], 'ens3')
        self.assertEqual(env['NCCL_IB_HCA'], 'ionic')
        self.assertEqual(env['NCCL_IB_GID_INDEX'], '1')
        self.assertEqual(env['NCCL_DMABUF_ENABLE'], '1')
        self.assertEqual(env['NCCL_NET_PLUGIN'], 'none')
        self.assertEqual(env['NCCL_DEBUG'], 'INFO')

    def test_managed_rccl_argv_sources_env_then_execs_binary(self):
        argv = rccl_lib._managed_rccl_argv(
            '/opt/rccl-tests/all_reduce_perf',
            ['-b', '8', '-e', '8G'],
            '/tmp/rccl env.sh',
        )
        self.assertEqual(argv[:2], ['bash', '-c'])
        self.assertIn("source '/tmp/rccl env.sh' && exec", argv[2])
        self.assertIn('/opt/rccl-tests/all_reduce_perf -b 8 -e 8G', argv[2])

    @patch(
        'cvs.lib.rccl_lib.JobStep',
        new=SimpleNamespace(rank=0, world_size=16, hosts=['n1', 'n2']),
    )
    def test_managed_layout_must_match_existing_srun_world(self):
        orch = MagicMock(hosts=['n1', 'n2'])
        self.assertEqual(
            rccl_lib._validate_managed_layout(orch, {'no_of_nodes': 2, 'no_of_local_ranks': 8}),
            16,
        )
        with self.assertRaisesRegex(RuntimeError, 'must match the existing srun step'):
            rccl_lib._validate_managed_layout(orch, {'no_of_nodes': 1, 'no_of_local_ranks': 8})

    @patch(
        'cvs.lib.rccl_lib.JobStep',
        new=SimpleNamespace(
            is_managed=True,
            kind=rccl_lib.Scheduler.SPUR,
            world_size=2,
            hosts=['n1', 'n2'],
        ),
    )
    @patch('cvs.lib.rccl_lib.get_model_from_rocm_smi_output')
    @patch('cvs.lib.rccl_lib.detect_rccl_output_flag', return_value='--output')
    def test_regression_managed_uses_launch_without_mpirun(self, _output_flag, _model):
        with tempfile.TemporaryDirectory() as root:
            result_file = Path(root) / 'regression.json'
            stdout = Path(root) / 'rank-0000.stdout'
            stderr = Path(root) / 'rank-0000.stderr'
            stdout.write_text('# Avg bus bandwidth : 100\n')
            stderr.write_text('')
            payload = [{'result': 'ok'}]

            orch = MagicMock(hosts=['n1', 'n2'], head_node='n1')

            def launch(argv, **_kwargs):
                Path(argv[-1]).write_text(json.dumps(payload))
                return [
                    SimpleNamespace(
                        rank=0,
                        exit_code=0,
                        timed_out=False,
                        error=None,
                        stdout_path=stdout,
                        stderr_path=stderr,
                    ),
                    SimpleNamespace(
                        rank=1,
                        exit_code=0,
                        timed_out=False,
                        error=None,
                        stdout_path=stdout,
                        stderr_path=stderr,
                    ),
                ]

            orch.launch.side_effect = launch
            shdl = MagicMock()
            shdl.exec.return_value = {'n1': 'gpu'}
            result = rccl_lib.rccl_regression(
                MagicMock(),
                shdl,
                'all_reduce_perf',
                None,
                {'no_of_nodes': 2, 'no_of_local_ranks': 1},
                {},
                {
                    'rccl_result_file': str(result_file),
                    'verify_bus_bw': 'False',
                    'verify_bw_dip': 'False',
                    'verify_lat_dip': 'False',
                },
                ['n1', 'n2'],
                ['n1', 'n2'],
                orch=orch,
            )

            self.assertEqual(result, payload)
            orch.launch.assert_called_once()
            self.assertNotIn('mpirun', ' '.join(call.args[0] for call in shdl.exec.call_args_list))

    @patch(
        'cvs.lib.rccl_lib.JobStep',
        new=SimpleNamespace(
            is_managed=True,
            kind=rccl_lib.Scheduler.SPUR,
            world_size=2,
            hosts=['n1', 'n2'],
        ),
    )
    @patch('cvs.lib.rccl_lib.aggregate_rccl_test_results', return_value=[])
    @patch('cvs.lib.rccl_lib.RcclTestsMultinodeRaw.model_validate', side_effect=lambda item: item)
    @patch('cvs.lib.rccl_lib.get_model_from_rocm_smi_output')
    @patch('cvs.lib.rccl_lib.detect_rccl_output_flag', return_value='--output')
    def test_perf_managed_uses_launch_without_mpirun(self, _output_flag, _model, _validate, _aggregate):
        with tempfile.TemporaryDirectory() as root:
            result_file = Path(root) / 'perf.json'
            stdout = Path(root) / 'rank-0000.stdout'
            stderr = Path(root) / 'rank-0000.stderr'
            stdout.write_text('# Avg bus bandwidth : 100\n')
            stderr.write_text('')
            payload = [{'name': 'all_reduce_perf', 'type': 'float'}]

            orch = MagicMock(hosts=['n1', 'n2'], head_node='n1')

            def launch(argv, **_kwargs):
                Path(argv[-1]).write_text(json.dumps(payload))
                return [
                    SimpleNamespace(
                        rank=rank,
                        exit_code=0,
                        timed_out=False,
                        error=None,
                        stdout_path=stdout,
                        stderr_path=stderr,
                    )
                    for rank in range(2)
                ]

            orch.launch.side_effect = launch
            shdl = MagicMock()
            shdl.exec.return_value = {'n1': 'gpu'}
            result = rccl_lib.rccl_perf(
                MagicMock(),
                shdl,
                'all_reduce_perf',
                None,
                {'no_of_nodes': 2, 'no_of_local_ranks': 1},
                {'data_types': ['float']},
                {
                    'rccl_result_file': str(result_file),
                    'verify_bus_bw': 'False',
                    'verify_bw_dip': 'False',
                    'verify_lat_dip': 'False',
                },
                ['n1', 'n2'],
                ['n1', 'n2'],
                orch=orch,
            )

            self.assertEqual(result, payload)
            orch.launch.assert_called_once()
            self.assertNotIn('mpirun', ' '.join(call.args[0] for call in shdl.exec.call_args_list))

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


if __name__ == '__main__':
    unittest.main()
