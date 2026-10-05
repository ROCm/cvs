# cvs/lib/unittests/test_rccl_lib.py
import os
import json
import shlex
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch
import cvs.lib.rccl_lib as rccl_lib
from cvs.core.run_layout import RunLayout


def _download_writes_suffixed_path(content, host='head'):
    """Mimic parallel-ssh's download_file: suffix the local path per host and return that path."""

    def _download(_remote, local):
        actual_path = f'{local}_{host}'
        Path(actual_path).write_text(content, encoding='utf-8')
        return {host: actual_path}

    return _download


class _FakeOrch:
    def __init__(self, ssh_port=22):
        self.exec = MagicMock()
        self.exec_on_host = MagicMock()
        self.exec_on_head = MagicMock()
        self.upload_to_head = MagicMock()
        self.download_from_head = MagicMock()
        self.all = MagicMock()
        self.head = MagicMock()
        self.ssh_port = ssh_port


class TestRcclLib(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'USER': 'cvsuser'}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        RunLayout._reset()
        self.addCleanup(RunLayout._reset)
        errors = patch.object(rccl_lib.globals, 'error_list', [])
        errors.start()
        self.addCleanup(errors.stop)

    @staticmethod
    def _shipped_config():
        config_file = Path(__file__).resolve().parents[2] / 'input/config_file/rccl/rccl_config.json'
        with config_file.open(encoding='utf-8') as stream:
            return json.load(stream)['rccl']

    @staticmethod
    def _configured_job(config):
        with patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True):
            return rccl_lib.RcclJob.from_config(_FakeOrch(), 'all_reduce_perf', config, ['n1'], ['n1'])

    @staticmethod
    def _taper_results():
        return [
            {'size': 8589934592, 'inPlace': 1, 'busBw': 340.0, 'time': 20.0},
            {'size': 17179869184, 'inPlace': 1, 'busBw': 300.0, 'time': 10.0},
        ]

    def test_expected_results_transposes_shipped_shape(self):
        config = self._shipped_config()
        job = self._configured_job(config)
        self.assertEqual(
            job._expected_results(['float']),
            {'8589934592': {'bus_bw': 330.0}, '17179869184': {'bus_bw': 350.0}},
        )

    def test_expected_results_accepts_transposed_shape(self):
        config = self._shipped_config()
        config['results'] = {'all_reduce_perf': {'1024': {'bus_bw': '80.0'}}}
        self.assertEqual(self._configured_job(config)._expected_results(['float']), {'1024': {'bus_bw': 80.0}})

    def test_expected_results_unknown_collective(self):
        job = self._configured_job(self._shipped_config())
        job.test_name = 'unknown_perf'
        self.assertIsNone(job._expected_results(['float']))

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_expected_results_invalid_bandwidth(self, mock_fail_test):
        config = self._shipped_config()
        config['results'] = {'all_reduce_perf': {'bus_bw': {'1024': 'invalid'}}}
        self.assertIsNone(self._configured_job(config)._expected_results(['float']))
        self.assertIn('rccl.results.all_reduce_perf.1024', mock_fail_test.call_args.args[0])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_invalid_bandwidth_does_not_suppress_valid_thresholds(self, mock_fail_test):
        config = self._shipped_config()
        config['results'] = {'all_reduce_perf': {'bus_bw': {'1024': 'invalid', '2048': '80.0'}}}
        job = self._configured_job(config)
        job.cvs_params['verify_bus_bw'] = 'True'
        job._verify_results([{'size': 2048, 'inPlace': 1, 'busBw': 10.0}], job._expected_results(['float']))
        self.assertEqual(mock_fail_test.call_count, 2)
        self.assertIn('rccl.results.all_reduce_perf.1024', mock_fail_test.call_args_list[0].args[0])
        self.assertIn('expected bus BW 80.0', mock_fail_test.call_args_list[1].args[0])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_expected_results_rejects_malformed_shape(self, mock_fail_test):
        config = self._shipped_config()
        for results, path in (
            ('invalid', 'rccl.results'),
            ({'all_reduce_perf': '330.0'}, 'rccl.results.all_reduce_perf'),
            ({'all_reduce_perf': {'bus_bw': '330.0'}}, 'rccl.results.all_reduce_perf.bus_bw'),
        ):
            with self.subTest(results=results):
                mock_fail_test.reset_mock()
                config['results'] = results
                self.assertIsNone(self._configured_job(config)._expected_results(['float']))
                self.assertIn(path, mock_fail_test.call_args.args[0])

    def test_legacy_results_fallback_warns(self):
        config = self._shipped_config()
        config['cvs_params']['results'] = config.pop('results')
        with self.assertLogs(rccl_lib.log, level='WARNING') as captured:
            self.assertTrue(self._configured_job(config)._expected_results(['float']))
        self.assertEqual(len(captured.output), 1)
        self.assertIn('rccl.results', captured.output[0])

    def test_legacy_results_fallback_with_empty_root_results(self):
        config = self._shipped_config()
        config['cvs_params']['results'] = config['results']
        config['results'] = {}
        job = self._configured_job(config)
        self.assertIsNone(job._expected_results(['float']))

    def test_malformed_results_report_one_failure_during_regression(self):
        config = self._shipped_config()
        config['results'] = {'all_reduce_perf': {'bus_bw': 'invalid'}}
        config['cvs_params']['verify_bus_bw'] = 'True'
        job = self._configured_job(config)
        rows = [{'size': 2048, 'inPlace': 1, 'busBw': 10.0}]
        with (
            patch.object(rccl_lib.globals, 'error_list', []),
            patch.object(job, 'prepare'),
            patch.object(job, 'execute', return_value='RCCL output'),
            patch.object(job, 'read_results', return_value=rows),
            patch.object(job, 'collect_gpu_info'),
            patch.object(job, '_save_topology_checks'),
        ):
            job.run_regression()
            self.assertEqual(len(rccl_lib.globals.error_list), 1)
            self.assertIn('rccl.results.all_reduce_perf.bus_bw', rccl_lib.globals.error_list[0])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_requested_bus_bw_without_thresholds_fails(self, mock_fail_test):
        verifier = rccl_lib.RcclVerifier('unknown_perf', self._taper_results(), None, {'verify_bus_bw': 'True'})
        verifier.check()
        mock_fail_test.assert_called_once_with(
            'No bus bandwidth thresholds for unknown_perf in rccl.results.unknown_perf'
        )

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_disabled_bus_bw_does_not_fail(self, mock_fail_test):
        verifier = rccl_lib.RcclVerifier(
            'all_reduce_perf',
            self._taper_results(),
            {'8589934592': {'bus_bw': 330.0}},
            {'verify_bus_bw': 'False', 'verify_bw_dip': 'False', 'verify_lat_dip': 'False'},
        )
        verifier.check()
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_omitted_dip_flags_are_disabled(self, mock_fail_test):
        job = self._configured_job(self._shipped_config())
        rccl_lib.RcclVerifier(job.test_name, self._taper_results(), job._expected_results(['float'])).check()
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_enabled_bus_bw_uses_shipped_thresholds(self, mock_fail_test):
        job = self._configured_job(self._shipped_config())
        job.cvs_params['verify_bus_bw'] = 'True'
        results = [
            {'size': 8589934592, 'inPlace': 1, 'busBw': 10.0, 'time': 20.0},
        ]
        job._verify_results(results, job._expected_results(['float']))
        mock_fail_test.assert_called_once()
        self.assertIn('expected bus BW 330.0', mock_fail_test.call_args.args[0])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_dip_flags_enable_their_checks(self, mock_fail_test):
        job = self._configured_job(self._shipped_config())
        expected = job._expected_results(['float'])
        for flag, failure in (
            ('verify_bw_dip', 'BusBW'),
            ('verify_lat_dip', 'latency'),
        ):
            with self.subTest(flag=flag):
                mock_fail_test.reset_mock()
                params = {'verify_bus_bw': 'False', 'verify_bw_dip': 'False', 'verify_lat_dip': 'False'}
                params[flag] = 'True'
                rccl_lib.RcclVerifier(job.test_name, self._taper_results(), expected, params).check()
                self.assertEqual(mock_fail_test.call_count, 1)
                self.assertIn(failure, mock_fail_test.call_args.args[0])
        mock_fail_test.reset_mock()
        rccl_lib.RcclVerifier(
            job.test_name,
            self._taper_results(),
            expected,
            {'verify_bus_bw': 'False', 'verify_bw_dip': 'False', 'verify_lat_dip': 'False'},
        ).check()
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_dip_checks_normalize_string_measurements(self, mock_fail_test):
        expected = {'1024': {'bus_bw': 80.0}, '2048': {'bus_bw': 85.0}}
        results = [
            {'size': 1024, 'inPlace': 1, 'busBw': '100.0', 'time': '20.0'},
            {'size': 2048, 'inPlace': 1, 'busBw': '90.0', 'time': '10.0'},
        ]
        rccl_lib.RcclVerifier(
            'all_reduce_perf', results, expected, {'verify_bw_dip': 'True', 'verify_lat_dip': 'True'}
        ).check()
        self.assertEqual(mock_fail_test.call_count, 2)

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_shipped_verification_is_disabled(self, mock_fail_test):
        job = self._configured_job(self._shipped_config())
        for flag in ('verify_bus_bw', 'verify_bw_dip', 'verify_lat_dip'):
            self.assertEqual(job.cvs_params[flag], 'False')
        rccl_lib.RcclVerifier(
            job.test_name, self._taper_results(), job._expected_results(['float']), job.cvs_params
        ).check()
        mock_fail_test.assert_not_called()

    def test_shipped_collectives_have_numeric_thresholds(self):
        config = self._shipped_config()
        job = self._configured_job(config)
        for collective in config['rccl_test_params']['rccl_collective']:
            with self.subTest(collective=collective):
                job.test_name = collective
                expected = job._expected_results(['float'])
                self.assertIsNotNone(expected)
                self.assertTrue(expected)
                for size, values in expected.items():
                    self.assertTrue(size.isdigit())
                    self.assertIsInstance(values['bus_bw'], float)

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
        self.assertTrue(rccl_lib.RcclVerifier.is_severe_wrong_corruption_error(mock_error))

        # Test with '#wrong' pattern
        class MockWrongError:
            def errors(self):
                return [{'msg': "Field validation failed: '#wrong' > 0"}]

            def __str__(self):
                return "ValidationError with wrong"

        mock_error = MockWrongError()
        self.assertTrue(rccl_lib.RcclVerifier.is_severe_wrong_corruption_error(mock_error))

        # Test fallback to string search with '#wrong' pattern
        class MockStringError:
            def errors(self):
                raise Exception("No structured errors")

            def __str__(self):
                return "ValidationError contains '#wrong' > 0"

        mock_error = MockStringError()
        self.assertTrue(rccl_lib.RcclVerifier.is_severe_wrong_corruption_error(mock_error))

        # Test normal error (should not be severe)
        class MockNormalError:
            def errors(self):
                raise Exception("No structured errors")

            def __str__(self):
                return "Normal validation error"

        mock_error = MockNormalError()
        self.assertFalse(rccl_lib.RcclVerifier.is_severe_wrong_corruption_error(mock_error))

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_scan_rccl_logs_success(self, mock_fail_test):
        """Test successful log scanning"""
        output = """
        INFO: Test starting
        NCCL WARN: Performance warning
        # Avg bus bandwidth    :   85.5
        Test completed successfully
        """

        rccl_lib.RcclVerifier.scan_logs(output)
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_scan_rccl_logs_orte_error(self, mock_fail_test):
        """Test log scanning with ORTE error"""
        output = """
        INFO: Test starting
        ORTE does not know how to route to destination
        """

        rccl_lib.RcclVerifier.scan_logs(output)
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_scan_rccl_logs_nccl_error(self, mock_fail_test):
        """Test log scanning with NCCL error"""
        output = """
        INFO: Test starting
        NCCL ERROR: Something went wrong
        """

        rccl_lib.RcclVerifier.scan_logs(output)
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_scan_rccl_logs_missing_bandwidth(self, mock_fail_test):
        """Test log scanning without bandwidth marker"""
        output = """
        INFO: Test starting
        Test completed but no bandwidth printed
        """

        rccl_lib.RcclVerifier.scan_logs(output)
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

        rccl_lib.RcclVerifier(test_name, output, exp_res_dict).check_bus_bw()
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

        rccl_lib.RcclVerifier(test_name, output, exp_res_dict).check_bus_bw()
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

        rccl_lib.RcclVerifier(test_name, output, exp_res_dict).check_bus_bw()
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

        rccl_lib.RcclVerifier(test_name, output, exp_res_dict).check_bw_dip()
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

        rccl_lib.RcclVerifier(test_name, output, exp_res_dict).check_bw_dip()
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bw_dip_no_reference(self, mock_fail_test):
        """Test bandwidth dip check without reference data"""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "inPlace": 1, "busBw": 100.0},
            {"size": 2048, "inPlace": 1, "busBw": 50.0},  # Big drop but no reference
        ]

        rccl_lib.RcclVerifier(test_name, output, None).check_bw_dip()
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_bw_dip_no_results(self, mock_fail_test):
        verifier = rccl_lib.RcclVerifier('all_reduce_perf', [], {'1024': {'bus_bw': 80.0}})
        with self.assertLogs(rccl_lib.log, level='WARNING') as captured:
            verifier.check_bw_dip()
        self.assertIn('No RCCL results available for BW dip check', captured.output[0])
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

        rccl_lib.RcclVerifier(test_name, output, exp_res_dict).check_lat_dip()
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

        rccl_lib.RcclVerifier(test_name, output, exp_res_dict).check_lat_dip()
        mock_fail_test.assert_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_lat_dip_no_reference(self, mock_fail_test):
        """Test latency dip check without reference data"""
        test_name = "all_reduce_perf"
        output = [
            {"size": 1024, "inPlace": 1, "time": 20.0},
            {"size": 2048, "inPlace": 1, "time": 5.0},  # Big drop but no reference
        ]

        rccl_lib.RcclVerifier(test_name, output, None).check_lat_dip()
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_check_lat_dip_no_results(self, mock_fail_test):
        verifier = rccl_lib.RcclVerifier('all_reduce_perf', [], {'1024': {'bus_bw': 80.0}})
        with self.assertLogs(rccl_lib.log, level='WARNING') as captured:
            verifier.check_lat_dip()
        self.assertIn('No RCCL results available for latency dip check', captured.output[0])
        mock_fail_test.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_verifier_records_empty_results_without_running_checks(self, mock_fail_test):
        verifier = rccl_lib.RcclVerifier(
            'all_reduce_perf',
            [],
            {'1024': {'bus_bw': 100}},
            {'verify_bus_bw': 'True', 'verify_bw_dip': 'True', 'verify_lat_dip': 'True'},
        )

        verifier.check()

        mock_fail_test.assert_called_once_with('RCCL test all_reduce_perf produced no result rows')

    def _openmpi(self):
        openmpi = rccl_lib.OpenMPI({'mpi_dir': '/opt/ompi', 'mpi_oob_port': 'eth0'})
        openmpi.pml = 'ob1'
        openmpi.prepared = True
        return openmpi

    def _srun(self, nodes, no_of_nodes, local_ranks, global_ranks, openmpi=None, orch=None):
        return rccl_lib.Srun(
            openmpi or self._openmpi(),
            orch or _FakeOrch(),
            nodes,
            no_of_nodes,
            local_ranks,
            global_ranks,
        )

    def test_mpirun_command(self):
        launcher = rccl_lib.MpiRun(self._openmpi(), _FakeOrch(), ['n1', 'n2'], ['v1', 'v2'], 16)
        cmd = launcher.command('all_reduce_perf')
        self.assertIn('mpirun', cmd)
        self.assertIn('--hostfile /tmp/rccl_hosts_file_', cmd)
        self.assertIn('--mca pml ob1', cmd)
        self.assertNotIn('--mpi=pmix', cmd)
        self.assertNotIn('spur run', cmd)

    def test_mpirun_command_env_overrides_win_over_env_file(self):
        launcher = rccl_lib.MpiRun(self._openmpi(), _FakeOrch(), ['n1', 'n2'], ['v1', 'v2'], 16)
        cmd = launcher.command('all_reduce_perf', '/home/user/env.sh', {'NCCL_ALGO': 'Ring'})
        self.assertIn('source /home/user/env.sh && export NCCL_ALGO=Ring && all_reduce_perf', cmd)
        self.assertNotIn('-x NCCL_ALGO', cmd)

    def test_mpirun_command_wraps_binary_without_env_file(self):
        launcher = rccl_lib.MpiRun(self._openmpi(), _FakeOrch(), ['n1', 'n2'], ['v1', 'v2'], 16)
        self.assertIn("bash -c all_reduce_perf", launcher.command('all_reduce_perf'))

    @patch('cvs.lib.rccl_lib.scheduler_hosts', return_value=['n1', 'n2'])
    @patch('cvs.lib.rccl_lib.Srun._cpus_per_nested_task', return_value=None)
    @patch('cvs.lib.rccl_lib.detect_scheduler', return_value=rccl_lib.Scheduler.SPUR)
    def test_srun_command_on_spur(self, _sched, _cpus, _hosts):
        cmd = self._srun(['n1', 'n2'], 2, 8, 16).command('all_reduce_perf')
        self.assertIn('spur run', cmd)
        self.assertIn('--overlap', cmd)
        self.assertIn('--mpi=pmix', cmd)
        self.assertIn('--gpu-bind=none', cmd)
        self.assertIn('-N 2', cmd)
        self.assertIn('-n 16', cmd)
        self.assertIn('--ntasks-per-node 8', cmd)
        self.assertNotIn('-w ', cmd)
        self.assertNotIn('--jobid', cmd)
        self.assertNotIn('--hostfile', cmd)
        self.assertNotIn('--mca', cmd)
        self.assertNotIn('mpirun', cmd)

    @patch('cvs.lib.rccl_lib.Srun._cpus_per_nested_task', return_value=None)
    @patch('cvs.lib.rccl_lib.detect_scheduler', return_value=rccl_lib.Scheduler.SLURM)
    def test_srun_command_on_slurm(self, _sched, _cpus):
        cmd = self._srun(['n1', 'n2'], 2, 8, 16).command('all_reduce_perf')
        self.assertTrue(cmd.startswith('srun '))
        self.assertIn('--overlap', cmd)
        self.assertIn('--mpi=pmix', cmd)
        self.assertIn('-w n1,n2', cmd)
        self.assertNotIn('spur run', cmd)
        self.assertNotIn('--jobid', cmd)

    @patch('cvs.lib.rccl_lib.Srun._cpus_per_nested_task', return_value=None)
    @patch('cvs.lib.rccl_lib.detect_scheduler', return_value=rccl_lib.Scheduler.SLURM)
    def test_srun_pairwise_nodelist_on_slurm(self, _sched, _cpus):
        cmd = self._srun(['ref', 'cand'], 2, 8, 16).command('all_reduce_perf')
        self.assertIn('-w ref,cand', cmd)

    @patch('cvs.lib.rccl_lib.scheduler_hosts', return_value=['n1', 'n2', 'n3'])
    @patch('cvs.lib.rccl_lib.Srun._cpus_per_nested_task', return_value=None)
    @patch('cvs.lib.rccl_lib.detect_scheduler', return_value=rccl_lib.Scheduler.SPUR)
    def test_srun_spur_rejects_subset_nodelist(self, _sched, _cpus, _hosts):
        with self.assertRaisesRegex(RuntimeError, 'does not apply --nodelist'):
            self._srun(['ref', 'cand'], 2, 8, 16).command('all_reduce_perf')

    def test_wrap_rccl_test_cmd_env_overrides(self):
        wrapped = self._srun(['n1'], 1, 1, 1).command(
            '/opt/all_reduce_perf -g 8',
            '/home/user/env.sh',
            {'NCCL_ALGO': 'Ring'},
        )
        self.assertIn('source /home/user/env.sh && export NCCL_ALGO=Ring &&', wrapped)
        self.assertNotRegex(wrapped, r'export NCCL_ALGO=Ring.*source ')
        self.assertIn('bash -c ', wrapped)

    def test_mpi_install_prefix_strips_bin(self):
        self.assertEqual(rccl_lib.OpenMPI._install_prefix('/opt/openmpi'), '/opt/openmpi')
        self.assertEqual(rccl_lib.OpenMPI._install_prefix('/opt/openmpi/bin'), '/opt/openmpi')
        self.assertEqual(rccl_lib.OpenMPI._install_prefix('/opt/openmpi/bin/'), '/opt/openmpi')

    @patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True)
    def test_rccl_job_from_config_composes_openmpi_and_srun(self, _managed):
        config = {
            'env_source_script': '/tmp/ainic.sh',
            'mpi_params': {
                'mpi_dir': '/opt/openmpi/bin',
                'no_of_nodes': '2',
                'no_of_local_ranks': '4',
                'mpi_oob_port': 'ens3',
            },
            'rccl_test_params': {'rccl_tests_dir': '/opt/rccl-tests/build'},
            'cvs_params': {'cvs_exec_timeout': '600'},
        }
        job = rccl_lib.RcclJob.from_config(_FakeOrch(), 'all_reduce_perf', config, ['n1', 'n2'], ['v1', 'v2'])
        self.assertEqual(job.env_file, '/tmp/ainic.sh')
        self.assertEqual(job.head_node, 'n1')
        self.assertEqual(job.no_of_global_ranks, 8)
        self.assertEqual(job.openmpi.oob_port, 'ens3')
        self.assertEqual(job.cvs_exec_timeout, 600)
        self.assertEqual(job.topology_check_mode, 'warn')
        self.assertIsInstance(job.openmpi, rccl_lib.OpenMPI)
        self.assertIsInstance(job.launcher, rccl_lib.Srun)

    def test_rccl_job_uses_orchestrator_head_node_regardless_of_list_order(self):
        orch = _FakeOrch()
        orch.head_node = 'n2'
        config = {'mpi_params': {}, 'rccl_test_params': {}, 'cvs_params': {}}

        job = rccl_lib.RcclJob.from_config(orch, 'all_reduce_perf', config, ['n1', 'n2'], ['v1', 'v2'])
        self.assertEqual(job.head_node, 'n2')

    @patch.object(rccl_lib.RcclJob, '_prepare_result_directory')
    @patch.object(rccl_lib.RcclJob, '_detect_output_flag', return_value='-X')
    @patch('cvs.lib.rccl_lib.Srun.prepare', autospec=True)
    @patch('cvs.lib.rccl_lib.OpenMPI.prepare', autospec=True)
    @patch.object(rccl_lib.RcclJob, '_require_spur_job_step')
    @patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True)
    def test_rccl_job_prepare_is_idempotent(
        self, _managed, require_step, prepare_openmpi, prepare_srun, detect_output, prepare_directory
    ):
        job = rccl_lib.RcclJob(
            _FakeOrch(),
            'all_reduce_perf',
            '/dev/null',
            {'mpi_pml': 'ob1'},
            {},
            {},
            ['n1'],
            ['v1'],
        )
        self.assertIs(job.prepare(), job)
        self.assertIs(job.prepare(), job)
        prepare_directory.assert_called_once()
        require_step.assert_called_once()
        prepare_openmpi.assert_called_once()
        prepare_srun.assert_called_once()
        detect_output.assert_called_once()

    def test_openmpi_process_env_cmds_from_mpi_params(self):
        openmpi = rccl_lib.OpenMPI({'mpi_dir': '/opt/openmpi/bin', 'mpi_oob_port': 'ens3'})
        openmpi.pml = 'ob1'
        openmpi.ucx_env = {'UCX_TLS': 'rc,self,sm,tcp'}
        cmds = self._srun(['n1'], 1, 1, 1, openmpi=openmpi).prepare().mpi_init_exports()
        joined = ' && '.join(cmds)
        self.assertIn('export OPAL_PREFIX=/opt/openmpi', joined)
        self.assertIn('export OMPI_MCA_btl_tcp_if_include=ens3', joined)
        self.assertIn('export OMPI_MCA_oob_tcp_if_include=ens3', joined)
        self.assertIn('export OMPI_MCA_pml=ob1', joined)
        self.assertIn('export PMIX_MCA_gds=hash', joined)
        self.assertIn('OMPI_MCA_orte_create_session_dirs=0', joined)
        self.assertIn('export UCX_TLS=rc,self,sm,tcp', joined)
        self.assertNotIn('NCCL_', joined)

    @patch.dict(os.environ, {'USER': 'cvsuser', 'SLURM_JOB_ID': '4242'}, clear=False)
    def test_srun_prepare_creates_session_dir_once_per_node(self):
        orch = _FakeOrch()
        launcher = self._srun(['n1', 'n2'], 2, 8, 16, orch=orch)
        self.assertIs(launcher.prepare(), launcher)
        launcher.prepare()
        self.assertEqual(launcher.session_dir, '/tmp/cvsuser/ompi-4242')
        orch.exec.assert_called_once()
        mkdir_cmd = orch.exec.call_args[0][0]
        self.assertIn('mkdir -p -m 700 /tmp/cvsuser', mkdir_cmd)
        self.assertIn('/tmp/cvsuser/ompi-4242', mkdir_cmd)

    @patch.dict(os.environ, {'USER': 'cvsuser', 'SLURM_JOB_ID': '4242'}, clear=False)
    def test_srun_exports_literal_session_dir(self):
        joined = ' && '.join(self._srun(['n1'], 1, 8, 8).prepare().mpi_init_exports())
        self.assertIn('export OMPI_MCA_orte_tmpdir_base=/tmp/cvsuser/ompi-4242', joined)
        self.assertIn('export OMPI_MCA_orte_top_session_dir=/tmp/cvsuser/ompi-4242', joined)
        self.assertNotIn('TMPDIR=', joined)
        self.assertNotIn('export TMP=', joined)
        self.assertNotIn('mkdir', joined)
        self.assertNotIn('$$', joined)

    def test_srun_cleanup_removes_session_dir(self):
        orch = _FakeOrch()
        launcher = self._srun(['n1'], 1, 8, 8, orch=orch).prepare()
        session_dir = launcher.session_dir
        launcher.cleanup()
        self.assertIn(f'rm -rf {session_dir}', orch.exec.call_args[0][0])
        self.assertEqual(launcher.session_dir, '')

    def test_wrap_rccl_test_cmd_mpi_setup_before_env_script(self):
        openmpi = rccl_lib.OpenMPI({'mpi_dir': '/opt/openmpi', 'mpi_oob_port': 'ens3'})
        openmpi.pml = 'ob1'
        wrapped = self._srun(['n1'], 1, 1, 1, openmpi=openmpi).command(
            '/opt/all_reduce_perf -g 8',
            '/home/user/ainic_env_script.sh',
            {'NCCL_ALGO': 'Ring'},
        )
        source_at = wrapped.find('source /home/user/ainic_env_script.sh')
        ompi_at = wrapped.find('OMPI_MCA_btl_tcp_if_include=ens3')
        nccl_at = wrapped.find('export NCCL_ALGO=Ring')
        self.assertGreater(source_at, 0)
        self.assertGreater(ompi_at, 0)
        self.assertLess(ompi_at, source_at)
        self.assertGreater(nccl_at, source_at)

    def test_mpirun_normalizes_mpi_dir_bin(self):
        launcher = rccl_lib.MpiRun(
            rccl_lib.OpenMPI({'mpi_dir': '/opt/openmpi/bin'}),
            _FakeOrch(),
            ['n1'],
            ['v1'],
            8,
        )
        cmd = launcher.command('all_reduce_perf')
        self.assertIn('/opt/openmpi/bin/mpirun', cmd)
        self.assertNotIn('/opt/openmpi/bin/bin/mpirun', cmd)

    def test_require_spur_job_step_rejects_bare_allocation(self):
        env = {'SPUR_JOB_ID': '99', 'SLURM_JOB_ID': '99'}
        with patch.dict('os.environ', env, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'requires a SPUR job step'):
                rccl_lib.RcclJob._require_spur_job_step()

    def test_require_spur_job_step_allows_managed_step(self):
        env = {'SPUR_JOB_ID': '99', 'SLURM_JOB_ID': '99', 'SLURM_STEP_ID': '0', 'SLURM_PROCID': '0'}
        with patch.dict('os.environ', env, clear=True):
            rccl_lib.RcclJob._require_spur_job_step()

    def test_require_spur_job_step_preserves_non_spur_runs(self):
        for env in ({'CVS_SCHEDULER': 'bare_metal'}, {'SLURM_JOB_ID': '99'}):
            with self.subTest(env=env), patch.dict('os.environ', env, clear=True):
                rccl_lib.RcclJob._require_spur_job_step()

    def _job(self, orch, head='head'):
        with patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True):
            return rccl_lib.RcclJob(
                orch,
                'all_reduce_perf',
                '/dev/null',
                {'mpi_pml': 'ob1'},
                {},
                {'cvs_exec_timeout': 60, 'rccl_result_file': '/shared/rccl.json'},
                [head],
                [head],
            )

    def test_exec_rccl_launch_nonzero_exit_does_not_scan(self):
        rccl_lib.globals.error_list = []
        orch = _FakeOrch()
        orch.exec_on_head.return_value = {
            'head': {'output': '# Avg bus bandwidth    : 1.8\n', 'exit_code': 3},
        }
        job = self._job(orch)
        with (
            patch.object(job, 'launch_command', return_value='spur run --overlap --mpi=pmix -- all_reduce_perf'),
            patch('cvs.lib.rccl_lib.RcclVerifier.scan_logs') as mock_scan,
        ):
            result = job.execute('all_reduce_perf', 'unit')
        self.assertIsNone(result)
        mock_scan.assert_not_called()
        orch.exec.assert_called()
        orch.exec_on_head.assert_called_once()
        self.assertTrue(orch.exec_on_head.call_args.kwargs.get('detailed'))
        self.assertTrue(any('exit code 3' in msg for msg in rccl_lib.globals.error_list))

    def test_exec_rccl_launch_timeout_with_bandwidth_is_failure(self):
        rccl_lib.globals.error_list = []
        orch = _FakeOrch()
        orch.exec_on_head.return_value = {
            'head': {'output': '# Avg bus bandwidth    : 1.8\nABORT: Timeout\n', 'exit_code': -1},
        }
        job = self._job(orch)
        with patch.object(job, 'launch_command', return_value='spur run --overlap --'):
            result = job.execute('all_reduce_perf', 'unit')
        self.assertIsNone(result)
        self.assertTrue(any('exit code -1' in msg for msg in rccl_lib.globals.error_list))

    def test_exec_rccl_launch_success_scans_logs(self):
        rccl_lib.globals.error_list = []
        orch = _FakeOrch()
        orch.exec_on_head.return_value = {
            'head': {'output': '# Avg bus bandwidth    : 1.8\n', 'exit_code': 0},
        }
        job = self._job(orch)
        with patch.object(job, 'launch_command', return_value='spur run --overlap --'):
            result = job.execute('all_reduce_perf', 'unit')
        self.assertIn('Avg bus bandwidth', result)
        self.assertEqual(rccl_lib.globals.error_list, [])
        orch.exec.assert_not_called()

    def test_rccl_regression_failed_launch_preserves_graph_generation(self):
        env = {
            'SPUR_JOB_ID': '99',
            'SLURM_JOB_ID': '99',
            'SLURM_STEP_ID': '0',
            'SLURM_PROCID': '0',
            'SPUR_NODES': 'n1,n2',
        }
        orch = _FakeOrch()
        orch.exec_on_head.side_effect = [
            {'n1': 'NEW'},
            {'n1': {'output': '# Avg bus bandwidth : 1.8\n', 'exit_code': 3}},
        ]
        with (
            patch.dict('os.environ', env, clear=True),
            patch.object(rccl_lib.globals, 'error_list', []),
            patch.object(rccl_lib.RcclJob, 'read_results') as read_results,
            patch.object(rccl_lib.RcclJob, '_prepare_result_directory'),
        ):
            failed_results = rccl_lib.RcclJob(
                orch,
                'all_reduce_perf',
                '/dev/null',
                {'no_of_nodes': 2, 'no_of_local_ranks': 1, 'mpi_pml': 'ob1', 'mpi_dir': '/opt/openmpi'},
                {},
                {'rccl_result_file': '/shared/rccl.json'},
                ['n1', 'n2'],
                ['n1', 'n2'],
            ).run_regression()
            self.assertEqual(failed_results, [])
            read_results.assert_not_called()
            self.assertTrue(any('exit code 3' in msg for msg in rccl_lib.globals.error_list))
            orch.exec.assert_called()

        successful_results = [
            {'size': 1024, 'name': 'all_reduce_perf', 'inPlace': 1, 'busBw': 100.0, 'algBw': 90.0, 'time': 1.0}
        ]
        graph = rccl_lib.convert_to_graph_dict({'failed': failed_results, 'successful': successful_results})
        self.assertEqual(graph['failed'], {})
        self.assertEqual(graph['successful'][1024], {'bus_bw': 100.0, 'alg_bw': 90.0, 'time': 1.0})

    def test_cpus_per_nested_task_from_slurm_env(self):
        with patch.dict('os.environ', {'SLURM_CPUS_ON_NODE': '236'}, clear=True):
            self.assertEqual(rccl_lib.Srun._cpus_per_nested_task(8), 29)

    def test_cpus_per_nested_task_override(self):
        with patch.dict('os.environ', {'RCCL_CPUS_PER_TASK': '16', 'SLURM_CPUS_ON_NODE': '236'}, clear=True):
            self.assertEqual(rccl_lib.Srun._cpus_per_nested_task(8), 16)

    def test_cleanup_does_not_pkill_scheduler(self):
        orch = _FakeOrch()
        self._job(orch)._cleanup_stale_processes('unit-test')
        commands = [call.args[0] for call in orch.exec.call_args_list]
        joined = ' '.join(commands)
        self.assertNotRegex(joined, r'(^|[^a-z])srun([^a-z]|$)')
        self.assertNotIn('spur', joined)
        self.assertIn('all_reduce_perf', joined)
        self.assertEqual(orch.exec.call_count, 3)
        orch.all.exec.assert_not_called()
        orch.head.exec.assert_not_called()

    def test_mpirun_uses_container_ssh_port_and_head_environment(self):
        orch = _FakeOrch(ssh_port=2224)
        launcher = rccl_lib.MpiRun(self._openmpi(), orch, ['n1', 'n2'], ['v1', 'v2'], 16)
        cmd = launcher.command('all_reduce_perf')
        args = shlex.split(cmd)
        self.assertEqual(
            args[args.index('plm_rsh_args') + 1],
            '-p 2224 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null',
        )
        self.assertEqual(orch.exec_on_head.call_count, 2)
        self.assertIn('v1 slots=8\nv2 slots=8', orch.exec_on_head.call_args.args[0])
        orch.head.exec.assert_not_called()
        orch.all.exec.assert_not_called()

    def test_openmpi_discovers_ucx_in_orchestrator_environment(self):
        orch = _FakeOrch()
        orch.exec_on_head.return_value = {'head': '1'}
        openmpi = rccl_lib.OpenMPI({'mpi_pml': 'auto'})
        with patch.object(rccl_lib.linux_utils, 'get_ucx_net_devices', return_value='rdma0:1') as devices:
            openmpi.prepare(orch, 'head')
        devices.assert_called_once_with(orch)
        self.assertEqual(openmpi.pml, 'ucx')
        self.assertEqual(openmpi.ucx_env['UCX_NET_DEVICES'], 'rdma0:1')
        orch.exec_on_head.assert_called_once()
        orch.head.exec.assert_not_called()

    def test_managed_prepare_rejects_result_directory_invisible_to_cvs(self):
        orch = _FakeOrch()
        orch.exec_on_head.return_value = {'head': {'output': '', 'exit_code': 0}}
        job = self._job(orch)
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(job.openmpi, 'prepare') as prepare_mpi:
            job.cvs_params['rccl_result_file'] = f'{tmpdir}/results.json'
            with self.assertRaisesRegex(RuntimeError, '--workspace or CVS_WORKSPACE'):
                job.prepare()
            prepare_mpi.assert_not_called()
            self.assertFalse(job.prepared)
            self.assertEqual(list(Path(tmpdir).iterdir()), [])

    def test_managed_prepare_checks_remote_result_directory_failures(self):
        for result in ({}, {'head': {'output': 'permission denied', 'exit_code': 1}}, {'head': 'no exit code'}):
            with self.subTest(result=result):
                orch = _FakeOrch()
                orch.exec_on_head.return_value = result
                job = self._job(orch)
                with self.assertRaisesRegex(RuntimeError, 'Cannot use RCCL result directory /shared'):
                    job.prepare()
                self.assertFalse(job.prepared)

    def test_managed_prepare_reports_head_connectivity_failure(self):
        orch = _FakeOrch()
        orch.exec_on_head.side_effect = OSError('head unreachable')
        with self.assertRaisesRegex(RuntimeError, 'head unreachable.*--workspace or CVS_WORKSPACE'):
            self._job(orch).prepare()

    def test_managed_result_directory_accepts_shared_writable_path_and_cleans_probe(self):
        orch = _FakeOrch()
        job = self._job(orch)
        with tempfile.TemporaryDirectory(prefix='rccl shared ') as tmpdir:
            job.cvs_params['rccl_result_file'] = f'{tmpdir}/results.json'
            sentinel = Path(tmpdir) / '.cvs-rccl-probe'
            orch.exec_on_head.return_value = {'head': {'output': '', 'exit_code': 0}}
            orch.download_from_head.side_effect = _download_writes_suffixed_path('probe')

            with patch.object(rccl_lib.uuid, 'uuid4', return_value=MagicMock(hex='probe')):
                job._prepare_result_directory()
            command = orch.exec_on_head.call_args_list[0].args[0]
            self.assertIn(shlex.quote(tmpdir), command)
            self.assertIn(shlex.quote(str(sentinel)), command)
            remote_arg, local_arg = orch.download_from_head.call_args.args
            self.assertEqual(remote_arg, str(sentinel))
            self.assertFalse(Path(local_arg).exists())
            self.assertFalse(Path(f'{local_arg}_head').exists())
            orch.all.exec.assert_not_called()
            orch.head.exec.assert_not_called()

    def test_managed_result_directory_rejects_mismatched_sentinel(self):
        orch = _FakeOrch()
        orch.exec_on_head.return_value = {'head': {'output': '', 'exit_code': 0}}
        orch.download_from_head.side_effect = _download_writes_suffixed_path('different filesystem')
        job = self._job(orch)
        with tempfile.TemporaryDirectory() as tmpdir:
            job.cvs_params['rccl_result_file'] = f'{tmpdir}/results.json'
            with patch.object(rccl_lib.uuid, 'uuid4', return_value=MagicMock(hex='probe')):
                with self.assertRaisesRegex(RuntimeError, 'sentinel is not visible'):
                    job._prepare_result_directory()

    def test_non_managed_result_directory_also_verifies_sentinel_via_download(self):
        orch = _FakeOrch()
        orch.exec_on_head.return_value = {'head': {'output': '', 'exit_code': 0}}
        orch.download_from_head.side_effect = _download_writes_suffixed_path('probe')
        job = self._job(orch)
        job.managed = False
        with patch.object(rccl_lib.uuid, 'uuid4', return_value=MagicMock(hex='probe')):
            job._prepare_result_directory()
        self.assertIn('mkdir -p -- /shared', orch.exec_on_head.call_args_list[0].args[0])
        orch.download_from_head.assert_called_once()

    def test_non_managed_result_directory_rejects_unmounted_result_path(self):
        orch = _FakeOrch()
        orch.exec_on_head.return_value = {'head': {'output': '', 'exit_code': 0}}
        job = self._job(orch)
        job.managed = False
        with self.assertRaisesRegex(RuntimeError, 'bind-mount the result directory'):
            job._prepare_result_directory()

    def test_default_result_file_uses_run_layout(self):
        job = self._job(_FakeOrch())
        job.cvs_params.pop('rccl_result_file')
        with patch('cvs.core.run_layout.RunLayout.get') as layout:
            layout.return_value.run_dir = Path('/shared/cvs_runs/1234')
            self.assertEqual(job.result_file, '/shared/cvs_runs/1234/rccl_result_file.json')

    def test_result_transfers_use_head_host_handle(self):
        orch = _FakeOrch()
        job = self._job(orch)
        payload = [{'size': 1024, 'busBw': 123}]
        uploads = []

        def upload(local_path, remote_path):
            uploads.append((json.loads(Path(local_path).read_text()), remote_path))

        orch.upload_to_head.side_effect = upload
        self.assertTrue(job.save_results('/shared/rccl.json', payload, 'unit'))
        self.assertEqual(uploads, [(payload, '/shared/rccl.json')])
        self.assertFalse(Path(orch.upload_to_head.call_args.args[0]).exists())

        def download(remote_path, local_path):
            Path(local_path).write_text(json.dumps(payload), encoding='utf-8')
            return {'head': local_path}

        orch.download_from_head.side_effect = download
        self.assertEqual(job.read_results('/shared/rccl.json'), payload)
        orch.download_from_head.assert_called_once()
        orch.head.upload_file.assert_not_called()
        orch.head.download_file.assert_not_called()
        orch.exec.assert_not_called()
        orch.exec_on_head.assert_not_called()

    def test_result_save_failure_records_test_failure(self):
        orch = _FakeOrch()
        orch.upload_to_head.side_effect = OSError('shared storage unavailable')
        self.assertFalse(self._job(orch).save_results('/shared/rccl.json', [], 'unit'))
        self.assertTrue(any('shared storage unavailable' in msg for msg in rccl_lib.globals.error_list))
        self.assertFalse(Path(orch.upload_to_head.call_args.args[0]).exists())

    def test_configured_collectives_uses_sample_config_and_supports_legacy(self):
        config_path = Path(rccl_lib.__file__).parents[1] / 'input/config_file/rccl/rccl_config.json'
        config = json.loads(config_path.read_text())['rccl']
        collectives = rccl_lib.configured_collectives(config)
        self.assertIn('all_gather_perf', collectives)
        self.assertGreater(len(collectives), 1)
        config['rccl_collective'] = ['reduce_perf']
        self.assertEqual(rccl_lib.configured_collectives(config), collectives)
        del config['rccl_test_params']['rccl_collective']
        self.assertEqual(rccl_lib.configured_collectives(config), ['reduce_perf'])
        self.assertEqual(rccl_lib.configured_collectives({}), rccl_lib.DEFAULT_COLLECTIVES)

    def test_thresholds_fail_perf_and_regression_for_both_config_shapes(self):
        references = [
            {'all_reduce_perf': {'bus_bw': {'1024': '100'}}},
            {'thor': {'all_reduce_perf-float-16': {'1024': {'bus_bw': '100'}}}},
        ]
        for reference in references:
            for method in ('run_perf', 'run_regression'):
                with self.subTest(reference=reference, method=method):
                    rccl_lib.globals.error_list = []
                    config = {
                        'mpi_params': {'no_of_nodes': 2, 'no_of_local_ranks': 8},
                        'rccl_test_params': {'data_types': ['float']},
                        'cvs_params': {
                            'nic_model': 'thor',
                            'verify_bus_bw': 'True',
                            'verify_bw_dip': 'False',
                            'verify_lat_dip': 'False',
                            'rccl_result_file': '/shared/rccl.json',
                        },
                        'results': reference,
                    }
                    job = rccl_lib.RcclJob.from_config(
                        _FakeOrch(), 'all_reduce_perf', config, ['n1', 'n2'], ['v1', 'v2']
                    )
                    rows = [{'size': 1024, 'inPlace': 1, 'busBw': 50}]
                    with (
                        patch.object(job, 'prepare'),
                        patch.object(job, 'execute', return_value='# Avg bus bandwidth : 50'),
                        patch.object(job, 'read_results', return_value=rows),
                        patch.object(job, '_run_perf_dtype', return_value=(rows, [])),
                        patch.object(job, 'save_results', return_value=True),
                        patch.object(job, 'collect_gpu_info'),
                    ):
                        self.assertEqual(getattr(job, method)(), rows)
                    self.assertTrue(any('lower than expected bus BW' in msg for msg in rccl_lib.globals.error_list))

    def test_threshold_config_location_precedence_and_legacy_fallback(self):
        legacy = {'all_reduce_perf': {'bus_bw': {'1024': '100'}}}
        current = {'all_reduce_perf': {'bus_bw': {'1024': '200'}}}
        config = {'mpi_params': {}, 'rccl_test_params': {}, 'cvs_params': {'results': legacy}}
        for top_level, expected in ((None, '100'), (current, '200'), ({}, None)):
            with self.subTest(top_level=top_level):
                if top_level is not None:
                    config['results'] = top_level
                job = rccl_lib.RcclJob.from_config(_FakeOrch(), 'all_reduce_perf', config, ['n1'], ['v1'])
                resolved = job._expected_results(['float'])
                self.assertEqual(resolved, {'1024': {'bus_bw': float(expected)}} if expected else None)

    def test_nic_thresholds_are_specific_to_rank_count_and_data_types(self):
        job = self._job(_FakeOrch())
        job.cvs_params['nic_model'] = 'Broadcom'
        job.expected_results = {'thor': {'all_reduce_perf-float_half-16': {'1024': {'bus_bw': '100'}}}}
        self.assertEqual(job._expected_results(['float', 'half']), {'1024': {'bus_bw': '100'}})
        self.assertIsNone(job._expected_results(['float']))
        job.no_of_global_ranks = 8
        self.assertIsNone(job._expected_results(['float', 'half']))


class TestRcclTopology(unittest.TestCase):
    def setUp(self):
        self.requested = {'nodes': 2, 'ranks': 2, 'ranksPerNode': 1, 'gpusPerRank': 8}
        self.row = {
            'numCycle': 0,
            'name': 'AllReduce',
            'nodes': 1,
            'ranks': 2,
            'ranksPerNode': 2,
            'gpusPerRank': 8,
            'size': 8,
            'type': 'float',
            'redop': 'sum',
            'inPlace': 0,
            'time': 65.3309,
            'algBw': 0.000122,
            'busBw': 0.00023,
            'wrong': '0',
        }
        self.job = self._job()

    def _job(self, cvs_params=None, mpi_params=None, nodes=None, managed=True):
        with patch('cvs.lib.rccl_lib.is_managed_compute', return_value=managed):
            return rccl_lib.RcclJob(
                _FakeOrch(),
                'all_reduce_perf',
                '/dev/null',
                mpi_params or {'no_of_nodes': '2', 'no_of_local_ranks': '1'},
                {'threads_per_gpu': '8'},
                cvs_params or {},
                nodes or ['n1', 'n2'],
                nodes or ['n1', 'n2'],
            )

    def _mock_run(self, rows):
        saved = {}

        def upload(local_path, remote_path):
            saved[remote_path] = json.loads(Path(local_path).read_text())

        for patcher in (
            patch.object(self.job, 'prepare'),
            patch.object(self.job, 'execute', return_value='completed'),
            patch.object(self.job, 'read_results', return_value=rows),
            patch.object(self.job, 'collect_gpu_info'),
            patch.object(self.job, '_verify_results'),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.job.orch.upload_to_head.side_effect = upload
        self.job.cvs_params['rccl_result_file'] = '/results/rccl.json'
        return saved

    def _mock_download(self, payload):
        def download(remote_path, local_path):
            Path(local_path).write_text(json.dumps(payload))
            return {self.job.head_node: local_path}

        self.job.orch.download_from_head.side_effect = download

    def test_compare_exact_match(self):
        self.assertEqual(rccl_lib.compare_topology(self.requested, self.requested), [])

    def test_compare_captured_mislabeled_topology(self):
        """Values captured from a real multi-node run where the launcher's requested rank/node counts and rccl-tests' reported values were swapped."""
        original = deepcopy(self.row)
        self.assertEqual(
            rccl_lib.compare_topology(self.requested, self.row),
            ['nodes: requested 2, reported 1', 'ranksPerNode: requested 1, reported 2'],
        )
        self.assertEqual(self.row, original)

    def test_compare_default_config_mislabeled_topology(self):
        requested = {**self.requested, 'ranks': 16, 'ranksPerNode': 8, 'gpusPerRank': 1}
        reported = {**requested, 'nodes': 1, 'ranksPerNode': 16}
        self.assertEqual(
            rccl_lib.compare_topology(requested, reported),
            ['nodes: requested 2, reported 1', 'ranksPerNode: requested 8, reported 16'],
        )

    def test_compare_missing_fields(self):
        for field in rccl_lib.TOPOLOGY_FIELDS:
            with self.subTest(field=field):
                observed = {key: value for key, value in self.requested.items() if key != field}
                self.assertEqual(
                    rccl_lib.compare_topology(self.requested, observed),
                    [f'{field}: requested {self.requested[field]}, reported <missing>'],
                )

    def test_compare_omits_fields_not_requested(self):
        expected = {key: value for key, value in self.requested.items() if key != 'gpusPerRank'}
        self.assertEqual(rccl_lib.compare_topology(expected, {**self.requested, 'gpusPerRank': 1}), [])

    def test_srun_topology_uses_launch_flags(self):
        job = self._job(mpi_params={'no_of_nodes': 2, 'no_of_local_ranks': 8}, nodes=['n1', 'n2', 'n3'])
        self.assertEqual(job.launcher.topology(), {'nodes': 2, 'ranks': 16, 'ranksPerNode': 8})

    def test_mpirun_topology_uses_cluster_nodes_and_hostfile_slots(self):
        job = self._job(
            mpi_params={'no_of_nodes': 2, 'no_of_local_ranks': 16},
            nodes=['n1', 'n2', 'n3', 'n4'],
            managed=False,
        )
        self.assertEqual(job.launcher.topology(), {'nodes': 4, 'ranks': 32, 'ranksPerNode': 8})
        job.launcher.prepare()
        self.assertIn('n4 slots=8', job.orch.exec_on_head.call_args.args[0])
        self.assertEqual(job.orch.exec_on_head.call_args.args[0].count('slots=8'), 4)

    def test_expected_topology_merges_binary_gpu_count(self):
        for gpus in (1, 8):
            with self.subTest(gpus=gpus):
                self.assertEqual(self.job.expected_topology(gpus), {**self.requested, 'gpusPerRank': gpus})

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_warn_mismatch_records_without_failing(self, fail):
        with patch.object(rccl_lib.log, 'warning') as warning:
            self.job._check_reported_topology([self.row], 'float', 8)
        fail.assert_not_called()
        warning.assert_called_once()
        self.assertIn('common.cu', warning.call_args.args[1])
        self.assertEqual(len(self.job.topology_checks), 1)
        record = self.job.topology_checks[0]
        self.assertEqual(record['mode'], 'warn')
        self.assertEqual(record['verdict'], 'mismatch')
        self.assertEqual(record['requested'], self.requested)
        self.assertEqual(len(record['mismatches']), 2)

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_strict_mismatch_fails_once(self, fail):
        job = self._job(cvs_params={'topology_check': 'STRICT'})
        job._check_reported_topology([self.row], 'float', 8)
        fail.assert_called_once()
        self.assertIn('nodes: requested 2, reported 1', fail.call_args.args[0])
        self.assertIn('ranksPerNode: requested 1, reported 2', fail.call_args.args[0])
        self.assertEqual(job.topology_checks[0]['mode'], 'strict')

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_strict_clean_topology_passes(self, fail):
        for nodes, local_ranks, gpus in ((1, 1, 8), (1, 8, 1), (2, 1, 8), (2, 8, 1)):
            with self.subTest(nodes=nodes, local_ranks=local_ranks, gpus=gpus):
                job = self._job(
                    cvs_params={'topology_check': 'strict'},
                    mpi_params={'no_of_nodes': nodes, 'no_of_local_ranks': local_ranks},
                    nodes=[f'n{i}' for i in range(nodes)],
                )
                reported = {
                    'nodes': nodes,
                    'ranks': nodes * local_ranks,
                    'ranksPerNode': local_ranks,
                    'gpusPerRank': gpus,
                }
                job._check_reported_topology([reported], 'float', gpus)
                self.assertEqual(job.topology_checks[0]['verdict'], 'pass')
        fail.assert_not_called()

    @patch('cvs.lib.rccl_lib.compare_topology')
    @patch('cvs.lib.rccl_lib.fail_test')
    def test_off_disables_comparison(self, fail, compare):
        job = self._job(cvs_params={'topology_check': 'off'})
        job._check_reported_topology([self.row], 'float', 8)
        fail.assert_not_called()
        compare.assert_not_called()
        self.assertEqual(job.topology_checks, [])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_invalid_mode_warns_and_defaults_to_warn(self, fail):
        with patch.object(rccl_lib.log, 'warning') as warning:
            job = self._job(cvs_params={'topology_check': 'typo'})
            self.assertIn('falling back to warn', warning.call_args.args[0])
            job._check_reported_topology([self.row], 'float', 8)
        fail.assert_not_called()
        self.assertEqual(job.topology_checks[0]['mode'], 'warn')

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_empty_rows_are_skipped(self, fail):
        self.job._check_reported_topology([], 'float', 8)
        fail.assert_not_called()
        self.assertEqual(self.job.topology_checks[0]['verdict'], 'skipped')
        self.assertIn('No result rows', self.job.topology_checks[0]['reason'])

    @patch('cvs.lib.rccl_lib.compare_topology')
    @patch('cvs.lib.rccl_lib.fail_test')
    def test_absent_topology_is_skipped_in_all_rows(self, fail, compare):
        job = self._job(cvs_params={'topology_check': 'strict'})
        with patch.object(rccl_lib.log, 'warning') as warning:
            job._check_reported_topology([{'size': 8}, {'size': 16}], 'float', 8)
        record = job.topology_checks[0]
        self.assertEqual(record['verdict'], 'skipped')
        self.assertEqual(record['reason'], 'Producer did not report topology fields')
        self.assertEqual(record['reported'], {})
        self.assertEqual(record['mismatches'], [])
        fail.assert_not_called()
        compare.assert_not_called()
        warning.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_partial_topology_still_reports_missing_fields(self, fail):
        job = self._job(cvs_params={'topology_check': 'strict'})
        job._check_reported_topology([{'gpusPerRank': 8}], 'float', 8)
        self.assertEqual(
            job.topology_checks[0]['mismatches'],
            [
                'nodes: requested 2, reported <missing>',
                'ranks: requested 2, reported <missing>',
                'ranksPerNode: requested 1, reported <missing>',
            ],
        )
        fail.assert_called_once()
        self.assertNotIn('common.cu', fail.call_args.args[0])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_absent_topology_in_only_some_rows_is_not_skipped(self, fail):
        for rows in ([{}, self.requested], [self.requested, {}]):
            with self.subTest(rows=rows):
                job = self._job(cvs_params={'topology_check': 'strict'})
                job._check_reported_topology(rows, 'regression', 8)
                self.assertEqual(job.topology_checks[0]['verdict'], 'pass')
        job = self._job(cvs_params={'topology_check': 'strict'})
        job._check_reported_topology([{}, {'gpusPerRank': 8}], 'regression', 8)
        self.assertEqual(job.topology_checks[0]['verdict'], 'mismatch')
        self.assertEqual(
            job.topology_checks[0]['mismatches'],
            [
                'nodes: requested 2, reported <missing>',
                'ranks: requested 2, reported <missing>',
                'ranksPerNode: requested 1, reported <missing>',
            ],
        )
        self.assertEqual(fail.call_count, 1)

    @patch('cvs.lib.rccl_lib.compare_topology')
    def test_topology_check_skips_invalid_result_shapes(self, compare):
        for rows in ({'nodes': 1}, {}, None, 1, 'invalid', [None], [self.row, 1]):
            with self.subTest(rows=rows):
                self.job._check_reported_topology(rows, 'regression', 8)
                record = self.job.topology_checks[-1]
                self.assertEqual(record['verdict'], 'skipped')
                self.assertIn('expected an array of result objects', record['reason'])
        compare.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_read_results_rejects_malformed_shapes(self, fail):
        for payload in ({'nodes': 1}, {}, None, 1, 'invalid', [None], [self.row, 1]):
            with self.subTest(payload=payload):
                fail.reset_mock()
                self._mock_download(payload)
                self.assertEqual(self.job.read_results('/results/rccl.json'), [])
                fail.assert_called_once()
                self.assertIn('/results/rccl.json', fail.call_args.args[0])
                self.assertIn('expected an array of result objects', fail.call_args.args[0])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_read_results_preserves_result_objects(self, fail):
        for payload in ([], [self.row], [{'gpusPerRank': 8}], [{'size': 8}]):
            with self.subTest(payload=payload):
                self._mock_download(payload)
                self.assertEqual(self.job.read_results('/results/rccl.json'), payload)
        fail.assert_not_called()

    @patch('cvs.lib.rccl_lib.compare_topology')
    @patch('cvs.lib.rccl_lib.fail_test')
    def test_nonuniform_mpirun_is_skipped(self, fail, compare):
        job = self._job(cvs_params={'topology_check': 'strict'}, nodes=['n1', 'n2', 'n3'], managed=False)
        job._check_reported_topology([self.row], 'float', 8)
        fail.assert_not_called()
        compare.assert_not_called()
        self.assertEqual(job.topology_checks[0]['verdict'], 'skipped')
        self.assertIn('not evenly divisible', job.topology_checks[0]['reason'])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_later_inconsistent_row_is_detected(self, fail):
        job = self._job(cvs_params={'topology_check': 'strict'})
        job._check_reported_topology([self.requested, self.row], 'regression', 8)
        fail.assert_called_once()
        self.assertIn('row 1', fail.call_args.args[0])
        mismatches = job.topology_checks[0]['mismatches']
        self.assertIn('nodes differs from row 0', mismatches[0])
        self.assertIn('row 0 reported 2, row 1 reported 1', mismatches[0])
        self.assertTrue(any('ranksPerNode differs from row 0' in mismatch for mismatch in mismatches))

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_later_inconsistent_row_is_detected_against_nonzero_baseline(self, fail):
        job = self._job(cvs_params={'topology_check': 'strict'})
        job._check_reported_topology([{}, self.requested, self.row], 'regression', 8)
        fail.assert_called_once()
        self.assertIn('row 2', fail.call_args.args[0])
        mismatches = job.topology_checks[0]['mismatches']
        self.assertIn('nodes differs from row 1', mismatches[0])
        self.assertIn('row 1 reported 2, row 2 reported 1', mismatches[0])
        self.assertTrue(any('ranksPerNode differs from row 1' in mismatch for mismatch in mismatches))

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_sparse_baseline_does_not_mask_later_conflicting_value(self, fail):
        job = self._job(cvs_params={'topology_check': 'strict'})
        job._check_reported_topology([{'gpusPerRank': 8}, {'nodes': 1}], 'regression', 8)
        record = job.topology_checks[0]
        self.assertEqual(record['reported'], {'gpusPerRank': 8})
        self.assertIn('nodes: requested 2, reported 1', record['mismatches'])
        fail.assert_called_once()

    def test_sparse_baseline_still_triggers_common_cu_hint(self):
        with patch.object(rccl_lib.log, 'warning') as warning:
            self.job._check_reported_topology([{'gpusPerRank': 8}, self.row], 'float', 8)
        self.assertIn('common.cu', warning.call_args.args[1])

    def test_result_model_accepts_single_node_and_multinode(self):
        single = {key: value for key, value in self.row.items() if key not in rccl_lib.TOPOLOGY_FIELDS}
        self.assertIs(rccl_lib._result_model(single), rccl_lib.RcclTests)
        self.assertIs(rccl_lib._result_model(self.row), rccl_lib.RcclTestsMultinodeRaw)
        self.assertEqual(rccl_lib._result_model(single).model_validate(single).wrong, 0)

    def test_mixed_result_shapes_fail_in_either_order(self):
        single = rccl_lib.RcclTests.model_validate(self.row)
        multi = rccl_lib.RcclTestsMultinodeRaw.model_validate(self.row)
        for rows in ([single, multi], [multi, single]):
            with self.subTest(first=type(rows[0]).__name__):
                with self.assertRaisesRegex(ValueError, 'Mixed single-node and multi-node results'):
                    rccl_lib.RcclJob.aggregate_results(rows)

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_perf_saves_all_dtype_checks_and_preserves_producer_metadata(self, fail):
        rows = [deepcopy(self.row)]
        saved = self._mock_run(rows)
        self.job.rccl_test_params['data_types'] = ['float', 'half']
        self.job.read_results.side_effect = [rows, [{**self.row, 'type': 'half'}]]
        result = self.job.run_perf()
        self.assertEqual(rows, [self.row])
        self.assertEqual(saved['/results/rccl.json'], result)
        artifact = saved['/results/rccl_topology_check.json']
        self.assertEqual(artifact['mode'], 'warn')
        self.assertEqual(
            [check['label'] for check in artifact['checks']], ['all_reduce_perf_float', 'all_reduce_perf_half']
        )
        for check in artifact['checks']:
            self.assertEqual(check['requested'], self.requested)
            self.assertEqual(check['verdict'], 'mismatch')
        for row in result + saved['/results/rccl_aggregated.json']:
            self.assertEqual({key: row[key] for key in rccl_lib.TOPOLOGY_FIELDS}, artifact['checks'][0]['reported'])
        self.assertEqual(
            set(saved), {'/results/rccl.json', '/results/rccl_aggregated.json', '/results/rccl_topology_check.json'}
        )
        self.assertIn('-g 8', self.job.execute.call_args.args[0])
        fail.assert_not_called()

    def test_perf_strict_records_failure_and_still_saves_results(self):
        saved = self._mock_run([self.row])
        self.job.topology_check_mode = 'strict'
        with patch.object(rccl_lib.globals, 'error_list', []):
            self.assertEqual(self.job.run_perf(), [self.row])
            self.assertEqual(len(rccl_lib.globals.error_list), 1)
        self.assertEqual(saved['/results/rccl_topology_check.json']['checks'][0]['verdict'], 'mismatch')
        self.assertIn('/results/rccl_aggregated.json', saved)

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_perf_single_node_without_topology_is_skipped_in_strict_mode(self, fail):
        self.job = self._job(
            cvs_params={'topology_check': 'strict'},
            mpi_params={'no_of_nodes': 1, 'no_of_local_ranks': 8},
            nodes=['n1'],
        )
        self.job.rccl_test_params['threads_per_gpu'] = '1'
        row = {key: value for key, value in self.row.items() if key not in rccl_lib.TOPOLOGY_FIELDS}
        saved = self._mock_run([row])
        self.assertEqual(self.job.run_perf(), [row])
        check = saved['/results/rccl_topology_check.json']['checks'][0]
        self.assertEqual(check['verdict'], 'skipped')
        self.assertEqual(check['reason'], 'Producer did not report topology fields')
        self.assertEqual(check['mismatches'], [])
        self.assertIsNone(saved['/results/rccl_aggregated.json'][0]['nodes'])
        fail.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_regression_without_topology_is_skipped_in_strict_mode(self, fail):
        self.job.topology_check_mode = 'strict'
        row = {key: value for key, value in self.row.items() if key not in rccl_lib.TOPOLOGY_FIELDS}
        saved = self._mock_run([row])
        self.assertEqual(self.job.run_regression(), [row])
        self.assertEqual(saved['/results/rccl_topology_check.json']['checks'][0]['verdict'], 'skipped')
        fail.assert_not_called()

    def test_perf_preserves_checks_when_a_later_launch_fails(self):
        saved = self._mock_run([self.row])
        self.job.rccl_test_params['data_types'] = ['float', 'half']
        self.job.execute.side_effect = ['completed', None]
        self.assertEqual(self.job.run_perf(), [self.row])
        checks = saved['/results/rccl_topology_check.json']['checks']
        self.assertEqual([check['verdict'] for check in checks], ['mismatch', 'skipped'])

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_off_mode_saves_disabled_audit(self, fail):
        saved = self._mock_run([self.row])
        self.job.topology_check_mode = 'off'
        self.job.run_perf()
        self.assertEqual(saved['/results/rccl_topology_check.json'], {'mode': 'off', 'checks': []})
        fail.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_off_mode_still_rejects_inconsistent_topology_during_aggregation(self, fail):
        saved = self._mock_run([self.row, {**self.row, **self.requested}])
        self.job.topology_check_mode = 'off'
        self.job.run_perf()
        fail.assert_called_once()
        self.assertIn('Inconsistent cluster config', fail.call_args.args[0])
        self.assertEqual(saved['/results/rccl_topology_check.json'], {'mode': 'off', 'checks': []})
        self.assertNotIn('/results/rccl_aggregated.json', saved)

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_schema_failure_still_aborts_and_saves_prior_checks(self, fail):
        saved = self._mock_run([self.row])
        self.job.rccl_test_params['data_types'] = ['float', 'half']
        self.job.read_results.side_effect = [[self.row], [{**self.row, 'wrong': '1'}]]
        with self.assertRaisesRegex(RuntimeError, 'schema validation failed'):
            self.job.run_perf()
        self.assertIn('SEVERE DATA CORRUPTION', fail.call_args.args[0])
        checks = saved['/results/rccl_topology_check.json']['checks']
        self.assertEqual(len(checks), 2)
        self.assertEqual([check['label'] for check in checks], ['all_reduce_perf_float', 'all_reduce_perf_half'])
        self.assertEqual(checks[1]['verdict'], 'mismatch')
        self.assertEqual(checks[1]['reported'], {field: self.row[field] for field in rccl_lib.TOPOLOGY_FIELDS})

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_regression_counts_gpus_across_threads_and_checks_all_raw_rows(self, fail):
        self.job.topology_check_mode = 'strict'
        rows = [self.requested, {'nodes': 1, 'ranks': 2, 'ranksPerNode': 2}]
        original = deepcopy(rows)
        saved = self._mock_run(rows)
        self.assertEqual(self.job.run_regression(), original)
        self.assertEqual(rows, original)
        self.assertIn('-t 8', self.job.execute.call_args.args[0])
        self.assertNotIn('-g ', self.job.execute.call_args.args[0])
        check = saved['/results/rccl_topology_check.json']['checks'][0]
        self.assertEqual(check['requested']['gpusPerRank'], 8)
        self.assertEqual(check['verdict'], 'mismatch')
        fail.assert_called_once()
        self.assertEqual(set(saved), {'/results/rccl_topology_check.json'})

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_regression_gpu_count_matches_thread_flag(self, fail):
        for threads in (None, '1', '8'):
            with self.subTest(threads=threads):
                self.job = self._job(cvs_params={'topology_check': 'strict'})
                if threads is None:
                    del self.job.rccl_test_params['threads_per_gpu']
                else:
                    self.job.rccl_test_params['threads_per_gpu'] = threads
                gpus = int(threads or 1)
                row = {**self.row, **self.requested, 'gpusPerRank': gpus}
                saved = self._mock_run([row])
                self.assertEqual(self.job.run_regression(), [row])
                self.assertIn(f'-t {gpus}', self.job.execute.call_args.args[0])
                check = saved['/results/rccl_topology_check.json']['checks'][0]
                self.assertEqual(check['requested']['gpusPerRank'], gpus)
                self.assertEqual(check['verdict'], 'pass')
        fail.assert_not_called()

    @patch('cvs.lib.rccl_lib.fail_test')
    def test_malformed_results_fail_loading_without_crashing_runs(self, fail):
        for method in ('run_perf', 'run_regression'):
            for mode in ('warn', 'strict', 'off'):
                with self.subTest(method=method, mode=mode):
                    fail.reset_mock()
                    self.job = self._job(cvs_params={'topology_check': mode})
                    saved = self._mock_run(None)
                    self._mock_download({'nodes': 1})
                    self.job.read_results.side_effect = lambda *args: rccl_lib.RcclJob.read_results(self.job, *args)
                    self.assertEqual(getattr(self.job, method)(), [])
                    fail.assert_called_once()
                    self.assertIn('expected an array of result objects', fail.call_args.args[0])
                    self.job._verify_results.assert_not_called()
                    checks = saved['/results/rccl_topology_check.json']['checks']
                    if mode == 'off':
                        self.assertEqual(checks, [])
                    else:
                        self.assertEqual(checks[0]['verdict'], 'skipped')

    def test_empty_result_set_returns_without_verifying(self):
        """An empty result array used to reach check_bw_dip and raise IndexError."""
        for method in ('run_perf', 'run_regression'):
            with self.subTest(method=method):
                self.job = self._job()
                saved = self._mock_run([])
                self.assertEqual(getattr(self.job, method)(), [])
                self.job._verify_results.assert_not_called()
                check = saved['/results/rccl_topology_check.json']['checks'][0]
                self.assertEqual(check['verdict'], 'skipped')
                self.assertIn('No result rows', check['reason'])

    def test_regression_saves_audit_when_execute_raises(self):
        saved = self._mock_run([])
        self.job.execute.side_effect = RuntimeError('launcher unavailable')
        with self.assertRaisesRegex(RuntimeError, 'launcher unavailable'):
            self.job.run_regression()
        self.assertEqual(saved['/results/rccl_topology_check.json'], {'mode': 'warn', 'checks': []})

    def test_regression_failed_launch_saves_skipped_check(self):
        saved = self._mock_run([])
        self.job.execute.return_value = None
        self.assertEqual(self.job.run_regression(), [])
        self.job.read_results.assert_not_called()
        self.assertEqual(saved['/results/rccl_topology_check.json']['checks'][0]['verdict'], 'skipped')

    def test_reused_job_resets_checks_for_each_run(self):
        saved = self._mock_run([self.row])
        for run in (self.job.run_perf, self.job.run_perf, self.job.run_regression, self.job.run_regression):
            run()
            self.assertEqual(len(saved['/results/rccl_topology_check.json']['checks']), 1)

    def test_save_topology_checks_reports_upload_failure(self):
        self.job.orch.upload_to_head.side_effect = IOError('unreachable head node')
        with patch.object(rccl_lib.log, 'error') as error:
            self.job._save_topology_checks('/results/rccl.json')
        self.assertIn('Failed to save topology checks', error.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
