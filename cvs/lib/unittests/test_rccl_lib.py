# cvs/lib/unittests/test_rccl_lib.py
import unittest
from unittest.mock import MagicMock, patch
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

    def _launch_kwargs(self, **overrides):
        kwargs = dict(
            mpi_dir='/opt/ompi',
            no_of_nodes=2,
            no_of_local_ranks=8,
            no_of_global_ranks=16,
            mpi_oob_port='eth0',
            pml_param='--mca pml ob1',
            ucx_params='',
            hosts_file_path='/tmp/rccl_hosts_file_cvs.txt',
            cluster_node_list=['n1', 'n2'],
            env_override_params='',
        )
        kwargs.update(overrides)
        return kwargs

    @patch('cvs.lib.rccl_lib.is_managed_compute', return_value=False)
    def test_build_launch_cmd_bare_metal_mpirun(self, _managed):
        cmd = rccl_lib._build_rccl_launch_cmd('bash -c all_reduce_perf', **self._launch_kwargs())
        self.assertIn('mpirun', cmd)
        self.assertIn('--hostfile /tmp/rccl_hosts_file_cvs.txt', cmd)
        self.assertIn('--mca pml ob1', cmd)
        self.assertNotIn('--mpi=pmix', cmd)
        self.assertNotIn('spur run', cmd)

    @patch('cvs.lib.rccl_lib.scheduler_hosts', return_value=['n1', 'n2'])
    @patch('cvs.lib.rccl_lib._cpus_per_nested_task', return_value=None)
    @patch('cvs.lib.rccl_lib.detect_scheduler', return_value=rccl_lib.Scheduler.SPUR)
    @patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True)
    def test_build_launch_cmd_managed_spur(self, _managed, _sched, _cpus, _hosts):
        cmd = rccl_lib._build_rccl_launch_cmd('bash -c all_reduce_perf', **self._launch_kwargs())
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

    @patch('cvs.lib.rccl_lib._cpus_per_nested_task', return_value=None)
    @patch('cvs.lib.rccl_lib.detect_scheduler', return_value=rccl_lib.Scheduler.SLURM)
    @patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True)
    def test_build_launch_cmd_managed_slurm(self, _managed, _sched, _cpus):
        cmd = rccl_lib._build_rccl_launch_cmd('bash -c all_reduce_perf', **self._launch_kwargs())
        self.assertTrue(cmd.startswith('srun '))
        self.assertIn('--overlap', cmd)
        self.assertIn('--mpi=pmix', cmd)
        self.assertIn('-w n1,n2', cmd)
        self.assertNotIn('spur run', cmd)
        self.assertNotIn('--jobid', cmd)

    @patch('cvs.lib.rccl_lib._cpus_per_nested_task', return_value=None)
    @patch('cvs.lib.rccl_lib.detect_scheduler', return_value=rccl_lib.Scheduler.SLURM)
    @patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True)
    def test_build_launch_cmd_pairwise_nodelist_slurm(self, _managed, _sched, _cpus):
        cmd = rccl_lib._build_rccl_launch_cmd(
            'bash -c all_reduce_perf',
            **self._launch_kwargs(cluster_node_list=['ref', 'cand'], no_of_nodes=2, no_of_global_ranks=16),
        )
        self.assertIn('-w ref,cand', cmd)

    @patch('cvs.lib.rccl_lib.scheduler_hosts', return_value=['n1', 'n2', 'n3'])
    @patch('cvs.lib.rccl_lib._cpus_per_nested_task', return_value=None)
    @patch('cvs.lib.rccl_lib.detect_scheduler', return_value=rccl_lib.Scheduler.SPUR)
    @patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True)
    def test_build_launch_cmd_spur_rejects_subset_nodelist(self, _managed, _sched, _cpus, _hosts):
        with self.assertRaisesRegex(RuntimeError, 'does not apply --nodelist'):
            rccl_lib._build_rccl_launch_cmd(
                'bash -c all_reduce_perf',
                **self._launch_kwargs(cluster_node_list=['ref', 'cand'], no_of_nodes=2, no_of_global_ranks=16),
            )

    def test_wrap_rccl_test_cmd_env_overrides(self):
        wrapped = rccl_lib._wrap_rccl_test_cmd(
            '/opt/all_reduce_perf -g 8',
            '/home/user/env.sh',
            {'NCCL_ALGO': 'Ring'},
        )
        self.assertIn('source /home/user/env.sh && export NCCL_ALGO=Ring &&', wrapped)
        self.assertNotRegex(wrapped, r'export NCCL_ALGO=Ring.*source ')
        self.assertTrue(wrapped.startswith('bash -c '))

    def test_require_spur_job_step_rejects_bare_allocation(self):
        env = {'SPUR_JOB_ID': '99', 'SLURM_JOB_ID': '99'}
        with patch.dict('os.environ', env, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'requires a SPUR job step'):
                rccl_lib._require_spur_job_step()

    def test_require_spur_job_step_allows_managed_step(self):
        env = {'SPUR_JOB_ID': '99', 'SLURM_JOB_ID': '99', 'SLURM_STEP_ID': '0', 'SLURM_PROCID': '0'}
        with patch.dict('os.environ', env, clear=True):
            rccl_lib._require_spur_job_step()

    def test_require_spur_job_step_preserves_non_spur_runs(self):
        for env in ({'CVS_SCHEDULER': 'bare_metal'}, {'SLURM_JOB_ID': '99'}):
            with self.subTest(env=env), patch.dict('os.environ', env, clear=True):
                rccl_lib._require_spur_job_step()

    def test_exec_rccl_launch_nonzero_exit_does_not_scan(self):
        rccl_lib.globals.error_list = []
        shdl = MagicMock()
        shdl.exec.return_value = {
            'head': {'output': '# Avg bus bandwidth    : 1.8\n', 'exit_code': 3},
        }
        phdl = MagicMock()
        with patch('cvs.lib.rccl_lib.scan_rccl_logs') as mock_scan:
            result = rccl_lib._exec_rccl_launch(
                phdl, shdl, 'head', 'spur run --overlap --mpi=pmix -- all_reduce_perf', 'all_reduce_perf', 60, 'unit'
            )
        self.assertIsNone(result)
        mock_scan.assert_not_called()
        phdl.exec.assert_called()
        shdl.exec.assert_called_once()
        self.assertTrue(shdl.exec.call_args.kwargs.get('detailed'))
        self.assertTrue(any('exit code 3' in msg for msg in rccl_lib.globals.error_list))

    def test_exec_rccl_launch_timeout_with_bandwidth_is_failure(self):
        rccl_lib.globals.error_list = []
        shdl = MagicMock()
        shdl.exec.return_value = {
            'head': {'output': '# Avg bus bandwidth    : 1.8\nABORT: Timeout\n', 'exit_code': -1},
        }
        phdl = MagicMock()
        result = rccl_lib._exec_rccl_launch(phdl, shdl, 'head', 'spur run --overlap --', 'all_reduce_perf', 60, 'unit')
        self.assertIsNone(result)
        self.assertTrue(any('exit code -1' in msg for msg in rccl_lib.globals.error_list))

    def test_exec_rccl_launch_success_scans_logs(self):
        rccl_lib.globals.error_list = []
        shdl = MagicMock()
        shdl.exec.return_value = {
            'head': {'output': '# Avg bus bandwidth    : 1.8\n', 'exit_code': 0},
        }
        phdl = MagicMock()
        result = rccl_lib._exec_rccl_launch(phdl, shdl, 'head', 'spur run --overlap --', 'all_reduce_perf', 60, 'unit')
        self.assertIn('Avg bus bandwidth', result)
        self.assertEqual(rccl_lib.globals.error_list, [])
        phdl.exec.assert_not_called()

    def test_rccl_regression_failed_launch_preserves_graph_generation(self):
        env = {
            'SPUR_JOB_ID': '99',
            'SLURM_JOB_ID': '99',
            'SLURM_STEP_ID': '0',
            'SLURM_PROCID': '0',
            'SPUR_NODES': 'n1,n2',
        }
        phdl = MagicMock()
        shdl = MagicMock()
        shdl.exec.side_effect = [
            {'n1': 'NEW'},
            {'n1': {'output': '# Avg bus bandwidth : 1.8\n', 'exit_code': 3}},
        ]
        with (
            patch.dict('os.environ', env, clear=True),
            patch.object(rccl_lib.globals, 'error_list', []),
            patch('cvs.lib.rccl_lib._read_json_from_head_node') as read_results,
        ):
            failed_results = rccl_lib.rccl_regression(
                phdl,
                shdl,
                'all_reduce_perf',
                '/dev/null',
                {'no_of_nodes': 2, 'no_of_local_ranks': 1},
                {},
                {},
                ['n1', 'n2'],
                ['n1', 'n2'],
            )
            self.assertEqual(failed_results, [])
            read_results.assert_not_called()
            self.assertTrue(any('exit code 3' in msg for msg in rccl_lib.globals.error_list))
            phdl.exec.assert_called()

        successful_results = [
            {'size': 1024, 'name': 'all_reduce_perf', 'inPlace': 1, 'busBw': 100.0, 'algBw': 90.0, 'time': 1.0}
        ]
        graph = rccl_lib.convert_to_graph_dict({'failed': failed_results, 'successful': successful_results})
        self.assertEqual(graph['failed'], {})
        self.assertEqual(graph['successful'][1024], {'bus_bw': 100.0, 'alg_bw': 90.0, 'time': 1.0})

    def test_cpus_per_nested_task_from_slurm_env(self):
        with patch.dict('os.environ', {'SLURM_CPUS_ON_NODE': '236'}, clear=True):
            self.assertEqual(rccl_lib._cpus_per_nested_task(8), 29)

    def test_cpus_per_nested_task_override(self):
        with patch.dict('os.environ', {'RCCL_CPUS_PER_TASK': '16', 'SLURM_CPUS_ON_NODE': '236'}, clear=True):
            self.assertEqual(rccl_lib._cpus_per_nested_task(8), 16)

    def test_cleanup_does_not_pkill_scheduler(self):
        phdl = MagicMock()
        rccl_lib._cleanup_stale_rccl_processes(phdl, 'all_reduce_perf', 'unit-test')
        commands = [call.args[0] for call in phdl.exec.call_args_list]
        joined = ' '.join(commands)
        self.assertNotRegex(joined, r'(^|[^a-z])srun([^a-z]|$)')
        self.assertNotIn('spur', joined)
        self.assertIn('all_reduce_perf', joined)


if __name__ == '__main__':
    unittest.main()
