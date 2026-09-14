# cvs/lib/unittests/test_rccl_lib.py
import os
import unittest
from unittest.mock import MagicMock, patch
import cvs.lib.rccl_lib as rccl_lib


class TestRcclLib(unittest.TestCase):
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

    def _openmpi(self):
        openmpi = rccl_lib.OpenMPI({'mpi_dir': '/opt/ompi', 'mpi_oob_port': 'eth0'})
        openmpi.pml = 'ob1'
        openmpi.prepared = True
        return openmpi

    def _srun(self, nodes, no_of_nodes, local_ranks, global_ranks, openmpi=None, phdl=None):
        return rccl_lib.Srun(
            openmpi or self._openmpi(),
            phdl or MagicMock(),
            nodes,
            no_of_nodes,
            local_ranks,
            global_ranks,
        )

    def test_mpirun_command(self):
        launcher = rccl_lib.MpiRun(self._openmpi(), MagicMock(), ['n1', 'n2'], ['v1', 'v2'], 16)
        cmd = launcher.command('all_reduce_perf')
        self.assertIn('mpirun', cmd)
        self.assertIn('--hostfile /tmp/rccl_hosts_file_', cmd)
        self.assertIn('--mca pml ob1', cmd)
        self.assertNotIn('--mpi=pmix', cmd)
        self.assertNotIn('spur run', cmd)

    def test_mpirun_command_env_overrides_win_over_env_file(self):
        launcher = rccl_lib.MpiRun(self._openmpi(), MagicMock(), ['n1', 'n2'], ['v1', 'v2'], 16)
        cmd = launcher.command('all_reduce_perf', '/home/user/env.sh', {'NCCL_ALGO': 'Ring'})
        self.assertIn('source /home/user/env.sh && export NCCL_ALGO=Ring && all_reduce_perf', cmd)
        self.assertNotIn('-x NCCL_ALGO', cmd)

    def test_mpirun_command_wraps_binary_without_env_file(self):
        launcher = rccl_lib.MpiRun(self._openmpi(), MagicMock(), ['n1', 'n2'], ['v1', 'v2'], 16)
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
        job = rccl_lib.RcclJob.from_config(
            MagicMock(), MagicMock(), 'all_reduce_perf', config, ['n1', 'n2'], ['v1', 'v2']
        )
        self.assertEqual(job.env_file, '/tmp/ainic.sh')
        self.assertEqual(job.head_node, 'n1')
        self.assertEqual(job.no_of_global_ranks, 8)
        self.assertEqual(job.openmpi.oob_port, 'ens3')
        self.assertEqual(job.cvs_exec_timeout, 600)
        self.assertIsInstance(job.openmpi, rccl_lib.OpenMPI)
        self.assertIsInstance(job.launcher, rccl_lib.Srun)

    @patch.object(rccl_lib.RcclJob, '_detect_output_flag', return_value='-X')
    @patch('cvs.lib.rccl_lib.Srun.prepare', autospec=True)
    @patch('cvs.lib.rccl_lib.OpenMPI.prepare', autospec=True)
    @patch.object(rccl_lib.RcclJob, '_require_spur_job_step')
    @patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True)
    def test_rccl_job_prepare_is_idempotent(self, _managed, require_step, prepare_openmpi, prepare_srun, detect_output):
        job = rccl_lib.RcclJob(
            MagicMock(),
            MagicMock(),
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
        phdl = MagicMock()
        launcher = self._srun(['n1', 'n2'], 2, 8, 16, phdl=phdl)
        self.assertIs(launcher.prepare(), launcher)
        launcher.prepare()
        self.assertEqual(launcher.session_dir, '/tmp/cvsuser/ompi-4242')
        phdl.exec.assert_called_once()
        mkdir_cmd = phdl.exec.call_args[0][0]
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
        phdl = MagicMock()
        launcher = self._srun(['n1'], 1, 8, 8, phdl=phdl).prepare()
        session_dir = launcher.session_dir
        launcher.cleanup()
        self.assertIn(f'rm -rf {session_dir}', phdl.exec.call_args[0][0])
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
            MagicMock(),
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

    def _job(self, phdl, shdl, head='head'):
        with patch('cvs.lib.rccl_lib.is_managed_compute', return_value=True):
            return rccl_lib.RcclJob(
                phdl,
                shdl,
                'all_reduce_perf',
                '/dev/null',
                {'mpi_pml': 'ob1'},
                {},
                {'cvs_exec_timeout': 60},
                [head],
                [head],
            )

    def test_exec_rccl_launch_nonzero_exit_does_not_scan(self):
        rccl_lib.globals.error_list = []
        shdl = MagicMock()
        shdl.exec.return_value = {
            'head': {'output': '# Avg bus bandwidth    : 1.8\n', 'exit_code': 3},
        }
        phdl = MagicMock()
        job = self._job(phdl, shdl)
        with (
            patch.object(job, 'launch_command', return_value='spur run --overlap --mpi=pmix -- all_reduce_perf'),
            patch('cvs.lib.rccl_lib.RcclVerifier.scan_logs') as mock_scan,
        ):
            result = job.execute('all_reduce_perf', 'unit')
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
        job = self._job(phdl, shdl)
        with patch.object(job, 'launch_command', return_value='spur run --overlap --'):
            result = job.execute('all_reduce_perf', 'unit')
        self.assertIsNone(result)
        self.assertTrue(any('exit code -1' in msg for msg in rccl_lib.globals.error_list))

    def test_exec_rccl_launch_success_scans_logs(self):
        rccl_lib.globals.error_list = []
        shdl = MagicMock()
        shdl.exec.return_value = {
            'head': {'output': '# Avg bus bandwidth    : 1.8\n', 'exit_code': 0},
        }
        phdl = MagicMock()
        job = self._job(phdl, shdl)
        with patch.object(job, 'launch_command', return_value='spur run --overlap --'):
            result = job.execute('all_reduce_perf', 'unit')
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
            patch.object(rccl_lib.RcclJob, 'read_results') as read_results,
        ):
            failed_results = rccl_lib.RcclJob(
                phdl,
                shdl,
                'all_reduce_perf',
                '/dev/null',
                {'no_of_nodes': 2, 'no_of_local_ranks': 1, 'mpi_pml': 'ob1', 'mpi_dir': '/opt/openmpi'},
                {},
                {},
                ['n1', 'n2'],
                ['n1', 'n2'],
            ).run_regression()
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
            self.assertEqual(rccl_lib.Srun._cpus_per_nested_task(8), 29)

    def test_cpus_per_nested_task_override(self):
        with patch.dict('os.environ', {'RCCL_CPUS_PER_TASK': '16', 'SLURM_CPUS_ON_NODE': '236'}, clear=True):
            self.assertEqual(rccl_lib.Srun._cpus_per_nested_task(8), 16)

    def test_cleanup_does_not_pkill_scheduler(self):
        phdl = MagicMock()
        self._job(phdl, MagicMock())._cleanup_stale_processes('unit-test')
        commands = [call.args[0] for call in phdl.exec.call_args_list]
        joined = ' '.join(commands)
        self.assertNotRegex(joined, r'(^|[^a-z])srun([^a-z]|$)')
        self.assertNotIn('spur', joined)
        self.assertIn('all_reduce_perf', joined)


if __name__ == '__main__':
    unittest.main()
