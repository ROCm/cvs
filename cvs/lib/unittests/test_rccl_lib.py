# cvs/lib/unittests/test_rccl_lib.py
import os
import json
import unittest
from copy import deepcopy
from pathlib import Path
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
        self.assertEqual(job.topology_check_mode, 'warn')
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
                MagicMock(),
                MagicMock(),
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
        self.job.shdl.upload_file.side_effect = upload
        self.job.cvs_params['rccl_result_file'] = '/results/rccl.json'
        return saved

    def _mock_download(self, payload):
        def download(remote_path, local_path):
            Path(local_path).write_text(json.dumps(payload))
            return {self.job.head_node: local_path}

        self.job.shdl.download_file.side_effect = download

    def test_compare_exact_match(self):
        self.assertEqual(rccl_lib.compare_topology(self.requested, self.requested), [])

    def test_compare_captured_mislabeled_topology(self):
        """Values captured from a real multi-node run exhibiting the AIMVT-334 label swap."""
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
        self.assertIn('n4 slots=8', job.shdl.exec.call_args.args[0])
        self.assertEqual(job.shdl.exec.call_args.args[0].count('slots=8'), 4)

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
        for rows in ([{}, self.requested], [self.requested, {}], [{}, {'gpusPerRank': 8}]):
            with self.subTest(rows=rows):
                job = self._job(cvs_params={'topology_check': 'strict'})
                job._check_reported_topology(rows, 'regression', 8)
                self.assertEqual(job.topology_checks[0]['verdict'], 'mismatch')
                self.assertIn('row 1', job.topology_checks[0]['mismatches'][-1])
        self.assertEqual(fail.call_count, 3)

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
        self.assertIn("'nodes': 1", job.topology_checks[0]['mismatches'][0])

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
        self.assertEqual(len(saved['/results/rccl_topology_check.json']['checks']), 1)

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
        self.job.shdl.upload_file.side_effect = IOError('unreachable head node')
        with patch.object(rccl_lib.log, 'error') as error:
            self.job._save_topology_checks('/results/rccl.json')
        self.assertIn('Failed to save topology checks', error.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
