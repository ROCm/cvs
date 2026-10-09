# cvs/lib/unittests/test_ibperf_lib.py
import unittest
from unittest.mock import patch, MagicMock
import cvs.lib.ibperf_lib as ibperf_lib
from cvs.lib import globals


class TestIbperfLib(unittest.TestCase):
    @patch('xlsxwriter.Workbook')
    def test_generate_ibperf_bw_chart(self, mock_workbook_class):
        mock_workbook = MagicMock()
        mock_workbook_class.return_value = mock_workbook
        mock_worksheet = MagicMock()
        mock_workbook.add_worksheet.return_value = mock_worksheet

        res_dict = {
            'ib_write_bw': {
                1024: {1: {'node1': {i: {'pps': str(10.0 + i), 'bw': str(1.0 + i * 0.1)} for i in range(8)}}}
            }
        }
        ibperf_lib.generate_ibperf_bw_chart(res_dict, 'test.xlsx')
        self.assertTrue(mock_workbook.add_worksheet.called)
        self.assertTrue(mock_workbook.close.called)

    @patch('xlsxwriter.Workbook')
    def test_generate_ibperf_lat_chart(self, mock_workbook_class):
        mock_workbook = MagicMock()
        mock_workbook_class.return_value = mock_workbook
        mock_worksheet = MagicMock()
        mock_workbook.add_worksheet.return_value = mock_worksheet

        res_dict = {
            'ib_write_lat': {
                1024: {
                    'node1': {
                        i: {
                            't_min': str(1.0 + i * 0.1),
                            't_max': str(2.0 + i * 0.1),
                            't_avg': str(1.5 + i * 0.1),
                            't_stdev': str(0.1 + i * 0.01),
                            't_99_pct': str(1.9 + i * 0.1),
                        }
                        for i in range(8)
                    }
                }
            }
        }
        ibperf_lib.generate_ibperf_lat_chart(res_dict, 'test.xlsx')
        self.assertTrue(mock_workbook.add_worksheet.called)
        self.assertTrue(mock_workbook.close.called)


class TestGetPerftestConfigureCmd(unittest.TestCase):
    def test_includes_rocm_dmabuf_by_default(self):
        self.assertEqual(
            ibperf_lib.get_perftest_configure_cmd('/opt/ibperf', '/opt/rocm'),
            'cd /opt/ibperf/perftest; ./configure --prefix=/opt/ibperf/perftest '
            '--with-rocm=/opt/rocm --enable-rocm --enable-rocm-dmabuf',
        )

    def test_omits_rocm_dmabuf_when_disabled(self):
        cmd = ibperf_lib.get_perftest_configure_cmd('/opt/ibperf', '/opt/rocm', rocm_dmabuf=False)
        self.assertEqual(
            cmd,
            'cd /opt/ibperf/perftest; ./configure --prefix=/opt/ibperf/perftest --with-rocm=/opt/rocm --enable-rocm',
        )
        self.assertNotIn('dmabuf', cmd)


class TestConfigurePerftest(unittest.TestCase):
    def setUp(self):
        globals.error_list = []

    def test_dmabuf_configure_succeeds(self):
        shdl = MagicMock()
        shdl.exec.return_value = {'node1': {'output': 'ok', 'exit_code': 0}}

        result = ibperf_lib.configure_perftest(shdl, '/opt/ibperf', '/opt/rocm')

        self.assertTrue(result)
        shdl.exec.assert_called_once()
        self.assertEqual(
            shdl.exec.call_args.args[0],
            ibperf_lib.get_perftest_configure_cmd('/opt/ibperf', '/opt/rocm', rocm_dmabuf=True),
        )
        self.assertEqual(shdl.exec.call_args.kwargs['timeout'], 200)
        self.assertTrue(shdl.exec.call_args.kwargs['detailed'])
        self.assertEqual(globals.error_list, [])

    def test_falls_back_when_dmabuf_configure_fails(self):
        shdl = MagicMock()
        shdl.exec.side_effect = [
            {'node1': {'output': 'configure: error: hsa_amd_portable_export_dmabuf not found', 'exit_code': 1}},
            {'node1': {'output': 'ok', 'exit_code': 0}},
        ]

        with self.assertLogs(ibperf_lib.log, level='WARNING'):
            result = ibperf_lib.configure_perftest(shdl, '/opt/ibperf', '/opt/rocm')

        self.assertFalse(result)
        self.assertEqual(shdl.exec.call_count, 2)
        fallback_cmd = shdl.exec.call_args_list[1].args[0]
        self.assertNotIn('--enable-rocm-dmabuf', fallback_cmd)
        self.assertEqual(
            fallback_cmd,
            ibperf_lib.get_perftest_configure_cmd('/opt/ibperf', '/opt/rocm', rocm_dmabuf=False),
        )
        self.assertEqual(globals.error_list, [])

    def test_records_failure_when_fallback_also_fails(self):
        shdl = MagicMock()
        failure = {
            'node1': {'output': 'line1\nline2\nconfigure: error: cannot include hip/hip_runtime_api.h', 'exit_code': 1}
        }
        shdl.exec.side_effect = [failure, failure]

        result = ibperf_lib.configure_perftest(shdl, '/opt/ibperf', '/opt/rocm')

        self.assertFalse(result)
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('node1', globals.error_list[0])
        self.assertIn('hip_runtime_api.h', globals.error_list[0])

    def test_aborted_exec_treated_as_failure(self):
        shdl = MagicMock()
        shdl.exec.side_effect = [
            {'node1': {'output': '', 'exit_code': -1}},
            {'node1': {'output': 'ok', 'exit_code': 0}},
        ]

        result = ibperf_lib.configure_perftest(shdl, '/opt/ibperf', '/opt/rocm')

        self.assertFalse(result)
        self.assertNotIn('--enable-rocm-dmabuf', shdl.exec.call_args_list[1].args[0])


class TestRunIbPerfLatTest(unittest.TestCase):
    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_lat_numb')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_builds_latency_commands(self, _dmabuf, mock_lat_numb, _sleep):
        nodes = ('node1', 'node2')
        lat = dict.fromkeys(('t_min', 't_max', 't_typical', 't_avg', 't_stdev', 't_99_pct', 't_99_9_pct'), '1.0')
        mock_lat_numb.return_value = {n: lat for n in nodes}
        gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
        phdl = MagicMock()

        ibperf_lib.run_ib_perf_lat_test(
            MagicMock(),
            phdl,
            'ib_write_lat',
            gpu_numa_dict,
            gpu_nic_dict,
            {n: {} for n in nodes},
            '/opt/perftest/bin',
            64,
            3,
        )

        server_cmd, client_cmd = phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertEqual(
            server_cmd,
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' -x 3 -F -p 1516 -s 64 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            client_cmd,
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' -x 3 -F -p 1516 -s 64 node1 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )


if __name__ == '__main__':
    unittest.main()
