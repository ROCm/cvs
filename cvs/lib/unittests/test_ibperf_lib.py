# cvs/lib/unittests/test_ibperf_lib.py
import unittest
from unittest.mock import patch, MagicMock
import cvs.lib.ibperf_lib as ibperf_lib


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


BW_LOG = '8192 bytes of GPU buffer\n 8192       5000           512.00             517.14             7.89\n'


class TestWaitForPerftestExit(unittest.TestCase):
    @patch.object(ibperf_lib.time, 'sleep')
    def test_returns_once_pgrep_confirms_exit(self, _sleep):
        phdl = MagicMock()
        phdl.exec.side_effect = [
            {'node1': 'pgrep_rc=0', 'node2': 'pgrep_rc=1'},
            {'node1': 'pgrep_rc=1', 'node2': 'pgrep_rc=1'},
        ]

        self.assertEqual(ibperf_lib.wait_for_perftest_exit(phdl, 'raw_ethernet_burst_lat', 90), [])
        self.assertEqual(phdl.exec.call_count, 2)
        self.assertIn('pgrep -x raw_ethernet_bu ', phdl.exec.call_args.args[0])

    def test_errored_poll_is_not_counted_as_exited(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': 'pgrep_rc=1', 'node2': 'ABORT: Timeout Error in Host: node2'}

        self.assertEqual(ibperf_lib.wait_for_perftest_exit(phdl, 'ib_write_bw', 0), ['node2'])


class TestGetIbBwPps(unittest.TestCase):
    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'fail_test')
    def test_stops_polling_once_every_node_reports(self, mock_fail, _sleep):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': BW_LOG, 'node2': BW_LOG}

        res = ibperf_lib.get_ib_bw_pps(phdl, 8192, 'cat /tmp/ib_perf_0_logs')

        self.assertEqual(res, {n: {'bw': '517.14', 'pps': '7.89'} for n in ('node1', 'node2')})
        self.assertEqual(phdl.exec.call_count, 2)
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'PERFTEST_BW_RESULT_TIMEOUT_S', 0)
    @patch.object(ibperf_lib, 'fail_test')
    def test_fails_once_per_node_after_retries_run_out(self, mock_fail):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': BW_LOG, 'node2': '8192 bytes of GPU buffer\n'}

        res = ibperf_lib.get_ib_bw_pps(phdl, 8192, 'cat /tmp/ib_perf_0_logs')

        self.assertEqual(res, {'node1': {'bw': '517.14', 'pps': '7.89'}})
        self.assertEqual(mock_fail.call_count, 2)
        self.assertTrue(all('node2' in c.args[0] for c in mock_fail.call_args_list))


class TestRunIbPerfLatTest(unittest.TestCase):
    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'wait_for_perftest_exit', return_value=[])
    @patch.object(ibperf_lib, 'get_ib_lat_numb')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_builds_latency_commands(self, _dmabuf, mock_lat_numb, mock_wait, _sleep):
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
        mock_wait.assert_called_once_with(phdl, 'ib_write_lat', ibperf_lib.PERFTEST_EXIT_SLACK_S)


if __name__ == '__main__':
    unittest.main()
