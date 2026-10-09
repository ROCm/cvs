# cvs/lib/unittests/test_ibperf_lib.py
import unittest
from unittest.mock import patch, MagicMock
import cvs.lib.ibperf_lib as ibperf_lib


class TestIbperfLib(unittest.TestCase):
    def test_get_ibperf_rdma_devices(self):
        mapping = {
            'nodeA': {
                'card0': {'rdma_dev': 'nic2'},
                'card1': {'rdma_dev': 'nic1'},
                'card2': {'rdma_dev': 'nic2'},
                'card3': {},
            },
            'other': {'card0': {'rdma_dev': 'nic9'}},
        }
        self.assertEqual(
            ibperf_lib.get_ibperf_rdma_devices(mapping, {'nodeA': {}}, gpu_count=4), {'nodeA': ['nic1', 'nic2']}
        )

    def test_resolve_explicit_gid(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'nodeA': 'DEVICE:nic0\nLINK_LAYER:Ethernet\nGID:0000:0000:0000:0000:0000:ffff:c000:0201\nGID_TYPE:RoCE v2\n'
        }
        with patch.object(ibperf_lib, 'fail_test') as fail:
            self.assertEqual(ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['nic0']}, '3'), '3')
            fail.assert_not_called()
        self.assertIn('gids/3', phdl.exec.call_args.args[0])

    def test_resolve_wrong_type_reports_node_nic_and_index(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'nodeA': 'DEVICE:nic0\nLINK_LAYER:Ethernet\nGID:0000:0000:0000:0000:0000:ffff:c000:0201\nGID_TYPE:IB/RoCE v1\n'
        }
        with patch.object(ibperf_lib, 'fail_test') as fail:
            self.assertIsNone(ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['nic0']}, '3'))
            fail.assert_called_once()
            for fragment in ('nodeA', 'nic0', 'index 3', 'IB/RoCE v1'):
                self.assertIn(fragment, fail.call_args.args[0])

    def test_resolve_auto_and_missing_index(self):
        table = 'GIDENT|nic0|1|fe80:0000:0000:0000:0000:0000:0000:0001|RoCE v2\nGIDENT|nic0|3|0000:0000:0000:0000:0000:ffff:c000:0201|RoCE v2\n'
        probe = 'DEVICE:nic0\nLINK_LAYER:Ethernet\nGID:0000:0000:0000:0000:0000:ffff:c000:0201\nGID_TYPE:RoCE v2\n'
        for requested in ('auto', None):
            phdl = MagicMock()
            phdl.exec.side_effect = [{'nodeA': table}, {'nodeA': probe}]
            with self.subTest(requested=requested), patch.object(ibperf_lib, 'fail_test') as fail:
                self.assertEqual(ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['nic0']}, requested), '3')
                fail.assert_not_called()
            self.assertIn('gids/3', phdl.exec.call_args_list[1].args[0])

    def test_resolve_auto_without_candidates(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'nodeA': 'GIDENT|nic0|1|fe80:0000:0000:0000:0000:0000:0000:0001|RoCE v2\n'}
        with patch.object(ibperf_lib, 'fail_test') as fail:
            self.assertIsNone(ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['nic0']}, 'auto'))
            fail.assert_called()

    def test_resolve_invalid_index_before_exec(self):
        phdl = MagicMock()
        with patch.object(ibperf_lib, 'fail_test') as fail:
            self.assertIsNone(ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['nic0']}, 'abc'))
            fail.assert_called_once()
        phdl.exec.assert_not_called()

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
