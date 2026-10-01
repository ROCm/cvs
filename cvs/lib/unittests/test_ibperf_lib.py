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


class TestRunIbPerfLatTest(unittest.TestCase):
    SERVER = '10.0.0.1'
    CLIENT = '10.0.0.2'
    APP_PATH = '/opt/perftest/bin'

    def setUp(self):
        nodes = [self.SERVER, self.CLIENT]
        self.gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        self.gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
        self.bck_nic_dict = {n: {} for n in nodes}
        lat_keys = ('t_min', 't_max', 't_typical', 't_avg', 't_stdev', 't_99_pct', 't_99_9_pct')
        self.lat_numb = {
            n: {k: f'{n_idx}.{k_idx}' for k_idx, k in enumerate(lat_keys)} for n_idx, n in enumerate(nodes)
        }
        self.phdl = MagicMock()
        self.shdl = MagicMock()

    def _run(self, dmabuf_supported=False):
        with (
            patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=dmabuf_supported),
            patch.object(ibperf_lib, 'get_ib_lat_numb', return_value=self.lat_numb),
            patch.object(ibperf_lib.time, 'sleep'),
        ):
            result = ibperf_lib.run_ib_perf_lat_test(
                self.shdl,
                self.phdl,
                'ib_write_lat',
                self.gpu_numa_dict,
                self.gpu_nic_dict,
                self.bck_nic_dict,
                self.APP_PATH,
                64,
                3,
                port_no=1516,
            )
        perftest_cmds = {self.SERVER: [], self.CLIENT: []}
        for call in self.phdl.exec_cmd_list.call_args_list:
            server_cmd, client_cmd = call.args[0]
            if 'numactl' in server_cmd:
                perftest_cmds[self.SERVER].append(server_cmd)
            if 'numactl' in client_cmd:
                perftest_cmds[self.CLIENT].append(client_cmd)
        return result, perftest_cmds

    def test_builds_latency_commands_without_bandwidth_flags(self):
        _, perftest_cmds = self._run()
        self.assertEqual(
            perftest_cmds[self.SERVER][0],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' -x 3 --report_gbits -F -p 1516 -s 64 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            perftest_cmds[self.CLIENT][0],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' -x 3 --report_gbits -F -p 1516 -s 64 10.0.0.1 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        for node, cmds in perftest_cmds.items():
            self.assertEqual(len(cmds), 8, node)
            for gpu_no, cmd in enumerate(cmds):
                tokens = cmd.split()
                self.assertEqual(tokens[tokens.index('-d') + 1], f'rdma{gpu_no}')
                self.assertEqual(tokens[tokens.index('-p') + 1], str(1516 + gpu_no))
                self.assertIn(f'--use_rocm={gpu_no}', tokens)
                self.assertNotIn('--use_rocm_dmabuf', tokens)
                for bw_only_flag in ('-b', '-D', '-q'):
                    self.assertNotIn(bw_only_flag, tokens)
        for cmd in perftest_cmds[self.CLIENT]:
            self.assertIn(self.SERVER, cmd.split())

    def test_adds_dmabuf_flag_when_supported(self):
        _, perftest_cmds = self._run(dmabuf_supported=True)
        for cmds in perftest_cmds.values():
            for cmd in cmds:
                self.assertIn('--use_rocm_dmabuf', cmd.split())

    def test_collects_latency_for_every_gpu(self):
        result, _ = self._run()
        for node in (self.SERVER, self.CLIENT):
            self.assertEqual(result[node], {i: self.lat_numb[node] for i in range(8)})


if __name__ == '__main__':
    unittest.main()
