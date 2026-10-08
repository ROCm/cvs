# cvs/lib/unittests/test_ibperf_lib.py
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import cvs.lib.ibperf_lib as ibperf_lib


SAMPLE_CONFIG = Path(ibperf_lib.__file__).parents[1] / 'input/config_file/ibperf/ibperf_config.json'

# perftest connection setup and ROCm buffer allocation happen before -D starts counting.
PERFTEST_SETUP_HEADROOM_S = 5


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


class TestRunIbPerfBwTest(unittest.TestCase):
    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_bw_pps', return_value={})
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_sample_duration_finishes_before_results_are_read(self, _dmabuf, _bw_pps, mock_sleep):
        config = json.loads(SAMPLE_CONFIG.read_text())['ibperf']
        duration = int(config['duration'])
        nodes = ('node1', 'node2')
        gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
        phdl = MagicMock()

        ibperf_lib.run_ib_perf_bw_test(
            MagicMock(),
            phdl,
            'ib_write_bw',
            gpu_numa_dict,
            gpu_nic_dict,
            {n: {} for n in nodes},
            '/opt/perftest/bin',
            config['msg_size_list'][0],
            config['gid_index'],
            config['qp_count_list'][0],
            int(config['port_no']),
            duration,
        )

        _, client_cmd = phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertIn(f' -D {duration} ', client_cmd)
        results_read_after = max(c.args[0] for c in mock_sleep.call_args_list)
        self.assertLess(duration + PERFTEST_SETUP_HEADROOM_S, results_read_after)


if __name__ == '__main__':
    unittest.main()
