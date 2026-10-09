# cvs/lib/unittests/test_ibperf_lib.py
import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import cvs.lib.ibperf_lib as ibperf_lib


class TestConfiguredTests(unittest.TestCase):
    def test_bw_tests_follow_config_order(self):
        tests = ibperf_lib.configured_bw_tests({'ib_bw_test_list': ['ib_send_bw', 'ib_write_bw']})
        self.assertEqual(tests, ['ib_send_bw', 'ib_write_bw'])

    def test_lat_tests_include_read_lat(self):
        requested = ['ib_write_lat', 'ib_send_lat', 'ib_read_lat']
        self.assertEqual(ibperf_lib.configured_lat_tests({'ib_lat_test_list': requested}), requested)

    def test_missing_key_uses_legacy_defaults(self):
        self.assertEqual(ibperf_lib.configured_bw_tests({}), ['ib_write_bw', 'ib_read_bw', 'ib_send_bw'])
        self.assertEqual(ibperf_lib.configured_lat_tests({}), ['ib_write_lat', 'ib_send_lat'])

    def test_defaults_are_copies(self):
        tests = ibperf_lib.configured_bw_tests({})
        tests.append('ib_bogus_bw')
        self.assertEqual(ibperf_lib.DEFAULT_BW_TESTS, ['ib_write_bw', 'ib_read_bw', 'ib_send_bw'])

    def test_empty_list_is_allowed(self):
        self.assertEqual(ibperf_lib.configured_bw_tests({'ib_bw_test_list': []}), [])

    def test_rejects_unknown_test(self):
        for value in ('ib_wrte_bw', 'ib_write_lat', 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, str(value)):
                    ibperf_lib.configured_bw_tests({'ib_bw_test_list': [value]})

    def test_rejects_non_list(self):
        with self.assertRaisesRegex(ValueError, 'ib_lat_test_list'):
            ibperf_lib.configured_lat_tests({'ib_lat_test_list': 'ib_write_lat'})

    def test_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, 'more than once'):
            ibperf_lib.configured_bw_tests({'ib_bw_test_list': ['ib_write_bw', 'ib_write_bw']})


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

    def test_latency_chart_uses_available_rows(self):
        latency = {'t_min': '10.66', 't_max': '13.76', 't_avg': '10.98', 't_stdev': '0.10', 't_99_pct': '12.01'}
        results = {'ib_read_lat': {8192: {'server': {}, 'client': {0: latency}}}}
        with tempfile.TemporaryDirectory() as temp_dir:
            chart_path = Path(temp_dir) / 'ib_lat_perf.xlsx'
            ibperf_lib.generate_ibperf_lat_chart(results, str(chart_path))
            with zipfile.ZipFile(chart_path) as chart:
                self.assertIn('ib_read_lat_lat', chart.read('xl/workbook.xml').decode())
                self.assertIn('10.98', chart.read('xl/worksheets/sheet1.xml').decode())

    def test_latency_chart_rejects_test_without_rows(self):
        results = {'ib_read_lat': {8192: {'server': {}, 'client': {}}}}
        with self.assertRaisesRegex(ValueError, 'No latency results for ib_read_lat'):
            ibperf_lib.generate_ibperf_lat_chart(results)

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
    def test_read_latency_records_client_without_server_table(self):
        server_log = 'allocated 65536 bytes of GPU buffer\n'
        client_log = server_log + '8192 1000 10.66 13.76 10.97 10.98 0.10 12.01 13.76\n'
        nodes = ('server', 'client')
        phdl = MagicMock()
        phdl.exec.side_effect = lambda cmd, **_kwargs: (
            {'server': server_log, 'client': client_log} if cmd.startswith('cat /tmp/ib_perf_') else {}
        )
        gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}

        with (
            patch.object(ibperf_lib.time, 'sleep'),
            patch.object(ibperf_lib, 'fail_test') as mock_fail,
            patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False),
        ):
            results = ibperf_lib.run_ib_perf_lat_test(
                MagicMock(),
                phdl,
                'ib_read_lat',
                gpu_numa_dict,
                gpu_nic_dict,
                {n: {} for n in nodes},
                '/opt/perftest/bin',
                8192,
                1,
            )

        mock_fail.assert_not_called()
        self.assertEqual(results['server'], {})
        self.assertEqual(len(results['client']), 8)
        self.assertTrue(all(row['t_avg'] == '10.98' for row in results['client'].values()))

    def test_write_latency_requires_server_table(self):
        server_log = 'allocated 65536 bytes of GPU buffer\n'
        client_log = server_log + '8192 1000 10.66 13.76 10.97 10.98 0.10 12.01 13.76\n'
        phdl = MagicMock()
        phdl.exec.return_value = {'server': server_log, 'client': client_log}
        with patch.object(ibperf_lib.time, 'sleep'), patch.object(ibperf_lib, 'fail_test') as mock_fail:
            results = ibperf_lib.get_ib_lat_numb(phdl, 8192, 'cat /tmp/ib_perf_0_logs')

        self.assertEqual(results['client']['t_avg'], '10.98')
        self.assertNotIn('server', results)
        mock_fail.assert_called_once()
        self.assertIn('server', mock_fail.call_args.args[0])

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
