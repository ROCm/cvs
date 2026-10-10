# cvs/lib/unittests/test_ibperf_lib.py
import json
import os
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


class TestVerifyExpectedBw(unittest.TestCase):
    def setUp(self):
        self.expected = {'ib_write_bw': {'8192': {'8': '180.0', '16': '200.0'}}}
        self.results = {
            'node1': {0: {'bw': '190.0', 'pps': '1.0'}, 1: {'bw': '190.0', 'pps': '1.0'}},
            'node2': {2: {'bw': '190.0', 'pps': '1.0'}, 3: {'bw': '190.0', 'pps': '1.0'}},
        }

    @patch.object(ibperf_lib, 'fail_test')
    def test_int_msg_size_and_qp_match_string_keys(self, mock_fail):
        self.assertTrue(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, self.expected))
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_string_qp_count_matches(self, mock_fail):
        for instances in self.results.values():
            for result in instances.values():
                result['bw'] = '210.0'
        self.assertTrue(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, '16', self.results, self.expected))
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_uses_qp_specific_threshold(self, mock_fail):
        self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 16, self.results, self.expected))
        self.assertEqual(mock_fail.call_count, 4)
        for call in mock_fail.call_args_list:
            self.assertIn('less than the expected BW', call.args[0])
            self.assertIn('qp=16', call.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_forced_threshold_fails_every_gpu_on_two_nodes(self, mock_fail):
        results = {node: {gpu: {'bw': '535.13'} for gpu in range(8)} for node in ('node1', 'node2')}
        expected = {'ib_write_bw': {'8192': {'8': '100000'}}}

        self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, results, expected))
        self.assertEqual(mock_fail.call_count, 16)
        for node in results:
            for gpu in results[node]:
                self.assertTrue(any(f'on node {node} gpu={gpu}' in call.args[0] for call in mock_fail.call_args_list))

    @patch.object(ibperf_lib, 'fail_test')
    def test_single_low_gpu_fails_with_node_and_gpu(self, mock_fail):
        self.results['node2'][3]['bw'] = '150.0'
        self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, self.expected))
        mock_fail.assert_called_once()
        self.assertIn('node2', mock_fail.call_args.args[0])
        self.assertIn('gpu=3', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_bw_equal_to_threshold_passes(self, mock_fail):
        for instances in self.results.values():
            for result in instances.values():
                result['bw'] = '180.0'
        self.assertTrue(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, self.expected))
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_not_gated_returns_none(self, mock_fail):
        for test, size, qp in (('ib_read_bw', 8192, 8), ('ib_write_bw', 4096, 8), ('ib_write_bw', 8192, 32)):
            with self.subTest(test=test, size=size, qp=qp):
                self.assertIsNone(ibperf_lib.verify_expected_bw(test, size, qp, self.results, self.expected))
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_empty_expected_results_returns_none(self, mock_fail):
        for expected in ({}, None):
            with self.subTest(expected=expected):
                self.assertIsNone(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, expected))
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_flat_threshold_is_config_error(self, mock_fail):
        expected = {'ib_write_bw': {'8192': '180.0'}}
        self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, expected))
        mock_fail.assert_called_once()
        self.assertIn('must map QP count', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_non_numeric_threshold_fails(self, mock_fail):
        expected = {'ib_write_bw': {'8192': {'8': '<changeme>'}}}
        self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, expected))
        mock_fail.assert_called_once()
        self.assertIn('Invalid expected BW', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_non_finite_threshold_fails(self, mock_fail):
        self.results['node1'][0]['bw'] = '1.0'
        for raw in ('NaN', 'nan', 'inf', '-Infinity', float('nan'), float('inf')):
            with self.subTest(raw=raw):
                mock_fail.reset_mock()
                expected = {'ib_write_bw': {'8192': {'8': raw}}}
                self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, expected))
                mock_fail.assert_called_once()
                self.assertIn('Invalid expected BW', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_node_without_results_fails(self, mock_fail):
        self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, {'node1': {}}, self.expected))
        mock_fail.assert_called_once()
        self.assertIn('No BW results', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_no_results_fails(self, mock_fail):
        self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, {}, self.expected))
        mock_fail.assert_called_once()
        self.assertIn('No BW results', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_non_numeric_measurement_fails(self, mock_fail):
        self.results['node2'][3]['bw'] = 'invalid'
        self.assertFalse(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, self.expected))
        mock_fail.assert_called_once()
        self.assertIn('Invalid actual BW', mock_fail.call_args.args[0])
        self.assertIn('node2 gpu=3', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_logs_verdict(self, mock_fail):
        with self.assertLogs(ibperf_lib.log, level='INFO') as captured:
            self.assertTrue(ibperf_lib.verify_expected_bw('ib_write_bw', 8192, 8, self.results, self.expected))
        self.assertTrue(any('Expected BW met' in record for record in captured.output))
        with self.assertLogs(ibperf_lib.log, level='INFO') as captured:
            self.assertIsNone(ibperf_lib.verify_expected_bw('ib_write_bw', 4096, 8, self.results, self.expected))
        self.assertTrue(any('not gated' in record for record in captured.output))
        mock_fail.assert_not_called()


class TestVerifyExpectedLat(unittest.TestCase):
    def setUp(self):
        self.expected = {'ib_write_lat': {'64': '5.0'}}
        self.results = {
            'node1': {0: {'t_avg': '3.0'}, 1: {'t_avg': '3.0'}},
            'node2': {2: {'t_avg': '3.0'}, 3: {'t_avg': '3.0'}},
        }

    @patch.object(ibperf_lib, 'fail_test')
    def test_int_msg_size_matches_and_passes(self, mock_fail):
        self.assertTrue(ibperf_lib.verify_expected_lat('ib_write_lat', 64, self.results, self.expected))
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_high_latency_fails_per_gpu(self, mock_fail):
        self.results['node2'][3]['t_avg'] = '6.5'
        self.assertFalse(ibperf_lib.verify_expected_lat('ib_write_lat', 64, self.results, self.expected))
        mock_fail.assert_called_once()
        self.assertIn('greater than the expected latency', mock_fail.call_args.args[0])
        self.assertIn('gpu=3', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_not_gated_returns_none(self, mock_fail):
        self.assertIsNone(ibperf_lib.verify_expected_lat('ib_write_lat', 128, self.results, self.expected))
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_dict_threshold_is_config_error(self, mock_fail):
        expected = {'ib_write_lat': {'64': {'8': '5.0'}}}
        self.assertFalse(ibperf_lib.verify_expected_lat('ib_write_lat', 64, self.results, expected))
        mock_fail.assert_called_once()
        self.assertIn('must be a latency', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_non_numeric_threshold_fails(self, mock_fail):
        expected = {'ib_write_lat': {'64': '<changeme>'}}
        self.assertFalse(ibperf_lib.verify_expected_lat('ib_write_lat', 64, self.results, expected))
        mock_fail.assert_called_once()
        self.assertIn('Invalid expected latency', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_non_finite_threshold_fails(self, mock_fail):
        self.results['node1'][0]['t_avg'] = '999.0'
        for raw in ('NaN', 'nan', 'inf', 'Infinity', float('nan'), float('inf')):
            with self.subTest(raw=raw):
                mock_fail.reset_mock()
                expected = {'ib_write_lat': {'64': raw}}
                self.assertFalse(ibperf_lib.verify_expected_lat('ib_write_lat', 64, self.results, expected))
                mock_fail.assert_called_once()
                self.assertIn('Invalid expected latency', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_node_without_results_fails(self, mock_fail):
        self.assertFalse(ibperf_lib.verify_expected_lat('ib_write_lat', 64, {'node1': {}}, self.expected))
        mock_fail.assert_called_once()
        self.assertIn('No latency results', mock_fail.call_args.args[0])
        self.assertIn('node1', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_no_results_fails(self, mock_fail):
        self.assertFalse(ibperf_lib.verify_expected_lat('ib_write_lat', 64, {}, self.expected))
        mock_fail.assert_called_once()
        self.assertIn('No latency results', mock_fail.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_non_numeric_measurement_fails(self, mock_fail):
        self.results['node2'][3]['t_avg'] = 'invalid'
        self.assertFalse(ibperf_lib.verify_expected_lat('ib_write_lat', 64, self.results, self.expected))
        mock_fail.assert_called_once()
        self.assertIn('Invalid actual latency', mock_fail.call_args.args[0])
        self.assertIn('node2 gpu=3', mock_fail.call_args.args[0])


class TestFindUnmatchedThresholds(unittest.TestCase):
    def test_sample_config_8388608_is_reported(self):
        expected = {'ib_write_bw': {'8192': {'8': '180', '16': '200'}, '8388608': {'8': '280'}}}
        self.assertEqual(
            ibperf_lib.find_unmatched_thresholds(expected, [2, 8192, 65536], ['8', '16']),
            ['ib_write_bw.8388608'],
        )

    def test_unmatched_qp_reported(self):
        expected = {'ib_write_bw': {'8192': {'8': '180', '16': '200'}}}
        self.assertEqual(
            ibperf_lib.find_unmatched_thresholds(expected, [8192], [8]),
            ['ib_write_bw.8192.16'],
        )

    def test_latency_entries_checked_by_msg_size_only(self):
        expected = {'ib_write_lat': {'64': '5.0', '128': '6.0'}}
        self.assertEqual(ibperf_lib.find_unmatched_thresholds(expected, [64], [8]), ['ib_write_lat.128'])

    def test_all_reachable_returns_empty(self):
        expected = {'ib_write_bw': {'8192': {'8': '180'}}, 'ib_write_lat': {'64': '5.0'}}
        self.assertEqual(ibperf_lib.find_unmatched_thresholds(expected, [64, 8192], [8]), [])
        self.assertEqual(ibperf_lib.find_unmatched_thresholds({}, [64, 8192], [8]), [])
        self.assertEqual(ibperf_lib.find_unmatched_thresholds(None, [64, 8192], [8]), [])

    def test_shipped_sample_config_has_no_unmatched_thresholds(self):
        config_path = os.path.join(
            os.path.dirname(ibperf_lib.__file__), '..', 'input', 'config_file', 'ibperf', 'ibperf_config.json'
        )
        with open(config_path) as config_file:
            config = json.load(config_file)['ibperf']
        self.assertEqual(
            ibperf_lib.find_unmatched_thresholds(
                config['expected_results'], config['msg_size_list'], config['qp_count_list']
            ),
            [],
        )


if __name__ == '__main__':
    unittest.main()
