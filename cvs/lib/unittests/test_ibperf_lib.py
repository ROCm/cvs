import json
import os
import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, call as mock_call, patch
import cvs.lib.ibperf_lib as ibperf_lib
from cvs.lib import globals


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


def _kfd_output(nodes):
    lines = []
    for node, simd_count, location_id, domain in sorted(nodes, key=lambda entry: str(entry[0])):
        path = f'/sys/class/kfd/kfd/topology/nodes/{node}/properties'
        lines.append(f'{path}:simd_count {simd_count}')
        lines.append(f'{path}:location_id {location_id}')
        if domain is not None:
            lines.append(f'{path}:domain {domain}')
    return '\n'.join(lines) + '\n'


def _env_output(**values):
    return ''.join(
        f'{ibperf_lib._GPU_ENV_TAG} {var}={values.get(var, "")}\n' for var in ibperf_lib._GPU_VISIBILITY_ENV_VARS
    )


_TICKET_KFD = _kfd_output(
    [(0, 0, 0, 0), (1, 0, 0, 0)]
    + [(node, 304, bus << 8, 0) for node, bus in enumerate((0x75, 0x05, 0x65, 0x15, 0xF5, 0x85, 0xE5, 0x95), 2)]
)
_TICKET_GPU_NIC = {
    node: {
        f'card{card}': {'rdma_dev': f'rdma{card}', 'gpu_bdf': f'0000:{bus:02x}:00.0'}
        for card, bus in enumerate((0x05, 0x15, 0x65, 0x75, 0x85, 0x95, 0xE5, 0xF5))
    }
    for node in ('node1', 'node2')
}
_TICKET_GPU_NIC['node1']['card7']['gpu_bdf'] = '0000:F5:00.0'
_TICKET_EXPECTED = {'card0': 1, 'card1': 3, 'card2': 2, 'card3': 0, 'card4': 5, 'card5': 7, 'card6': 6, 'card7': 4}


class TestParseKfdGpuBdfList(unittest.TestCase):
    def test_orders_gpus_by_kfd_node_and_skips_cpus(self):
        self.assertEqual(
            ibperf_lib.parse_kfd_gpu_bdf_list(_TICKET_KFD),
            [
                '0000:75:00.0',
                '0000:05:00.0',
                '0000:65:00.0',
                '0000:15:00.0',
                '0000:f5:00.0',
                '0000:85:00.0',
                '0000:e5:00.0',
                '0000:95:00.0',
            ],
        )

    def test_numeric_node_sort(self):
        output = _kfd_output([(10, 304, 0x75 << 8, 0), (2, 304, 0x05 << 8, 0)])
        self.assertEqual(ibperf_lib.parse_kfd_gpu_bdf_list(output), ['0000:05:00.0', '0000:75:00.0'])

    def test_decodes_domain_device_function(self):
        output = _kfd_output([(2, 304, (0x05 << 8) | (1 << 3) | 1, 1)])
        self.assertEqual(ibperf_lib.parse_kfd_gpu_bdf_list(output), ['0001:05:01.1'])

    def test_missing_domain_defaults_to_zero(self):
        self.assertEqual(ibperf_lib.parse_kfd_gpu_bdf_list(_kfd_output([(2, 304, 0x05 << 8, None)])), ['0000:05:00.0'])

    def test_empty_output(self):
        self.assertEqual(ibperf_lib.parse_kfd_gpu_bdf_list(''), [])

    def test_ignores_env_tag_lines(self):
        env_output = _env_output(ROCR_VISIBLE_DEVICES='0,1')
        self.assertEqual(
            ibperf_lib.parse_kfd_gpu_bdf_list(_TICKET_KFD + env_output), ibperf_lib.parse_kfd_gpu_bdf_list(_TICKET_KFD)
        )
        self.assertEqual(ibperf_lib.parse_kfd_gpu_bdf_list(env_output), [])


class TestParseGpuVisibilityEnv(unittest.TestCase):
    def test_all_empty_returns_empty_dict(self):
        self.assertEqual(ibperf_lib.parse_gpu_visibility_env(_TICKET_KFD + _env_output()), {})

    def test_reports_non_empty_vars(self):
        output = _env_output(ROCR_VISIBLE_DEVICES='0,1', GPU_DEVICE_ORDINAL='3')
        self.assertEqual(
            ibperf_lib.parse_gpu_visibility_env(output), {'ROCR_VISIBLE_DEVICES': '0,1', 'GPU_DEVICE_ORDINAL': '3'}
        )

    def test_no_tag_lines_returns_empty_dict(self):
        self.assertEqual(ibperf_lib.parse_gpu_visibility_env(_TICKET_KFD), {})


class TestGetHipDeviceDict(unittest.TestCase):
    @patch.object(ibperf_lib, 'fail_test')
    def test_maps_cards_to_hip_ordinal_by_bdf(self, mock_fail_test):
        phdl = MagicMock()
        phdl.exec.return_value = {node: _TICKET_KFD + _env_output() for node in _TICKET_GPU_NIC}
        self.assertEqual(
            ibperf_lib.get_hip_device_dict(phdl, _TICKET_GPU_NIC),
            {node: _TICKET_EXPECTED for node in _TICKET_GPU_NIC},
        )
        phdl.exec.assert_called_once_with(ibperf_lib._HIP_DEVICE_PROBE_CMD, print_console=False)
        mock_fail_test.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_probe_cmd_reports_all_visibility_vars(self, _mock_fail_test):
        self.assertTrue(ibperf_lib._HIP_DEVICE_PROBE_CMD.startswith(f'{GPU_ENV_UNSET} sh -c '))
        self.assertIn('grep -HE', ibperf_lib._HIP_DEVICE_PROBE_CMD)
        for var in ibperf_lib._GPU_VISIBILITY_ENV_VARS:
            self.assertIn(f'"${{{var}-}}"', ibperf_lib._HIP_DEVICE_PROBE_CMD)

    @patch.object(ibperf_lib, 'fail_test')
    def test_visibility_env_fails_node_and_falls_back(self, mock_fail_test):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _TICKET_KFD + _env_output(ROCR_VISIBLE_DEVICES='0,1,2,3'),
            'node2': _TICKET_KFD + _env_output(),
        }
        result = ibperf_lib.get_hip_device_dict(phdl, _TICKET_GPU_NIC)
        self.assertEqual(result['node1'], {f'card{card}': card for card in range(8)})
        self.assertEqual(result['node2'], _TICKET_EXPECTED)
        mock_fail_test.assert_called_once()
        message = mock_fail_test.call_args.args[0]
        for part in ('node1', 'ROCR_VISIBLE_DEVICES=0,1,2,3', 'KFD'):
            self.assertIn(part, message)

    @patch.object(ibperf_lib, 'fail_test')
    def test_unmatched_bdf_fails_and_falls_back_to_card_index(self, mock_fail_test):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _TICKET_KFD + _env_output()}
        gpu_nic_dict = {'node1': {card: entry.copy() for card, entry in _TICKET_GPU_NIC['node1'].items()}}
        gpu_nic_dict['node1']['card3']['gpu_bdf'] = '0000:aa:00.0'
        result = ibperf_lib.get_hip_device_dict(phdl, gpu_nic_dict)['node1']
        self.assertEqual(result, {**_TICKET_EXPECTED, 'card3': 3})
        mock_fail_test.assert_called_once()
        for part in ('node1', 'card3', '0000:aa:00.0'):
            self.assertIn(part, mock_fail_test.call_args.args[0])

    @patch.object(ibperf_lib, 'fail_test')
    def test_missing_node_output(self, mock_fail_test):
        phdl = MagicMock()
        phdl.exec.return_value = {}
        result = ibperf_lib.get_hip_device_dict(phdl, _TICKET_GPU_NIC)
        self.assertEqual(result, {node: {f'card{card}': card for card in range(8)} for node in _TICKET_GPU_NIC})
        self.assertEqual(mock_fail_test.call_count, 16)


SAMPLE_CONFIG = Path(ibperf_lib.__file__).parents[1] / 'input/config_file/ibperf/ibperf_config.json'

# perftest connection setup and ROCm buffer allocation happen before -D starts counting.
PERFTEST_SETUP_HEADROOM_S = 5


class TestIbperfLib(unittest.TestCase):
    def test_get_ibperf_rdma_devices_defaults_to_discovered_cards(self):
        _, nic, _ = gpu_maps(['four'], 4)
        _, ten, _ = gpu_maps(['ten'], 10)
        nic.update(ten)
        self.assertEqual(
            ibperf_lib.get_ibperf_rdma_devices(nic, {'four': {}, 'ten': {}}),
            {'four': [f'rdma{g}' for g in range(4)], 'ten': sorted((f'rdma{g}' for g in range(10)))},
        )

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

    def test_resolve_auto_on_infiniband_selects_port_gid(self):
        table = (
            'GIDENT|mlx5_0|0|fe80:0000:0000:0000:a088:c203:0001:0001|IB/RoCE v1|InfiniBand\n'
            'GIDENT|mlx5_0|1|fe80:0000:0000:0000:0000:0000:0000:0000|IB/RoCE v1|InfiniBand\n'
            'GIDENT|mlx5_0|3|fe80:0000:0000:0000:0000:0000:0000:0000|IB/RoCE v1|InfiniBand\n'
        )
        probe = (
            'DEVICE:mlx5_0\nLINK_LAYER:InfiniBand\nGID:fe80:0000:0000:0000:a088:c203:0001:0001\nGID_TYPE:IB/RoCE v1\n'
        )
        phdl = MagicMock()
        phdl.exec.side_effect = [{'nodeA': table, 'nodeB': table}, {'nodeA': probe, 'nodeB': probe}]
        with patch.object(ibperf_lib, 'fail_test') as fail:
            self.assertEqual(
                ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['mlx5_0'], 'nodeB': ['mlx5_0']}, 'auto'), '0'
            )
            fail.assert_not_called()
        self.assertIn('link_layer', phdl.exec.call_args_list[0].args[0])
        self.assertIn('gids/0', phdl.exec.call_args_list[1].args[0])

    def test_resolve_explicit_index_on_infiniband(self):
        gids = {
            '0': 'fe80:0000:0000:0000:a088:c203:0001:0001',
            '3': 'fe80:0000:0000:0000:0000:0000:0000:0000',
        }
        for requested, value in gids.items():
            phdl = MagicMock()
            phdl.exec.return_value = {
                'nodeA': f'DEVICE:mlx5_0\nLINK_LAYER:InfiniBand\nGID:{value}\nGID_TYPE:IB/RoCE v1\n'
            }
            with self.subTest(requested=requested), patch.object(ibperf_lib, 'fail_test') as fail:
                self.assertEqual(ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['mlx5_0']}, requested), requested)
                fail.assert_not_called()
            self.assertEqual(phdl.exec.call_count, 1)

    def test_resolve_explicit_index_on_infiniband_rejects_zero_and_missing(self):
        outputs = {
            'EMPTY': 'DEVICE:mlx5_0\nLINK_LAYER:InfiniBand\nGID:0000:0000:0000:0000:0000:0000:0000:0000\nGID_TYPE:\n',
            'MISSING': 'DEVICE:mlx5_0\nLINK_LAYER:InfiniBand\nGID_MISSING\n',
        }
        for name, output in outputs.items():
            phdl = MagicMock()
            phdl.exec.return_value = {'nodeA': output}
            with self.subTest(name=name), patch.object(ibperf_lib, 'fail_test') as fail:
                self.assertIsNone(ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['mlx5_0']}, '3'))
                fail.assert_called_once()
                self.assertIn('mlx5_0', fail.call_args.args[0])

    def test_resolve_invalid_index_before_exec(self):
        phdl = MagicMock()
        with patch.object(ibperf_lib, 'fail_test') as fail:
            self.assertIsNone(ibperf_lib.resolve_gid_index(phdl, {'nodeA': ['nic0']}, 'abc'))
            fail.assert_called_once()
        phdl.exec.assert_not_called()

    @patch('xlsxwriter.Workbook')
    def test_generate_ibperf_bw_chart(self, mock_workbook_class):
        worksheet = mock_workbook_class.return_value.add_worksheet.return_value
        res_dict = {
            'ib_write_bw': {
                1024: {1: {'node1': {i: {'pps': str(10.0 + i), 'bw': str(1.0 + i * 0.1)} for i in range(8)}}}
            }
        }
        ibperf_lib.generate_ibperf_bw_chart(res_dict, 'test.xlsx', dmabuf_dict={'ib_write_bw': True})
        worksheet.merge_range.assert_any_call(
            'A1:T1', 'Test ib_write_bw - BW, MPPS Numbers for 1 QPs (DMA-BUF on)', ANY
        )
        self.assertTrue(mock_workbook_class.return_value.close.called)

        worksheet.reset_mock()
        ibperf_lib.generate_ibperf_bw_chart(res_dict, 'test.xlsx')
        worksheet.merge_range.assert_any_call('A1:T1', 'Test ib_write_bw - BW, MPPS Numbers for 1 QPs', ANY)

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
        worksheet = mock_workbook_class.return_value.add_worksheet.return_value
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
        ibperf_lib.generate_ibperf_lat_chart(res_dict, 'test.xlsx', dmabuf_dict={'ib_write_lat': False})
        worksheet.merge_range.assert_any_call('A1:Z1', 'Test ib_write_lat - latency results (DMA-BUF off)', ANY)
        self.assertTrue(mock_workbook_class.return_value.close.called)

        worksheet.reset_mock()
        ibperf_lib.generate_ibperf_lat_chart(res_dict, 'test.xlsx')
        worksheet.merge_range.assert_any_call('A1:Z1', 'Test ib_write_lat - latency results', ANY)


BW_LOG = '8192 bytes of GPU buffer\n 8192       5000           512.00             517.14             7.89\n'
GPU_ENV_UNSET = 'env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL'
LAUNCH = f'{GPU_ENV_UNSET} bash /tmp/ib_cmds_file.txt'


class TestPerRunCleanup(unittest.TestCase):
    def test_clears_stale_files_without_sudo_before_writing_commands(self):
        for run in (ibperf_lib.run_ib_perf_bw_test, ibperf_lib.run_ib_perf_lat_test):
            with self.subTest(run=run.__name__):
                phdl = MagicMock()
                phdl.exec.side_effect = [None, None, RuntimeError('stop after cleanup')]
                with self.assertRaises(RuntimeError):
                    run(phdl, 'ib_write_bw', {}, {}, {}, '/opt/pt/bin', 64, 3, hip_dev_dict={})
                self.assertEqual(
                    [c.args[0] for c in phdl.exec.call_args_list[:3]],
                    ['rm -rf /tmp/ib_cmds_file.txt', 'rm -rf /tmp/ib_perf*', 'touch /tmp/ib_cmds_file.txt'],
                )


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
    @patch.object(ibperf_lib, 'fail_test')
    def test_stops_polling_once_every_node_reports(self, mock_fail):
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

    @patch.object(ibperf_lib, 'fail_test')
    def test_exited_node_without_result_is_read_once(self, mock_fail):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': BW_LOG, 'node2': "Couldn't initialize ROCm device\n"}

        res = ibperf_lib.get_ib_bw_pps(phdl, 8192, 'cat /tmp/ib_perf_0_logs', running=[])

        self.assertEqual(res, {'node1': {'bw': '517.14', 'pps': '7.89'}})
        self.assertEqual(phdl.exec.call_count, 2)
        self.assertTrue(any('perftest exited without a result' in c.args[0] for c in mock_fail.call_args_list))

    @patch.object(ibperf_lib, 'fail_test')
    def test_nonzero_exit_fails_even_with_result_row(self, mock_fail):
        phdl = MagicMock()
        phdl.exec.side_effect = [
            {'node1': BW_LOG + 'perftest_rc=0\n', 'node2': BW_LOG},
            {'node1': BW_LOG + 'perftest_rc=0\n', 'node2': BW_LOG + 'perftest_rc=1\n'},
        ]

        ibperf_lib.get_ib_bw_pps(phdl, 8192, 'cat /tmp/ib_perf_3_logs', instance_no=3, running=['node2'])

        mock_fail.assert_called_once_with('perftest instance 3 on node node2 exited with code 1')


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


class TestReadLatencyResults(unittest.TestCase):
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
        ):
            results = ibperf_lib.run_ib_perf_lat_test(
                phdl,
                'ib_read_lat',
                gpu_numa_dict,
                gpu_nic_dict,
                {n: {} for n in nodes},
                '/opt/perftest/bin',
                8192,
                1,
                hip_dev_dict={n: {f'card{g}': g for g in range(8)} for n in nodes},
            )

        mock_fail.assert_not_called()
        self.assertEqual(results['server'], {})
        self.assertEqual(len(results['client']), 8)
        self.assertTrue(all(row['t_avg'] == '10.98' for row in results['client'].values()))

    @patch.object(ibperf_lib, 'PERFTEST_LAT_RESULT_TIMEOUT_S', 0)
    def test_write_latency_requires_server_table(self):
        server_log = 'allocated 65536 bytes of GPU buffer\n'
        client_log = server_log + '8192 1000 10.66 13.76 10.97 10.98 0.10 12.01 13.76\n'
        phdl = MagicMock()
        phdl.exec.return_value = {'server': server_log, 'client': client_log}
        with patch.object(ibperf_lib, 'fail_test') as mock_fail:
            results = ibperf_lib.get_ib_lat_numb(phdl, 8192, 'cat /tmp/ib_perf_0_logs')

        self.assertEqual(results['client']['t_avg'], '10.98')
        self.assertNotIn('server', results)
        mock_fail.assert_called_once()
        self.assertIn('server', mock_fail.call_args.args[0])

    def test_read_latency_server_nonzero_exit_fails(self):
        server_log = 'allocated 65536 bytes of GPU buffer\npartial line'
        client_log = 'allocated 65536 bytes of GPU buffer\n8192 1000 10.66 13.76 10.97 10.98 0.10 12.01 13.76\n'
        phdl = MagicMock()
        phdl.exec.side_effect = [
            {'server': server_log, 'client': client_log},
            {'server': server_log + 'perftest_rc=2\n', 'client': client_log},
        ]
        with patch.object(ibperf_lib, 'fail_test') as mock_fail:
            ibperf_lib.get_ib_lat_numb(
                phdl, 8192, 'cat /tmp/ib_perf_0_logs', instance_no=0, result_nodes=['client'], running=['server']
            )

        mock_fail.assert_called_once_with('perftest instance 0 on node server exited with code 2')


class IbPerfCommandFixtures:
    def active_nodes(self):
        excluded = self.phdl.prune_nodes.call_args.args[0] if self.phdl.prune_nodes.called else []
        return [node for node in self.bck_nic_dict if node not in excluded]

    def setUp(self):
        globals.error_list = []
        nodes = ('node1', 'node2')
        self.phdl = MagicMock()
        self.gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        self.gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
        self.bck_nic_dict = {n: {} for n in nodes}
        self.hip_dev_dict = {n: {f'card{g}': g for g in range(8)} for n in nodes}
        lat = dict.fromkeys(('t_min', 't_max', 't_typical', 't_avg', 't_stdev', 't_99_pct', 't_99_9_pct'), '1.0')
        lat_patch = patch.object(
            ibperf_lib, 'get_ib_lat_numb', side_effect=lambda *args, **kwargs: {n: lat for n in self.active_nodes()}
        )
        bw_patch = patch.object(
            ibperf_lib,
            'get_ib_bw_pps',
            side_effect=lambda *args, **kwargs: {n: {'bw': '1.0', 'pps': '1.0'} for n in self.active_nodes()},
        )
        sleep_patch = patch.object(ibperf_lib.time, 'sleep')
        for patcher in (lat_patch, bw_patch, sleep_patch):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_latency(self, **kwargs):
        kwargs.setdefault('hip_dev_dict', self.hip_dev_dict)
        return ibperf_lib.run_ib_perf_lat_test(
            self.phdl,
            'ib_write_lat',
            self.gpu_numa_dict,
            self.gpu_nic_dict,
            self.bck_nic_dict,
            '/opt/perftest/bin',
            64,
            3,
            **kwargs,
        )

    def run_bandwidth(self, **kwargs):
        kwargs.setdefault('hip_dev_dict', self.hip_dev_dict)
        return ibperf_lib.run_ib_perf_bw_test(
            self.phdl,
            'ib_write_bw',
            self.gpu_numa_dict,
            self.gpu_nic_dict,
            self.bck_nic_dict,
            '/opt/perftest/bin',
            64,
            3,
            **kwargs,
        )

    def commands(self):
        return self.phdl.exec_cmd_list.call_args_list[1].args[0]


class TestRunIbPerfLatTest(IbPerfCommandFixtures, unittest.TestCase):
    def test_default_pairs_latency_commands_unchanged(self):
        result = self.run_latency(use_dmabuf=False)
        calls = self.phdl.exec_cmd_list.call_args_list
        self.assertEqual(len(calls), 9)
        self.assertEqual(
            calls[0].args[0],
            [
                'echo "sleep 1" >> /tmp/ib_cmds_file.txt',
                'echo "sleep 5" >> /tmp/ib_cmds_file.txt',
            ],
        )
        for g in range(8):
            stem = (
                f'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat '
                f'-d rdma{g} --use_rocm={g} -x 3 -F -p {1516 + g} -s 64'
            )
            suffix = f' > /tmp/ib_perf_{g}_logs 2>&1 &" >> /tmp/ib_cmds_file.txt'
            self.assertEqual(calls[g + 1].args[0], [stem + suffix, stem + ' node1' + suffix])
        self.assertEqual({n: set(result[n]) for n in result}, {n: set(range(8)) for n in result})
        self.phdl.prune_nodes.assert_not_called()
        self.assertEqual(globals.error_list, [])

    def test_four_gpu_node(self):
        self.gpu_numa_dict, self.gpu_nic_dict, self.hip_dev_dict = gpu_maps(('node1', 'node2'), 4)
        result = self.run_latency(use_dmabuf=False)
        calls = self.phdl.exec_cmd_list.call_args_list
        self.assertEqual(len(calls), 5)
        for cmd in calls[4].args[0]:
            self.assertIn('rdma3 --use_rocm=3', cmd)
            self.assertIn('-p 1519', cmd)
            self.assertIn('/tmp/ib_perf_3_logs', cmd)
        self.assertEqual({n: set(result[n]) for n in result}, {n: set(range(4)) for n in result})
        self.phdl.prune_nodes.assert_not_called()
        self.assertEqual(globals.error_list, [])

    def test_explicit_pairs_and_read_clients(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        self.gpu_numa_dict, self.gpu_nic_dict, self.hip_dev_dict = gpu_maps(nodes)
        self.bck_nic_dict = {n: {} for n in nodes}
        with patch.object(ibperf_lib, 'get_ib_lat_numb', return_value={}) as get_lat:
            ibperf_lib.run_ib_perf_lat_test(
                self.phdl,
                'ib_read_lat',
                self.gpu_numa_dict,
                self.gpu_nic_dict,
                self.bck_nic_dict,
                '/opt/perftest/bin',
                64,
                3,
                hip_dev_dict=self.hip_dev_dict,
                node_pairs=[('n1', 'n3'), ('n2', 'n4')],
            )
        commands = self.phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertIn('-s 64 n1 >', commands[2])
        self.assertIn('-s 64 n2 >', commands[3])
        self.assertEqual(get_lat.call_args.kwargs['result_nodes'], ['n3', 'n4'])

    def test_orphan_and_no_active_pairs(self):
        nodes = ['n1', 'n2', 'n3']
        self.gpu_numa_dict, self.gpu_nic_dict, self.hip_dev_dict = gpu_maps(nodes)
        self.bck_nic_dict = {n: {} for n in nodes}
        result = self.run_latency(node_pairs=[('n1', 'n2'), ('n3', 'n9')])
        self.phdl.prune_nodes.assert_called_once_with(['n3'])
        self.assertEqual(set(result), {'n1', 'n2'})
        self.assertEqual({n: set(result[n]) for n in result}, {'n1': set(range(8)), 'n2': set(range(8))})
        self.assertIn('n3', globals.error_list[0])
        self.phdl.reset_mock()
        globals.error_list = []
        self.bck_nic_dict = {'n1': {}}
        result = self.run_latency(node_pairs=[('n1', 'n2')])
        self.assertEqual(result, {})
        self.phdl.prune_nodes.assert_called_once_with(['n1'])
        self.phdl.exec_cmd_list.assert_not_called()

    def test_builds_latency_commands(self):
        hip_dev_dict = {n: {f'card{g}': (g + 1) % 8 for g in range(8)} for n in ('node1', 'node2')}
        self.run_latency(use_dmabuf=False, hip_dev_dict=hip_dev_dict)
        self.assertEqual(
            self.commands()[0],
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=1'
            ' -x 3 -F -p 1516 -s 64; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=1'
            ' -x 3 -F -p 1516 -s 64 node1; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )

    def test_dmabuf_flag_added_when_enabled(self):
        self.run_latency(use_dmabuf=True)
        self.assertEqual(
            self.commands()[0],
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' --use_rocm_dmabuf -x 3 -F -p 1516 -s 64; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' --use_rocm_dmabuf -x 3 -F -p 1516 -s 64 node1; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )

    def test_dmabuf_defaults_on(self):
        self.run_latency()
        self.assertIn('--use_rocm_dmabuf', self.commands()[0])

    def test_does_not_probe_help(self):
        self.run_latency()
        self.assertFalse(any('--help' in call.args[0] for call in self.phdl.exec.call_args_list))

    @patch.object(ibperf_lib, 'wait_for_perftest_exit', return_value=[])
    def test_launches_then_waits_for_perftest_exit(self, mock_wait):
        self.run_latency()
        self.phdl.exec.assert_any_call(LAUNCH, print_console=False)
        mock_wait.assert_called_once_with(self.phdl, 'ib_write_lat', ibperf_lib.PERFTEST_LAT_EXIT_TIMEOUT_S)


class TestRunIbPerfBwTest(IbPerfCommandFixtures, unittest.TestCase):
    def test_default_pairs_commands_unchanged(self):
        result = self.run_bandwidth(use_dmabuf=False)
        calls = self.phdl.exec_cmd_list.call_args_list
        self.assertEqual(len(calls), 9)
        self.assertEqual(
            calls[0].args[0],
            [
                'echo "sleep 1" >> /tmp/ib_cmds_file.txt',
                'echo "sleep 5" >> /tmp/ib_cmds_file.txt',
            ],
        )
        for g in range(8):
            stem = (
                f'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw '
                f'-d rdma{g} --use_rocm={g} -x 3 --report_gbits -b -F -D 60 '
                f'-p {1516 + g} -s 64 -q 8'
            )
            suffix = f' > /tmp/ib_perf_{g}_logs 2>&1 &" >> /tmp/ib_cmds_file.txt'
            self.assertEqual(calls[g + 1].args[0], [stem + suffix, stem + ' node1' + suffix])
        self.assertEqual(
            [c.args[0] for c in self.phdl.exec.call_args_list[:3]],
            ['rm -rf /tmp/ib_cmds_file.txt', 'rm -rf /tmp/ib_perf*', 'touch /tmp/ib_cmds_file.txt'],
        )
        self.phdl.prune_nodes.assert_not_called()
        self.assertEqual(globals.error_list, [])
        self.assertEqual({n: set(result[n]) for n in result}, {n: set(range(8)) for n in result})

    def test_default_pairs_four_nodes_unchanged(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        self.gpu_numa_dict, self.gpu_nic_dict, self.hip_dev_dict = gpu_maps(nodes)
        self.bck_nic_dict = {n: {} for n in nodes}
        self.run_bandwidth(use_dmabuf=False)
        for g, call_args in enumerate(self.phdl.exec_cmd_list.call_args_list[1:]):
            commands = call_args.args[0]
            self.assertEqual(len(commands), 4)
            suffix = f'-q 8 > /tmp/ib_perf_{g}_logs 2>&1 &" >> /tmp/ib_cmds_file.txt'
            self.assertTrue(commands[0].endswith(suffix))
            self.assertTrue(commands[2].endswith(suffix))
            self.assertTrue(commands[1].endswith(suffix.replace('-q 8 >', '-q 8 n1 >')))
            self.assertTrue(commands[3].endswith(suffix.replace('-q 8 >', '-q 8 n3 >')))

    def test_four_gpu_node(self):
        self.gpu_numa_dict, self.gpu_nic_dict, self.hip_dev_dict = gpu_maps(('node1', 'node2'), 4)
        result = self.run_bandwidth(use_dmabuf=False)
        calls = self.phdl.exec_cmd_list.call_args_list
        self.assertEqual(len(calls), 5)
        for cmd in calls[4].args[0]:
            self.assertIn('rdma3 --use_rocm=3', cmd)
            self.assertIn('-p 1519', cmd)
            self.assertIn('/tmp/ib_perf_3_logs', cmd)
        self.assertEqual({n: set(result[n]) for n in result}, {n: set(range(4)) for n in result})
        self.phdl.prune_nodes.assert_not_called()
        self.assertEqual(globals.error_list, [])

    def test_explicit_pairs_and_orphan(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        self.gpu_numa_dict, self.gpu_nic_dict, self.hip_dev_dict = gpu_maps(nodes)
        self.bck_nic_dict = {n: {} for n in nodes}
        self.run_bandwidth(node_pairs=[('n1', 'n3'), ('n2', 'n4')])
        commands = self.phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertIn('-q 8 n1 >', commands[2])
        self.assertIn('-q 8 n2 >', commands[3])
        self.phdl.reset_mock()
        result = self.run_bandwidth(node_pairs=[('n1', 'n2'), ('n3', 'n9')])
        self.phdl.prune_nodes.assert_called_once_with(['n3', 'n4'])
        self.assertEqual(set(result), {'n1', 'n2'})
        self.assertIn('n3', globals.error_list[0])

    def test_mismatch_pair_pruned_and_no_active_pairs(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        self.gpu_numa_dict, self.gpu_nic_dict, self.hip_dev_dict = gpu_maps(nodes)
        self.gpu_nic_dict['n4'] = dict(list(self.gpu_nic_dict['n4'].items())[:4])
        self.bck_nic_dict = {n: {} for n in nodes}
        self.run_bandwidth(node_pairs=[('n1', 'n2'), ('n3', 'n4')])
        self.phdl.prune_nodes.assert_called_once_with(['n3', 'n4'])
        self.assertTrue(all(len(call_args.args[0]) == 2 for call_args in self.phdl.exec_cmd_list.call_args_list))
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('n3', globals.error_list[0])
        self.assertIn('n4', globals.error_list[0])
        self.phdl.reset_mock()
        globals.error_list = []
        self.bck_nic_dict = {'n1': {}}
        self.assertEqual(self.run_bandwidth(node_pairs=[('n1', 'n2')]), {})
        self.phdl.prune_nodes.assert_called_once_with(['n1'])
        self.phdl.exec_cmd_list.assert_not_called()

    def test_first_pair_has_fewer_cards_than_majority(self):
        nodes = ['a', 'b', 'c', 'd', 'e', 'f']
        self.gpu_numa_dict, self.gpu_nic_dict, self.hip_dev_dict = gpu_maps(nodes)
        for node in ('a', 'b'):
            self.gpu_numa_dict[node] = dict(list(self.gpu_numa_dict[node].items())[:4])
            self.gpu_nic_dict[node] = dict(list(self.gpu_nic_dict[node].items())[:4])
            self.hip_dev_dict[node] = dict(list(self.hip_dev_dict[node].items())[:4])
        self.bck_nic_dict = {node: {} for node in nodes}
        pairs = [('a', 'b'), ('c', 'd'), ('e', 'f')]

        result = self.run_bandwidth(node_pairs=pairs)

        self.assertEqual(set(result), {'c', 'd', 'e', 'f'})
        self.assertEqual(len(self.phdl.exec_cmd_list.call_args_list), 9)
        self.assertTrue(all(len(call.args[0]) == 4 for call in self.phdl.exec_cmd_list.call_args_list))
        self.phdl.prune_nodes.assert_called_once_with(['a', 'b'])
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('a, b', globals.error_list[0])

        self.phdl.reset_mock()
        self.run_bandwidth(node_pairs=pairs)
        self.phdl.prune_nodes.assert_not_called()
        self.assertEqual(len(globals.error_list), 1)

    @patch.object(ibperf_lib, 'wait_for_perftest_exit', return_value=[])
    def test_launches_then_waits_for_perftest_exit_bounded_by_duration(self, mock_wait):
        self.run_bandwidth(duration=30)
        self.phdl.exec.assert_any_call(LAUNCH, print_console=False)
        mock_wait.assert_called_once_with(self.phdl, 'ib_write_bw', 30 + ibperf_lib.PERFTEST_EXIT_SLACK_S)

    def test_builds_bw_commands_without_dmabuf(self):
        self.run_bandwidth(use_dmabuf=False)
        self.assertEqual(
            self.commands()[0],
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &"'
            ' >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8 node1; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &"'
            ' >> /tmp/ib_cmds_file.txt',
        )

    def test_builds_bw_commands_with_dmabuf(self):
        self.run_bandwidth(use_dmabuf=True)
        self.assertEqual(
            self.commands()[0],
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' --use_rocm_dmabuf -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8'
            '; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' --use_rocm_dmabuf -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8 node1'
            '; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )


class TestResultReaderException(IbPerfCommandFixtures, unittest.TestCase):
    def test_instance_lost_to_exception_fails_test(self):
        for reader, run in (('get_ib_bw_pps', self.run_bandwidth), ('get_ib_lat_numb', self.run_latency)):
            with (
                self.subTest(reader=reader),
                patch.object(ibperf_lib, reader, side_effect=[KeyError('node2')] + [{}] * 7),
                patch.object(ibperf_lib, 'fail_test') as mock_fail,
                self.assertLogs(ibperf_lib.log, 'ERROR'),
            ):
                run()
                mock_fail.assert_called_once()
                self.assertIn('instance=0', mock_fail.call_args.args[0])


class TestProbePerftestDmabufSupport(unittest.TestCase):
    def setUp(self):
        self.phdl = MagicMock()

    def test_parses_each_node(self):
        self.phdl.exec.return_value = {
            'n1': 'dmabuf=supported\n',
            'n2': 'dmabuf=unsupported\n',
            'n3': 'dmabuf=probe_failed\n',
            'n4': 'ssh: connect to host n4 port 22: Connection refused',
        }
        self.assertEqual(
            ibperf_lib.probe_perftest_dmabuf_support(self.phdl, '/bin/ib_write_bw'),
            {
                'n1': ibperf_lib.DMABUF_SUPPORTED,
                'n2': ibperf_lib.DMABUF_UNSUPPORTED,
                'n3': ibperf_lib.DMABUF_PROBE_FAILED,
                'n4': ibperf_lib.DMABUF_PROBE_FAILED,
            },
        )

    def test_command_includes_rocm_ld_path(self):
        self.phdl.exec.return_value = {'n1': 'dmabuf=supported'}
        ibperf_lib.probe_perftest_dmabuf_support(self.phdl, '/bin/ib_write_bw', rocm_path='/opt/rocm')
        cmd = self.phdl.exec.call_args.args[0]
        self.assertIn('LD_LIBRARY_PATH=/opt/rocm/lib:', cmd)
        self.assertIn('/bin/ib_write_bw --help', cmd)
        self.assertIn('*use_rocm_dmabuf*', cmd)
        self.phdl.exec.assert_called_once_with(cmd, print_console=False)

    def test_command_without_rocm_path(self):
        self.phdl.exec.return_value = {'n1': 'dmabuf=supported'}
        ibperf_lib.probe_perftest_dmabuf_support(self.phdl, '/bin/ib_write_bw')
        self.assertNotIn('LD_LIBRARY_PATH', self.phdl.exec.call_args.args[0])


class TestCheckPerftestDmabuf(unittest.TestCase):
    def setUp(self):
        self.phdl = MagicMock()
        fail_patch = patch.object(ibperf_lib, 'fail_test')
        self.fail_test = fail_patch.start()
        self.addCleanup(fail_patch.stop)

    def test_all_nodes_supported(self):
        self.phdl.exec.return_value = {'n1': 'dmabuf=supported', 'n2': 'dmabuf=supported'}
        self.assertTrue(ibperf_lib.check_perftest_dmabuf(self.phdl, '/bin/ib_write_bw'))
        self.fail_test.assert_not_called()

    def test_one_unsupported_node_is_named(self):
        self.phdl.exec.return_value = {'n1': 'dmabuf=supported', 'n2': 'dmabuf=unsupported'}
        self.assertFalse(ibperf_lib.check_perftest_dmabuf(self.phdl, '/bin/ib_write_bw', require_dmabuf=True))
        self.fail_test.assert_called_once()
        message = self.fail_test.call_args.args[0]
        self.assertIn('n2', message)
        self.assertIn('/bin/ib_write_bw', message)
        self.assertIn('--enable-rocm-dmabuf', message)
        self.assertNotIn('n1', message)

    def test_each_unsupported_node_is_reported(self):
        self.phdl.exec.return_value = {'n1': 'dmabuf=unsupported', 'n2': 'dmabuf=unsupported'}
        self.assertFalse(ibperf_lib.check_perftest_dmabuf(self.phdl, '/bin/ib_write_bw', require_dmabuf=True))
        self.assertEqual(self.fail_test.call_count, 2)
        self.assertIn('n1', self.fail_test.call_args_list[0].args[0])
        self.assertIn('n2', self.fail_test.call_args_list[1].args[0])

    def test_unsupported_node_can_disable_dmabuf(self):
        self.phdl.exec.return_value = {'n1': 'dmabuf=supported', 'n2': 'dmabuf=unsupported'}
        with self.assertLogs(ibperf_lib.log, 'WARNING') as logs:
            self.assertFalse(ibperf_lib.check_perftest_dmabuf(self.phdl, '/bin/ib_write_bw', require_dmabuf=False))
        self.fail_test.assert_not_called()
        self.assertIn('n2', logs.output[0])

    def test_probe_failure_is_reported_even_when_optional(self):
        self.phdl.exec.return_value = {'n1': 'dmabuf=probe_failed'}
        self.assertFalse(ibperf_lib.check_perftest_dmabuf(self.phdl, '/bin/ib_write_bw', require_dmabuf=False))
        self.fail_test.assert_called_once()
        self.assertIn('n1', self.fail_test.call_args.args[0])

    def test_empty_result_disables_dmabuf(self):
        self.phdl.exec.return_value = {}
        self.assertFalse(ibperf_lib.check_perftest_dmabuf(self.phdl, '/bin/ib_write_bw'))
        self.fail_test.assert_called_once()
        self.assertIn('no probe output', self.fail_test.call_args.args[0])


class TestIsDmabufRequired(unittest.TestCase):
    def test_default_is_required(self):
        self.assertTrue(ibperf_lib.is_dmabuf_required({}))

    def test_true_values(self):
        for value in ('True', 'true', True, '1', 'yes', 'on'):
            with self.subTest(value=value):
                self.assertTrue(ibperf_lib.is_dmabuf_required({'require_dmabuf': value}))

    def test_false_values(self):
        for value in ('False', False, '0', 'no', 'off'):
            with self.subTest(value=value):
                self.assertFalse(ibperf_lib.is_dmabuf_required({'require_dmabuf': value}))

    def test_unrecognized_value_is_required(self):
        with self.assertLogs(ibperf_lib.log, 'WARNING') as logs:
            self.assertTrue(ibperf_lib.is_dmabuf_required({'require_dmabuf': 'bogus'}))
        self.assertIn('bogus', logs.output[0])


class TestRunIbPerfBwHipDevice(unittest.TestCase):
    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_bw_pps')
    def test_bw_commands_use_hip_device_for_card(self, mock_bw_pps, _sleep):
        nodes = ('node1', 'node2')
        mock_bw_pps.return_value = {node: {'bw': '1.0', 'pps': '1.0'} for node in nodes}
        gpu_nic_dict = {node: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for node in nodes}
        gpu_numa_dict = {node: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for node in nodes}
        hip_dev_dict = {node: {f'card{g}': (g + 1) % 8 for g in range(8)} for node in nodes}
        phdl = MagicMock()

        ibperf_lib.run_ib_perf_bw_test(
            phdl,
            'ib_write_bw',
            gpu_numa_dict,
            gpu_nic_dict,
            {node: {} for node in nodes},
            '/opt/perftest/bin',
            65536,
            3,
            qp_count=16,
            duration=10,
            hip_dev_dict=hip_dev_dict,
            use_dmabuf=False,
        )

        server_cmd, client_cmd = phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertEqual(
            server_cmd,
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=1'
            ' -x 3 --report_gbits -b -F -D 10 -p 1516 -s 65536 -q 16; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            client_cmd,
            'echo "( numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=1'
            ' -x 3 --report_gbits -b -F -D 10 -p 1516 -s 65536 -q 16 node1; echo perftest_rc=\\$? ) > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )

    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_bw_pps')
    @patch.object(ibperf_lib, 'get_hip_device_dict')
    def test_computes_hip_dev_dict_when_not_given(self, mock_get_hip_device_dict, mock_bw_pps, _sleep):
        nodes = ('node1', 'node2')
        mock_bw_pps.return_value = {node: {'bw': '1.0', 'pps': '1.0'} for node in nodes}
        gpu_nic_dict = {node: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for node in nodes}
        gpu_numa_dict = {node: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for node in nodes}
        mock_get_hip_device_dict.return_value = {node: {f'card{g}': (g + 1) % 8 for g in range(8)} for node in nodes}
        phdl = MagicMock()

        ibperf_lib.run_ib_perf_bw_test(
            phdl,
            'ib_write_bw',
            gpu_numa_dict,
            gpu_nic_dict,
            {node: {} for node in nodes},
            '/opt/perftest/bin',
            65536,
            3,
            use_dmabuf=False,
        )

        mock_get_hip_device_dict.assert_called_once_with(phdl, gpu_nic_dict)
        server_cmd = phdl.exec_cmd_list.call_args_list[1].args[0][0]
        self.assertIn('--use_rocm=1', server_cmd)


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

    @patch.object(ibperf_lib, 'fail_test')
    def test_read_latency_skips_passive_node_without_results(self, mock_fail):
        expected = {'ib_read_lat': {'8192': '50'}}
        results = {'server': {}, 'client': {0: {'t_avg': '10.98'}, 1: {'t_avg': '10.98'}}}
        self.assertTrue(ibperf_lib.verify_expected_lat('ib_read_lat', 8192, results, expected))
        mock_fail.assert_not_called()

    @patch.object(ibperf_lib, 'fail_test')
    def test_write_latency_still_fails_node_without_results(self, mock_fail):
        expected = {'ib_write_lat': {'8192': '50'}}
        results = {'server': {}, 'client': {0: {'t_avg': '10.98'}, 1: {'t_avg': '10.98'}}}
        self.assertFalse(ibperf_lib.verify_expected_lat('ib_write_lat', 8192, results, expected))
        mock_fail.assert_called_once()
        self.assertIn('No latency results', mock_fail.call_args.args[0])
        self.assertIn('server', mock_fail.call_args.args[0])


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


class TestSampleConfigDuration(unittest.TestCase):
    @patch.object(ibperf_lib, 'wait_for_perftest_exit', return_value=[])
    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_bw_pps', return_value={})
    def test_sample_duration_finishes_before_results_are_read(self, _bw_pps, _sleep, mock_wait):
        config = json.loads(SAMPLE_CONFIG.read_text())['ibperf']
        duration = int(config['duration'])
        nodes = ('node1', 'node2')
        gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
        hip_dev_dict = {n: {f'card{g}': g for g in range(8)} for n in nodes}
        phdl = MagicMock()

        ibperf_lib.run_ib_perf_bw_test(
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
            use_dmabuf=False,
            hip_dev_dict=hip_dev_dict,
        )

        _, client_cmd = phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertIn(f' -D {duration} ', client_cmd)
        self.assertLess(duration + PERFTEST_SETUP_HEADROOM_S, mock_wait.call_args.args[2])


AFM_FIXTURE = Path(__file__).parent / 'fixtures' / 'afmctl_show_device_mi450.json'
MI450_POD = '7699065b-19a2-4934-bac8-434ba1054400'


def afm_output(accel_offset=0, physical_pod=None, virtual_pod=None, vpod_accelerators=None, drop_pod_ids=False):
    document = json.loads(AFM_FIXTURE.read_text())
    for record in document['device']:
        spec = record['spec']
        spec['accelerator_id'] = str(int(spec['accelerator_id']) + accel_offset)
        fabric = spec['scaleup_fabric']
        if physical_pod is not None:
            fabric['physical_pod']['id'] = physical_pod
        if virtual_pod is not None:
            fabric['virtual_pod']['id'] = virtual_pod
        if vpod_accelerators is not None:
            spec['vpod_accelerators'] = list(vpod_accelerators)
        if drop_pod_ids:
            del fabric['physical_pod']['id']
            del fabric['virtual_pod']['id']
    return json.dumps(document, indent=4)


def afm_devices(accelerators, membership):
    return json.dumps(
        {
            'devices': [
                {
                    'bdf': f'0000:{index + 1:02x}:00.1',
                    'accelerator_id': index,
                    'config_phase': 'ACTIVE',
                    'virtualization_mode': 'bare-metal',
                    'local_accelerators': list(accelerators),
                    'vpod_accelerators': list(membership),
                    'num_network_ports': 36,
                }
                for index in accelerators
            ]
        }
    )


def gpu_maps(nodes, gpu_count=8):
    numa = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(gpu_count)} for n in nodes}
    nic = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(gpu_count)} for n in nodes}
    hip = {n: {f'card{g}': g for g in range(gpu_count)} for n in nodes}
    return numa, nic, hip


class TestBuildIbperfNodePairs(unittest.TestCase):
    def test_sequential_even_odd_and_tuple_input(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        self.assertEqual(ibperf_lib.build_ibperf_node_pairs(nodes), ([('n1', 'n2'), ('n3', 'n4')], []))
        self.assertEqual(ibperf_lib.build_ibperf_node_pairs(tuple(nodes + ['n5']))[1], ['n5'])

    def test_intra_vpod_with_missing_labels(self):
        nodes = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']
        labels = {'n1': 'A', 'n2': 'A', 'n3': 'A', 'n4': 'B', 'n5': 'B', 'n6': None}
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(nodes, 'intra_vpod', labels),
            ([('n1', 'n2'), ('n4', 'n5')], ['n3', 'n6', 'n7']),
        )

    def test_inter_vpod_equal_and_unequal_groups(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        labels = {'n1': (0, 1), 'n2': (0, 1), 'n3': (2, 3), 'n4': (2, 3)}
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(nodes, 'inter_vpod', labels), ([('n1', 'n3'), ('n2', 'n4')], [])
        )
        nodes = ['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'c0', 'c1']
        labels = {n: n[0].upper() for n in nodes}
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(nodes, 'inter_vpod', labels)[0],
            [('a0', 'b0'), ('a1', 'c0'), ('a2', 'b1'), ('a3', 'c1')],
        )

    def test_pair_invariants(self):
        for labels in (
            {'a': 'A', 'b': 'A', 'c': 'B', 'd': 'B'},
            {'a': 'A', 'b': 'A', 'c': 'A', 'd': 'B', 'e': 'C'},
            {'a': (1, 2), 'b': (1, 2), 'c': (2, 2), 'd': (3, 2)},
            {
                f'{group}{i}': group
                for group, size in (('A', 3), ('B', 1), ('C', 4), ('D', 2), ('E', 1))
                for i in range(size)
            },
            {f'a{i}': 'A' for i in range(5)} | {'b': 'B', 'c': 'C'},
        ):
            with self.subTest(labels=labels):
                nodes = list(labels)
                pairs, _ = ibperf_lib.build_ibperf_node_pairs(nodes, 'inter_vpod', labels)
                self.assertTrue(all(labels[s] != labels[c] for s, c in pairs))
                self.assertEqual(len({n for pair in pairs for n in pair}), 2 * len(pairs))
                largest = max(list(labels.values()).count(key) for key in labels.values())
                self.assertEqual(len(pairs), min(len(nodes) // 2, len(nodes) - largest))
                intra, _ = ibperf_lib.build_ibperf_node_pairs(nodes, 'intra_vpod', labels)
                self.assertTrue(all(labels[s] == labels[c] for s, c in intra))

    def test_invalid_and_single_group(self):
        nodes = ['n1', 'n2', 'n3']
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(nodes, 'inter_vpod', dict.fromkeys(nodes, 'A')), ([], nodes)
        )
        with self.assertRaisesRegex(ValueError, 'pairing_mode'):
            ibperf_lib.build_ibperf_node_pairs(nodes, 'bad')
        with self.assertRaisesRegex(ValueError, 'vpod_map'):
            ibperf_lib.build_ibperf_node_pairs(nodes, 'inter_vpod')


class TestGetClusterFileVpodMap(unittest.TestCase):
    def test_labels_and_errors(self):
        labels, errors = ibperf_lib.get_cluster_file_vpod_map(
            {
                'node_dict': {
                    'n1': {'vpod_id': ' A '},
                    'n2': {'vpod_id': 'B'},
                    'n3': {},
                    'n4': {'vpod_id': '  '},
                    'n5': {'vpod_id': None},
                }
            }
        )
        self.assertEqual(labels, {'n1': 'A', 'n2': 'B'})
        self.assertEqual(len(errors), 3)
        self.assertEqual([error.split(':')[0] for error in errors], ['n3', 'n4', 'n5'])


class TestExtractAfmPodKeys(unittest.TestCase):
    def test_pod_id_normalization(self):
        self.assertEqual(ibperf_lib._afm_pod_id({'id': ' 3 '}), '3')
        self.assertEqual(ibperf_lib._afm_pod_id({'id': 3}), '3')
        self.assertEqual(ibperf_lib._afm_pod_id({'id': None}), '')
        self.assertEqual(ibperf_lib._afm_pod_id(None), '')

    def test_real_and_missing_ids(self):
        self.assertEqual(ibperf_lib.extract_afm_pod_keys(AFM_FIXTURE.read_text()), [(MI450_POD, '3')] * 4)
        self.assertEqual(ibperf_lib.extract_afm_pod_keys(afm_output(drop_pod_ids=True)), [None] * 4)
        document = json.loads(afm_output())
        del document['device'][0]['spec']['scaleup_fabric']
        self.assertEqual(ibperf_lib.extract_afm_pod_keys(json.dumps(document))[0], None)
        self.assertEqual(ibperf_lib.extract_afm_pod_keys(json.dumps(document['device']))[1:], [(MI450_POD, '3')] * 3)

    def test_flat_and_invalid_documents(self):
        self.assertEqual(ibperf_lib.extract_afm_pod_keys(afm_devices(range(2), range(4))), [None] * 2)
        flat = {'devices': [{'scaleup_fabric': {'physical_pod': {'id': MI450_POD}, 'virtual_pod': {'id': '3'}}}]}
        self.assertEqual(ibperf_lib.extract_afm_pod_keys(json.dumps(flat)), [(MI450_POD, '3')])
        for output in ('bad json', 'ERR - afmctl requires root privileges', '{}', None):
            self.assertEqual(ibperf_lib.extract_afm_pod_keys(output), [])


class TestParseAfmVpodMembership(unittest.TestCase):
    def test_real_fixture_and_same_pod(self):
        with self.assertLogs(ibperf_lib.log, 'INFO') as logs:
            labels, errors = ibperf_lib.parse_afm_vpod_membership({'tray': AFM_FIXTURE.read_text()})
        self.assertEqual((labels, errors), ({'tray': (MI450_POD, '3')}, []))
        self.assertIn('physical_pod.id', '\n'.join(logs.output))
        labels, errors = ibperf_lib.parse_afm_vpod_membership({'tray': afm_output(), 'n2': afm_output(4)})
        self.assertEqual(labels['tray'], labels['n2'])
        self.assertEqual(errors, [])
        self.assertEqual(ibperf_lib.build_ibperf_node_pairs(['tray', 'n2'], 'intra_vpod', labels)[0], [('tray', 'n2')])
        self.assertEqual(ibperf_lib.build_ibperf_node_pairs(['tray', 'n2'], 'inter_vpod', labels)[0], [])

    def test_distinct_virtual_and_physical_pods(self):
        outputs = {
            'tray': afm_output(),
            'virtual': afm_output(4, virtual_pod='4'),
            'physical': afm_output(physical_pod='00000000-0000-4000-8000-000000000002'),
        }
        labels, errors = ibperf_lib.parse_afm_vpod_membership(outputs)
        self.assertEqual(errors, [])
        self.assertEqual(labels['virtual'], (MI450_POD, '4'))
        self.assertNotEqual(labels['tray'], labels['physical'])
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(['tray', 'physical'], 'inter_vpod', labels)[0], [('tray', 'physical')]
        )

    def test_fallback_and_overlap_guard(self):
        outputs = {
            'n1': afm_output(drop_pod_ids=True, vpod_accelerators=range(8)),
            'n2': afm_output(4, drop_pod_ids=True, vpod_accelerators=range(8)),
        }
        with self.assertLogs(ibperf_lib.log, 'WARNING') as logs:
            labels, errors = ibperf_lib.parse_afm_vpod_membership(outputs)
        self.assertEqual(labels, {'n1': tuple(range(8)), 'n2': tuple(range(8))})
        self.assertEqual(errors, [])
        self.assertIn('vpod_accelerators', '\n'.join(logs.output))
        outputs['n2'] = afm_output(2, drop_pod_ids=True, vpod_accelerators=range(8))
        with self.assertRaisesRegex(ValueError, 'cluster_file'):
            ibperf_lib.parse_afm_vpod_membership(outputs)
        labels, errors = ibperf_lib.parse_afm_vpod_membership(
            {'n1': afm_devices(range(4), range(8)), 'n2': afm_devices(range(4, 8), range(8))}
        )
        self.assertEqual(labels['n1'], labels['n2'])
        self.assertEqual(errors, [])

    def test_membership_errors(self):
        document = json.loads(afm_output())
        document['device'][1]['spec']['scaleup_fabric']['virtual_pod']['id'] = '5'
        labels, errors = ibperf_lib.parse_afm_vpod_membership({'n': json.dumps(document)})
        self.assertEqual(labels, {})
        self.assertIn('multiple vPODs', errors[0])
        document = json.loads(afm_output())
        del document['device'][2]['spec']['scaleup_fabric']['physical_pod']['id']
        labels, errors = ibperf_lib.parse_afm_vpod_membership({'n': json.dumps(document)})
        self.assertEqual(labels, {})
        self.assertIn('only some devices', errors[0])
        labels, errors = ibperf_lib.parse_afm_vpod_membership({'n1': afm_output(), 'n2': afm_output(drop_pod_ids=True)})
        self.assertEqual(list(labels), ['n1'])
        self.assertIn('n2', errors[0])
        mixed = json.loads(afm_devices(range(2), range(4)))
        mixed['devices'][1]['vpod_accelerators'] = list(range(4, 8))
        labels, errors = ibperf_lib.parse_afm_vpod_membership(
            {'n1': afm_devices(range(2), []), 'n2': json.dumps(mixed)}
        )
        self.assertEqual(labels, {})
        self.assertIn('no vPOD membership', errors[0])
        self.assertIn('multiple vPOD memberships', errors[1])

    def test_root_and_command_errors(self):
        outputs = {
            'n1': 'ERR - afmctl requires root privileges, please login as superuser or use sudo\n',
            'n2': 'sudo: a password is required\n',
            'n3': 'sudo: Passwort erforderlich',
            'n4': 'CVS_ERROR: ibperf.vpod_source afm needs passwordless sudo for afmctl',
        }
        labels, errors = ibperf_lib.parse_afm_vpod_membership(outputs)
        self.assertEqual(labels, {})
        self.assertEqual(len(errors), 4)
        self.assertIn('root', errors[0])
        self.assertNotIn('Invalid afmctl', errors[0])
        for error in errors[1:3]:
            self.assertIn('AFM vPOD discovery failed', error)
            self.assertNotIn('needs root', error)
        self.assertIn('n4: ibperf.vpod_source afm needs passwordless sudo for afmctl', errors[3])


class TestDiscoverAfmVpodMap(unittest.TestCase):
    def test_relative_command_is_resolved_before_sudo(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'n': AFM_FIXTURE.read_text()}
        self.assertEqual(ibperf_lib.discover_afm_vpod_map(phdl), ({'n': (MI450_POD, '3')}, []))
        phdl.exec.assert_called_once_with(
            '{ if ! sudo -n true >/dev/null 2>&1; then '
            "printf '%s\\n' 'CVS_ERROR: ibperf.vpod_source afm needs passwordless sudo for afmctl' >&2; exit 1; fi; "
            '_cvs_afmctl="$(command -v afmctl)" || '
            "{ printf '%s\\n' 'Unable to find afmctl in PATH' >&2; exit 127; }; "
            'case "$_cvs_afmctl" in /*) ;; *) '
            "printf '%s\\n' 'Resolved afmctl path is not absolute' >&2; exit 127 ;; esac; "
            'sudo -n "$_cvs_afmctl" show device --json; } 2>&1',
            print_console=False,
        )

    def test_absolute_and_quoted_commands(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'n': AFM_FIXTURE.read_text()}
        ibperf_lib.discover_afm_vpod_map(phdl, '/path with space/afmctl')
        phdl.exec.assert_called_once_with(
            '{ if ! sudo -n true >/dev/null 2>&1; then '
            "printf '%s\\n' 'CVS_ERROR: ibperf.vpod_source afm needs passwordless sudo for afmctl' >&2; exit 1; fi; "
            "sudo -n '/path with space/afmctl' show device --json; } 2>&1",
            print_console=False,
        )
        phdl.reset_mock()
        ibperf_lib.discover_afm_vpod_map(phdl, 'my afmctl')
        command = phdl.exec.call_args.args[0]
        self.assertIn("command -v 'my afmctl'", command)
        self.assertIn('sudo -n "$_cvs_afmctl"', command)


class TestResolveIbperfNodePairing(unittest.TestCase):
    def test_modes_and_handle_lifecycle(self):
        nodes = ['tray', 's1', 's2', 's3']
        factory = MagicMock()
        factory.return_value.exec.return_value = {
            'tray': afm_output(),
            's1': afm_output(4),
            's2': afm_output(8, virtual_pod='4'),
            's3': afm_output(physical_pod='00000000-0000-4000-8000-000000000002'),
        }
        with self.assertLogs(ibperf_lib.log, 'INFO') as logs:
            inter = ibperf_lib.resolve_ibperf_node_pairing(nodes, {'pairing_mode': 'inter_vpod'}, {}, factory)
        self.assertEqual(inter['pairs'], [('tray', 's2'), ('s1', 's3')])
        self.assertIn('ibperf vPOD map', '\n'.join(logs.output))
        factory.return_value.destroy_clients.assert_called_once()
        intra = ibperf_lib.resolve_ibperf_node_pairing(nodes, {'pairing_mode': 'intra_vpod'}, {}, factory)
        self.assertEqual(intra['pairs'], [('tray', 's1')])
        self.assertEqual(intra['unpaired'], ['s2', 's3'])
        factory.reset_mock()
        sequential = ibperf_lib.resolve_ibperf_node_pairing(nodes, {}, {}, factory)
        self.assertEqual(sequential['pairs'], [('tray', 's1'), ('s2', 's3')])
        factory.assert_not_called()
        factory.return_value.exec.side_effect = RuntimeError('discovery failed')
        with self.assertRaisesRegex(RuntimeError, 'discovery failed'):
            ibperf_lib.resolve_ibperf_node_pairing(nodes, {'pairing_mode': 'inter_vpod'}, {}, factory)
        factory.return_value.destroy_clients.assert_called_once()

    def test_cluster_file_errors_and_validation(self):
        nodes = ['n1', 'n2', 'n3', 'n4', 'n5']
        cluster = {
            'node_dict': {
                'n1': {'vpod_id': 'A'},
                'n2': {'vpod_id': 'B'},
                'n3': {},
                'n4': {'vpod_id': 'A'},
                'n5': {'vpod_id': 'B'},
            }
        }
        result = ibperf_lib.resolve_ibperf_node_pairing(
            nodes, {'pairing_mode': 'inter_vpod', 'vpod_source': 'cluster_file'}, cluster
        )
        self.assertEqual(result['pairs'], [('n1', 'n2'), ('n4', 'n5')])
        self.assertEqual(result['unpaired'], ['n3'])
        self.assertIn('n3', result['errors'][0])
        for key in ('pairing_mode', 'vpod_source'):
            with self.assertRaisesRegex(ValueError, key):
                ibperf_lib.resolve_ibperf_node_pairing(['a', 'b'], {key: 'bad'}, {})
        with self.assertRaisesRegex(ValueError, 'at least 2 nodes'):
            ibperf_lib.resolve_ibperf_node_pairing(['a'], {}, {})
        with self.assertRaisesRegex(ValueError, 'intra_vpod needs'):
            ibperf_lib.resolve_ibperf_node_pairing(
                ['a', 'b'],
                {'pairing_mode': 'intra_vpod', 'vpod_source': 'cluster_file'},
                {'node_dict': {'a': {'vpod_id': 'A'}, 'b': {'vpod_id': 'B'}}},
            )
        with self.assertRaisesRegex(ValueError, 'phdl_factory'):
            ibperf_lib.resolve_ibperf_node_pairing(['a', 'b'], {'pairing_mode': 'inter_vpod'}, {})

    def test_errors_logged_before_zero_pair_error(self):
        factory = MagicMock()
        factory.return_value.exec.return_value = {
            'a': 'ERR - afmctl requires root privileges',
            'b': 'CVS_ERROR: ibperf.vpod_source afm needs passwordless sudo for afmctl',
        }
        with self.assertLogs(ibperf_lib.log, 'ERROR') as logs:
            with self.assertRaisesRegex(ValueError, 'No ibperf node pairs') as failure:
                ibperf_lib.resolve_ibperf_node_pairing(['a', 'b'], {'pairing_mode': 'inter_vpod'}, {}, factory)
        for node in ('a', 'b'):
            self.assertIn(node, str(failure.exception))
            self.assertIn(node, '\n'.join(logs.output))


class TestAssignPairRoles(unittest.TestCase):
    def test_order_orphans_and_duplicate(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        roles, orphans = ibperf_lib._assign_pair_roles([('n1', 'n3'), ('n2', 'n4')], nodes)
        self.assertEqual(roles, {'n1': None, 'n2': None, 'n3': 'n1', 'n4': 'n2'})
        self.assertEqual(list(roles), nodes)
        self.assertEqual(orphans, [])
        roles, orphans = ibperf_lib._assign_pair_roles([('n1', 'n2'), ('n3', 'n9')], nodes)
        self.assertEqual(roles, {'n1': None, 'n2': 'n1'})
        self.assertEqual(orphans, ['n3', 'n4'])
        with self.assertRaisesRegex(ValueError, 'more than one'):
            ibperf_lib._assign_pair_roles([('n1', 'n2'), ('n2', 'n3')], nodes)
        with self.assertRaisesRegex(ValueError, 'more than one'):
            ibperf_lib._assign_pair_roles([('n1', 'n1')], nodes)


class TestNodeGpuCards(unittest.TestCase):
    def test_numeric_order_and_missing_node(self):
        self.assertEqual(
            ibperf_lib.get_node_gpu_cards({'n': {'card10': {}, 'card2': {}, 'card0': {}}}, 'n'),
            ['card0', 'card2', 'card10'],
        )
        self.assertEqual(ibperf_lib.get_node_gpu_cards({}, 'n'), [])


class TestPreparePairRoles(unittest.TestCase):
    def setUp(self):
        self.fail_patch = patch.object(ibperf_lib, 'fail_test')
        self.fail = self.fail_patch.start()
        self.addCleanup(self.fail_patch.stop)

    def test_equal_and_mismatch_cards(self):
        numa, cards, hip = gpu_maps(['n1', 'n2'])
        roles, selected, excluded = ibperf_lib._prepare_pair_roles([('n1', 'n2')], ['n1', 'n2'], cards, numa, hip, 'bw')
        self.assertEqual(roles, {'n1': None, 'n2': 'n1'})
        self.assertEqual(selected, [f'card{g}' for g in range(8)])
        self.assertEqual(excluded, [])
        self.fail.assert_not_called()
        cards['n2'] = dict(list(cards['n2'].items())[:4])
        self.assertEqual(
            ibperf_lib._prepare_pair_roles([('n1', 'n2')], ['n1', 'n2'], cards, numa, hip, 'bw')[2],
            ['n1', 'n2'],
        )
        self.assertIn('n1', self.fail.call_args.args[0])
        self.assertIn('n2', self.fail.call_args.args[0])
        del cards['n2']
        self.assertEqual(
            ibperf_lib._prepare_pair_roles([('n1', 'n2')], ['n1', 'n2'], cards, numa, hip, 'bw')[2],
            ['n1', 'n2'],
        )

    def test_majority_tie_and_orphan(self):
        numa, cards, hip = gpu_maps(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        for node in ('a', 'b'):
            cards[node] = dict(list(cards[node].items())[:4])
        pairs = [('a', 'b'), ('c', 'd'), ('e', 'f')]
        roles, selected, excluded = ibperf_lib._prepare_pair_roles(pairs, list(cards), cards, numa, hip, 'bw')
        self.assertEqual(selected, [f'card{g}' for g in range(8)])
        self.assertEqual(excluded, ['a', 'b', 'g'])
        self.assertEqual(list(roles), ['c', 'd', 'e', 'f'])
        self.assertEqual(self.fail.call_count, 2)
        roles, selected, excluded = ibperf_lib._prepare_pair_roles(
            pairs[:2], ['a', 'b', 'c', 'd'], cards, numa, hip, 'bw'
        )
        self.assertEqual(selected, [f'card{g}' for g in range(4)])
        self.assertEqual(excluded, ['c', 'd'])
        self.assertEqual(list(roles), ['a', 'b'])

    def test_missing_command_inputs_excludes_pair(self):
        numa, nic, hip = gpu_maps(['a', 'b', 'c', 'd'])
        del numa['b']['card4']
        del hip['b']['card5']
        del nic['b']['card6']['rdma_dev']
        roles, cards, excluded = ibperf_lib._prepare_pair_roles(
            [('a', 'b'), ('c', 'd')], list(nic), nic, numa, hip, 'bw'
        )
        self.assertEqual(list(roles), ['c', 'd'])
        self.assertEqual(cards, [f'card{g}' for g in range(8)])
        self.assertEqual(excluded, ['a', 'b'])
        message = self.fail.call_args.args[0]
        self.assertIn('b/card4 NUMA CPU list', message)
        self.assertIn('b/card5 HIP device', message)
        self.assertIn('b/card6 RDMA device', message)


class TestDynamicIbperfCharts(unittest.TestCase):
    def setUp(self):
        globals.error_list = []

    def bw_results(self, gpu_count=8):
        return {
            'ib_write_bw': {
                size: {
                    8: {
                        node: {
                            g: {
                                'bw': str(100 * node_index + 10 * size_index + g + 1),
                                'pps': str(1000 * node_index + 100 * size_index + g + 1),
                            }
                            for g in range(gpu_count)
                        }
                        for node_index, node in enumerate(('node1', 'node2'))
                    }
                }
                for size_index, size in enumerate((1024, 2048))
            }
        }

    def lat_results(self, gpu_count=8):
        data = {metric: '1.0' for metric in ('t_min', 't_max', 't_avg', 't_stdev', 't_99_pct')}
        return {'ib_write_lat': {1024: {'node1': {g: dict(data) for g in range(gpu_count)}}}}

    @patch('xlsxwriter.Workbook')
    def test_bw_chart_8_gpu_golden(self, workbook_class):
        worksheet = workbook_class.return_value.add_worksheet.return_value
        ibperf_lib.generate_ibperf_bw_chart(self.bw_results(), 'test.xlsx')
        merges = [
            mock_call('A1:T1', 'Test ib_write_bw - BW, MPPS Numbers for 8 QPs', ANY),
            mock_call('A2:A3', 'Node IP', ANY),
            mock_call('B2:B3', 'Msg Size', ANY),
            mock_call('C2:D2', 'Total', ANY),
            mock_call('E2:F2', 'GPU0', ANY),
            mock_call('G2:H2', 'GPU1', ANY),
            mock_call('I2:J2', 'GPU2', ANY),
            mock_call('K2:L2', 'GPU3', ANY),
            mock_call('M2:N2', 'GPU4', ANY),
            mock_call('O2:P2', 'GPU5', ANY),
            mock_call('Q2:R2', 'GPU6', ANY),
            mock_call('S2:T2', 'GPU7', ANY),
            mock_call('A4:A5', 'node1', ANY),
            mock_call('A6:A7', 'node2', ANY),
            mock_call('A8:A9', 'Cluster Avg'),
        ]
        self.assertEqual(worksheet.merge_range.call_args_list, merges)
        self.assertEqual(worksheet.write_row.call_args_list, [mock_call('C3', ['BW', 'PPS'] * 9, ANY)])
        gpu_bw = {g: [float(100 * n + 10 * s + g + 1) for n in range(2) for s in range(2)] for g in range(8)}
        gpu_pps = {g: [float(1000 * n + 100 * s + g + 1) for n in range(2) for s in range(2)] for g in range(8)}
        total_bw = [sum(gpu_bw[g][i] for g in range(8)) for i in range(4)]
        total_pps = [sum(gpu_pps[g][i] for g in range(8)) for i in range(4)]
        expected = [
            mock_call('B4', [1024.0, 2048.0, 1024.0, 2048.0]),
            mock_call('C4', total_bw),
            mock_call('D4', total_pps),
        ]
        for g in range(8):
            first = ibperf_lib.xl_col_to_name(4 + 2 * g)
            second = ibperf_lib.xl_col_to_name(5 + 2 * g)
            expected.extend([mock_call(f'{first}4', gpu_bw[g]), mock_call(f'{second}4', gpu_pps[g])])
        expected.extend(
            [
                mock_call('B8', [1024, 2048]),
                mock_call('C8', [(total_bw[i] + total_bw[i + 2]) / 2 for i in range(2)]),
                mock_call('D8', [(total_pps[i] + total_pps[i + 2]) / 2 for i in range(2)]),
            ]
        )
        for g in range(8):
            first = ibperf_lib.xl_col_to_name(4 + 2 * g)
            second = ibperf_lib.xl_col_to_name(5 + 2 * g)
            expected.extend(
                [
                    mock_call(f'{first}8', [(gpu_bw[g][i] + gpu_bw[g][i + 2]) / 2 for i in range(2)]),
                    mock_call(f'{second}8', [(gpu_pps[g][i] + gpu_pps[g][i + 2]) / 2 for i in range(2)]),
                ]
            )
        self.assertEqual(worksheet.write_column.call_args_list, expected)

    @patch('xlsxwriter.Workbook')
    def test_bw_chart_four_gpus(self, workbook_class):
        worksheet = workbook_class.return_value.add_worksheet.return_value
        ibperf_lib.generate_ibperf_bw_chart(self.bw_results(4), 'test.xlsx')
        self.assertEqual(
            worksheet.merge_range.call_args_list[0],
            mock_call('A1:L1', 'Test ib_write_bw - BW, MPPS Numbers for 8 QPs', ANY),
        )
        self.assertIn(mock_call('K2:L2', 'GPU3', ANY), worksheet.merge_range.call_args_list)
        self.assertNotIn(mock_call('M2:N2', 'GPU4', ANY), worksheet.merge_range.call_args_list)
        self.assertEqual(worksheet.write_row.call_args_list, [mock_call('C3', ['BW', 'PPS'] * 5, ANY)])

    @patch('xlsxwriter.Workbook')
    def test_lat_chart_8_gpu_golden(self, workbook_class):
        worksheet = workbook_class.return_value.add_worksheet.return_value
        ibperf_lib.generate_ibperf_lat_chart(self.lat_results(), 'test.xlsx')
        self.assertEqual(
            worksheet.merge_range.call_args_list,
            [
                mock_call('A1:Z1', 'Test ib_write_lat - latency results', ANY),
                mock_call('A2:A3', 'Node IP', ANY),
                mock_call('B2:B3', 'Msg Size', ANY),
                mock_call('C2:G2', 'Node Avg ', ANY),
                mock_call(1, 7, 1, 11, 'GPU0', ANY),
                mock_call(1, 12, 1, 16, 'GPU1', ANY),
                mock_call(1, 17, 1, 21, 'GPU2', ANY),
                mock_call(1, 22, 1, 26, 'GPU3', ANY),
                mock_call(1, 27, 1, 31, 'GPU4', ANY),
                mock_call(1, 32, 1, 36, 'GPU5', ANY),
                mock_call(1, 37, 1, 41, 'GPU6', ANY),
                mock_call(1, 42, 1, 46, 'GPU7', ANY),
            ],
        )
        self.assertEqual(
            worksheet.write_row.call_args_list, [mock_call(2, 2, ['Tmin', 'Tmax', 'TAvg', 'TStdev', 'T99pct'] * 9, ANY)]
        )

    @patch('xlsxwriter.Workbook')
    def test_lat_chart_four_gpus(self, workbook_class):
        worksheet = workbook_class.return_value.add_worksheet.return_value
        ibperf_lib.generate_ibperf_lat_chart(self.lat_results(4), 'test.xlsx')
        self.assertEqual(
            worksheet.write_row.call_args_list, [mock_call(2, 2, ['Tmin', 'Tmax', 'TAvg', 'TStdev', 'T99pct'] * 5, ANY)]
        )
        self.assertIn(mock_call(1, 22, 1, 26, 'GPU3', ANY), worksheet.merge_range.call_args_list)
        self.assertNotIn(mock_call(1, 27, 1, 31, 'GPU4', ANY), worksheet.merge_range.call_args_list)

    def test_result_gpu_list(self):
        self.assertEqual(ibperf_lib._result_gpu_list([{}, {0: {}, 3: {}}]), [0, 1, 2, 3])
        self.assertEqual(ibperf_lib._result_gpu_list([{}, {}]), [])


if __name__ == '__main__':
    unittest.main()
