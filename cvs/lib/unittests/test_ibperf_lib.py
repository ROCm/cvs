import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

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


class IbPerfCommandFixtures:
    def setUp(self):
        nodes = ('node1', 'node2')
        self.phdl = MagicMock()
        self.gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        self.gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
        self.bck_nic_dict = {n: {} for n in nodes}
        self.hip_dev_dict = {n: {f'card{g}': g for g in range(8)} for n in nodes}
        lat = dict.fromkeys(('t_min', 't_max', 't_typical', 't_avg', 't_stdev', 't_99_pct', 't_99_9_pct'), '1.0')
        lat_patch = patch.object(ibperf_lib, 'get_ib_lat_numb', return_value={n: lat for n in nodes})
        bw_patch = patch.object(
            ibperf_lib, 'get_ib_bw_pps', return_value={n: {'bw': '1.0', 'pps': '1.0'} for n in nodes}
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
    def test_builds_latency_commands(self):
        hip_dev_dict = {n: {f'card{g}': (g + 1) % 8 for g in range(8)} for n in ('node1', 'node2')}
        self.run_latency(use_dmabuf=False, hip_dev_dict=hip_dev_dict)
        self.assertEqual(
            self.commands()[0],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=1'
            ' -x 3 -F -p 1516 -s 64 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=1'
            ' -x 3 -F -p 1516 -s 64 node1 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )

    def test_dmabuf_flag_added_when_enabled(self):
        self.run_latency(use_dmabuf=True)
        self.assertEqual(
            self.commands()[0],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' --use_rocm_dmabuf -x 3 -F -p 1516 -s 64 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' --use_rocm_dmabuf -x 3 -F -p 1516 -s 64 node1 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )

    def test_dmabuf_defaults_on(self):
        self.run_latency()
        self.assertIn('--use_rocm_dmabuf', self.commands()[0])

    def test_does_not_probe_help(self):
        self.run_latency()
        self.assertFalse(any('--help' in call.args[0] for call in self.phdl.exec.call_args_list))


class TestRunIbPerfBwTest(IbPerfCommandFixtures, unittest.TestCase):
    def test_builds_bw_commands_without_dmabuf(self):
        self.run_bandwidth(use_dmabuf=False)
        self.assertEqual(
            self.commands()[0],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8 > /tmp/ib_perf_0_logs 2>&1 &"'
            ' >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8 node1 > /tmp/ib_perf_0_logs 2>&1 &"'
            ' >> /tmp/ib_cmds_file.txt',
        )

    def test_builds_bw_commands_with_dmabuf(self):
        self.run_bandwidth(use_dmabuf=True)
        self.assertEqual(
            self.commands()[0],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' --use_rocm_dmabuf -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8'
            ' > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' --use_rocm_dmabuf -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8 node1'
            ' > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )


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
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=1'
            ' -x 3 --report_gbits -b -F -D 10 -p 1516 -s 65536 -q 16 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            client_cmd,
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=1'
            ' -x 3 --report_gbits -b -F -D 10 -p 1516 -s 65536 -q 16 node1 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
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


if __name__ == '__main__':
    unittest.main()
