# cvs/lib/unittests/test_ibperf_lib.py
import unittest
from unittest.mock import patch, MagicMock
import cvs.lib.ibperf_lib as ibperf_lib


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
        hip_dev_dict = {n: {f'card{g}': (g + 1) % 8 for g in range(8)} for n in nodes}
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
            hip_dev_dict=hip_dev_dict,
        )

        server_cmd, client_cmd = phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertEqual(
            server_cmd,
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=1'
            ' -x 3 -F -p 1516 -s 64 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            client_cmd,
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=1'
            ' -x 3 -F -p 1516 -s 64 node1 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )


class TestRunIbPerfBwTest(unittest.TestCase):
    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_bw_pps')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_bw_commands_use_hip_device_for_card(self, _dmabuf, mock_bw_pps, _sleep):
        nodes = ('node1', 'node2')
        mock_bw_pps.return_value = {node: {'bw': '1.0', 'pps': '1.0'} for node in nodes}
        gpu_nic_dict = {node: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for node in nodes}
        gpu_numa_dict = {node: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for node in nodes}
        hip_dev_dict = {node: {f'card{g}': (g + 1) % 8 for g in range(8)} for node in nodes}
        phdl = MagicMock()

        ibperf_lib.run_ib_perf_bw_test(
            MagicMock(),
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
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    @patch.object(ibperf_lib, 'get_hip_device_dict')
    def test_computes_hip_dev_dict_when_not_given(self, mock_get_hip_device_dict, _dmabuf, mock_bw_pps, _sleep):
        nodes = ('node1', 'node2')
        mock_bw_pps.return_value = {node: {'bw': '1.0', 'pps': '1.0'} for node in nodes}
        gpu_nic_dict = {node: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for node in nodes}
        gpu_numa_dict = {node: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for node in nodes}
        mock_get_hip_device_dict.return_value = {node: {f'card{g}': (g + 1) % 8 for g in range(8)} for node in nodes}
        phdl = MagicMock()

        ibperf_lib.run_ib_perf_bw_test(
            MagicMock(),
            phdl,
            'ib_write_bw',
            gpu_numa_dict,
            gpu_nic_dict,
            {node: {} for node in nodes},
            '/opt/perftest/bin',
            65536,
            3,
        )

        mock_get_hip_device_dict.assert_called_once_with(phdl, gpu_nic_dict)
        server_cmd = phdl.exec_cmd_list.call_args_list[1].args[0][0]
        self.assertIn('--use_rocm=1', server_cmd)


if __name__ == '__main__':
    unittest.main()
