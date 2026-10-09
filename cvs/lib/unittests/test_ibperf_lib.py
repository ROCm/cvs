import unittest
from unittest.mock import ANY, MagicMock, patch

import cvs.lib.ibperf_lib as ibperf_lib


class TestIbperfLib(unittest.TestCase):
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


class IbPerfCommandFixtures:
    def setUp(self):
        nodes = ('node1', 'node2')
        self.phdl = MagicMock()
        self.gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        self.gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
        self.bck_nic_dict = {n: {} for n in nodes}
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
        self.run_latency(use_dmabuf=False)
        self.assertEqual(
            self.commands()[0],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
            ' -x 3 -F -p 1516 -s 64 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            self.commands()[1],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_lat -d rdma0 --use_rocm=0'
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


if __name__ == '__main__':
    unittest.main()
