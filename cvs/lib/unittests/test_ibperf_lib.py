# cvs/lib/unittests/test_ibperf_lib.py
import json
import unittest
from unittest.mock import patch, MagicMock
from cvs.lib import globals
import cvs.lib.ibperf_lib as ibperf_lib


def gpu_maps(nodes):
    gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
    gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
    return gpu_numa_dict, gpu_nic_dict


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


class TestBuildIbperfNodePairs(unittest.TestCase):
    def test_sequential_even_and_odd(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        self.assertEqual(ibperf_lib.build_ibperf_node_pairs(nodes), ([('n1', 'n2'), ('n3', 'n4')], []))
        self.assertEqual(ibperf_lib.build_ibperf_node_pairs(nodes + ['n5']), ([('n1', 'n2'), ('n3', 'n4')], ['n5']))
        self.assertEqual(ibperf_lib.build_ibperf_node_pairs(tuple(nodes + ['n5']))[1], ['n5'])
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(nodes, vpod_map={'n1': 'A'}), ([('n1', 'n2'), ('n3', 'n4')], [])
        )

    def test_intra_vpod_and_missing_labels(self):
        nodes = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']
        labels = {'n1': 'A', 'n2': 'A', 'n3': 'A', 'n4': 'B', 'n5': 'B', 'n6': None}
        pairs, unpaired = ibperf_lib.build_ibperf_node_pairs(nodes, 'intra_vpod', labels)
        self.assertEqual(pairs, [('n1', 'n2'), ('n4', 'n5')])
        self.assertEqual(unpaired, ['n3', 'n6', 'n7'])

    def test_inter_vpod_equal_groups_and_tuple_keys(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        labels = {'n1': (0, 1), 'n2': (0, 1), 'n3': (2, 3), 'n4': (2, 3)}
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(nodes, 'inter_vpod', labels), ([('n1', 'n3'), ('n2', 'n4')], [])
        )

    def test_inter_vpod_unequal_groups(self):
        nodes = ['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'c0', 'c1']
        labels = {n: n[0].upper() for n in nodes}
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(nodes, 'inter_vpod', labels),
            ([('a0', 'b0'), ('a1', 'c0'), ('a2', 'b1'), ('a3', 'c1')], []),
        )

    def test_inter_vpod_single_group(self):
        nodes = ['n1', 'n2', 'n3']
        self.assertEqual(
            ibperf_lib.build_ibperf_node_pairs(nodes, 'inter_vpod', dict.fromkeys(nodes, 'A')), ([], nodes)
        )

    def test_invalid_mode_or_missing_map(self):
        with self.assertRaisesRegex(ValueError, 'pairing_mode'):
            ibperf_lib.build_ibperf_node_pairs(['n1', 'n2'], 'unknown')
        with self.assertRaisesRegex(ValueError, 'vpod_map'):
            ibperf_lib.build_ibperf_node_pairs(['n1', 'n2'], 'inter_vpod')


class TestGetClusterFileVpodMap(unittest.TestCase):
    def test_labels_and_errors(self):
        cluster = {
            'node_dict': {
                'n1': {'vpod_id': ' A '},
                'n2': {'vpod_id': 'B'},
                'n3': {},
                'n4': {'vpod_id': '  '},
                'n5': {'vpod_id': None},
            }
        }
        labels, errors = ibperf_lib.get_cluster_file_vpod_map(cluster)
        self.assertEqual(labels, {'n1': 'A', 'n2': 'B'})
        self.assertEqual(len(errors), 3)
        for node, error in zip(('n3', 'n4', 'n5'), errors):
            self.assertIn(node, error)
            self.assertIn('no vpod_id', error)


class TestParseAfmVpodMembership(unittest.TestCase):
    def test_same_and_distinct_groups(self):
        output = {'n1': afm_devices(range(4), range(8)), 'n2': afm_devices(range(4, 8), range(8))}
        labels, errors = ibperf_lib.parse_afm_vpod_membership(output)
        self.assertEqual(labels, {'n1': tuple(range(8)), 'n2': tuple(range(8))})
        self.assertEqual(errors, [])
        output['n2'] = afm_devices(range(8, 12), range(8, 16))
        labels, errors = ibperf_lib.parse_afm_vpod_membership(output)
        self.assertNotEqual(labels['n1'], labels['n2'])
        self.assertEqual(errors, [])

    def test_command_error_and_multiple_memberships(self):
        mixed = json.loads(afm_devices(range(2), range(4)))
        mixed['devices'][1]['vpod_accelerators'] = list(range(4, 8))
        labels, errors = ibperf_lib.parse_afm_vpod_membership(
            {'n1': 'sudo: afmctl: command not found', 'n2': json.dumps(mixed)}
        )
        self.assertEqual(labels, {})
        self.assertEqual(len(errors), 2)
        self.assertIn('n1', errors[0])
        self.assertIn('multiple vPOD memberships', errors[1])

    def test_no_membership(self):
        labels, errors = ibperf_lib.parse_afm_vpod_membership({'n1': afm_devices(range(2), [])})
        self.assertEqual(labels, {})
        self.assertIn('no vPOD membership', errors[0])

    def test_overlapping_accelerator_ids(self):
        output = {'n1': afm_devices(range(4), range(8)), 'n2': afm_devices(range(2, 6), range(8))}
        with self.assertRaisesRegex(ValueError, 'cluster_file'):
            ibperf_lib.parse_afm_vpod_membership(output)


class TestDiscoverAfmVpodMap(unittest.TestCase):
    def test_default_command_and_result(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': afm_devices(range(4), range(8))}
        labels, errors = ibperf_lib.discover_afm_vpod_map(phdl)
        phdl.exec.assert_called_once_with('sudo -n afmctl show device --json 2>&1', print_console=False)
        self.assertEqual(labels, {'n1': tuple(range(8))})
        self.assertEqual(errors, [])

    def test_custom_path_is_quoted(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': afm_devices(range(4), range(8))}
        ibperf_lib.discover_afm_vpod_map(phdl, '/path with space/afmctl')
        phdl.exec.assert_called_once_with(
            "sudo -n '/path with space/afmctl' show device --json 2>&1", print_console=False
        )


class TestResolveIbperfNodePairing(unittest.TestCase):
    def test_sequential_does_not_create_discovery_handle(self):
        factory = MagicMock()
        result = ibperf_lib.resolve_ibperf_node_pairing(['n1', 'n2', 'n3'], {}, {}, factory)
        factory.assert_not_called()
        self.assertEqual(result['pairs'], [('n1', 'n2')])
        self.assertEqual(result['unpaired'], ['n3'])
        self.assertEqual(result['errors'], [])

    def test_afm_handle_destroyed_on_success_and_error(self):
        nodes = ['n1', 'n2']
        factory = MagicMock()
        factory.return_value.exec.return_value = {
            'n1': afm_devices(range(4), range(8)),
            'n2': afm_devices(range(8, 12), range(8, 16)),
        }
        result = ibperf_lib.resolve_ibperf_node_pairing(nodes, {'pairing_mode': 'inter_vpod'}, {}, factory)
        self.assertEqual(result['pairs'], [('n1', 'n2')])
        factory.assert_called_once_with(nodes)
        factory.return_value.destroy_clients.assert_called_once_with()
        factory.reset_mock()
        factory.return_value.exec.side_effect = RuntimeError('discovery failed')
        with self.assertRaisesRegex(RuntimeError, 'discovery failed'):
            ibperf_lib.resolve_ibperf_node_pairing(nodes, {'pairing_mode': 'inter_vpod'}, {}, factory)
        factory.return_value.destroy_clients.assert_called_once_with()

    def test_cluster_file_partial_labels(self):
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

    def test_invalid_configuration_and_zero_pairs(self):
        for key, value in [('pairing_mode', 'bad'), ('vpod_source', 'bad')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                ibperf_lib.resolve_ibperf_node_pairing(['n1', 'n2'], {key: value}, {})
        with self.assertRaisesRegex(ValueError, 'No ibperf node pairs'):
            ibperf_lib.resolve_ibperf_node_pairing(['n1'], {}, {})
        with self.assertRaisesRegex(ValueError, 'No ibperf node pairs'):
            ibperf_lib.resolve_ibperf_node_pairing(
                ['n1', 'n2'],
                {'pairing_mode': 'inter_vpod', 'vpod_source': 'cluster_file'},
                {'node_dict': {'n1': {'vpod_id': 'A'}, 'n2': {'vpod_id': 'A'}}},
            )
        with self.assertRaisesRegex(ValueError, 'phdl_factory'):
            ibperf_lib.resolve_ibperf_node_pairing(['n1', 'n2'], {'pairing_mode': 'inter_vpod'}, {})


class TestAssignPairRoles(unittest.TestCase):
    def test_roles_follow_node_order(self):
        nodes = ['n1', 'n2', 'n3', 'n4']
        roles, orphans = ibperf_lib._assign_pair_roles([('n1', 'n3'), ('n2', 'n4')], nodes)
        self.assertEqual(roles, {'n1': None, 'n2': None, 'n3': 'n1', 'n4': 'n2'})
        self.assertEqual(list(roles), nodes)
        self.assertEqual(orphans, [])

    def test_missing_partner_and_unpaired_node(self):
        roles, orphans = ibperf_lib._assign_pair_roles([('n1', 'n2'), ('n3', 'n9')], ['n1', 'n2', 'n3', 'n4'])
        self.assertEqual(roles, {'n1': None, 'n2': 'n1'})
        self.assertEqual(orphans, ['n3', 'n4'])

    def test_duplicate_pair_node(self):
        with self.assertRaisesRegex(ValueError, 'more than one'):
            ibperf_lib._assign_pair_roles([('n1', 'n2'), ('n2', 'n3')], ['n1', 'n2', 'n3'])


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
    def setUp(self):
        globals.error_list = []

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

    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_lat_numb')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_explicit_pairs(self, _dmabuf, mock_lat_numb, _sleep):
        nodes = ['n1', 'n2', 'n3', 'n4']
        lat = dict.fromkeys(('t_min', 't_max', 't_typical', 't_avg', 't_stdev', 't_99_pct', 't_99_9_pct'), '1.0')
        mock_lat_numb.return_value = {n: lat for n in nodes}
        gpu_numa_dict, gpu_nic_dict = gpu_maps(nodes)
        phdl = MagicMock()

        result = ibperf_lib.run_ib_perf_lat_test(
            MagicMock(),
            phdl,
            'ib_write_lat',
            gpu_numa_dict,
            gpu_nic_dict,
            {n: {} for n in nodes},
            '/opt/perftest/bin',
            64,
            3,
            node_pairs=[('n1', 'n3'), ('n2', 'n4')],
        )

        self.assertEqual(phdl.exec_cmd_list.call_count, 9)
        commands = phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertIn('-s 64 n1 > /tmp/ib_perf_0_logs', commands[2])
        self.assertIn('-s 64 n2 > /tmp/ib_perf_0_logs', commands[3])
        self.assertEqual({n: set(result[n]) for n in nodes}, {n: set(range(8)) for n in nodes})

    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_lat_numb')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_orphan_results_collected(self, _dmabuf, mock_lat_numb, _sleep):
        nodes = ['n1', 'n2', 'n3']
        lat = dict.fromkeys(('t_min', 't_max', 't_typical', 't_avg', 't_stdev', 't_99_pct', 't_99_9_pct'), '1.0')
        mock_lat_numb.return_value = {n: lat for n in nodes[:2]}
        gpu_numa_dict, gpu_nic_dict = gpu_maps(nodes)
        phdl = MagicMock()

        result = ibperf_lib.run_ib_perf_lat_test(
            MagicMock(),
            phdl,
            'ib_write_lat',
            gpu_numa_dict,
            gpu_nic_dict,
            {n: {} for n in nodes},
            '/opt/perftest/bin',
            64,
            3,
            node_pairs=[('n1', 'n2'), ('n3', 'n9')],
        )

        phdl.prune_nodes.assert_called_once_with(['n3'])
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('n3', globals.error_list[0])
        self.assertEqual(set(result), {'n1', 'n2'})
        self.assertEqual({n: set(result[n]) for n in result}, {n: set(range(8)) for n in nodes[:2]})

    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_no_active_pairs_skips_launch(self, _dmabuf, _sleep):
        gpu_numa_dict, gpu_nic_dict = gpu_maps(['n1'])
        phdl = MagicMock()
        result = ibperf_lib.run_ib_perf_lat_test(
            MagicMock(),
            phdl,
            'ib_write_lat',
            gpu_numa_dict,
            gpu_nic_dict,
            {'n1': {}},
            '/opt/perftest/bin',
            64,
            3,
            node_pairs=[('n1', 'n2')],
        )
        self.assertEqual(result, {})
        phdl.prune_nodes.assert_called_once_with(['n1'])
        phdl.exec_cmd_list.assert_not_called()
        self.assertEqual(len(globals.error_list), 1)


class TestRunIbPerfBwTest(unittest.TestCase):
    def setUp(self):
        globals.error_list = []

    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_bw_pps')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_default_pairs_commands_unchanged(self, _dmabuf, mock_bw_pps, _sleep):
        nodes = ('node1', 'node2')
        mock_bw_pps.return_value = {n: {'bw': '1.0', 'pps': '1.0'} for n in nodes}
        gpu_nic_dict = {n: {f'card{g}': {'rdma_dev': f'rdma{g}'} for g in range(8)} for n in nodes}
        gpu_numa_dict = {n: {f'card{g}': {'local_cpulist': '0-63'} for g in range(8)} for n in nodes}
        phdl = MagicMock()

        result = ibperf_lib.run_ib_perf_bw_test(
            MagicMock(),
            phdl,
            'ib_write_bw',
            gpu_numa_dict,
            gpu_nic_dict,
            {n: {} for n in nodes},
            '/opt/perftest/bin',
            64,
            3,
        )

        calls = phdl.exec_cmd_list.call_args_list
        self.assertEqual(len(calls), 9)
        self.assertEqual(
            calls[0].args[0],
            [
                'echo "sleep 1" >> /tmp/ib_cmds_file.txt',
                'echo "sleep 5" >> /tmp/ib_cmds_file.txt',
            ],
        )
        self.assertEqual(
            calls[1].args[0][0],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        self.assertEqual(
            calls[1].args[0][1],
            'echo "numactl --physcpubind=0-63 --localalloc /opt/perftest/bin/ib_write_bw -d rdma0 --use_rocm=0'
            ' -x 3 --report_gbits -b -F -D 60 -p 1516 -s 64 -q 8 node1 > /tmp/ib_perf_0_logs 2>&1 &" >> /tmp/ib_cmds_file.txt',
        )
        for command in calls[8].args[0]:
            self.assertIn('rdma7 --use_rocm=7', command)
            self.assertIn('-p 1523', command)
            self.assertIn('/tmp/ib_perf_7_logs', command)
        phdl.prune_nodes.assert_not_called()
        self.assertEqual(globals.error_list, [])
        self.assertEqual({n: set(result[n]) for n in nodes}, {n: set(range(8)) for n in nodes})

    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_bw_pps')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_explicit_pairs(self, _dmabuf, mock_bw_pps, _sleep):
        nodes = ['n1', 'n2', 'n3', 'n4']
        mock_bw_pps.return_value = {n: {'bw': '1.0', 'pps': '1.0'} for n in nodes}
        gpu_numa_dict, gpu_nic_dict = gpu_maps(nodes)
        phdl = MagicMock()

        result = ibperf_lib.run_ib_perf_bw_test(
            MagicMock(),
            phdl,
            'ib_write_bw',
            gpu_numa_dict,
            gpu_nic_dict,
            {n: {} for n in nodes},
            '/opt/perftest/bin',
            64,
            3,
            node_pairs=[('n1', 'n3'), ('n2', 'n4')],
        )

        calls = phdl.exec_cmd_list.call_args_list
        self.assertEqual(len(calls), 9)
        self.assertEqual(
            calls[0].args[0],
            [
                'echo "sleep 1" >> /tmp/ib_cmds_file.txt',
                'echo "sleep 1" >> /tmp/ib_cmds_file.txt',
                'echo "sleep 5" >> /tmp/ib_cmds_file.txt',
                'echo "sleep 5" >> /tmp/ib_cmds_file.txt',
            ],
        )
        for gpu in range(8):
            commands = calls[gpu + 1].args[0]
            self.assertEqual(len(commands), 4)
            suffix = f'-q 8 > /tmp/ib_perf_{gpu}_logs 2>&1 &" >> /tmp/ib_cmds_file.txt'
            self.assertTrue(commands[0].endswith(suffix))
            self.assertTrue(commands[1].endswith(suffix))
            self.assertTrue(commands[2].endswith(f'-q 8 n1 > /tmp/ib_perf_{gpu}_logs 2>&1 &" >> /tmp/ib_cmds_file.txt'))
            self.assertTrue(commands[3].endswith(f'-q 8 n2 > /tmp/ib_perf_{gpu}_logs 2>&1 &" >> /tmp/ib_cmds_file.txt'))
        self.assertEqual({n: set(result[n]) for n in nodes}, {n: set(range(8)) for n in nodes})

    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'get_ib_bw_pps')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_orphan_is_pruned(self, _dmabuf, mock_bw_pps, _sleep):
        nodes = ['n1', 'n2', 'n3']
        mock_bw_pps.return_value = {n: {'bw': '1.0', 'pps': '1.0'} for n in nodes[:2]}
        gpu_numa_dict, gpu_nic_dict = gpu_maps(nodes)
        phdl = MagicMock()

        result = ibperf_lib.run_ib_perf_bw_test(
            MagicMock(),
            phdl,
            'ib_write_bw',
            gpu_numa_dict,
            gpu_nic_dict,
            {n: {} for n in nodes},
            '/opt/perftest/bin',
            64,
            3,
            node_pairs=[('n1', 'n2'), ('n3', 'n9')],
        )

        phdl.prune_nodes.assert_called_once_with(['n3'])
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('n3', globals.error_list[0])
        self.assertEqual(phdl.exec_cmd_list.call_count, 9)
        for call in phdl.exec_cmd_list.call_args_list:
            self.assertEqual(len(call.args[0]), 2)
        self.assertEqual(set(result), {'n1', 'n2'})

    @patch.object(ibperf_lib.time, 'sleep')
    @patch.object(ibperf_lib, 'check_perftest_dmabuf_support', return_value=False)
    def test_no_active_pairs_skips_launch(self, _dmabuf, _sleep):
        gpu_numa_dict, gpu_nic_dict = gpu_maps(['n1'])
        phdl = MagicMock()
        result = ibperf_lib.run_ib_perf_bw_test(
            MagicMock(),
            phdl,
            'ib_write_bw',
            gpu_numa_dict,
            gpu_nic_dict,
            {'n1': {}},
            '/opt/perftest/bin',
            64,
            3,
            node_pairs=[('n1', 'n2')],
        )
        self.assertEqual(result, {})
        phdl.prune_nodes.assert_called_once_with(['n1'])
        phdl.exec_cmd_list.assert_not_called()
        self.assertEqual(len(globals.error_list), 1)


if __name__ == '__main__':
    unittest.main()
