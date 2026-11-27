"""Unit tests for the cross-node NIC bandwidth matrix.

Node names, NIC names and bandwidth values are synthetic.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.topology import nic_matrix


def _inventory(nodes, nics_per_node=2):
    # TransferBench's index is deliberately not the list position here:
    # a regression to enumerate() ordering must fail these tests.
    return {
        node: [
            {'nic': f'nic{index}', 'index': nics_per_node - 1 - index, 'gpu': index // 2}
            for index in range(nics_per_node)
        ]
        for node in nodes
    }


def _uniform_matrix(value=100.0):
    """Two nodes, two NICs each: every cross-node ordered pair measured."""
    pairs = nic_matrix.build_pairs(_inventory(['nodeA', 'nodeB']))
    matrix = {}
    for pair in pairs:
        source = (pair['src_node'], pair['src_nic'])
        matrix.setdefault(source, {})[(pair['dst_node'], pair['dst_nic'])] = value
    return matrix


class TestCountFullPairs(unittest.TestCase):
    def test_uniform_cluster(self):
        # N nodes x K NICs, ordered and cross-node only: N(N-1)K^2.
        self.assertEqual(nic_matrix.count_full_pairs(_inventory(['a', 'b', 'c'], nics_per_node=8)), 3 * 2 * 64)

    def test_counts_both_directions_when_nic_counts_differ(self):
        inventory = {'a': [{}] * 8, 'b': [{}] * 4}
        self.assertEqual(nic_matrix.count_full_pairs(inventory), 8 * 4 + 4 * 8)

    def test_nodes_without_nics_contribute_nothing(self):
        self.assertEqual(nic_matrix.count_full_pairs({'a': [{}, {}], 'b': [{}, {}], 'c': []}), 8)

    def test_a_single_node_has_no_cross_node_pairs(self):
        self.assertEqual(nic_matrix.count_full_pairs({'a': [{}] * 8}), 0)


class TestResolveCoverage(unittest.TestCase):
    def test_explicit_values_pass_through(self):
        self.assertEqual(nic_matrix.resolve_coverage('full', 10**6), 'full')
        self.assertEqual(nic_matrix.resolve_coverage('quick', 2), 'quick')

    def test_auto_switches_at_the_pair_budget(self):
        budget = nic_matrix.DEFAULT_AUTO_FULL_MAX_PAIRS
        self.assertEqual(nic_matrix.resolve_coverage('auto', budget), 'full')
        self.assertEqual(nic_matrix.resolve_coverage('auto', budget + 1), 'quick')

    def test_the_budget_follows_nics_per_node_not_just_node_count(self):
        # The same node count is affordable with few NICs and not with many;
        # a node-count cutoff cannot express that.
        thin = nic_matrix.count_full_pairs(_inventory([f'n{i}' for i in range(10)], nics_per_node=2))
        fat = nic_matrix.count_full_pairs(_inventory([f'n{i}' for i in range(10)], nics_per_node=8))
        self.assertEqual(nic_matrix.resolve_coverage('auto', thin), 'full')
        self.assertEqual(nic_matrix.resolve_coverage('auto', fat), 'quick')

    def test_unknown_coverage_is_rejected(self):
        with self.assertRaises(ValueError):
            nic_matrix.resolve_coverage('sample', 4)


class TestResolveTrafficClass(unittest.TestCase):
    def test_auto_follows_the_cluster_nccl_traffic_class(self):
        self.assertEqual(nic_matrix.resolve_traffic_class('auto', {'NCCL_IB_TC': '128'}), 128)

    def test_auto_without_cluster_setting_is_zero(self):
        self.assertEqual(nic_matrix.resolve_traffic_class('auto', {}), 0)
        self.assertEqual(nic_matrix.resolve_traffic_class('auto', None), 0)
        self.assertEqual(nic_matrix.resolve_traffic_class('auto', {'NCCL_IB_TC': '  '}), 0)

    def test_explicit_integer_is_used(self):
        self.assertEqual(nic_matrix.resolve_traffic_class(96, {'NCCL_IB_TC': '128'}), 96)
        self.assertEqual(nic_matrix.resolve_traffic_class('96', None), 96)

    def test_out_of_range_and_wrong_types_are_rejected(self):
        for bad in [-1, 256, 'tc', True, None, [0], {}]:
            with self.assertRaises(ValueError):
                nic_matrix.resolve_traffic_class(bad, None)

    def test_unusable_cluster_value_is_rejected_not_silently_defaulted(self):
        with self.assertRaises(ValueError):
            nic_matrix.resolve_traffic_class('auto', {'NCCL_IB_TC': 'high'})


class TestSelectReferenceNodes(unittest.TestCase):
    def test_prefers_nodes_on_different_switches(self):
        switches = {'nodeA': 'sw1', 'nodeB': 'sw1', 'nodeC': 'sw2'}
        self.assertEqual(nic_matrix.select_reference_nodes(switches, switches), ['nodeA', 'nodeC'])

    def test_without_switch_data_the_choice_is_deterministic(self):
        self.assertEqual(nic_matrix.select_reference_nodes(['nodeC', 'nodeA', 'nodeB']), ['nodeA', 'nodeB'])

    def test_falls_back_when_every_node_shares_a_switch(self):
        switches = {'nodeA': 'sw1', 'nodeB': 'sw1'}
        self.assertEqual(nic_matrix.select_reference_nodes(switches, switches), ['nodeA', 'nodeB'])


class TestBuildPairs(unittest.TestCase):
    def test_never_pairs_a_node_with_itself(self):
        pairs = nic_matrix.build_pairs(_inventory(['nodeA', 'nodeB', 'nodeC'], nics_per_node=4))
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertNotEqual(pair['src_node'], pair['dst_node'])
            self.assertNotEqual(pair['src_rank'], pair['dst_rank'])

    def test_full_coverage_measures_every_ordered_cross_node_pair(self):
        pairs = nic_matrix.build_pairs(_inventory(['nodeA', 'nodeB'], nics_per_node=2))
        # 2 source NICs per node x 2 destination NICs x 2 directions.
        self.assertEqual(len(pairs), 8)
        self.assertEqual(len({(p['src_node'], p['src_nic'], p['dst_node'], p['dst_nic']) for p in pairs}), 8)

    def test_quick_coverage_sources_only_reference_nodes(self):
        inventory = _inventory(['nodeA', 'nodeB', 'nodeC'], nics_per_node=1)
        pairs = nic_matrix.build_pairs(inventory, coverage='quick')
        self.assertEqual({p['src_node'] for p in pairs}, {'nodeA', 'nodeB'})
        self.assertEqual({p['dst_node'] for p in pairs}, {'nodeA', 'nodeB', 'nodeC'})

    def test_rank_zero_is_the_first_node_in_sorted_order(self):
        pairs = nic_matrix.build_pairs(_inventory(['nodeB', 'nodeA'], nics_per_node=1))
        ranks = {p['src_node']: p['src_rank'] for p in pairs}
        self.assertEqual(ranks['nodeA'], 0)
        self.assertEqual(ranks['nodeB'], 1)

    def test_nodes_without_usable_nics_are_left_out(self):
        inventory = _inventory(['nodeA', 'nodeB'], nics_per_node=1)
        inventory['nodeC'] = []
        pairs = nic_matrix.build_pairs(inventory)
        self.assertNotIn('nodeC', {p['src_node'] for p in pairs} | {p['dst_node'] for p in pairs})

    def test_single_node_cluster_yields_nothing_to_measure(self):
        self.assertEqual(nic_matrix.build_pairs(_inventory(['nodeA'])), [])

    def test_nic_and_gpu_indices_come_from_the_entry_not_its_list_position(self):
        # TransferBench numbers NICs and GPUs positionally in its own table,
        # which matches neither device-name order nor amd-smi's GPU order.
        # Here every entry's index is the opposite of its list position, so
        # deriving either from enumerate() produces the wrong device.
        inventory = {
            'nodeA': [{'nic': 'first', 'index': 5, 'gpu': 7}, {'nic': 'second', 'index': 2, 'gpu': 3}],
            'nodeB': [{'nic': 'only', 'index': 4, 'gpu': 6}],
        }
        pairs = nic_matrix.build_pairs(inventory)
        by_nic = {p['src_nic']: (p['src_nic_index'], p['src_gpu']) for p in pairs}
        self.assertEqual(by_nic['first'], (5, 7))
        self.assertEqual(by_nic['second'], (2, 3))
        self.assertEqual(by_nic['only'], (4, 6))

    def test_an_entry_without_a_transferbench_index_is_refused(self):
        # Silently falling back to list position is the defect this guards.
        with self.assertRaises(ValueError):
            nic_matrix.build_pairs({'a': [{'nic': 'n0', 'gpu': 0}], 'b': [{'nic': 'n0', 'gpu': 0}]})

    def test_a_node_missing_from_the_rank_map_is_refused(self):
        inventory = _inventory(['nodeA', 'nodeB'], nics_per_node=1)
        with self.assertRaises(ValueError):
            nic_matrix.build_pairs(inventory, rank_by_node={'nodeA': 0})


class TestRenderTransferFile(unittest.TestCase):
    def test_line_format_matches_the_transferbench_grammar(self):
        inventory = {
            'nodeA': [{'nic': 'nic0', 'index': 0, 'gpu': 0}],
            'nodeB': [{'nic': 'nic0', 'index': 0, 'gpu': 0}],
        }
        rendered = nic_matrix.render_transfer_file(nic_matrix.build_pairs(inventory), queue_pairs=4)
        self.assertEqual(
            rendered.splitlines(),
            ['1 4 (R0G0->R0I0.0->R1G0)', '1 4 (R1G0->R1I0.0->R0G0)'],
        )

    def test_no_generated_line_is_a_same_node_transfer(self):
        pairs = nic_matrix.build_pairs(_inventory(['nodeA', 'nodeB', 'nodeC'], nics_per_node=4))
        for line in nic_matrix.render_transfer_file(pairs, queue_pairs=4).splitlines():
            source_rank = line.split('(R')[1].split('G')[0]
            dest_rank = line.rsplit('->R', 1)[1].split('G')[0]
            self.assertNotEqual(source_rank, dest_rank, line)

    def test_a_same_rank_pair_is_refused(self):
        bad = [
            {
                'src_node': 'nodeA',
                'src_rank': 0,
                'src_nic': 'nic0',
                'src_nic_index': 0,
                'src_gpu': 0,
                'dst_node': 'nodeA',
                'dst_rank': 0,
                'dst_nic': 'nic1',
                'dst_nic_index': 1,
                'dst_gpu': 0,
            }
        ]
        with self.assertRaises(ValueError):
            nic_matrix.render_transfer_file(bad, queue_pairs=4)

    def test_queue_pairs_must_be_a_positive_integer(self):
        pairs = nic_matrix.build_pairs(_inventory(['nodeA', 'nodeB'], nics_per_node=1))
        for bad in [0, -1, '4', 1.5, True, None]:
            with self.assertRaises(ValueError):
                nic_matrix.render_transfer_file(pairs, queue_pairs=bad)

    def test_no_pairs_renders_nothing(self):
        self.assertEqual(nic_matrix.render_transfer_file([], queue_pairs=4), '')


class TestParseResults(unittest.TestCase):
    def setUp(self):
        self.inventory = {
            'nodeA': [{'nic': 'nic0', 'index': 0, 'gpu': 0}],
            'nodeB': [{'nic': 'nic0', 'index': 0, 'gpu': 0}],
        }
        self.pairs = nic_matrix.build_pairs(self.inventory)

    def test_results_map_onto_the_pairs_that_produced_them(self):
        output = 'Executor: Rank 0 NIC 0 | 94.36 GB/s\nExecutor: Rank 1 NIC 0 | 93.69 GB/s\n'
        matrix = nic_matrix.parse_results(output, self.pairs)
        self.assertEqual(matrix[('nodeA', 'nic0')][('nodeB', 'nic0')], 94.36)
        self.assertEqual(matrix[('nodeB', 'nic0')][('nodeA', 'nic0')], 93.69)

    def test_surrounding_output_is_tolerated(self):
        output = 'banner\nwarmup done\nExecutor: Rank 0 NIC 0 | 94.36 GB/s\ntrailing\nExecutor: Rank 1 NIC 0 | 93.69 GB/s\nbye\n'
        self.assertEqual(len(nic_matrix.parse_results(output, self.pairs)), 2)

    def test_a_short_read_is_an_error_not_a_partial_matrix(self):
        with self.assertRaises(ValueError):
            nic_matrix.parse_results('Executor: Rank 0 NIC 0 | 94.36 GB/s\n', self.pairs)

    def test_results_out_of_order_are_rejected(self):
        output = 'Executor: Rank 1 NIC 0 | 93.69 GB/s\nExecutor: Rank 0 NIC 0 | 94.36 GB/s\n'
        with self.assertRaises(ValueError):
            nic_matrix.parse_results(output, self.pairs)

    def test_empty_output_with_expected_pairs_is_an_error(self):
        with self.assertRaises(ValueError):
            nic_matrix.parse_results('', self.pairs)

    def test_matches_the_box_drawing_separator_real_transferbench_prints(self):
        # Real TransferBench (confirmed on hardware, v1.69.01) renders its result
        # table with U+2502, not an ASCII pipe; a parser built only against
        # hand-written ASCII fixtures never matches a single real result line.
        output = (
            '  Executor: Rank 0 NIC 00 │  47.531 GB/s │   5.648 ms │   268435456 bytes │  '
            '47.549 GB/s (sum) [tus1-p3-g40] \n'
            '  Executor: Rank 1 NIC 00 │  44.825 GB/s │   5.988 ms │   268435456 bytes │  '
            '44.839 GB/s (sum) [tus1-p3-g42] \n'
        )
        matrix = nic_matrix.parse_results(output, self.pairs)
        self.assertEqual(matrix[('nodeA', 'nic0')][('nodeB', 'nic0')], 47.531)
        self.assertEqual(matrix[('nodeB', 'nic0')][('nodeA', 'nic0')], 44.825)


class TestFlagOutliers(unittest.TestCase):
    def test_healthy_matrix_flags_nothing(self):
        flags = nic_matrix.flag_outliers(_uniform_matrix())
        self.assertEqual((flags['nodes'], flags['nics'], flags['pairs']), ([], [], []))
        self.assertEqual(flags['median'], 100.0)

    def test_single_weak_link_is_reported_as_one_pair(self):
        matrix = _uniform_matrix()
        matrix[('nodeA', 'nic0')][('nodeB', 'nic1')] = 50.0
        flags = nic_matrix.flag_outliers(matrix)

        self.assertEqual(flags['pairs'], [(('nodeA', 'nic0'), ('nodeB', 'nic1'))])
        self.assertEqual(flags['nics'], [])
        self.assertEqual(flags['nodes'], [])
        self.assertIn('NIC pair nodeA:nic0 -> nodeB:nic1', nic_matrix.describe_flags(flags)[0])

    def test_a_nic_low_on_every_pair_is_reported_once_as_a_nic(self):
        matrix = _uniform_matrix()
        for destination in matrix[('nodeA', 'nic0')]:
            matrix[('nodeA', 'nic0')][destination] = 50.0
        flags = nic_matrix.flag_outliers(matrix)

        self.assertEqual(flags['nics'], [('nodeA', 'nic0')])
        self.assertEqual(flags['pairs'], [])
        self.assertIn('NIC nic0 on nodeA', nic_matrix.describe_flags(flags)[0])

    def test_a_destination_nic_low_on_every_pair_is_reported_as_that_nic(self):
        matrix = _uniform_matrix()
        for source in matrix:
            if ('nodeB', 'nic1') in matrix[source]:
                matrix[source][('nodeB', 'nic1')] = 50.0
        flags = nic_matrix.flag_outliers(matrix)

        self.assertEqual(flags['nics'], [('nodeB', 'nic1')])
        self.assertEqual(flags['pairs'], [])

    def test_a_node_pair_low_on_every_nic_is_reported_as_the_node_pair(self):
        matrix = _uniform_matrix()
        for source, row in matrix.items():
            if source[0] != 'nodeA':
                continue
            for destination in row:
                row[destination] = 50.0
        flags = nic_matrix.flag_outliers(matrix)

        self.assertEqual(flags['nodes'], [('nodeA', 'nodeB')])
        self.assertEqual(flags['pairs'], [])
        self.assertEqual(flags['nics'], [], 'the node pair already explains these NICs')
        self.assertIn('node pair nodeA -> nodeB', nic_matrix.describe_flags(flags)[0])

    def test_threshold_follows_the_configured_percentage(self):
        matrix = _uniform_matrix()
        matrix[('nodeA', 'nic0')][('nodeB', 'nic1')] = 95.0
        self.assertEqual(nic_matrix.flag_outliers(matrix, 90)['pairs'], [])
        self.assertEqual(nic_matrix.flag_outliers(matrix, 99)['pairs'], [(('nodeA', 'nic0'), ('nodeB', 'nic1'))])

    def test_empty_matrix_is_not_an_error(self):
        flags = nic_matrix.flag_outliers({})
        self.assertIsNone(flags['median'])
        self.assertEqual(flags['pairs'], [])


class TestToTopologyJson(unittest.TestCase):
    def test_carries_the_full_matrix_and_the_flags(self):
        matrix = _uniform_matrix()
        matrix[('nodeA', 'nic0')][('nodeB', 'nic1')] = 50.0
        flags = nic_matrix.flag_outliers(matrix)
        document = nic_matrix.to_topology_json(matrix, flags, meta={'traffic_class': 0})

        self.assertEqual(len(document['matrix']), 8)
        self.assertEqual(document['meta']['traffic_class'], 0)
        self.assertEqual(document['median_gbps'], 100.0)
        self.assertEqual(
            document['flagged']['pairs'],
            [{'src_node': 'nodeA', 'src_nic': 'nic0', 'dst_node': 'nodeB', 'dst_nic': 'nic1'}],
        )


if __name__ == '__main__':
    unittest.main()
