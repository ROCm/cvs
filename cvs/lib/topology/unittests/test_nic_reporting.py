"""Unit tests for the NIC bandwidth Run Deck records."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.topology import nic_matrix, nic_reporting


def _matrix(value=100.0):
    inventory = {node: [{'nic': f'nic{i}', 'index': i, 'gpu': 0} for i in range(2)] for node in ('nodeA', 'nodeB')}
    matrix = {}
    for pair in nic_matrix.build_pairs(inventory):
        matrix.setdefault((pair['src_node'], pair['src_nic']), {})[(pair['dst_node'], pair['dst_nic'])] = value
    return matrix


class TestBuildHeatmap(unittest.TestCase):
    def test_opts_into_the_graded_ramp_and_higher_is_better(self):
        heat = nic_reporting.build_heatmap('m', ['r'], ['c'], [[95.0]], threshold=90)
        self.assertEqual(heat['scale'], 'graded')
        self.assertEqual(heat['direction'], 'higher')
        self.assertEqual(heat['threshold'], 90)
        self.assertEqual(heat['unit'], nic_reporting.UNIT)

    def test_a_ragged_grid_is_an_error_not_a_silent_drop(self):
        with self.assertRaises(ValueError):
            nic_reporting.build_heatmap('m', ['r1', 'r2'], ['c'], [[1.0]], threshold=90)
        with self.assertRaises(ValueError):
            nic_reporting.build_heatmap('m', ['r'], ['c1', 'c2'], [[1.0]], threshold=90)

    def test_empty_axes_yield_no_heatmap(self):
        self.assertIsNone(nic_reporting.build_heatmap('m', [], ['c'], [], threshold=90))


class TestNodeHeatmap(unittest.TestCase):
    def test_rows_are_the_nodes_own_nics_and_columns_are_peers_only(self):
        heat = nic_reporting.node_heatmap('nodeA', _matrix(), median=100.0, threshold=90)
        self.assertEqual(heat['rows'], ['nic0', 'nic1'])
        self.assertEqual(heat['cols'], ['nodeB:nic0', 'nodeB:nic1'])
        self.assertNotIn('nodeA:nic0', heat['cols'], 'same-node pairs are never measured')

    def test_values_are_percentages_of_the_median(self):
        matrix = _matrix()
        matrix[('nodeA', 'nic0')][('nodeB', 'nic1')] = 50.0
        heat = nic_reporting.node_heatmap('nodeA', matrix, median=100.0, threshold=90)
        self.assertEqual(heat['values'][0], [100.0, 50.0])

    def test_node_with_no_measurements_has_no_heatmap(self):
        self.assertIsNone(nic_reporting.node_heatmap('nodeZ', _matrix(), median=100.0, threshold=90))


class TestClusterHeatmap(unittest.TestCase):
    def test_grid_covers_every_nic_and_leaves_same_node_cells_empty(self):
        heat = nic_reporting.cluster_heatmap(_matrix(), median=100.0, threshold=90)
        self.assertEqual(heat['rows'], ['nodeA:nic0', 'nodeA:nic1', 'nodeB:nic0', 'nodeB:nic1'])
        self.assertEqual(heat['cols'], heat['rows'])
        # nodeA:nic0 -> nodeA:nic1 was never measured.
        self.assertIsNone(heat['values'][0][1])
        self.assertEqual(heat['values'][0][2], 100.0)


class TestNodeRollup(unittest.TestCase):
    def test_reports_the_worst_nic_pair_between_each_node_pair(self):
        matrix = _matrix()
        matrix[('nodeA', 'nic0')][('nodeB', 'nic1')] = 40.0
        heat = nic_reporting.node_rollup(matrix, median=100.0, threshold=90)
        row = heat['rows'].index('nodeA')
        col = heat['cols'].index('nodeB')
        self.assertEqual(heat['values'][row][col], 40.0)


class TestRecordResults(unittest.TestCase):
    def _record(self, matrix, excluded_by_node=None):
        flags = nic_matrix.flag_outliers(matrix)
        store = {}
        nic_reporting.record_results(
            store, matrix, flags, meta={'suite': 'nic_bandwidth'}, excluded_by_node=excluded_by_node
        )
        return store

    def test_healthy_cluster_records_a_passing_group_for_every_node(self):
        store = self._record(_matrix())
        nodes = store['groups'][nic_reporting.GROUP]['nodes']
        self.assertEqual(sorted(nodes), ['nodeA', 'nodeB'])
        self.assertEqual({record['status'] for record in nodes.values()}, {'pass'})

    def test_each_node_record_carries_its_grid_and_the_rollup(self):
        store = self._record(_matrix())
        heatmaps = store['groups'][nic_reporting.GROUP]['nodes']['nodeA']['heatmaps']
        self.assertEqual([heat['name'] for heat in heatmaps], ['NIC bandwidth', 'Node to node (worst NIC pair)'])

    def test_cluster_wide_grid_is_published_through_meta(self):
        store = self._record(_matrix())
        grid = store['_meta'][nic_reporting.CLUSTER_GRID_META_KEY]
        self.assertEqual(grid['heatmaps'][0]['name'], 'NIC to NIC bandwidth')
        self.assertEqual(grid['heatmaps'][0]['scale'], 'graded')

    def test_a_flagged_pair_fails_both_of_its_nodes(self):
        matrix = _matrix()
        matrix[('nodeA', 'nic0')][('nodeB', 'nic1')] = 10.0
        nodes = self._record(matrix)['groups'][nic_reporting.GROUP]['nodes']
        self.assertEqual(nodes['nodeA']['status'], 'fail')
        self.assertEqual(nodes['nodeB']['status'], 'fail')

    def test_excluded_nics_are_listed_with_their_reason(self):
        excluded = {'nodeA': [{'nic': 'nic9', 'reason': 'link is not ACTIVE'}]}
        nodes = self._record(_matrix(), excluded_by_node=excluded)['groups'][nic_reporting.GROUP]['nodes']
        items = nodes['nodeA']['items']
        self.assertEqual(items[0]['detail'], 'link is not ACTIVE')
        self.assertEqual(items[0]['status'], 'na')

    def test_empty_matrix_publishes_no_cluster_grid(self):
        store = self._record({})
        self.assertEqual(store['_meta'][nic_reporting.CLUSTER_GRID_META_KEY], {})


if __name__ == '__main__':
    unittest.main()
