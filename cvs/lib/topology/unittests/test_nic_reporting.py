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

    def test_a_node_record_carries_only_its_own_grid(self):
        # The node-to-node rollup is cluster-wide. Copying it into all N node
        # records would put N^3 cells in the deck on a large cluster.
        store = self._record(_matrix())
        heatmaps = store['groups'][nic_reporting.GROUP]['nodes']['nodeA']['heatmaps']
        self.assertEqual([heat['name'] for heat in heatmaps], ['NIC bandwidth'])

    def test_the_rollup_is_published_once_beside_the_cluster_grid(self):
        store = self._record(_matrix())
        names = [heat['name'] for heat in store['_meta'][nic_reporting.CLUSTER_GRID_META_KEY]['heatmaps']]
        self.assertEqual(names, ['NIC to NIC bandwidth', 'Node to node (worst NIC pair)'])

    def test_a_node_whose_nics_were_all_excluded_is_still_reported(self):
        # Otherwise the only record of why the node was left out disappears.
        store = self._record(_matrix(), excluded_by_node={'nodeZ': [{'nic': 'nic0', 'reason': 'link is not ACTIVE'}]})
        record = store['groups'][nic_reporting.GROUP]['nodes']['nodeZ']
        self.assertEqual(record['status'], 'na')
        self.assertEqual(record['items'][0]['message'], 'link is not ACTIVE')

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
        self.assertEqual(items[0]['message'], 'link is not ACTIVE')
        self.assertEqual(items[0]['status'], 'na')

    def test_empty_matrix_publishes_no_cluster_grid(self):
        store = self._record({})
        self.assertEqual(store['_meta'][nic_reporting.CLUSTER_GRID_META_KEY], {})


if __name__ == '__main__':
    unittest.main()


class TestExplanationsReachTheRenderedReport(unittest.TestCase):
    """Asserting on the dict is not enough: the deck reads a specific key."""

    def _results_rows(self, store):
        from cvs.lib.report.profile import load_json_profile
        from cvs.lib.report.rundeck.payload import build_rundeck_payload

        payload = build_rundeck_payload(
            profile=load_json_profile('nic_bandwidth'),
            store={'cvs_results_dict': store, 'nic_bandwidth_variant_config': {'hosts': ['nodeA']}},
            cvs_version='0.4.0',
        )
        return payload['datasets']['status_matrix']['results_table']['rows']

    def test_an_excluded_nic_reason_reaches_the_rendered_results_row(self):
        # Asserting on the payload as a whole would pass even with the wrong
        # key, because the raw record is embedded in it; the builder reads one
        # specific field for the row a reader actually sees.
        matrix = _matrix()
        store = {}
        nic_reporting.record_results(
            store,
            matrix,
            nic_matrix.flag_outliers(matrix),
            meta={'suite': 'nic_bandwidth'},
            excluded_by_node={'nodeA': [{'nic': 'nic9', 'reason': 'link is not ACTIVE'}]},
        )
        row = next(row for row in self._results_rows(store) if 'nic9 excluded' in row)
        self.assertIn('link is not ACTIVE', row, f'the reason must render in the row, got {row}')


class TestDestinationOnlyNodes(unittest.TestCase):
    def test_a_node_measured_only_as_a_destination_still_gets_a_heatmap(self):
        # Under quick coverage only the reference nodes are sources; the rest
        # have incoming measurements and would otherwise show no grid at all.
        inv = {n: [{'nic': 'nic0', 'index': 0, 'gpu': 0}] for n in ('nodeA', 'nodeB', 'nodeC')}
        pairs = nic_matrix.build_pairs(inv, coverage='quick')
        matrix = {}
        for pair in pairs:
            source = (pair['src_node'], pair['src_nic'])
            matrix.setdefault(source, {})[(pair['dst_node'], pair['dst_nic'])] = 100.0
        self.assertNotIn('nodeC', {s[0] for s in matrix}, 'nodeC must be destination-only for this test')

        heat = nic_reporting.node_heatmap('nodeC', matrix, 100.0, 90)
        self.assertIsNotNone(heat)
        self.assertEqual(heat['rows'], ['nic0'], "the node's own NICs stay on the rows")
        self.assertEqual(heat['row_label'], 'destination NIC')
        self.assertEqual(heat['col_label'], 'source NIC')
        self.assertTrue(all(cell is not None for row in heat['values'] for cell in row))
