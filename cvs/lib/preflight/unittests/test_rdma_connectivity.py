# cvs/lib/preflight/unittests/test_rdma_connectivity.py
"""Unit tests for RDMA preflight connectivity (class-based implementation)."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add cvs package root (four levels up: unittests -> preflight -> lib -> cvs)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.base import partition_nodes_into_groups
from cvs.lib.preflight.rdma_connectivity import RdmaConnectivityCheck


def _make_checker(
    phdl=None,
    node_list=None,
    *,
    expected_interfaces=None,
    parallel_group_size=2,
    **kwargs,
):
    """Build RdmaConnectivityCheck with sensible test defaults."""
    phdl = phdl or MagicMock()
    node_list = node_list or ['A', 'B', 'C', 'D']
    return RdmaConnectivityCheck(
        phdl,
        node_list,
        mode=kwargs.get('mode', 'full_mesh'),
        port_range=kwargs.get('port_range', '5000-6000'),
        timeout=kwargs.get('timeout', 30),
        expected_interfaces=expected_interfaces,
        gid_index=kwargs.get('gid_index', '3'),
        parallel_group_size=parallel_group_size,
        config_dict=kwargs.get('config_dict', {}),
    )


class TestPartitionNodesIntoGroups(unittest.TestCase):
    """Test node partitioning (PreflightCheck.partition_nodes_into_groups / base alias)."""

    def test_partition_4_nodes_group_size_2(self):
        nodes = ['A', 'B', 'C', 'D']
        groups = partition_nodes_into_groups(nodes, group_size=2)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups['group_1'], ['A', 'B'])
        self.assertEqual(groups['group_2'], ['C', 'D'])

    def test_partition_4_nodes_group_size_4(self):
        nodes = ['A', 'B', 'C', 'D']
        groups = partition_nodes_into_groups(nodes, group_size=4)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups['group_1'], ['A', 'B', 'C', 'D'])

    def test_partition_6_nodes_group_size_4(self):
        nodes = ['A', 'B', 'C', 'D', 'E', 'F']
        groups = partition_nodes_into_groups(nodes, group_size=4)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups['group_1'], ['A', 'B', 'C', 'D'])
        self.assertEqual(groups['group_2'], ['E', 'F'])

    def test_partition_5_nodes_group_size_2(self):
        """Odd node count: last group is smaller (common with parallel_group_size=2)."""
        nodes = ['A', 'B', 'C', 'D', 'E']
        groups = partition_nodes_into_groups(nodes, group_size=2)

        self.assertEqual(len(groups), 3)
        self.assertEqual(groups['group_1'], ['A', 'B'])
        self.assertEqual(groups['group_2'], ['C', 'D'])
        self.assertEqual(groups['group_3'], ['E'])

    def test_partition_empty_list(self):
        groups = partition_nodes_into_groups([], group_size=2)
        self.assertEqual(len(groups), 0)

    def test_partition_single_node(self):
        groups = partition_nodes_into_groups(['A'], group_size=2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups['group_1'], ['A'])


class TestCalculatePortAssignments(unittest.TestCase):
    """Test RdmaConnectivityCheck._calculate_port_assignments."""

    def setUp(self):
        self.interfaces = ['if1', 'if2', 'if3', 'if4', 'if5', 'if6', 'if7', 'if8']
        self.port_start = 5000

    def test_intragroup_assignments_2_nodes(self):
        checker = _make_checker(expected_interfaces=self.interfaces)
        test_groups = {'group_1': ['A', 'B']}

        assignments = checker._calculate_port_assignments(test_groups, 'intra_group', self.port_start)

        self.assertEqual(len(assignments), 128)

        a_to_b = [a for a in assignments if a['server_node'] == 'A' and a['client_node'] == 'B']
        b_to_a = [a for a in assignments if a['server_node'] == 'B' and a['client_node'] == 'A']

        self.assertEqual(len(a_to_b), 64)
        self.assertEqual(len(b_to_a), 64)

        a_to_b_interfaces = [(a['server_iface'], a['client_iface']) for a in a_to_b]
        expected_combinations = [(s_if, c_if) for s_if in self.interfaces for c_if in self.interfaces]
        self.assertEqual(len(a_to_b_interfaces), len(expected_combinations))
        for combo in expected_combinations:
            self.assertIn(combo, a_to_b_interfaces)

    def test_intragroup_assignments_4_nodes(self):
        checker = _make_checker(expected_interfaces=self.interfaces)
        test_groups = {'group_1': ['A', 'B', 'C', 'D']}

        assignments = checker._calculate_port_assignments(test_groups, 'intra_group', self.port_start)

        self.assertEqual(len(assignments), 768)

        self_connections = [a for a in assignments if a['server_node'] == a['client_node']]
        self.assertEqual(len(self_connections), 0)

        node_pairs = {(a['server_node'], a['client_node']) for a in assignments}
        self.assertEqual(len(node_pairs), 12)

    def test_intergroup_assignments(self):
        checker = _make_checker(expected_interfaces=self.interfaces)
        test_groups = {'group_1_to_group_2': {'group1': ['A', 'B'], 'group2': ['C', 'D']}}

        assignments = checker._calculate_port_assignments(test_groups, 'inter_group', self.port_start)

        self.assertEqual(len(assignments), 256)

        for assignment in assignments:
            self.assertIn(assignment['server_node'], ['A', 'B'])
            self.assertIn(assignment['client_node'], ['C', 'D'])

    def test_port_assignment_uniqueness_per_server(self):
        checker = _make_checker(expected_interfaces=self.interfaces)
        test_groups = {'group_1': ['A', 'B']}

        assignments = checker._calculate_port_assignments(test_groups, 'intra_group', self.port_start)

        a_assignments = [a for a in assignments if a['server_node'] == 'A']
        a_ports = [a['port'] for a in a_assignments]

        self.assertEqual(len(set(a_ports)), 64)
        self.assertEqual(min(a_ports), self.port_start)
        self.assertEqual(max(a_ports), self.port_start + 63)

        b_assignments = [a for a in assignments if a['server_node'] == 'B']
        b_ports = [a['port'] for a in b_assignments]

        self.assertEqual(len(set(b_ports)), 64)
        self.assertEqual(min(b_ports), self.port_start)
        self.assertEqual(max(b_ports), self.port_start + 63)


class TestConnectivityMathInvariant(unittest.TestCase):
    """Total mocked tests should be invariant to parallel_group_size when mesh is full."""

    def setUp(self):
        self.nodes = ['A', 'B', 'C', 'D']
        self.interfaces = ['if1', 'if2', 'if3', 'if4', 'if5', 'if6', 'if7', 'if8']

    @patch.object(RdmaConnectivityCheck, '_execute_round_with_coordination')
    def test_execute_full_mesh_group_size_invariant(self, mock_execute_round):
        def make_side_effect(checker):
            invocation = [0]

            def mock_execute_side_effect(test_groups, round_type, port_start, **kwargs):
                assignments = checker._calculate_port_assignments(test_groups, round_type, port_start)
                n = invocation[0]
                invocation[0] += 1
                return {f'assignment_{n}_{i}': f'mock_result_{n}_{i}' for i in range(len(assignments))}

            return mock_execute_side_effect

        mock_phdl = MagicMock()
        mock_phdl.reachable_hosts = self.nodes

        c2 = _make_checker(
            phdl=mock_phdl,
            node_list=self.nodes,
            expected_interfaces=self.interfaces,
            parallel_group_size=2,
        )
        mock_execute_round.side_effect = make_side_effect(c2)
        results_group2, _, _ = c2._execute_full_mesh(5000)

        c4 = _make_checker(
            phdl=mock_phdl,
            node_list=self.nodes,
            expected_interfaces=self.interfaces,
            parallel_group_size=4,
        )
        mock_execute_round.side_effect = make_side_effect(c4)
        results_group4, _, _ = c4._execute_full_mesh(5000)

        self.assertEqual(len(results_group2), len(results_group4))

        expected_total = 4 * 3 * 8 * 8
        self.assertEqual(len(results_group2), expected_total)
        self.assertEqual(len(results_group4), expected_total)

    @patch.object(RdmaConnectivityCheck, '_execute_round_with_coordination')
    def test_full_mesh_five_nodes_eight_ifaces_parallel_group_size_2(self, mock_execute_round):
        """
        Full mesh total tests = N × (N − 1) × I²; parallel_group_size only batches work, not totals.

        With N=5, I=8: 5 × 4 × 64 = 1280. (If a live run shows 768 with five hosts, that is 4×3×8×8 —
        typically only four reachable nodes participated in RDMA, or a stale report.)
        """
        nodes = ['A', 'B', 'C', 'D', 'E']
        interfaces = ['if1', 'if2', 'if3', 'if4', 'if5', 'if6', 'if7', 'if8']
        n, i = len(nodes), len(interfaces)
        expected_total = n * (n - 1) * i * i

        def make_side_effect(checker):
            invocation = [0]

            def mock_execute_side_effect(test_groups, round_type, port_start, **kwargs):
                assignments = checker._calculate_port_assignments(test_groups, round_type, port_start)
                inv = invocation[0]
                invocation[0] += 1
                return {f'assignment_{inv}_{j}': f'mock_{inv}_{j}' for j in range(len(assignments))}

            return mock_execute_side_effect

        mock_phdl = MagicMock()
        mock_phdl.reachable_hosts = nodes

        checker = _make_checker(
            phdl=mock_phdl,
            node_list=nodes,
            expected_interfaces=interfaces,
            parallel_group_size=2,
        )
        mock_execute_round.side_effect = make_side_effect(checker)

        results, _, _ = checker._execute_full_mesh(5000)

        self.assertEqual(len(results), expected_total)
        self.assertEqual(expected_total, 1280)

    @patch.object(RdmaConnectivityCheck, '_execute_round_with_coordination')
    def test_execute_full_mesh_completeness(self, mock_execute_round):
        generated_assignments = []

        mock_phdl = MagicMock()
        mock_phdl.reachable_hosts = self.nodes

        checker = _make_checker(
            phdl=mock_phdl,
            node_list=self.nodes,
            expected_interfaces=self.interfaces,
            parallel_group_size=2,
        )

        def capture_assignments(test_groups, round_type, port_start, **kwargs):
            assignments = checker._calculate_port_assignments(test_groups, round_type, port_start)
            generated_assignments.extend(assignments)
            return {f'result_{i}': f'mock_{i}' for i in range(len(assignments))}

        mock_execute_round.side_effect = capture_assignments

        checker._execute_full_mesh(5000)

        connections = {(a['server_node'], a['client_node']) for a in generated_assignments}

        expected_connections = {
            ('A', 'B'),
            ('A', 'C'),
            ('A', 'D'),
            ('B', 'A'),
            ('B', 'C'),
            ('B', 'D'),
            ('C', 'A'),
            ('C', 'B'),
            ('C', 'D'),
            ('D', 'A'),
            ('D', 'B'),
            ('D', 'C'),
        }

        self.assertEqual(connections, expected_connections, 'Algorithm should test all directed node pairs')


class TestAlgorithmBugDetection(unittest.TestCase):
    """Inter-group port assignments for one round (same helper as production)."""

    def test_incomplete_connectivity_with_2_groups(self):
        nodes = ['A', 'B', 'C', 'D']
        interfaces = ['if1', 'if2']

        groups = partition_nodes_into_groups(nodes, group_size=2)
        checker = _make_checker(expected_interfaces=interfaces, node_list=nodes)
        # Round 1 is the first inter-group round used by _execute_intergroup (range(1, num_groups))
        inter_group_tests = checker._generate_intergroup_pairs(groups, round_num=1)
        all_inter_assignments = checker._calculate_port_assignments(inter_group_tests, 'inter_group', 5000)

        # Two directed inter-group tests (g1→g2 and g2→g1) × 2×2 nodes × 2×2 iface pairs = 2×16 = 32
        self.assertEqual(len(all_inter_assignments), 32)


class TestGenerateIntergroupPairs(unittest.TestCase):
    """RdmaConnectivityCheck._generate_intergroup_pairs round-robin behavior."""

    def setUp(self):
        self.groups_2 = {'group_1': ['A', 'B'], 'group_2': ['C', 'D']}
        self.groups_3 = {'group_1': ['A', 'B'], 'group_2': ['C', 'D'], 'group_3': ['E', 'F']}

    def test_round_2_with_2_groups(self):
        checker = _make_checker()
        # For 2 groups, round_num=2 yields no pairs (group2_idx == i for both i); round 1 is used in practice.
        pairs_empty = checker._generate_intergroup_pairs(self.groups_2, round_num=2)
        self.assertEqual(pairs_empty, {})
        pairs_r1 = checker._generate_intergroup_pairs(self.groups_2, round_num=1)
        self.assertTrue(pairs_r1)

    def test_all_rounds_with_2_groups(self):
        checker = _make_checker()
        all_pairs = {}
        num_groups = len(self.groups_2)

        # Match _execute_intergroup: range(1, num_groups) → only round 1 when num_groups == 2
        for round_num in range(1, num_groups):
            round_pairs = checker._generate_intergroup_pairs(self.groups_2, round_num)
            all_pairs.update(round_pairs)

        connections = set()
        for _test_name, test_config in all_pairs.items():
            group1_nodes = test_config['group1']
            group2_nodes = test_config['group2']
            for node1 in group1_nodes:
                for node2 in group2_nodes:
                    connections.add((node1, node2))

        expected_connections = {
            ('A', 'C'),
            ('A', 'D'),
            ('B', 'C'),
            ('B', 'D'),
            ('C', 'A'),
            ('C', 'B'),
            ('D', 'A'),
            ('D', 'B'),
        }

        self.assertEqual(
            connections,
            expected_connections,
            'Round-robin algorithm should generate all inter-group directed pairs',
        )

    def test_round_generation_with_3_groups(self):
        checker = _make_checker()
        all_pairs = {}
        num_groups = len(self.groups_3)

        for round_num in range(2, num_groups + 1):
            round_pairs = checker._generate_intergroup_pairs(self.groups_3, round_num)
            all_pairs.update(round_pairs)

        self.assertTrue(all_pairs)


class TestExecuteIntragroup(unittest.TestCase):
    """RdmaConnectivityCheck._execute_intragroup delegates to coordination."""

    @patch.object(RdmaConnectivityCheck, '_execute_round_with_coordination')
    def test_execute_intragroup(self, mock_execute_round):
        mock_execute_round.return_value = {'test1': 'result1'}

        groups = {'group_1': ['A', 'B'], 'group_2': ['C', 'D']}
        mock_phdl = MagicMock()
        checker = _make_checker(phdl=mock_phdl, node_list=['A', 'B', 'C', 'D'], expected_interfaces=['if1', 'if2'])

        result = checker._execute_intragroup(groups, 5000)

        mock_execute_round.assert_called_once_with(groups, 'intra_group', 5000, work_segment='intra_group')
        self.assertEqual(result, {'test1': 'result1'})


class TestExecuteIntergroup(unittest.TestCase):
    """RdmaConnectivityCheck._execute_intergroup runs all ordered group pairs in one coordination round."""

    @patch.object(RdmaConnectivityCheck, '_execute_round_with_coordination')
    def test_execute_intergroup_with_2_groups(self, mock_execute_round):
        mock_execute_round.return_value = {'test1': 'result1'}

        groups = {'group_1': ['A', 'B'], 'group_2': ['C', 'D']}
        mock_phdl = MagicMock()
        checker = _make_checker(phdl=mock_phdl, node_list=['A', 'B', 'C', 'D'], expected_interfaces=['if1', 'if2'])

        checker._execute_intergroup(groups, 5000)

        self.assertEqual(mock_execute_round.call_count, 1)
        test_groups, round_type, port = mock_execute_round.call_args[0]
        self.assertEqual(mock_execute_round.call_args.kwargs.get('work_segment'), 'inter_group_legacy')
        self.assertEqual(round_type, 'inter_group')
        self.assertEqual(port, 5000)
        self.assertIn('group_1_to_group_2', test_groups)
        self.assertIn('group_2_to_group_1', test_groups)


class TestIntraPeerPrune(unittest.TestCase):
    """Round 1 prune: peer failure fraction ≥ threshold (default 50%)."""

    def test_prunes_nodes_at_half_threshold_three_node_group(self):
        """One failed pair A–B: A and B each have 1/2 peers failed → prune; C has 0/2 → keep."""
        groups = {'group_1': ['A', 'B', 'C']}
        intra_results = {
            't1': {'status': 'FAIL', 'server_node': 'A', 'client_node': 'B'},
        }
        checker = _make_checker(config_dict={'rdma_prune_peer_failure_threshold': 0.5})
        pruned_set, records, new_g = checker._apply_intra_prune(intra_results, groups)
        self.assertEqual(pruned_set, {'A', 'B'})
        self.assertEqual(new_g['group_1'], ['C'])
        self.assertEqual(len(records), 2)

    def test_no_prune_when_below_threshold(self):
        groups = {'group_1': ['A', 'B', 'C']}
        intra_results = {
            't1': {'status': 'FAIL', 'server_node': 'A', 'client_node': 'B'},
        }
        checker = _make_checker(config_dict={'rdma_prune_peer_failure_threshold': 0.6})
        pruned_set, _records, new_g = checker._apply_intra_prune(intra_results, groups)
        self.assertEqual(pruned_set, set())
        self.assertEqual(new_g['group_1'], ['A', 'B', 'C'])


class TestGenerateServerCommandsScriptletDebug(unittest.TestCase):
    """Server scriptlet lines: optional strace when scriptlet_debug is set."""

    def test_default_no_strace(self):
        checker = _make_checker(expected_interfaces=['if1'], config_dict={})
        groups = {'group_1': ['A', 'B']}
        cmds = checker._generate_server_commands('A', groups, 'intra_group', 5000, '/tmp/preflight')
        self.assertTrue(cmds)
        self.assertTrue(all('strace' not in c for c in cmds))

    def test_scriptlet_debug_adds_strace(self):
        checker = _make_checker(expected_interfaces=['if1'], config_dict={'scriptlet_debug': True})
        groups = {'group_1': ['A', 'B']}
        cmds = checker._generate_server_commands('A', groups, 'intra_group', 5000, '/tmp/preflight')
        self.assertTrue(cmds)
        for c in cmds:
            with self.subTest(cmd=c[:80]):
                self.assertIn('strace', c)
                self.assertIn('-e trace=bind,socket,setsockopt,listen,accept', c)
                self.assertIn('-o /tmp/preflight/strace_server_', c)

    def test_scriptlet_debug_accepts_string_true(self):
        checker = _make_checker(expected_interfaces=['if1'], config_dict={'scriptlet_debug': 'true'})
        groups = {'group_1': ['A', 'B']}
        cmds = checker._generate_server_commands('A', groups, 'intra_group', 5000, '/tmp/preflight')
        self.assertTrue(any('strace' in c for c in cmds))


class TestEdgeCases(unittest.TestCase):
    """Edge cases for port assignments."""

    def setUp(self):
        self.interfaces = ['if1', 'if2']
        self.port_start = 5000

    def test_single_node_group(self):
        checker = _make_checker(expected_interfaces=self.interfaces)
        test_groups = {'group_1': ['A']}

        assignments = checker._calculate_port_assignments(test_groups, 'intra_group', self.port_start)
        self.assertEqual(len(assignments), 0)

    def test_empty_interfaces(self):
        checker = _make_checker(expected_interfaces=[])
        test_groups = {'group_1': ['A', 'B']}

        assignments = checker._calculate_port_assignments(test_groups, 'intra_group', self.port_start)
        self.assertEqual(len(assignments), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
