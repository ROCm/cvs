"""Unit tests for reading TransferBench's own NIC table.

The two tables below are real output from TransferBench v1.71.01 on two
different NIC families. They are kept verbatim because the whole point of this
module is that the tool's ordering cannot be predicted.
"""

import os
import re
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.transferbench import multirank, nic_table

# Pensando/ionic adapter: one active management NIC, backend NICs down.
IONIC_TABLE = """\
 NIC | Device Name | Active | PCIe Bus ID  | NUMA | Closest GPU(s) | MaxMsgSz   | GID Index | GID Descriptor
-----+-------------+--------+--------------+------+----------------+------------+-----------+---------------
 0   | ionic_0     | Yes    | 0000:04:00.0 | 0    | 0,1,2,3        | 2147483648 | 1         | RoCEv2 IPv4-mapped IPv6
 1   | ionic_1     | No     | 0001:44:00.0 | 1    |                | -          | N/A       | N/A
 2   | ionic_2     | No     | 0001:48:00.0 | 1    |                | -          | N/A       | N/A
 3   | ionic_3     | No     | 0002:44:00.0 | 1    |                | -          | N/A       | N/A
"""

# Broadcom adapter: every NIC active, and the tool's index order is NOT the
# device-name order.
BROADCOM_TABLE = """\
 NIC | Device Name | Active | PCIe Bus ID  | NUMA | Closest GPU(s) | MaxMsgSz   | GID Index | GID Descriptor
-----+-------------+--------+--------------+------+----------------+------------+-----------+---------------
 0   | rdma3       | Yes    | 0000:76:00.0 | 0    |                | 1073741824 | 3         | RoCEv2 IPv4-mapped IPv6
 1   | rdma0       | Yes    | 0000:06:00.0 | 0    |                | 1073741824 | 3         | RoCEv2 IPv4-mapped IPv6
 2   | rdma2       | Yes    | 0000:66:00.0 | 0    |                | 1073741824 | 3         | RoCEv2 IPv4-mapped IPv6
 3   | rdma1       | Yes    | 0000:16:00.0 | 0    |                | 1073741824 | 3         | RoCEv2 IPv4-mapped IPv6
"""

# What the same tool prints once TB_NIC_FILTER restricts it to a subset.
FILTERED_TABLE = """\
 NIC | Device Name | Active | PCIe Bus ID  | NUMA | Closest GPU(s) | MaxMsgSz   | GID Index | GID Descriptor
-----+-------------+--------+--------------+------+----------------+------------+-----------+---------------
 0   | ionic_4     | No     | 0003:44:00.0 | 0    |                | -          | N/A       | N/A
 1   | ionic_6     | No     | 0004:44:00.0 | 0    |                | -          | N/A       | N/A
 2   | ionic_7     | No     | 0004:48:00.0 | 0    |                | -          | N/A       | N/A
"""


class TestParseNicTable(unittest.TestCase):
    def test_reads_every_field_of_a_real_table(self):
        rows = nic_table.parse_nic_table(IONIC_TABLE)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            rows[0],
            {
                'index': 0,
                'nic': 'ionic_0',
                'active': True,
                'bdf': '0000:04:00.0',
                'numa': '0',
                'closest_gpus': [0, 1, 2, 3],
            },
        )
        self.assertFalse(rows[1]['active'])
        self.assertEqual(rows[1]['closest_gpus'], [], 'an inactive NIC lists no GPUs')

    def test_tool_index_order_is_not_device_name_order(self):
        # The defect this module exists to prevent: assuming sorted device
        # names map to the tool's indices mis-addresses most NICs here.
        mapping = nic_table.index_by_name(nic_table.parse_nic_table(BROADCOM_TABLE))
        self.assertEqual(mapping, {'rdma3': 0, 'rdma0': 1, 'rdma2': 2, 'rdma1': 3})

        naive = {name: i for i, name in enumerate(sorted(mapping))}
        self.assertNotEqual(naive, mapping, 'fixture must exercise a scrambled order')

    def test_filtered_table_is_renumbered_from_zero(self):
        mapping = nic_table.index_by_name(nic_table.parse_nic_table(FILTERED_TABLE))
        self.assertEqual(mapping, {'ionic_4': 0, 'ionic_6': 1, 'ionic_7': 2})

    def test_banner_and_preamble_are_ignored(self):
        noisy = '*** WELCOME ***\nDetected Topology:\n  4 GPU device(s)\n\n' + IONIC_TABLE
        self.assertEqual(len(nic_table.parse_nic_table(noisy)), 4)

    def test_output_after_the_table_does_not_become_rows(self):
        trailing = IONIC_TABLE + '\nSome closing summary line\n 99 not a row\n'
        rows = nic_table.parse_nic_table(trailing)
        self.assertEqual([row['index'] for row in rows], [0, 1, 2, 3])

    def test_output_without_a_table_yields_nothing(self):
        self.assertEqual(nic_table.parse_nic_table('no table here'), [])
        self.assertEqual(nic_table.parse_nic_table(''), [])

    def test_non_string_output_yields_nothing(self):
        self.assertEqual(nic_table.parse_nic_table(None), [])


class TestBuildNicFilter(unittest.TestCase):
    def test_matches_exactly_the_requested_devices(self):
        pattern = re.compile(nic_table.build_nic_filter(['ionic_4', 'ionic_6']))
        self.assertTrue(pattern.search('ionic_4'))
        self.assertTrue(pattern.search('ionic_6'))
        self.assertFalse(pattern.search('ionic_5'))

    def test_anchors_prevent_a_prefix_collision(self):
        # Without anchors, ionic_1 would also select ionic_10 on a node with
        # ten or more NICs.
        pattern = re.compile(nic_table.build_nic_filter(['ionic_1']))
        self.assertTrue(pattern.search('ionic_1'))
        self.assertFalse(pattern.search('ionic_10'))

    def test_regex_metacharacters_in_a_name_are_escaped(self):
        pattern = re.compile(nic_table.build_nic_filter(['mlx5.0']))
        self.assertTrue(pattern.search('mlx5.0'))
        self.assertFalse(pattern.search('mlx5x0'))

    def test_matches_the_filter_that_was_verified_on_hardware(self):
        self.assertEqual(
            nic_table.build_nic_filter(['ionic_4', 'ionic_6', 'ionic_7']),
            '^(ionic_4|ionic_6|ionic_7)$',
        )

    def test_empty_selection_is_an_error_not_a_match_everything_filter(self):
        # An empty alternation would match every NIC, silently widening the run.
        for empty in [[], ['', '  ']]:
            with self.assertRaises(ValueError):
                nic_table.build_nic_filter(empty)


def _with_sentinel(output, exit_code):
    return f'{output}\n{multirank.EXIT_SENTINEL}={exit_code}\n'


def _make_orch(exec_result):
    phdl = MagicMock()
    phdl.exec.return_value = exec_result
    orch = MagicMock()
    orch.all = phdl
    return orch


class TestQueryNicTable(unittest.TestCase):
    def test_successful_query_parses_the_table(self):
        orch = _make_orch({'nodeA': _with_sentinel(IONIC_TABLE, 0)})
        nics_by_node, failures = nic_table.query_nic_table(orch, 'TransferBench')

        self.assertEqual(len(nics_by_node['nodeA']), 4)
        self.assertEqual(failures, {})

    def test_nic_filter_is_passed_through_to_the_command(self):
        orch = _make_orch({'nodeA': _with_sentinel(IONIC_TABLE, 0)})
        nic_table.query_nic_table(orch, 'TransferBench', nic_filter='^(ionic_0)$')
        cmd = orch.all.exec.call_args[0][0]
        inner = re.split(r'bash -c ', cmd, maxsplit=1)[1]
        self.assertIn("TB_NIC_FILTER=", inner)
        self.assertIn('^(ionic_0)$', inner)

    def test_unreachable_node_is_distinguished_from_an_empty_table(self):
        orch = _make_orch({'nodeA': '\nABORT: Host Unreachable Error'})
        _, failures = nic_table.query_nic_table(orch, 'TransferBench')
        self.assertEqual(failures['nodeA'], 'host unreachable')

    def test_nonzero_exit_is_distinguished_from_an_empty_table(self):
        # Reaching the sentinel with a nonzero exit means the binary ran and
        # failed (bad filter, binary not found); that is not "found nothing".
        orch = _make_orch({'nodeA': _with_sentinel('command not found', 127)})
        _, failures = nic_table.query_nic_table(orch, 'TransferBench')
        self.assertEqual(failures['nodeA'], 'TransferBench exited 127')

    def test_missing_sentinel_is_distinguished_from_an_empty_table(self):
        # No sentinel means the command never completed (e.g. timed out),
        # which is a different problem than a tool that ran cleanly and found
        # nothing.
        orch = _make_orch({'nodeA': 'partial output, connection dropped'})
        _, failures = nic_table.query_nic_table(orch, 'TransferBench')
        self.assertEqual(failures['nodeA'], 'TransferBench did not complete (no exit status recovered)')

    def test_zero_exit_with_no_rows_is_reported_as_an_empty_table(self):
        orch = _make_orch({'nodeA': _with_sentinel('no NICs on this node', 0)})
        _, failures = nic_table.query_nic_table(orch, 'TransferBench')
        self.assertEqual(failures['nodeA'], 'TransferBench reported no NIC table')

    def test_partial_table_without_a_sentinel_is_not_accepted_as_the_full_table(self):
        # The node's real table has 4 rows; the connection dropped after only
        # 1 was printed, so the sentinel never ran. Accepting this row would
        # silently understate how many NICs the node actually has.
        partial = IONIC_TABLE.splitlines()[:3]  # header + separator + one row, no sentinel
        orch = _make_orch({'nodeA': '\n'.join(partial)})
        nics_by_node, failures = nic_table.query_nic_table(orch, 'TransferBench')

        self.assertNotIn('nodeA', nics_by_node, 'a truncated table must not be reported as complete')
        self.assertIn('nodeA', failures)

    def test_extra_env_key_cannot_inject_through_the_runtime_prefix(self):
        # build_env_prefix quotes values but interpolates keys as shell
        # syntax; query_nic_table must validate before handing it a raw dict.
        orch = _make_orch({'nodeA': _with_sentinel(IONIC_TABLE, 0)})
        with self.assertRaises(ValueError):
            nic_table.query_nic_table(orch, 'TransferBench', extra_env={'X; touch /tmp/pwned; #': '1'})
        orch.all.exec.assert_not_called()

    def test_well_formed_extra_env_still_reaches_the_command(self):
        orch = _make_orch({'nodeA': _with_sentinel(IONIC_TABLE, 0)})
        nic_table.query_nic_table(orch, 'TransferBench', extra_env={'LD_LIBRARY_PATH': '/opt/rocm/lib'})
        cmd = orch.all.exec.call_args[0][0]
        self.assertIn('LD_LIBRARY_PATH', cmd)


if __name__ == '__main__':
    unittest.main()
