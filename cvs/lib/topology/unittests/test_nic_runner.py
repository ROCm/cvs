"""Unit tests for the NIC bandwidth TransferBench run."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.topology import nic_runner
from cvs.lib.transferbench import staging

INVENTORY = {
    'nodeA': [{'nic': 'nic0', 'gpu': 0}],
    'nodeB': [{'nic': 'nic0', 'gpu': 0}],
}


class _Phdl:
    """Minimal stand-in: staging uses exec, dispatch and removal use exec_cmd_list."""

    def __init__(self, hosts, stage_ok=True, run_output=None):
        self.reachable_hosts = list(hosts)
        self.stage_ok = stage_ok
        self.run_output = run_output or {}
        self.cmd_lists = []
        self.removed = False

    def exec(self, cmd, **kwargs):
        if not self.stage_ok:
            return {host: {'output': '', 'exit_code': 1} for host in self.reachable_hosts}
        return {
            host: {'output': f'/tmp/{nic_runner.STAGED_FILE_PREFIX}{host}', 'exit_code': 0}
            for host in self.reachable_hosts
        }

    def exec_cmd_list(self, cmd_list, **kwargs):
        self.cmd_lists.append(list(cmd_list))
        if any('rm -f --' in cmd for cmd in cmd_list):
            self.removed = True
            return {host: f'{staging.REMOVE_SENTINEL}=0' for host in self.reachable_hosts}
        return dict(self.run_output)


def _orch(phdl):
    orch = MagicMock()
    orch.all = phdl
    return orch


def _results(count):
    # Transfer-file order for a 2-node, 1-NIC-each cluster.
    return 'Executor: Rank 0 NIC 0 | 94.36 GB/s\nExecutor: Rank 1 NIC 0 | 93.69 GB/s\n' if count == 2 else ''


class TestBuildSettings(unittest.TestCase):
    def test_defaults_apply_when_nothing_is_configured(self):
        settings = nic_runner.build_settings({})
        self.assertEqual(settings['message_size'], nic_runner.DEFAULT_MESSAGE_SIZE)
        self.assertEqual(settings['queue_pairs'], nic_runner.DEFAULT_QUEUE_PAIRS)
        self.assertEqual(settings['iterations'], nic_runner.DEFAULT_ITERATIONS)
        self.assertEqual(settings['warmup_iterations'], nic_runner.DEFAULT_WARMUP_ITERATIONS)
        self.assertEqual(settings['traffic_class'], 0)

    def test_advanced_block_overrides_defaults(self):
        settings = nic_runner.build_settings({'advanced': {'message_size': '16M', 'queue_pairs': 8}})
        self.assertEqual(settings['message_size'], '16M')
        self.assertEqual(settings['queue_pairs'], 8)

    def test_traffic_class_auto_reads_the_cluster_file(self):
        settings = nic_runner.build_settings({'traffic_class': 'auto'}, {'env_vars': {'NCCL_IB_TC': '128'}})
        self.assertEqual(settings['traffic_class'], 128)

    def test_bad_configuration_is_rejected_at_load(self):
        with self.assertRaises(ValueError):
            nic_runner.build_settings({'traffic_class': 999})
        with self.assertRaises(ValueError):
            nic_runner.build_settings({'tb_binary': ''})
        with self.assertRaises(ValueError):
            nic_runner.build_settings({'transferbench_env': {'BAD KEY': '1'}})

    def test_non_dict_advanced_block_does_not_crash(self):
        self.assertEqual(nic_runner.build_settings({'advanced': 'nope'})['queue_pairs'], nic_runner.DEFAULT_QUEUE_PAIRS)


class TestBuildRunEnv(unittest.TestCase):
    def test_carries_traffic_class_and_message_size(self):
        env = nic_runner.build_run_env(nic_runner.build_settings({'traffic_class': 96}))
        self.assertEqual(env['NIC_TRAFFIC_CLASS'], '96')
        self.assertEqual(env['SIZE_LIST'], nic_runner.DEFAULT_MESSAGE_SIZE)

    def test_nic_filter_is_omitted_unless_configured(self):
        self.assertNotIn('TB_NIC_FILTER', nic_runner.build_run_env(nic_runner.build_settings({})))
        env = nic_runner.build_run_env(nic_runner.build_settings({'nic_filter': 'ionic'}))
        self.assertEqual(env['TB_NIC_FILTER'], 'ionic')


class TestRunMatrix(unittest.TestCase):
    def _settings(self, **overrides):
        config = {'coverage': 'full'}
        config.update(overrides)
        return nic_runner.build_settings(config)

    def test_measures_every_pair_and_cleans_up(self):
        phdl = _Phdl(['nodeA', 'nodeB'], run_output={'nodeA': _results(2), 'nodeB': ''})
        matrix, pairs, problems = nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings())

        self.assertEqual(len(pairs), 2)
        self.assertEqual(matrix[('nodeA', 'nic0')][('nodeB', 'nic0')], 94.36)
        self.assertEqual(problems, [])
        self.assertTrue(phdl.removed, 'staged transfer files must be removed')

    def test_each_rank_reads_its_own_staged_copy(self):
        phdl = _Phdl(['nodeB', 'nodeA'], run_output={'nodeA': _results(2), 'nodeB': ''})
        nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings())

        run_cmds = phdl.cmd_lists[0]
        # exec_cmd_list is aligned to reachable_hosts, so nodeB's command comes first.
        self.assertIn(f'/tmp/{nic_runner.STAGED_FILE_PREFIX}nodeB', run_cmds[0])
        self.assertIn(f'/tmp/{nic_runner.STAGED_FILE_PREFIX}nodeA', run_cmds[1])

    def test_traffic_class_reaches_the_command(self):
        phdl = _Phdl(['nodeA', 'nodeB'], run_output={'nodeA': _results(2), 'nodeB': ''})
        nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings(traffic_class=128))
        self.assertIn('NIC_TRAFFIC_CLASS=128', phdl.cmd_lists[0][0])

    def test_staging_failure_is_reported_without_running(self):
        phdl = _Phdl(['nodeA', 'nodeB'], stage_ok=False)
        matrix, _, problems = nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings())

        self.assertEqual(matrix, {})
        self.assertEqual(len(problems), 2)
        self.assertEqual(phdl.cmd_lists, [], 'nothing should run without a transfer file')

    def test_nothing_to_measure_on_a_single_node(self):
        phdl = _Phdl(['nodeA'])
        matrix, pairs, problems = nic_runner.run_matrix(_orch(phdl), {'nodeA': INVENTORY['nodeA']}, self._settings())
        self.assertEqual((matrix, pairs, problems), ({}, [], []))

    def test_staged_files_are_removed_even_when_parsing_fails(self):
        phdl = _Phdl(['nodeA', 'nodeB'], run_output={'nodeA': 'no results at all', 'nodeB': ''})
        with self.assertRaises(ValueError):
            nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings())
        self.assertTrue(phdl.removed)


if __name__ == '__main__':
    unittest.main()
