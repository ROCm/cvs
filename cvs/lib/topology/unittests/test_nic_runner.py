"""Unit tests for the NIC bandwidth TransferBench run."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.topology import nic_runner
from cvs.lib.transferbench import multirank, staging

INVENTORY = {
    'nodeA': [{'nic': 'nic0', 'index': 0, 'gpu': 0}],
    'nodeB': [{'nic': 'nic0', 'index': 0, 'gpu': 0}],
}


class _Phdl:
    """Minimal stand-in matching the real routing: staging's write/remove go
    through the orchestrator's public exec methods; the TransferBench run
    dispatch still goes through orch.all.exec_cmd_list (preserved as-is)."""

    def __init__(self, hosts, stage_ok=True, run_output=None):
        self.reachable_hosts = list(hosts)
        self.stage_ok = stage_ok
        self.run_output = run_output or {}
        self.cmd_lists = []
        self.removed = False

    def stage_exec(self, cmd, **kwargs):
        if not self.stage_ok:
            return {host: {'output': '', 'exit_code': 1} for host in self.reachable_hosts}
        return {
            host: {'output': f'{staging.PATH_MARKER}|/tmp/{nic_runner.STAGED_FILE_PREFIX}{host}', 'exit_code': 0}
            for host in self.reachable_hosts
        }

    def run_exec_cmd_list(self, cmd_list, **kwargs):
        self.cmd_lists.append(list(cmd_list))
        return dict(self.run_output)

    def remove_exec_cmd_list(self, cmd_list, **kwargs):
        self.removed = True
        return {host: f'{staging.REMOVE_SENTINEL}=0' for host in self.reachable_hosts}


def _orch(phdl):
    orch = MagicMock()
    orch.exec.side_effect = phdl.stage_exec
    orch.exec_cmd_list.side_effect = phdl.remove_exec_cmd_list
    orch.all.reachable_hosts = phdl.reachable_hosts
    orch.all.exec_cmd_list.side_effect = phdl.run_exec_cmd_list
    return orch


def _results(count, exit_code=0):
    # Transfer-file order for a 2-node, 1-NIC-each cluster. Rank 0's own
    # output carries the sentinel run_matrix now requires before trusting it.
    if count != 2:
        return ''
    body = 'Executor: Rank 0 NIC 0 | 94.36 GB/s\nExecutor: Rank 1 NIC 0 | 93.69 GB/s\n'
    return f'{body}{multirank.EXIT_SENTINEL}={exit_code}\n'


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
    def test_carries_traffic_class(self):
        env = nic_runner.build_run_env(nic_runner.build_settings({'traffic_class': 96}))
        self.assertEqual(env['NIC_TRAFFIC_CLASS'], '96')

    def test_message_size_is_not_an_environment_variable(self):
        # TransferBench has no environment variable for the byte count; setting
        # one silently leaves every transfer at the built-in default size.
        env = nic_runner.build_run_env(nic_runner.build_settings({'advanced': {'message_size': '64M'}}))
        self.assertNotIn('SIZE_LIST', env)
        for key, value in env.items():
            self.assertNotEqual(value, '64M', f'message size must not be passed via {key}')

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

    def test_message_size_is_passed_as_the_transferbench_size_argument(self):
        phdl = _Phdl(['nodeA', 'nodeB'], run_output={'nodeA': _results(2), 'nodeB': ''})
        nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings(advanced={'message_size': '64M'}))
        command = phdl.cmd_lists[0][0]
        self.assertRegex(command, r'cvs_topo_nic_\w+ 64M', 'size must follow the transfer file as argv')

    def test_ranks_stay_aligned_when_a_reachable_node_has_no_backend_nics(self):
        # nodeB is reachable but has nothing to measure. If the transfer file
        # numbered ranks over the inventory while the launcher numbered them
        # over reachable hosts, 'R1' would mean nodeC in the file and nodeB on
        # the wire, and every measurement would name the wrong host.
        phdl = _Phdl(['nodeA', 'nodeB', 'nodeC'], run_output={'nodeA': _results(2), 'nodeC': ''})
        inventory = {'nodeA': INVENTORY['nodeA'], 'nodeC': INVENTORY['nodeB']}
        _, pairs, _ = nic_runner.run_matrix(_orch(phdl), inventory, self._settings())

        rank_of = {p['src_node']: p['src_rank'] for p in pairs}
        self.assertEqual(rank_of, {'nodeA': 0, 'nodeC': 1})
        run_cmds = phdl.cmd_lists[0]
        # Commands are positional against reachable_hosts: A, B, C.
        self.assertIn('TB_RANK=0', run_cmds[0])
        self.assertEqual(run_cmds[1], 'true', 'a node with no NICs must not be given a rank')
        self.assertIn('TB_RANK=1', run_cmds[2])
        self.assertIn('TB_NUM_RANKS=2', run_cmds[2])

    def test_nothing_to_measure_on_a_single_node(self):
        phdl = _Phdl(['nodeA'])
        matrix, pairs, problems = nic_runner.run_matrix(_orch(phdl), {'nodeA': INVENTORY['nodeA']}, self._settings())
        self.assertEqual((matrix, pairs, problems), ({}, [], []))

    def test_staged_files_are_removed_even_when_parsing_fails(self):
        # Clean exit recovered (so the sentinel check passes), but TransferBench's
        # own output does not match the transfer file: a genuine tool-level
        # anomaly, distinct from a run that never completed.
        bad_output = f'no results at all\n{multirank.EXIT_SENTINEL}=0\n'
        phdl = _Phdl(['nodeA', 'nodeB'], run_output={'nodeA': bad_output, 'nodeB': ''})
        with self.assertRaises(ValueError):
            nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings())
        self.assertTrue(phdl.removed)

    def test_missing_exit_sentinel_is_reported_as_a_problem_not_treated_as_success(self):
        # dispatch_multi_rank's own contract: non-empty output is not proof the
        # run finished. No sentinel at all must not reach parse_results.
        phdl = _Phdl(['nodeA', 'nodeB'], run_output={'nodeA': 'Executor: Rank 0 NIC 0 | 94.36 GB/s\n', 'nodeB': ''})
        matrix, _, problems = nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings())

        self.assertEqual(matrix, {})
        self.assertIn('no exit status recovered', problems[0])
        self.assertTrue(phdl.removed, 'staged files must still be cleaned up')

    def test_nonzero_exit_is_reported_as_a_problem(self):
        phdl = _Phdl(['nodeA', 'nodeB'], run_output={'nodeA': _results(2, exit_code=1), 'nodeB': ''})
        matrix, _, problems = nic_runner.run_matrix(_orch(phdl), INVENTORY, self._settings())

        self.assertEqual(matrix, {})
        self.assertIn('exited 1', problems[0])
        self.assertTrue(phdl.removed, 'staged files must still be cleaned up')


if __name__ == '__main__':
    unittest.main()


class TestValidateMessageSize(unittest.TestCase):
    def test_plain_and_suffixed_sizes_are_accepted(self):
        for good in ('64M', '1024', '2G', '512k'):
            self.assertEqual(nic_runner.validate_message_size(good), good.strip())

    def test_non_ascii_digits_are_refused(self):
        # \d matches Unicode digits, which TransferBench cannot parse; the
        # failure belongs at config load, not mid-run on every node.
        with self.assertRaises(ValueError):
            nic_runner.validate_message_size('٥M')

    def test_shell_metacharacters_and_junk_are_refused(self):
        for bad in ('64M; rm -rf /', '64 M', '', 'M', '64X', '64M\nrm -rf /'):
            with self.assertRaises(ValueError):
                nic_runner.validate_message_size(bad)
