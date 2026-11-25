"""Unit tests for the shared multi-rank TransferBench runner.

All fixtures are synthetic: generic node names and binary paths, no real
hostnames, usernames or home paths.
"""

import os
import shlex
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.transferbench import multirank


def _make_orch(phdl):
    orch = MagicMock()
    orch.all = phdl
    return orch


def _make_phdl(reachable_hosts, cmd_list_response=None):
    phdl = MagicMock()
    phdl.reachable_hosts = list(reachable_hosts)
    phdl.exec_cmd_list.return_value = cmd_list_response or {host: '' for host in reachable_hosts}
    return phdl


class TestValidateBinaryName(unittest.TestCase):
    def test_strips_surrounding_whitespace(self):
        self.assertEqual(multirank.validate_binary_name('tb_binary', '  TransferBench '), 'TransferBench')

    def test_rejects_empty_and_non_string(self):
        for bad in ['', '   ', None, 7, ['TransferBench']]:
            with self.assertRaises(ValueError):
                multirank.validate_binary_name('tb_binary', bad)


class TestValidateExtraEnv(unittest.TestCase):
    def test_none_means_absent(self):
        self.assertEqual(multirank.validate_extra_env(None), {})

    def test_rejects_non_mapping(self):
        for bad in [[], 'LD_LIBRARY_PATH=/x', 5]:
            with self.assertRaises(ValueError):
                multirank.validate_extra_env(bad)

    def test_rejects_key_that_could_change_command_syntax(self):
        for bad_key in ['LD;rm -rf /', 'A B', '1ABC', '', 'A-B']:
            with self.assertRaises(ValueError):
                multirank.validate_extra_env({bad_key: 'x'})

    def test_rejects_null_value_but_coerces_other_types(self):
        with self.assertRaises(ValueError):
            multirank.validate_extra_env({'LD_LIBRARY_PATH': None})
        self.assertEqual(multirank.validate_extra_env({'TB_DEBUG': 1}), {'TB_DEBUG': '1'})


class TestResolveSocketMasterPort(unittest.TestCase):
    def test_default_when_nothing_supplied(self):
        self.assertEqual(multirank.resolve_socket_master_port(environ={}), multirank.DEFAULT_SOCKET_MASTER_PORT)

    def test_configured_value_used_without_environment(self):
        self.assertEqual(multirank.resolve_socket_master_port(configured=41337, environ={}), 41337)

    def test_environment_overrides_configured_value(self):
        env = {multirank.SOCKET_MASTER_PORT_ENV: '45000'}
        self.assertEqual(multirank.resolve_socket_master_port(configured=41337, environ=env), 45000)

    def test_blank_environment_falls_back_to_configured(self):
        env = {multirank.SOCKET_MASTER_PORT_ENV: '  '}
        self.assertEqual(multirank.resolve_socket_master_port(configured=41337, environ=env), 41337)

    def test_rejects_non_integer_and_out_of_range(self):
        for bad in ['not-a-port', '0', '65536', '-1', [1234]]:
            with self.assertRaises(ValueError):
                multirank.resolve_socket_master_port(configured=bad, environ={})


class TestSudoExecutableArgument(unittest.TestCase):
    def test_bare_name_is_resolved_by_the_outer_shell(self):
        self.assertEqual(multirank.sudo_executable_argument('TransferBench'), '"$(command -v TransferBench)"')

    def test_explicit_path_is_quoted(self):
        self.assertEqual(multirank.sudo_executable_argument('/opt/tb/TransferBench'), '/opt/tb/TransferBench')

    def test_path_containing_a_space_stays_a_single_argument(self):
        rendered = multirank.sudo_executable_argument('/opt/my tools/TransferBench')
        self.assertEqual(shlex.split(rendered), ['/opt/my tools/TransferBench'])


class TestRankEnv(unittest.TestCase):
    def test_single_rank_gets_no_socket_environment(self):
        self.assertEqual(multirank.rank_env(0, 1, '127.0.0.1'), {})

    def test_multi_rank_carries_rank_and_master(self):
        env = multirank.rank_env(2, 4, '10.0.0.1', master_port=41337)
        self.assertEqual(
            env,
            {
                'TB_NUM_RANKS': '4',
                'TB_RANK': '2',
                'TB_MASTER_ADDR': '10.0.0.1',
                'TB_MASTER_PORT': '41337',
            },
        )


class TestRenderEnvAssignments(unittest.TestCase):
    def test_values_are_quoted_so_they_cannot_expand(self):
        rendered = multirank.render_env_assignments({'SIZE_LIST': '1K,16M', 'EVIL': '$(touch /tmp/pwned)'})
        self.assertIn("EVIL='$(touch /tmp/pwned)'", rendered)
        self.assertEqual(rendered[0], 'SIZE_LIST=1K,16M')


class TestBuildCommand(unittest.TestCase):
    def test_plain_command_wraps_payload_and_appends_sentinel(self):
        cmd = multirank.build_command('TransferBench', ['smoketest'], {'NUM_ITERATIONS': '2'})
        self.assertTrue(cmd.startswith('bash -c '))
        inner = shlex.split(cmd)[2]
        self.assertEqual(inner, f'NUM_ITERATIONS=2 TransferBench smoketest; echo "{multirank.EXIT_SENTINEL}=$?"')

    def test_environment_is_inside_the_privileged_shell(self):
        cmd = multirank.build_command('TransferBench', ['smoketest'], {'TB_RANK': '1'}, use_sudo=True)
        self.assertTrue(cmd.startswith('sudo bash -c '))
        # The env assignment must sit inside the quoted inner script, not before
        # sudo, or the privileged child never sees it.
        inner = shlex.split(cmd)[3]
        self.assertIn('TB_RANK=1', inner)
        self.assertIn('"$1" smoketest', inner)
        self.assertTrue(cmd.endswith('transferbench "$(command -v TransferBench)"'))

    def test_sudo_argv0_is_caller_supplied(self):
        cmd = multirank.build_command('TransferBench', ['x'], {}, use_sudo=True, sudo_argv0='topology-nic')
        self.assertIn(' topology-nic ', cmd)

    def test_binary_path_with_a_space_survives_quoting(self):
        cmd = multirank.build_command('/opt/my tools/TransferBench', ['smoketest'], {})
        inner = shlex.split(cmd)[2]
        self.assertEqual(shlex.split(inner)[0], '/opt/my tools/TransferBench')

    def test_runtime_prefix_precedes_the_invocation(self):
        cmd = multirank.build_command('TransferBench', ['p'], {}, runtime_env_prefix='export LD_LIBRARY_PATH=/x')
        inner = shlex.split(cmd)[2]
        self.assertTrue(inner.startswith('export LD_LIBRARY_PATH=/x; '))

    def test_empty_environment_leaves_no_leading_space(self):
        cmd = multirank.build_command('TransferBench', ['smoketest'], {})
        inner = shlex.split(cmd)[2]
        self.assertTrue(inner.startswith('TransferBench smoketest;'))

    def test_arguments_are_quoted_individually(self):
        cmd = multirank.build_command('TransferBench', ['/tmp/a b.txt', '--flag'], {})
        invocation = shlex.split(cmd)[2].split('; echo ')[0]
        self.assertEqual(shlex.split(invocation), ['TransferBench', '/tmp/a b.txt', '--flag'])


class TestExtractExitCode(unittest.TestCase):
    def test_reads_the_sentinel(self):
        self.assertEqual(multirank.extract_exit_code(f'output\n{multirank.EXIT_SENTINEL}=3\n'), 3)

    def test_missing_sentinel_is_none(self):
        self.assertIsNone(multirank.extract_exit_code('output with no sentinel'))

    def test_non_string_is_none(self):
        self.assertIsNone(multirank.extract_exit_code(None))

    def test_last_sentinel_wins(self):
        text = f'{multirank.EXIT_SENTINEL}=0\nmore\n{multirank.EXIT_SENTINEL}=7\n'
        self.assertEqual(multirank.extract_exit_code(text), 7)


class TestMasterSelection(unittest.TestCase):
    def test_preferred_master_used_when_reachable(self):
        self.assertEqual(multirank.resolve_master_node(['nodeB', 'nodeA'], preferred='nodeB'), 'nodeB')

    def test_unreachable_preference_falls_back_to_sorted_first(self):
        self.assertEqual(multirank.resolve_master_node(['nodeB', 'nodeA'], preferred='nodeZ'), 'nodeA')

    def test_no_hosts_yields_none(self):
        self.assertIsNone(multirank.resolve_master_node([]))

    def test_rank_order_puts_master_first(self):
        self.assertEqual(multirank.rank_order(['nodeC', 'nodeA', 'nodeB'], 'nodeC'), ['nodeC', 'nodeA', 'nodeB'])


class TestDispatchMultiRank(unittest.TestCase):
    def _command_for_rank(self, rank, num_ranks, master_addr):
        return f'run rank={rank} of={num_ranks} master={master_addr}'

    def test_commands_align_with_reachable_host_order_not_rank_order(self):
        # exec_cmd_list pairs commands to reachable_hosts positionally, so a
        # list built in rank order would hand each node another node's rank.
        phdl = _make_phdl(['nodeB', 'nodeA'])
        multirank.dispatch_multi_rank(_make_orch(phdl), self._command_for_rank)

        phdl.exec_cmd_list.assert_called_once()
        sent = phdl.exec_cmd_list.call_args[0][0]
        self.assertEqual(sent, ['run rank=1 of=2 master=nodeA', 'run rank=0 of=2 master=nodeA'])

    def test_records_rank_and_output_per_node(self):
        phdl = _make_phdl(['nodeB', 'nodeA'], cmd_list_response={'nodeA': 'out-a', 'nodeB': 'out-b'})
        per_node = multirank.dispatch_multi_rank(_make_orch(phdl), self._command_for_rank)

        self.assertEqual(per_node['nodeA']['rank'], 0)
        self.assertEqual(per_node['nodeB']['rank'], 1)
        self.assertEqual(per_node['nodeA']['output'], 'out-a')
        self.assertEqual(per_node['nodeB']['output'], 'out-b')

    def test_no_reachable_hosts_runs_nothing(self):
        phdl = _make_phdl([])
        self.assertEqual(multirank.dispatch_multi_rank(_make_orch(phdl), self._command_for_rank), {})
        phdl.exec_cmd_list.assert_not_called()

    def test_preferred_master_becomes_rank_zero(self):
        phdl = _make_phdl(['nodeA', 'nodeB'])
        per_node = multirank.dispatch_multi_rank(_make_orch(phdl), self._command_for_rank, preferred_master='nodeB')
        self.assertEqual(per_node['nodeB']['rank'], 0)
        self.assertIn('master=nodeB', per_node['nodeA']['command'])

    def test_output_for_an_unexpected_host_is_still_recorded(self):
        phdl = _make_phdl(['nodeA'], cmd_list_response={'nodeA': 'out-a', 'nodeZ': 'late'})
        per_node = multirank.dispatch_multi_rank(_make_orch(phdl), self._command_for_rank)
        self.assertEqual(per_node['nodeZ']['output'], 'late')


if __name__ == '__main__':
    unittest.main()
