"""Unit tests for staging a generated TransferBench input file.

All fixtures are synthetic: generic node names and temp paths.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.transferbench import staging


def _make_orch(reachable_hosts, exec_result=None, cmd_list_result=None):
    phdl = MagicMock()
    phdl.reachable_hosts = list(reachable_hosts)
    phdl.exec.return_value = exec_result or {}
    phdl.exec_cmd_list.return_value = cmd_list_result or {}
    orch = MagicMock()
    orch.all = phdl
    return orch


def _ok(path):
    return {'output': path, 'exit_code': 0}


class TestStageFile(unittest.TestCase):
    def test_rejects_prefix_that_could_alter_the_command(self):
        orch = _make_orch(['nodeA'])
        for bad in ['tb;rm -rf /', 'tb file', 'tb$(id)', '', 'tb`id`', None]:
            with self.assertRaises(ValueError):
                staging.stage_file(orch, 'content', bad)
        orch.all.exec.assert_not_called()

    def test_rejects_non_string_content(self):
        orch = _make_orch(['nodeA'])
        with self.assertRaises(ValueError):
            staging.stage_file(orch, ['line'], 'cvs_tb_')

    def test_creates_a_private_file_and_returns_its_path(self):
        orch = _make_orch(
            ['nodeA', 'nodeB'], exec_result={'nodeA': _ok('/tmp/cvs_tb_abc'), 'nodeB': _ok('/tmp/cvs_tb_def')}
        )
        paths, failures = staging.stage_file(orch, '1 4 (R0G0->R0I0.0->R1G0)\n', 'cvs_tb_')

        self.assertEqual(paths, {'nodeA': '/tmp/cvs_tb_abc', 'nodeB': '/tmp/cvs_tb_def'})
        self.assertEqual(failures, {})
        cmd = orch.all.exec.call_args[0][0]
        self.assertIn('umask 077', cmd)
        self.assertIn('mktemp -p "${TMPDIR:-/tmp}" cvs_tb_XXXXXXXXXX', cmd)

    def test_content_with_shell_metacharacters_is_quoted(self):
        orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok('/tmp/cvs_tb_abc')})
        staging.stage_file(orch, '$(touch /tmp/pwned)`id`\n', 'cvs_tb_')
        cmd = orch.all.exec.call_args[0][0]
        self.assertIn("'$(touch /tmp/pwned)`id`", cmd)

    def test_nonzero_exit_is_reported_not_raised(self):
        orch = _make_orch(
            ['nodeA', 'nodeB'], exec_result={'nodeA': _ok('/tmp/cvs_tb_abc'), 'nodeB': {'output': '', 'exit_code': 1}}
        )
        paths, failures = staging.stage_file(orch, 'x', 'cvs_tb_')

        self.assertEqual(list(paths), ['nodeA'])
        self.assertIn('nodeB', failures)

    def test_path_outside_the_expected_shape_is_rejected(self):
        bad_paths = ['relative/cvs_tb_abc', '/tmp/other_file', '/tmp/cvs_tb_a\nb', '']
        for bad in bad_paths:
            orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok(bad)})
            paths, failures = staging.stage_file(orch, 'x', 'cvs_tb_')
            self.assertEqual(paths, {}, f'accepted bad path {bad!r}')
            self.assertIn('nodeA', failures)

    def test_unreachable_shaped_result_is_a_failure(self):
        orch = _make_orch(['nodeA'], exec_result={'nodeA': '\nABORT: Host Unreachable Error'})
        paths, failures = staging.stage_file(orch, 'x', 'cvs_tb_')
        self.assertEqual(paths, {})
        self.assertIn('nodeA', failures)


class TestRemoveFiles(unittest.TestCase):
    def test_commands_align_with_reachable_hosts_and_skip_unstaged(self):
        orch = _make_orch(
            ['nodeB', 'nodeA'],
            cmd_list_result={'nodeA': f'{staging.REMOVE_SENTINEL}=0', 'nodeB': f'{staging.REMOVE_SENTINEL}=0'},
        )
        staging.remove_files(orch, {'nodeA': '/tmp/cvs_tb_abc'})

        sent = orch.all.exec_cmd_list.call_args[0][0]
        self.assertEqual(sent[0], 'true')
        self.assertIn('rm -f -- /tmp/cvs_tb_abc', sent[1])

    def test_confirmed_removal_leaves_nothing_outstanding(self):
        orch = _make_orch(['nodeA'], cmd_list_result={'nodeA': f'{staging.REMOVE_SENTINEL}=0'})
        self.assertEqual(staging.remove_files(orch, {'nodeA': '/tmp/cvs_tb_abc'}), [])

    def test_failed_removal_is_reported(self):
        orch = _make_orch(['nodeA'], cmd_list_result={'nodeA': f'{staging.REMOVE_SENTINEL}=1'})
        self.assertEqual(staging.remove_files(orch, {'nodeA': '/tmp/cvs_tb_abc'}), ['nodeA'])

    def test_missing_output_is_reported(self):
        orch = _make_orch(['nodeA'], cmd_list_result={})
        self.assertEqual(staging.remove_files(orch, {'nodeA': '/tmp/cvs_tb_abc'}), ['nodeA'])

    def test_path_with_a_space_is_quoted(self):
        orch = _make_orch(['nodeA'], cmd_list_result={'nodeA': f'{staging.REMOVE_SENTINEL}=0'})
        staging.remove_files(orch, {'nodeA': '/tmp/dir with space/cvs_tb_abc'})
        sent = orch.all.exec_cmd_list.call_args[0][0]
        self.assertIn("rm -f -- '/tmp/dir with space/cvs_tb_abc'", sent[0])

    def test_nothing_staged_runs_nothing(self):
        orch = _make_orch(['nodeA'])
        self.assertEqual(staging.remove_files(orch, {}), [])
        orch.all.exec_cmd_list.assert_not_called()


if __name__ == '__main__':
    unittest.main()
