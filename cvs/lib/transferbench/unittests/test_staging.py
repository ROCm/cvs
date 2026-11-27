"""Unit tests for staging a generated TransferBench input file.

All fixtures are synthetic: generic node names and temp paths.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.transferbench import staging
from cvs.unittests.support import run_trusted


def _make_orch(reachable_hosts, exec_result=None, cmd_list_result=None):
    """orch.exec/orch.exec_cmd_list are what staging must call; orch.all is the
    legacy phdl path, kept distinct so a test can assert staging never touches it."""
    orch = MagicMock()
    orch.exec.return_value = exec_result or {}
    orch.exec_cmd_list.return_value = cmd_list_result or {}
    orch.all = MagicMock()
    orch.all.reachable_hosts = list(reachable_hosts)
    return orch


def _ok(path, banner=False):
    """A successful stage echoes the path behind a marker, after any banner."""
    prefix = '*** WELCOME ***\nLast login: never\n' if banner else ''
    return {'output': f'{prefix}{staging.PATH_MARKER}|{path}', 'exit_code': 0}


class TestStageFile(unittest.TestCase):
    def test_rejects_prefix_that_could_alter_the_command(self):
        orch = _make_orch(['nodeA'])
        for bad in ['tb;rm -rf /', 'tb file', 'tb$(id)', '', 'tb`id`', None]:
            with self.assertRaises(ValueError):
                staging.stage_file(orch, 'content', bad)
        orch.exec.assert_not_called()

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
        cmd = orch.exec.call_args[0][0]
        self.assertIn('umask 077', cmd)
        self.assertIn('mktemp -p "${TMPDIR:-/tmp}" cvs_tb_XXXXXXXXXX', cmd)

    def test_runs_through_the_orchestrators_public_exec_not_the_legacy_handle(self):
        # orch.all.exec bypasses the container under the container orchestrator
        # (it is the bare-host phdl, inherited unmodified by ContainerOrchestrator);
        # staging must go through orch.exec so it runs wherever TransferBench runs.
        orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok('/tmp/cvs_tb_abc')})
        staging.stage_file(orch, 'x', 'cvs_tb_')
        orch.exec.assert_called_once()
        orch.all.exec.assert_not_called()

    def test_content_with_shell_metacharacters_is_quoted(self):
        orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok('/tmp/cvs_tb_abc')})
        staging.stage_file(orch, '$(touch /tmp/pwned)`id`\n', 'cvs_tb_')
        cmd = orch.exec.call_args[0][0]
        self.assertIn("'$(touch /tmp/pwned)`id`", cmd)

    def test_content_with_shell_metacharacters_cannot_execute(self):
        # A string match on the rendered command proves it looks quoted, not
        # that it is safe to run; actually running it is the real proof.
        workdir = tempfile.mkdtemp()
        try:
            witness = os.path.join(workdir, 'injected')
            hostile = f'$(touch {witness})`touch {witness}`\n'
            orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok(os.path.join(workdir, 'cvs_tb_abc'))})
            staging.stage_file(orch, hostile, 'cvs_tb_')
            cmd = orch.exec.call_args[0][0]

            run_trusted(['bash', '-c', cmd], capture_output=True, text=True, env=dict(os.environ, TMPDIR=workdir))
            self.assertFalse(os.path.exists(witness), f'content executed via: {cmd}')
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_write_failure_after_mktemp_leaves_no_orphaned_file(self):
        # mktemp can succeed and the write still fail (e.g. a quota); without
        # the trap, that leaves an untracked file the caller never learns of
        # and can never clean up.
        workdir = tempfile.mkdtemp()
        try:
            orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok(os.path.join(workdir, 'cvs_tb_abc'))})
            staging.stage_file(orch, 'x' * 1000, 'cvs_tb_')
            cmd = orch.exec.call_args[0][0]

            # ulimit -f 0 forces the printf write to fail after mktemp has
            # already created the file, the same failure a real quota causes.
            result = run_trusted(
                ['bash', '-c', f'ulimit -f 0; {cmd}'],
                capture_output=True,
                text=True,
                env=dict(os.environ, TMPDIR=workdir),
            )
            self.assertNotEqual(result.returncode, 0)
            orphans = [name for name in os.listdir(workdir) if name.startswith('cvs_tb_')]
            self.assertEqual(orphans, [], f'orphaned file(s) left behind: {orphans}')
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_nonzero_exit_is_reported_not_raised(self):
        orch = _make_orch(
            ['nodeA', 'nodeB'], exec_result={'nodeA': _ok('/tmp/cvs_tb_abc'), 'nodeB': {'output': '', 'exit_code': 1}}
        )
        paths, failures = staging.stage_file(orch, 'x', 'cvs_tb_')

        self.assertEqual(list(paths), ['nodeA'])
        self.assertIn('nodeB', failures)

    def test_path_outside_the_expected_shape_is_rejected(self):
        # A newline is not in this list because the marked record is read a
        # line at a time, so an embedded newline cannot reach the validator.
        bad_paths = ['relative/cvs_tb_abc', '/tmp/other_file', '/tmp/wrong_prefix_abc', '']
        for bad in bad_paths:
            orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok(bad)})
            paths, failures = staging.stage_file(orch, 'x', 'cvs_tb_')
            self.assertEqual(paths, {}, f'accepted bad path {bad!r}')
            self.assertIn('nodeA', failures)

    def test_login_banner_before_the_path_is_ignored(self):
        orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok('/tmp/cvs_tb_abc', banner=True)})
        paths, failures = staging.stage_file(orch, 'x', 'cvs_tb_')

        self.assertEqual(paths, {'nodeA': '/tmp/cvs_tb_abc'})
        self.assertEqual(failures, {})

    def test_trailing_output_after_the_path_is_ignored(self):
        noisy = {'output': f'{staging.PATH_MARKER}|/tmp/cvs_tb_abc\nConnection to node closed.', 'exit_code': 0}
        orch = _make_orch(['nodeA'], exec_result={'nodeA': noisy})
        self.assertEqual(staging.stage_file(orch, 'x', 'cvs_tb_')[0], {'nodeA': '/tmp/cvs_tb_abc'})

    def test_output_without_the_marker_is_a_failure(self):
        orch = _make_orch(['nodeA'], exec_result={'nodeA': {'output': '/tmp/cvs_tb_abc', 'exit_code': 0}})
        paths, failures = staging.stage_file(orch, 'x', 'cvs_tb_')
        self.assertEqual(paths, {})
        self.assertIn('nodeA', failures)

    def test_command_marks_the_path_it_echoes(self):
        orch = _make_orch(['nodeA'], exec_result={'nodeA': _ok('/tmp/cvs_tb_abc')})
        staging.stage_file(orch, 'x', 'cvs_tb_')
        self.assertIn(staging.PATH_MARKER, orch.exec.call_args[0][0])

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

        sent = orch.exec_cmd_list.call_args[0][0]
        self.assertEqual(sent[0], 'true')
        self.assertIn('rm -f -- /tmp/cvs_tb_abc', sent[1])

    def test_runs_through_the_orchestrators_public_exec_cmd_list_not_the_legacy_handle(self):
        orch = _make_orch(['nodeA'], cmd_list_result={'nodeA': f'{staging.REMOVE_SENTINEL}=0'})
        staging.remove_files(orch, {'nodeA': '/tmp/cvs_tb_abc'})
        orch.exec_cmd_list.assert_called_once()
        orch.all.exec_cmd_list.assert_not_called()

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
        sent = orch.exec_cmd_list.call_args[0][0]
        self.assertIn("rm -f -- '/tmp/dir with space/cvs_tb_abc'", sent[0])

    def test_nothing_staged_runs_nothing(self):
        orch = _make_orch(['nodeA'])
        self.assertEqual(staging.remove_files(orch, {}), [])
        orch.exec_cmd_list.assert_not_called()

    def test_every_host_unreachable_still_reports_the_staged_files(self):
        # No removal could be attempted, so the caller must keep the paths for
        # a retry rather than be told nothing is outstanding.
        orch = _make_orch([])
        outstanding = staging.remove_files(orch, {'nodeA': '/tmp/cvs_tb_abc', 'nodeB': '/tmp/cvs_tb_def'})

        self.assertEqual(outstanding, ['nodeA', 'nodeB'])
        orch.exec_cmd_list.assert_not_called()

    def test_host_that_went_unreachable_is_reported_outstanding(self):
        orch = _make_orch(['nodeA'], cmd_list_result={'nodeA': f'{staging.REMOVE_SENTINEL}=0'})
        outstanding = staging.remove_files(orch, {'nodeA': '/tmp/cvs_tb_abc', 'nodeB': '/tmp/cvs_tb_def'})
        self.assertEqual(outstanding, ['nodeB'])


if __name__ == '__main__':
    unittest.main()


class TestPrefixValidation(unittest.TestCase):
    def test_a_prefix_starting_with_a_hyphen_is_refused(self):
        # Not shell syntax, but mktemp would parse it as its own option.
        with self.assertRaises(ValueError):
            staging.stage_file(MagicMock(), 'body', '-rf')

    def test_ordinary_prefixes_are_still_accepted(self):
        for good in ('cvs_topo_nic_', 'a.b-c', 'X9_'):
            staging._validate_prefix(good)


class TestChunkedStaging(unittest.TestCase):
    """A cluster-sized transfer file exceeds the 128 KiB single-argv cap."""

    def _orch(self, hosts, append_ok=True, remove_ok=True):
        orch = MagicMock()
        orch.all = MagicMock()
        orch.all.reachable_hosts = list(hosts)
        orch.exec.return_value = {h: {'output': f'{staging.PATH_MARKER}|/tmp/cvs_x_{h}', 'exit_code': 0} for h in hosts}
        self.commands = []

        def exec_cmd_list(cmd_list, **kwargs):
            self.commands.append(list(cmd_list))
            joined = ' '.join(cmd_list)
            if 'rm -f --' in joined:
                token = f'{staging.REMOVE_SENTINEL}=0' if remove_ok else f'{staging.REMOVE_SENTINEL}=1'
                return {h: token for h in hosts}
            token = f'{staging.APPEND_SENTINEL}=0' if append_ok else f'{staging.APPEND_SENTINEL}=1'
            return {h: token for h in hosts}

        orch.exec_cmd_list.side_effect = exec_cmd_list
        return orch

    def test_a_large_file_is_written_in_commands_under_the_argv_cap(self):
        orch = self._orch(['n1'])
        paths, failures = staging.stage_file(orch, 'x' * 200_000, 'cvs_x_')
        self.assertEqual(failures, {})
        self.assertEqual(paths, {'n1': '/tmp/cvs_x_n1'})
        issued = [orch.exec.call_args.args[0]] + [c[0] for c in self.commands]
        self.assertGreater(len(issued), 1, 'a 200 KB file must not go in one command')
        self.assertLess(max(len(c) for c in issued), 131072, 'every command must fit one argv entry')

    def test_a_small_file_still_takes_exactly_one_command(self):
        orch = self._orch(['n1'])
        staging.stage_file(orch, '1 4 (R0G0->R0I0.0->R1G0)\n', 'cvs_x_')
        self.assertEqual(self.commands, [], 'a small file needs no append round trips')

    def test_a_failed_append_removes_the_partial_file_and_reports_the_host(self):
        orch = self._orch(['n1'], append_ok=False)
        paths, failures = staging.stage_file(orch, 'x' * 200_000, 'cvs_x_')
        self.assertEqual(paths, {})
        self.assertIn('could not write staged file', failures['n1'])
        self.assertTrue(any('rm -f --' in ' '.join(c) for c in self.commands), 'partial file must be removed')

    def test_a_partial_file_that_cannot_be_removed_is_named(self):
        orch = self._orch(['n1'], append_ok=False, remove_ok=False)
        _, failures = staging.stage_file(orch, 'x' * 200_000, 'cvs_x_')
        self.assertIn('could not be removed', failures['n1'])
        self.assertIn('/tmp/cvs_x_n1', failures['n1'])
