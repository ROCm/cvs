"""Unit tests for Spur/container sudo handling in AGFHC health."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cvs.core.runtimes.docker import DockerRuntime


def _load(name, relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agfhc = _load('agfhc_cvs_under_test', 'tests/health/agfhc_cvs.py')
csp_qual = _load('csp_qual_agfhc_under_test', 'tests/health/csp_qual_agfhc.py')
install_agfhc = _load('install_agfhc_under_test', 'tests/health/install/install_agfhc.py')


class _HostExec:
    def __init__(self, captured, result):
        self.captured = captured
        self.result = result
        self.host_list = ('host1',)
        self.reachable_hosts = ('host1',)

    def exec(self, cmd, timeout=None, detailed=False, print_console=True):
        self.captured.append(cmd)
        return self.result

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        self.captured.extend(cmd_list)
        return {self.host_list[0]: self.result.get(self.host_list[0], '')}


class _ContainerOrch:
    orchestrator_type = 'container'
    hosts = ('host1',)

    def __init__(self, captured, result):
        self.all = _HostExec(captured, result)

    def sudo_prefix(self):
        return 'sudo -n '

    def exec(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        return DockerRuntime(MagicMock(), self).exec('cvs_health', cmd, hosts, timeout, detailed, print_console)


class TestAgfhcPayloadSudo(unittest.TestCase):
    def test_container_payload_omits_sudo(self):
        orch = MagicMock()
        orch.orchestrator_type = 'container'
        orch.sudo_prefix.return_value = 'sudo -n '
        self.assertEqual(agfhc._payload_sudo_prefix(orch), '')
        cmd = agfhc._build_agfhc_cmd(orch, '/opt/amd/agfhc', '-r hbm')
        self.assertTrue(cmd.startswith('bash -c '))
        self.assertIn('exec /opt/amd/agfhc/agfhc -r hbm', cmd)
        self.assertNotIn('sudo', cmd)

    def test_baremetal_payload_uses_sudo_prefix(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = 'sudo -n '
        self.assertEqual(agfhc._payload_sudo_prefix(orch), 'sudo -n ')
        cmd = agfhc._build_agfhc_cmd(orch, '/opt/amd/agfhc', '-r hbm')
        self.assertTrue(cmd.startswith('sudo -n bash -c '))
        self.assertIn('exec /opt/amd/agfhc/agfhc -r hbm', cmd)

    def test_container_run_keeps_sudo_on_docker_exec(self):
        captured = []
        orch = _ContainerOrch(captured, {'host1': 'return code AGFHC_SUCCESS\n'})
        with (
            patch.object(agfhc, 'scan_agfc_results'),
            patch.object(agfhc, 'print_test_output'),
            patch.object(agfhc, '_capture_agfhc_rundeck'),
            patch.object(agfhc, 'update_test_result'),
            patch.object(agfhc, 'timed_stage') as timed,
        ):
            timed.return_value.__enter__ = MagicMock(return_value=None)
            timed.return_value.__exit__ = MagicMock(return_value=False)
            agfhc._run_agfhc(
                orch,
                {'path': '/opt/amd/agfhc'},
                '-r hbm',
                60,
                'hbm',
                {},
                {},
                MagicMock(),
            )
        rendered = captured[0]
        self.assertTrue(rendered.startswith('sudo -n docker exec cvs_health bash -c '))
        payload = rendered.split(' bash -c ', 1)[1]
        self.assertNotIn('sudo', payload)
        self.assertIn('/opt/amd/agfhc/agfhc -r hbm', payload)


class TestCspQualAgfhcOrch(unittest.TestCase):
    def test_build_cmd_matches_agfhc_cvs(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = 'sudo -n '
        bare = csp_qual._build_agfhc_cmd(orch, '/opt/amd/agfhc', '-v')
        self.assertEqual(bare, agfhc._build_agfhc_cmd(orch, '/opt/amd/agfhc', '-v'))

    def test_get_log_results_uses_payload_sudo_prefix(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.hosts = ['host1']
        orch.sudo_prefix.return_value = 'sudo -n '
        orch.all.exec_cmd_list.return_value = {'host1': '"total_failed": 0,\n'}
        out_dict = {'host1': 'Log directory: /home/first.last/agfhc_logs/run1\n'}
        res = csp_qual.get_log_results(orch, out_dict)
        self.assertIn('total_failed', res['host1'])
        cmd_list = orch.all.exec_cmd_list.call_args.args[0]
        self.assertEqual(len(cmd_list), 1)
        self.assertTrue(cmd_list[0].startswith('sudo -n cat '))
        self.assertIn('/home/first.last/agfhc_logs/run1/results.json', cmd_list[0])
        orch.exec_cmd_list.assert_not_called()

    def test_get_log_results_container_reads_dict_output_in_parallel(self):
        orch = MagicMock()
        orch.orchestrator_type = 'container'
        orch.hosts = ['host2', 'host1']
        orch.sudo_prefix.return_value = 'sudo -n '
        orch.exec_cmd_list.return_value = {
            'host2': {'output': '"total_failed": 0,\n'},
            'host1': {'output': '"total_failed": 0,\n'},
        }
        out_dict = {
            'host1': 'Log directory: /var/tmp/agfhc_logs/run1\n',
            'host2': 'Log directory: /var/tmp/agfhc_logs/run2\n',
        }
        res = csp_qual.get_log_results(orch, out_dict)
        self.assertIn('total_failed', res['host1'])
        self.assertIn('total_failed', res['host2'])
        cmd_list = orch.exec_cmd_list.call_args.args[0]
        self.assertEqual(
            cmd_list,
            [
                'cat /var/tmp/agfhc_logs/run2/results.json',
                'cat /var/tmp/agfhc_logs/run1/results.json',
            ],
        )
        self.assertEqual(orch.exec_cmd_list.call_count, 1)
        orch.all.exec_cmd_list.assert_not_called()

    def test_get_log_results_dumps_once_after_a_failure(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.hosts = ['host1', 'host2']
        orch.sudo_prefix.return_value = 'sudo -n '

        def exec_cmd_list(cmd_list, timeout=None, print_console=True):
            joined = ' '.join(cmd_list)
            if 'results.json' in joined:
                return {'host1': '"total_failed": 1,\n', 'host2': '"total_failed": 2,\n'}
            if 'journal.log' in joined:
                return {'host1': 'journal-1', 'host2': 'journal-2'}
            return {'host1': 'error-1', 'host2': 'error-2'}

        orch.all.exec_cmd_list.side_effect = exec_cmd_list
        out_dict = {
            'host1': 'Log directory: /tmp/a\n',
            'host2': 'Log directory: /tmp/b\n',
            'host3': 'no log directory here',
        }
        with patch.object(csp_qual, 'fail_test') as fail, patch.object(csp_qual.log, 'info') as info:
            csp_qual.get_log_results(orch, out_dict)
        self.assertEqual(orch.all.exec_cmd_list.call_count, 3)
        messages = [call.args[0] for call in fail.call_args_list]
        self.assertTrue(any('host3' in message for message in messages))
        self.assertTrue(any('host1' in message for message in messages))
        dumped = [call.args[0] for call in info.call_args_list if 'journal.log' in call.args[0]]
        self.assertEqual(len(dumped), 2)

    def test_get_log_results_noop_fills_a_host_with_no_log_directory(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.hosts = ['h1', 'h2', 'h3']
        orch.sudo_prefix.return_value = 'sudo -n '
        orch.all.exec_cmd_list.return_value = {
            'h1': '"total_failed": 0,\n',
            'h3': '"total_failed": 0,\n',
        }
        out_dict = {
            'h1': 'Log directory: /tmp/a\n',
            'h2': 'no log directory here',
            'h3': 'Log directory: /tmp/c\n',
        }
        with patch.object(csp_qual, 'fail_test') as fail:
            res = csp_qual.get_log_results(orch, out_dict)
        cmd_list = orch.all.exec_cmd_list.call_args.args[0]
        self.assertEqual(
            cmd_list,
            [
                'sudo -n cat /tmp/a/results.json',
                'true',
                'sudo -n cat /tmp/c/results.json',
            ],
        )
        self.assertEqual(orch.all.exec_cmd_list.call_count, 1)
        self.assertNotIn('h2', res)
        fail.assert_called_once()
        self.assertIn('h2', fail.call_args.args[0])

    def test_recipe_quotes_log_dir_with_a_dollar(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = ''
        orch.exec.return_value = {'host1': 'return code AGFHC_SUCCESS\n'}
        with (
            patch.object(csp_qual, 'scan_agfc_results'),
            patch.object(csp_qual, 'get_log_results', return_value={}),
            patch.object(csp_qual, 'print_test_output'),
            patch.object(csp_qual, '_capture_agfhc_rundeck'),
            patch.object(csp_qual, 'update_test_result'),
            patch.object(csp_qual, 'timed_stage') as timed,
        ):
            timed.return_value.__enter__ = MagicMock(return_value=None)
            timed.return_value.__exit__ = MagicMock(return_value=False)
            csp_qual._run_agfhc_recipe(
                orch,
                {'path': '/opt/amd/agfhc', 'log_dir': '/tmp/$logs'},
                '-r hbm',
                30,
                'hbm',
                'out',
                {},
                {},
                MagicMock(),
            )
        cmd = orch.exec.call_args.args[0]
        self.assertIn('/tmp/$logs/out', cmd)
        self.assertNotIn('-o /tmp/$logs/out', cmd)

    def test_version_check_quotes_log_dir_for_ls(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = ''
        calls = []

        def exec_cmd(cmd, timeout=None, **_kwargs):
            calls.append(cmd)
            if 'test -w' in cmd:
                return {'host1': 'writable'}
            if 'agfhc' in cmd and '-v' in cmd:
                return {'host1': 'agfhc version: 1.0'}
            return {'host1': ''}

        orch.exec.side_effect = exec_cmd
        with (
            patch.object(csp_qual.time, 'sleep'),
            patch.object(csp_qual, 'update_test_result'),
            patch.object(csp_qual, '_capture_agfhc_rundeck'),
            patch.object(csp_qual, 'timed_stage') as timed,
        ):
            timed.return_value.__enter__ = MagicMock(return_value=None)
            timed.return_value.__exit__ = MagicMock(return_value=False)
            csp_qual.test_version_check(
                orch,
                {'path': '/opt/amd/agfhc', 'log_dir': '/tmp/$logs'},
                {},
                {},
                MagicMock(),
            )
        self.assertTrue(any("ls -ld '/tmp/$logs'" in cmd for cmd in calls))

    def test_log_dir_probe_returns_early(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = ''
        orch.exec.side_effect = [
            {'host1': ''},
            {'host1': 'not-writable'},
        ]
        with (
            patch.object(csp_qual, 'fail_test') as fail,
            patch.object(csp_qual, 'update_test_result') as update,
            patch.object(csp_qual, 'timed_stage') as timed,
        ):
            csp_qual.test_version_check(
                orch,
                {'path': '/opt/amd/agfhc', 'log_dir': '/root/agfhc_logs'},
                {},
                {},
                MagicMock(),
            )
        fail.assert_called_once()
        self.assertIn('/root/agfhc_logs', fail.call_args.args[0])
        update.assert_called_once()
        timed.assert_not_called()


class TestInstallAgfhcPayloadSudo(unittest.TestCase):
    def test_container_install_omits_payload_sudo(self):
        orch = MagicMock()
        orch.orchestrator_type = 'container'
        orch.sudo_prefix.return_value = 'sudo -n '
        self.assertEqual(install_agfhc._payload_sudo_prefix(orch), '')

    def test_baremetal_install_uses_sudo_prefix(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = 'sudo -n '
        self.assertEqual(install_agfhc._payload_sudo_prefix(orch), 'sudo -n ')

    def test_unwritable_output_dir_returns_before_install(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = 'sudo -n '
        calls = []

        def exec_cmd(cmd, timeout=None, **_kwargs):
            calls.append(cmd)
            if 'test -w' in cmd:
                return {'host1': 'not-writable'}
            return {'host1': 'ok'}

        orch.exec.side_effect = exec_cmd
        config = {
            'install_dir': '/home/user/INSTALL/agfhc/',
            'package_tar_ball': '/home/user/PACKAGES/agfhc.tar.bz2',
            'path': '/home/user/INSTALL/agfhc/prefix/agfhc',
        }
        with (
            patch.object(install_agfhc.time, 'sleep'),
            patch.object(install_agfhc, 'fail_test') as fail,
            patch.object(install_agfhc, 'update_test_result') as update,
        ):
            install_agfhc.test_install_agfhc(orch, config)
        update.assert_called_once()
        fail.assert_called_once()
        self.assertIn('/home/user/INSTALL/agfhc/prefix', fail.call_args.args[0])
        self.assertFalse(any('./install' in cmd for cmd in calls))
        self.assertTrue(any('test -w' in cmd for cmd in calls))


if __name__ == '__main__':
    unittest.main()
