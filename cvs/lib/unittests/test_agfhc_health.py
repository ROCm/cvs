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
        self.assertEqual(cmd, '/opt/amd/agfhc/agfhc -r hbm')
        self.assertNotIn('sudo', cmd)

    def test_baremetal_payload_uses_sudo_prefix(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = 'sudo -n '
        self.assertEqual(agfhc._payload_sudo_prefix(orch), 'sudo -n ')
        cmd = agfhc._build_agfhc_cmd(orch, '/opt/amd/agfhc', '-r hbm')
        self.assertEqual(cmd, 'sudo -n /opt/amd/agfhc/agfhc -r hbm')

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
        self.assertEqual(
            csp_qual._build_agfhc_cmd(orch, '/opt/amd/agfhc', '-v'),
            'sudo -n /opt/amd/agfhc/agfhc -v',
        )

    def test_get_log_results_uses_payload_sudo_prefix(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = 'sudo -n '
        handle = MagicMock()
        handle.reachable_hosts = ['host1']
        handle.host_list = ['host1']
        handle.exec_cmd_list.return_value = {'host1': '"total_failed": 0,\n'}
        orch.all = handle
        out_dict = {'host1': 'Log directory: /var/tmp/agfhc_logs/run1\n'}
        res = csp_qual.get_log_results(orch, out_dict)
        self.assertIn('total_failed', res['host1'])
        cmd = handle.exec_cmd_list.call_args.args[0][0]
        self.assertTrue(cmd.startswith('sudo -n cat '))
        self.assertIn('/var/tmp/agfhc_logs/run1/results.json', cmd)

    def test_get_log_results_container_omits_sudo(self):
        orch = MagicMock()
        orch.orchestrator_type = 'container'
        orch.sudo_prefix.return_value = 'sudo -n '
        handle = MagicMock()
        handle.reachable_hosts = ['host1']
        handle.host_list = ['host1']
        handle.exec_cmd_list.return_value = {'host1': '"total_failed": 0,\n'}
        orch.all = handle
        out_dict = {'host1': 'Log directory: /var/tmp/agfhc_logs/run1\n'}
        csp_qual.get_log_results(orch, out_dict)
        cmd = handle.exec_cmd_list.call_args.args[0][0]
        self.assertTrue(cmd.startswith('cat '))
        self.assertNotIn('sudo', cmd)


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


if __name__ == '__main__':
    unittest.main()
