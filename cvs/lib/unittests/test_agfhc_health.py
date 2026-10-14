"""Unit tests for Spur/container sudo handling in AGFHC health."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load(name, relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agfhc = _load('agfhc_cvs_under_test', 'tests/health/agfhc_cvs.py')
install_agfhc = _load('install_agfhc_under_test', 'tests/health/install/install_agfhc.py')
DockerRuntime = _load('docker_runtime_under_test', 'core/runtimes/docker.py').DockerRuntime

_SUCCESS = 'Program exiting with return code AGFHC_SUCCESS [0]\n'
_INSTALL_DIR = '/home/u/INSTALL/agfhc'
_CONFIG = {
    'install_dir': _INSTALL_DIR,
    'package_tar_ball': '/home/u/PACKAGES/agfhc.tar.bz2',
    'path': '/opt/amd/agfhc',
}


class _HostExec:
    def __init__(self, captured, result):
        self.captured = captured
        self.result = result

    def exec(self, cmd, timeout=None, detailed=False, print_console=True):
        self.captured.append(cmd)
        return self.result


class _ContainerOrch:
    orchestrator_type = 'container'
    hosts = ('host1',)

    def __init__(self, captured, result):
        self.all = _HostExec(captured, result)

    def sudo_prefix(self):
        return 'sudo -n '

    def exec(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        return DockerRuntime(MagicMock(), self).exec('cvs_health', cmd, hosts, timeout, detailed, print_console)


def _baremetal(sudo_prefix, result):
    orch = MagicMock()
    orch.orchestrator_type = 'baremetal'
    orch.sudo_prefix.return_value = sudo_prefix
    orch.exec.return_value = result
    return orch


class TestAgfhcPayloadSudo(unittest.TestCase):
    def test_build_agfhc_cmd_uses_sudo_prefix(self):
        cmd = agfhc._build_agfhc_cmd('/opt/amd/agfhc', '-r dma_lvl1', sudo_prefix='sudo -n ')
        self.assertEqual(cmd, 'sudo -n /opt/amd/agfhc/agfhc -r dma_lvl1')

    def test_build_agfhc_cmd_omits_sudo_when_prefix_empty(self):
        cmd = agfhc._build_agfhc_cmd('/opt/amd/agfhc', '-r dma_lvl1', sudo_prefix='')
        self.assertEqual(cmd, '/opt/amd/agfhc/agfhc -r dma_lvl1')
        self.assertNotIn('sudo', cmd)

    def test_run_agfhc_baremetal_uses_sudo_prefix(self):
        orch = _baremetal('sudo -n ', {'host1': _SUCCESS})
        agfhc.globals.error_list = []
        agfhc._run_agfhc(orch, {'path': '/opt/amd/agfhc'}, '-r dma_lvl1', 30, 'dma_lvl1', {}, {}, None)
        cmd = orch.exec.call_args.args[0]
        self.assertTrue(cmd.startswith('sudo -n /opt/amd/agfhc/agfhc '))

    def test_run_agfhc_omits_sudo_when_prefix_empty(self):
        orch = _baremetal('', {'host1': _SUCCESS})
        agfhc.globals.error_list = []
        agfhc._run_agfhc(orch, {'path': '/opt/amd/agfhc'}, '-r dma_lvl1', 30, 'dma_lvl1', {}, {}, None)
        cmd = orch.exec.call_args.args[0]
        self.assertEqual(cmd, '/opt/amd/agfhc/agfhc -r dma_lvl1')

    def test_run_agfhc_container_sudo_stays_on_docker_exec(self):
        captured = []
        orch = _ContainerOrch(captured, {'host1': _SUCCESS})
        agfhc.globals.error_list = []
        agfhc._run_agfhc(orch, {'path': '/opt/amd/agfhc'}, '-r dma_lvl1', 30, 'dma_lvl1', {}, {}, None)
        self.assertEqual(len(captured), 1)
        rendered = captured[0]
        self.assertTrue(rendered.startswith('sudo -n docker exec cvs_health bash -c '))
        payload = rendered.split(' bash -c ', 1)[1]
        self.assertNotIn('sudo', payload)
        self.assertIn('/opt/amd/agfhc/agfhc', payload)

    def test_recipe_info_uses_sudo_prefix(self):
        orch = _baremetal('sudo -n ', {'host1': _SUCCESS})
        with patch.object(agfhc, 'update_test_result'):
            agfhc.test_agfhc_all_lvl5(orch, {'path': '/opt/amd/agfhc'})
        cmd = orch.exec.call_args.args[0]
        self.assertEqual(cmd, 'sudo -n /opt/amd/agfhc/agfhc --recipe-info all_lvl5')


class TestInstallAgfhcSudo(unittest.TestCase):
    def test_install_cmd_uses_sudo_prefix(self):
        cmd = install_agfhc._build_agfhc_install_cmd(_INSTALL_DIR, sudo_prefix='sudo -n ')
        self.assertEqual(
            cmd,
            f"sudo -n bash -c 'cd {_INSTALL_DIR} && ./install --rocm-tar'",
        )

    def test_install_cmd_omits_sudo_when_prefix_empty(self):
        cmd = install_agfhc._build_agfhc_install_cmd(_INSTALL_DIR, sudo_prefix='')
        self.assertEqual(cmd, f"bash -c 'cd {_INSTALL_DIR} && ./install --rocm-tar'")
        self.assertNotIn('sudo', cmd)

    def _install_commands(self, orch):
        with patch.object(install_agfhc.time, 'sleep'), patch.object(install_agfhc, 'update_test_result'):
            install_agfhc.test_install_agfhc(orch, dict(_CONFIG))
        return [call.args[0] for call in orch.exec.call_args_list]

    def test_install_baremetal_uses_sudo_prefix(self):
        orch = _baremetal('sudo -n ', {'host1': '/opt/amd/agfhc/agfhc'})
        cmds = self._install_commands(orch)
        install_cmds = [cmd for cmd in cmds if './install --rocm-tar' in cmd]
        self.assertEqual(install_cmds, [f"sudo -n bash -c 'cd {_INSTALL_DIR} && ./install --rocm-tar'"])

    def test_install_omits_sudo_when_prefix_empty(self):
        orch = _baremetal('', {'host1': '/opt/amd/agfhc/agfhc'})
        cmds = self._install_commands(orch)
        install_cmds = [cmd for cmd in cmds if './install --rocm-tar' in cmd]
        self.assertEqual(install_cmds, [f"bash -c 'cd {_INSTALL_DIR} && ./install --rocm-tar'"])
        self.assertNotIn('sudo', install_cmds[0])

    def test_install_container_sudo_stays_on_docker_exec(self):
        captured = []
        orch = _ContainerOrch(captured, {'host1': '/opt/amd/agfhc/agfhc'})
        with patch.object(install_agfhc.time, 'sleep'), patch.object(install_agfhc, 'update_test_result'):
            install_agfhc.test_install_agfhc(orch, dict(_CONFIG))
        install_rendered = [cmd for cmd in captured if 'install --rocm-tar' in cmd]
        self.assertEqual(len(install_rendered), 1)
        rendered = install_rendered[0]
        self.assertTrue(rendered.startswith('sudo -n docker exec cvs_health bash -c '))
        payload = rendered.split(' bash -c ', 1)[1]
        self.assertNotIn('sudo', payload)
        self.assertIn('./install --rocm-tar', payload)

    def test_exec_error_is_fail_test_not_uncaught(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = ''

        def _exec(cmd, timeout=None, detailed=False, print_console=True):
            if './install' in cmd:
                raise OSError('ssh dropped')
            return {'host1': '/opt/amd/agfhc/agfhc'}

        orch.exec.side_effect = _exec
        with (
            patch.object(install_agfhc, 'fail_test') as fail_test,
            patch.object(install_agfhc, 'update_test_result'),
            patch.object(install_agfhc.time, 'sleep'),
        ):
            install_agfhc.test_install_agfhc(orch, dict(_CONFIG))
        messages = [call.args[0] for call in fail_test.call_args_list]
        self.assertTrue(any('ssh dropped' in message for message in messages))
