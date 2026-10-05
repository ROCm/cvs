"""Unit tests for Spur/container sudo handling in RVS health."""

import importlib.util
import json
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


rvs = _load('rvs_cvs_under_test', 'tests/health/rvs_cvs.py')
install_rvs = _load('install_rvs_under_test', 'tests/health/install/install_rvs.py')

_GPU_JSON = json.dumps({'gpu_data': [{'asic': {'market_name': 'AMD Instinct MI355X'}}]})


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


class TestRvsPayloadSudo(unittest.TestCase):
    def test_container_amd_smi_sudo_stays_on_docker_exec(self):
        captured = []
        orch = _ContainerOrch(captured, {'host1': _GPU_JSON})
        device_map = rvs.get_gpu_device_name(orch)
        self.assertEqual(device_map['host1'], 'MI355X')
        rendered = captured[0]
        self.assertTrue(rendered.startswith('sudo -n docker exec cvs_health bash -c '))
        payload = rendered.split(' bash -c ', 1)[1]
        self.assertNotIn('sudo', payload)
        self.assertIn('amd-smi', payload)

    def test_baremetal_amd_smi_uses_sudo_prefix(self):
        orch = MagicMock()
        orch.orchestrator_type = 'baremetal'
        orch.sudo_prefix.return_value = 'sudo -n '
        orch.exec.return_value = {'host1': _GPU_JSON}
        device_map = rvs.get_gpu_device_name(orch)
        self.assertEqual(device_map['host1'], 'MI355X')
        cmd = orch.exec.call_args.args[0]
        self.assertTrue(cmd.startswith('sudo -n amd-smi '))

    def test_build_rvs_cmd_uses_env_for_sudo_ld_path(self):
        cmd = rvs._build_rvs_cmd('/opt/rocm/bin', '-r 4', sudo_prefix='sudo -n ', ld_path='/opt/rocm/lib')
        self.assertTrue(cmd.startswith('sudo -n env LD_LIBRARY_PATH='))
        self.assertNotIn('bash -c', cmd)

    def test_build_rvs_cmd_omits_sudo_when_prefix_empty(self):
        cmd = rvs._build_rvs_cmd('/opt/rocm/bin', '-g', sudo_prefix='', ld_path='/opt/rocm/lib')
        self.assertTrue(cmd.startswith('LD_LIBRARY_PATH='))
        self.assertNotIn('sudo', cmd)


class TestInstallRvsTarball(unittest.TestCase):
    def test_exec_error_is_fail_test_not_uncaught(self):
        orch = MagicMock()
        orch.exec.side_effect = OSError('ssh dropped')
        config = {'rocm_runtime_lib_path': ''}
        with patch.object(install_rvs, 'fail_test') as fail_test:
            result = install_rvs._install_rvs_tarball(orch, config, '/opt/rocm', '/tmp/install', '')
        fail_test.assert_called_once()
        self.assertIn('ssh dropped', fail_test.call_args.args[0])
        self.assertEqual(result, '/opt/rocm')
