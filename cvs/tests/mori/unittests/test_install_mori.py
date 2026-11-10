'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/tests/mori/install_mori.py: per_run fails before any command, the storage
# probe picks head-only or every-node builds, apt runs only where allowed, the guard and a
# mixed-visibility mori_dir stop before building, and the marker is always removed.

import re
import unittest
from unittest.mock import MagicMock, patch

from _pytest.outcomes import Failed, Skipped

from cvs.lib import globals
from cvs.lib.mori_lib import CHECK_DONE, MORI_IN_DIR, apt_packages
from cvs.tests.mori import install_mori as suite
from cvs.tests.mori.conftest import _Lifecycle

MORI_DICT = {
    'mori_dir': '/m',
    'git_url': 'https://example.com/mori',
    'git_ref': 'main',
    'install_system_packages': True,
    'log_dir': '/logs',
}


class _FakeOrch:
    """Answers install_mori's commands and records each as (where, cmd, hosts)."""

    def __init__(
        self, orchestrator_type='baremetal', hosts=('n0', 'n1'), sees_marker=None, missing=None, sudo='sudo -n '
    ):
        self.orchestrator_type = orchestrator_type
        self.hosts = list(hosts)
        self.head_node = self.hosts[0]
        self.container_config = {'name': 'mori_c', 'image': 'img', 'lifetime': 'persistent'}
        self.sees_marker = set(self.hosts if sees_marker is None else sees_marker)
        self.missing = {host: list((missing or {}).get(host, [])) for host in self.hosts}
        self.env_mori = {}
        self.build_exit = 0
        self.calls = []
        self.token = None
        self._sudo = sudo

    def sudo_prefix(self):
        return self._sudo

    def _answer(self, cmd, host):
        if 'find_spec' in cmd:
            return self.env_mori.get(host, 'MORI_AT=/usr/lib/python3/dist-packages/mori'), 0
        if 'apt-get install' in cmd:
            installed = cmd.split('apt-get install -y -q ')[1].split()
            self.missing[host] = [i for i in self.missing[host] if not set(apt_packages([i])) & set(installed)]
            return '', 0
        if 'MISSING:git' in cmd:
            return ''.join(f'MISSING:{item}\n' for item in self.missing[host]) + CHECK_DONE, 0
        if 'build/examples' in cmd:
            return CHECK_DONE, 0
        if '.cvs_storage_probe' in cmd and cmd.startswith('mkdir'):
            self.token = re.search(r'echo (\w+) >', cmd).group(1)
            return '', 0
        if cmd.startswith('cat '):
            return (self.token if host in self.sees_marker else ''), 0
        if 'BUILD_EXAMPLES=ON' in cmd:
            return 'build output', self.build_exit
        if 'rev-parse --short' in cmd:
            return '38dbdd8f', 0
        return '', 0

    def _run(self, where, cmd, hosts, detailed):
        self.calls.append((where, cmd, list(hosts)))
        out = {}
        for host in hosts:
            text, code = self._answer(cmd, host)
            out[host] = {'output': text, 'exit_code': code} if detailed else text
        return out

    def exec(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        return self._run('all', cmd, hosts or self.hosts, detailed)

    def exec_on_head(self, cmd, timeout=None, detailed=False, print_console=True):
        return self._run('head', cmd, [self.head_node], detailed)

    def ran(self, text):
        return [(where, hosts) for where, cmd, hosts in self.calls if text in cmd]


class _InstallCase(unittest.TestCase):
    def setUp(self):
        globals.error_list = []
        managed = patch('cvs.tests.mori.install_mori.is_managed_compute', return_value=False)
        self.managed = managed.start()
        self.addCleanup(managed.stop)
        self.addCleanup(setattr, globals, 'error_list', [])

    def _install(self, orch, **overrides):
        suite.test_install_mori(orch, dict(MORI_DICT, **overrides), _Lifecycle())

    def _install_fails(self, orch, **overrides):
        with self.assertRaises(Failed) as ctx:
            self._install(orch, **overrides)
        return str(ctx.exception)


class TestLaunch(unittest.TestCase):
    def test_per_run_container_fails_before_any_command(self):
        orch = MagicMock(orchestrator_type='container', container_config={'image': 'img', 'lifetime': 'per_run'})
        lifecycle = _Lifecycle()
        with self.assertRaises(Failed) as ctx:
            suite.test_launch_mori_container(orch, lifecycle)
        self.assertIn('per_run', str(ctx.exception))
        self.assertTrue(lifecycle.failed)
        self.assertEqual(orch.method_calls, [])
        with self.assertRaises(Skipped):
            suite.test_install_mori(orch, dict(MORI_DICT), lifecycle)
        self.assertEqual(orch.method_calls, [])


class TestBuildPlacement(_InstallCase):
    def test_shared_mori_dir_builds_only_on_the_head_node(self):
        orch = _FakeOrch()
        self._install(orch)
        self.assertEqual(orch.ran('git checkout'), [('head', ['n0'])])
        self.assertEqual(orch.ran('BUILD_EXAMPLES=ON'), [('head', ['n0'])])
        self.assertEqual(orch.ran('build/examples'), [('all', ['n0', 'n1'])])

    def test_node_local_mori_dir_builds_on_every_node(self):
        orch = _FakeOrch(sees_marker=['n0'])
        self._install(orch)
        self.assertEqual(orch.ran('git checkout'), [('all', ['n0', 'n1'])])
        self.assertEqual(orch.ran('BUILD_EXAMPLES=ON'), [('all', ['n0', 'n1'])])

    def test_mori_dir_seen_by_some_nodes_fails_without_building(self):
        orch = _FakeOrch(hosts=('n0', 'n1', 'n2'), sees_marker=['n0', 'n1'])
        message = self._install_fails(orch)
        self.assertIn('n2', message)
        self.assertEqual(orch.ran('git checkout'), [])
        self.assertEqual(orch.ran('BUILD_EXAMPLES=ON'), [])
        self.assertEqual(orch.ran('rm -f'), [('head', ['n0'])])


class TestInstallRules(_InstallCase):
    def test_managed_run_never_runs_apt(self):
        self.managed.return_value = True
        orch = _FakeOrch(orchestrator_type='container', missing={'n1': ['git']})
        message = self._install_fails(orch)
        self.assertIn('n1: git', message)
        self.assertEqual(orch.ran('apt-get'), [])
        self.assertEqual(orch.ran('BUILD_EXAMPLES=ON'), [])

    def test_baremetal_without_passwordless_sudo_never_runs_apt(self):
        orch = _FakeOrch(missing={'n1': ['git']}, sudo='')
        self.assertIn('n1: git', self._install_fails(orch))
        self.assertEqual(orch.ran('apt-get'), [])

    def test_install_system_packages_false_never_runs_apt(self):
        orch = _FakeOrch(orchestrator_type='container', missing={'n1': ['git']})
        message = self._install_fails(orch, install_system_packages=False)
        self.assertIn('n1: git; install them, or set install_system_packages to true', message)
        self.assertEqual(orch.ran('apt-get'), [])

    def test_items_apt_never_installs_get_no_install_system_packages_hint(self):
        orch = _FakeOrch(missing={'n0': ['pci.h', 'hip-lang', 'torch']})
        message = self._install_fails(orch, install_system_packages=False)
        self.assertIn(
            'n0: pci.h, hip-lang, torch; the ROCm dev stack and torch must come from the image or host', message
        )
        self.assertNotIn('install_system_packages', message)

    def test_apt_installs_only_what_each_node_misses_and_fails_on_the_rest(self):
        orch = _FakeOrch(orchestrator_type='container', missing={'n1': ['git', 'hipcc']})
        message = self._install_fails(orch)
        self.assertIn('n1: hipcc', message)
        self.assertNotIn('git', message.split('n1: ')[1].split(';')[0])
        (apt,) = [cmd for _, cmd, _ in orch.calls if 'apt-get install' in cmd]
        self.assertTrue(apt.endswith('apt-get install -y -q git'))
        self.assertNotIn('sudo', apt)
        self.assertEqual(orch.ran('apt-get install'), [('all', ['n1'])])
        self.assertEqual(orch.ran('BUILD_EXAMPLES=ON'), [])

    def test_baremetal_apt_uses_passwordless_sudo_and_then_builds(self):
        orch = _FakeOrch(missing={'n0': ['mpicc']})
        self._install(orch)
        (apt,) = [cmd for _, cmd, _ in orch.calls if 'apt-get install' in cmd]
        self.assertIn('sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q libopenmpi-dev', apt)
        self.assertEqual(orch.ran('BUILD_EXAMPLES=ON'), [('head', ['n0'])])

    def test_guard_tripped_on_any_node_builds_nothing(self):
        orch = _FakeOrch()
        orch.env_mori['n1'] = MORI_IN_DIR
        self.assertIn('n1', self._install_fails(orch))
        self.assertEqual(orch.ran('.cvs_storage_probe'), [])
        self.assertEqual(orch.ran('git checkout'), [])
        self.assertEqual(orch.ran('BUILD_EXAMPLES=ON'), [])


class TestConfigDefaults(_InstallCase):
    def test_config_without_install_keys_clones_mori_main_and_allows_apt(self):
        orch = _FakeOrch(orchestrator_type='container', missing={'n1': ['git']})
        config = {k: v for k, v in MORI_DICT.items() if k not in ('git_url', 'git_ref', 'install_system_packages')}
        suite.test_install_mori(orch, config, _Lifecycle())
        (checkout,) = [cmd for _, cmd, _ in orch.calls if 'git checkout' in cmd]
        self.assertIn('git remote add origin https://github.com/ROCm/mori', checkout)
        self.assertIn('origin/main', checkout)
        self.assertEqual(orch.ran('apt-get install'), [('all', ['n1'])])

    def test_install_system_packages_accepts_strings_and_booleans(self):
        for value, runs_apt in (('False', False), ('false', False), (False, False), ('True', True), (True, True)):
            with self.subTest(value=value):
                orch = _FakeOrch(orchestrator_type='container', missing={'n1': ['git']})
                if runs_apt:
                    self._install(orch, install_system_packages=value)
                else:
                    self._install_fails(orch, install_system_packages=value)
                self.assertEqual(bool(orch.ran('apt-get install')), runs_apt)


class TestCleanup(_InstallCase):
    def test_marker_is_removed_after_a_build(self):
        orch = _FakeOrch()
        self._install(orch)
        self.assertEqual(orch.calls[-1][:1], ('head',))
        self.assertIn('rm -f /m/.cvs_storage_probe', orch.calls[-1][1])

    def test_marker_is_removed_after_a_failed_build(self):
        orch = _FakeOrch()
        orch.build_exit = 1
        self.assertIn('install_mori_', self._install_fails(orch))
        self.assertIn('rm -f /m/.cvs_storage_probe', orch.calls[-1][1])
        self.assertEqual(orch.ran('build/examples'), [])

    def test_baremetal_puts_prettytable_in_site_not_host_python(self):
        orch = _FakeOrch()
        self._install(orch)
        pips = [cmd for _, cmd, _ in orch.calls if 'pip install' in cmd and 'BUILD_EXAMPLES' not in cmd]
        self.assertEqual(pips, ['python3 -m pip install --target /m/site prettytable'])

    def test_container_leaves_prettytable_to_the_benchmark(self):
        orch = _FakeOrch(orchestrator_type='container')
        self._install(orch)
        self.assertEqual(orch.ran('prettytable'), [])


if __name__ == '__main__':
    unittest.main()
