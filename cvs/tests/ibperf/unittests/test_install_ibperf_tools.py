"""Unit tests for the ibperf perftest install."""

import unittest
from unittest.mock import MagicMock, patch

from cvs.lib import globals
from cvs.tests.ibperf import install_ibperf_tools

_NODES = ('n1', 'n2')


class TestInstallIbPerf(unittest.TestCase):
    def setUp(self):
        for name in ('detect_rocm_path', 'configure_perftest', 'check_perftest_dmabuf'):
            patcher = patch.object(install_ibperf_tools.ibperf_lib, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(globals, 'error_list', [])
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, config, deps_out=None, managed=False):
        deps_out = deps_out or {}

        def all_exec(cmd, **_kwargs):
            if cmd == install_ibperf_tools.RUNTIME_DEPS_CHECK:
                return {n: deps_out.get(n, '') for n in _NODES}
            if 'ib_write_bw -h' in cmd:
                return {n: 'GPUDirect RDMA' for n in _NODES}
            return {n: '' for n in _NODES}

        orch = MagicMock()
        orch.all.exec.side_effect = all_exec
        config = {'install_perf_package': 'True', 'install_dir': '/shared', **config}
        with patch.object(install_ibperf_tools, 'is_managed_compute', return_value=managed):
            try:
                install_ibperf_tools.test_install_ib_perf(orch, config)
            finally:
                self.all_cmds = [c.args[0] for c in orch.all.exec.call_args_list]
                self.head_cmds = [c.args[0] for c in orch.head.exec.call_args_list]

    def _apt_cmds(self):
        return [c for c in self.all_cmds if 'apt' in c]

    def test_config_without_key_checks_deps_and_never_runs_apt(self):
        self._run({})
        self.assertEqual(self._apt_cmds(), [])
        self.assertIn(install_ibperf_tools.RUNTIME_DEPS_CHECK, self.all_cmds)
        self.assertIn('cd /shared/perftest; make install', self.head_cmds)

    def test_missing_runtime_deps_fail_naming_node_before_build(self):
        with self.assertRaises(BaseException) as ctx:
            self._run({'install_system_packages': 'False'}, {'n2': 'MISSING:numactl\nMISSING:librdmacm\n'})
        self.assertIn('node n2: numactl, librdmacm', str(ctx.exception))
        self.assertNotIn('node n1', str(ctx.exception))
        self.assertFalse(any('make' in c for c in self.head_cmds))

    def test_opt_in_runs_apt_except_in_managed_job_step(self):
        self._run({'install_system_packages': 'true'})
        self.assertIn('sudo apt install -y numactl', self._apt_cmds())
        self.assertNotIn(install_ibperf_tools.RUNTIME_DEPS_CHECK, self.all_cmds)

        self._run({'install_system_packages': 'True'}, managed=True)
        self.assertEqual(self._apt_cmds(), [])
        self.assertIn(install_ibperf_tools.RUNTIME_DEPS_CHECK, self.all_cmds)


if __name__ == '__main__':
    unittest.main()
