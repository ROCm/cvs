"""Unit tests for the ibperf perftest install."""

import unittest
from unittest.mock import MagicMock, patch

from cvs.lib import rocm_plib
from cvs.tests.ibperf import install_ibperf_tools

_NODES = ('n1', 'n2')


class TestInstallIbPerf(unittest.TestCase):
    def setUp(self):
        for name in ('detect_rocm_path', 'configure_perftest', 'check_perftest_dmabuf'):
            patcher = patch.object(install_ibperf_tools.ibperf_lib, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, config, deps_out=None, managed=False, gpu_tool_out=None):
        deps_out = deps_out or {}
        gpu_tool_out = gpu_tool_out or {}

        def all_exec(cmd, **_kwargs):
            if cmd == install_ibperf_tools.RUNTIME_DEPS_CHECK:
                return {n: deps_out.get(n, '') for n in _NODES}
            if cmd == install_ibperf_tools.GPU_TOOL_CHECK:
                return {n: gpu_tool_out.get(n, '') for n in _NODES}
            if 'ib_write_bw -h' in cmd:
                return {n: 'GPUDirect RDMA' for n in _NODES}
            return {n: '' for n in _NODES}

        orch = MagicMock()
        orch.exec.side_effect = all_exec
        config = {'install_perf_package': 'True', 'install_dir': '/shared', **config}
        with patch.object(install_ibperf_tools, 'is_managed_compute', return_value=managed):
            try:
                install_ibperf_tools.test_install_ib_perf(orch, config)
            finally:
                self.all_cmds = [c.args[0] for c in orch.exec.call_args_list]
                self.head_cmds = [c.args[0] for c in orch.exec_on_head.call_args_list]

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
        self.assertIn(
            'node n2: numactl, librdmacm; install them or set install_system_packages to True', str(ctx.exception)
        )
        self.assertNotIn('node n1', str(ctx.exception))
        self.assertFalse(any('make' in c for c in self.head_cmds))

    def test_missing_nic_discovery_tools_fail_install(self):
        self.assertIn('MISSING:lshw', install_ibperf_tools.RUNTIME_DEPS_CHECK)
        self.assertIn('MISSING:rdma', install_ibperf_tools.RUNTIME_DEPS_CHECK)
        with self.assertRaises(BaseException) as ctx:
            self._run({'install_system_packages': 'False'}, {'n1': 'MISSING:lshw\nMISSING:rdma\n'})
        self.assertIn('node n1: lshw, rdma;', str(ctx.exception))
        self.assertFalse(any('make' in c for c in self.head_cmds))

    def test_gpu_tool_check_uses_gpu_discovery_lookup(self):
        check = install_ibperf_tools.GPU_TOOL_CHECK
        self.assertIn('command -v rocm-smi', check)
        self.assertIn('[ -x /opt/rocm/bin/rocm-smi ]', check)
        self.assertIn('/opt/rocm/bin/rocm-smi', rocm_plib._rocm_smi_showbus_cmd(use_sudo=False))

    def test_missing_gpu_tool_fails_naming_node_whether_or_not_apt_runs(self):
        for flag, managed in (('False', False), ('True', False), ('True', True)):
            with self.subTest(flag=flag, managed=managed):
                with self.assertRaises(BaseException) as ctx:
                    self._run(
                        {'install_system_packages': flag}, managed=managed, gpu_tool_out={'n2': 'MISSING:rocm-smi\n'}
                    )
                self.assertIn('rocm-smi not found on node n2 (PATH or /opt/rocm/bin)', str(ctx.exception))
                self.assertNotIn('node n1', str(ctx.exception))
                self.assertEqual(self._apt_cmds(), [])
                self.assertFalse(any('make' in c for c in self.head_cmds))

    def test_missing_gpu_tool_and_runtime_deps_are_both_reported(self):
        with self.assertRaises(BaseException) as ctx:
            self._run({}, {'n1': 'MISSING:numactl\n'}, gpu_tool_out={'n1': 'MISSING:rocm-smi\n'})
        self.assertIn('rocm-smi not found on node n1', str(ctx.exception))
        self.assertIn('node n1: numactl;', str(ctx.exception))

    def test_missing_runtime_deps_in_job_step_say_install_outside_step(self):
        for flag in ('False', 'True'):
            with self.subTest(flag=flag):
                with self.assertRaises(BaseException) as ctx:
                    self._run({'install_system_packages': flag}, {'n1': 'MISSING:numactl\n'}, managed=True)
                self.assertIn('node n1: numactl; install them outside the job step', str(ctx.exception))
                self.assertNotIn('install_system_packages', str(ctx.exception))

    def test_opt_in_runs_apt_except_in_managed_job_step(self):
        self._run({'install_system_packages': 'true'})
        self.assertIn('sudo apt install -y numactl lshw iproute2', self._apt_cmds())
        self.assertNotIn(install_ibperf_tools.RUNTIME_DEPS_CHECK, self.all_cmds)
        self.assertLess(
            self.all_cmds.index(install_ibperf_tools.GPU_TOOL_CHECK), self.all_cmds.index('sudo apt update -y')
        )

        self._run({'install_system_packages': 'True'}, managed=True)
        self.assertEqual(self._apt_cmds(), [])
        self.assertIn(install_ibperf_tools.RUNTIME_DEPS_CHECK, self.all_cmds)


if __name__ == '__main__':
    unittest.main()
