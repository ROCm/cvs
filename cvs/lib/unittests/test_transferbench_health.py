"""Unit tests for health TransferBench helpers (AIMVT-314)."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


_TB_PATH = Path(__file__).resolve().parents[2] / 'tests' / 'health' / 'transferbench_cvs.py'
_SPEC = importlib.util.spec_from_file_location('transferbench_cvs_under_test', _TB_PATH)
tb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tb)

HELIOS_R_P2P_ABORT = """\
NUM_CPU_DEVICES = 38
Transfer 0: DST 0: CPU 2 on rank 0 cannot allocate memory due to process memory policy/cpuset
"""

HELIOS_R_P2P_SUCCESS = """\
NUM_CPU_DEVICES = 2
Averages (During UniDir): 1.1 2.2 3.3 40.5
Averages (During BiDir): 1.1 2.2 3.3 50.5
"""

P2P_EXPECT = {
    'avg_gpu_to_gpu_p2p_unidir_bw': '33.9',
    'avg_gpu_to_gpu_p2p_bidir_bw': '43.9',
}


class TestCpuDeviceCount(unittest.TestCase):
    def test_mems_allowed_list_helios_r(self):
        self.assertEqual(tb.parse_cpu_device_count('0-1'), 2)

    def test_numeric_and_int(self):
        self.assertEqual(tb.parse_cpu_device_count(2), 2)
        self.assertEqual(tb.parse_cpu_device_count('2'), 2)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            tb.count_linux_id_list('')


class TestTransferBenchEnvAndCommand(unittest.TestCase):
    def test_config_num_cpu_devices_wins(self):
        env = tb.resolve_configured_tb_env({'num_cpu_devices': 2, 'env': {'FOO': 'bar'}})
        self.assertEqual(env['NUM_CPU_DEVICES'], '2')
        self.assertEqual(env['FOO'], 'bar')

    def test_env_num_cpu_devices_without_knob(self):
        env = tb.resolve_configured_tb_env({'env': {'NUM_CPU_DEVICES': '2'}})
        self.assertEqual(env['NUM_CPU_DEVICES'], '2')

    def test_command_exports_num_cpu_devices_inside_sudo(self):
        cmd = tb.build_transferbench_command(
            '/opt/amdtools/transferbench',
            '/opt/rocm',
            'p2p',
            {'NUM_CPU_DEVICES': '2'},
        )
        self.assertTrue(cmd.startswith('sudo bash -c '))
        self.assertIn('NUM_CPU_DEVICES', cmd)
        self.assertIn('TransferBench p2p', cmd)
        self.assertIn('LD_LIBRARY_PATH', cmd)

    def test_detect_command_counts_populated_cpulists(self):
        self.assertIn('for f in /sys/devices/system/node/node*/cpulist', tb._DETECT_NUM_CPU_DEVICES_CMD)
        self.assertIn('Mems_allowed_list', tb._DETECT_NUM_CPU_DEVICES_CMD)

    def test_auto_detect_when_hosts_agree(self):
        orch = MagicMock()
        orch.exec.return_value = {'nodeA': '2\n', 'nodeB': '2\n'}
        env = tb.resolve_runtime_tb_env(orch, {})
        self.assertEqual(env['NUM_CPU_DEVICES'], '2')
        orch.exec.assert_called_once()

    def test_config_skips_auto_detect(self):
        orch = MagicMock()
        env = tb.resolve_runtime_tb_env(orch, {'num_cpu_devices': 2})
        self.assertEqual(env['NUM_CPU_DEVICES'], '2')
        orch.exec.assert_not_called()

    def test_disagreeing_hosts_do_not_auto_set(self):
        orch = MagicMock()
        orch.exec.return_value = {'nodeA': '2\n', 'nodeB': '4\n'}
        env = tb.resolve_runtime_tb_env(orch, {})
        self.assertNotIn('NUM_CPU_DEVICES', env)


class TestParseTbP2pBw(unittest.TestCase):
    def test_helios_r_abort_fail_tests_without_attributeerror(self):
        with patch.object(tb, 'fail_test') as fail_test:
            tb.parse_tb_p2p_bw({'ctheliosr-rck-g02-k19-1': HELIOS_R_P2P_ABORT}, P2P_EXPECT)
        fail_test.assert_called_once()
        message = fail_test.call_args.args[0]
        self.assertIn('UniDir averages not found', message)
        self.assertIn('NUM_CPU_DEVICES', message)
        self.assertIn('cannot allocate memory', message)

    def test_success_fixture_does_not_fail(self):
        with patch.object(tb, 'fail_test') as fail_test:
            tb.parse_tb_p2p_bw({'nodeA': HELIOS_R_P2P_SUCCESS}, P2P_EXPECT)
        fail_test.assert_not_called()


class TestScanTestResultsNumaAbort(unittest.TestCase):
    @patch('cvs.lib.utils_lib.fail_test')
    def test_cpuset_allocation_abort_is_a_failure(self, mock_fail_test):
        from cvs.lib import utils_lib

        utils_lib.scan_test_results({'nodeA': HELIOS_R_P2P_ABORT})
        mock_fail_test.assert_called()
        self.assertIn('allocate', mock_fail_test.call_args.args[0].lower())


if __name__ == '__main__':
    unittest.main()
