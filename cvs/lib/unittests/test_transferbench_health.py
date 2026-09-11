"""Unit tests for health TransferBench helpers (AIMVT-314)."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


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

A2A_OUTPUT = """\
RTotal 320.0 321.0 322.0 323.0 324.0 325.0 326.0 327.0
"""

P2P_MATRIX_OUTPUT = """\
NUM_CPU_DEVICES = 2 : Using 2 CPUs
Bytes Per Direction 268435456
Unidirectional copy peak bandwidth GB/s
GPU 00 -> 55.0 55.0 1600.0 48.0
GPU 01 -> 55.0 55.0 49.0 1700.0
Averages (During UniDir): 37.0 34.0 55.0 48.5
Bidirectional copy peak bandwidth GB/s
GPU 00 <-> 80.0 80.0 N/A 92.0
GPU 01 <-> 80.0 80.0 93.0 N/A
Averages (During BiDir): 70.0 70.0 80.0 92.5
"""

A2A_SWEEP_OUTPUT = """\
Blocksize: 256
#CUs\\Unroll 1(Min) 2(Min)
4 120.38 209.89
8 229.32 314.99
"""

SCALING_OUTPUT = """\
NumCUs CPU00 CPU01 GPU00 GPU01
1 20.22 20.41 18.68 25.91
32 57.15 56.35 493.17 49.23
Best 57.15( 32) 56.35( 32) 493.17( 32) 49.23( 32)
"""

SCHMOO_OUTPUT = """\
#CUs |G00->G00->N00|N00->G00->G00|G00->G00->G00|G01->G00->N00|N00->G00->G01|G00->G00->G01|
1 51.0 53.0 54.0 23.0 46.0 26.0
32 1700.0 1300.0 1300.0 49.0 49.0 49.0
"""


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


class TestTransferBenchReportMetrics(unittest.TestCase):
    def test_extracts_supported_bandwidth_shapes(self):
        a2a = tb.extract_tb_a2a_metrics({'nodeA': A2A_OUTPUT})
        p2p = tb.extract_tb_p2p_metrics({'nodeA': P2P_MATRIX_OUTPUT})
        sweep = tb.extract_tb_a2asweep_metrics({'nodeA': A2A_SWEEP_OUTPUT})
        scaling = tb.extract_tb_scaling_metrics({'nodeA': SCALING_OUTPUT})
        schmoo = tb.extract_tb_schmoo_metrics({'nodeA': SCHMOO_OUTPUT})

        self.assertEqual(a2a['a2a RTotal · nodeA']['0']['a2a_bw'], 320.0)
        self.assertEqual(p2p['p2p UniDir GPU 0 · nodeA']['1']['p2p_bw'], 48.0)
        self.assertEqual(sweep['a2asweep B256 U2 Min · nodeA']['8']['a2asweep_bw'], 314.99)
        self.assertEqual(scaling['scaling GPU00 · nodeA']['32']['scaling_bw'], 493.17)
        self.assertEqual(schmoo['schmoo Local Read · nodeA']['32']['schmoo_bw'], 1700.0)

    def test_a2a_test_populates_session_results_with_mocked_orch(self):
        orch = MagicMock()
        orch.exec.side_effect = [
            {'nodeA': '/opt/rocm/lib/libamdhip64.so.7'},
            {'nodeA': A2A_OUTPUT},
        ]
        config = {
            'path': '/opt/TransferBench',
            'rocm_path': '/opt/rocm',
            'num_cpu_devices': 2,
            'results': {'gpu_to_gpu_a2a_rtotal': '300.0'},
        }
        results = {}
        variant = {
            'rocm_path': '—',
            'tests_enabled': ['a2a'],
            'duration_seconds': 0.0,
        }

        tb.test_transfer_bench_a2a(orch, config, results, variant)

        self.assertIn('a2a RTotal · nodeA', results)
        self.assertEqual(variant['rocm_path'], '/opt/rocm')
        self.assertGreaterEqual(variant['duration_seconds'], 0.0)
        self.assertEqual(orch.exec.call_count, 2)

    def test_real_profile_builder_payload_and_render(self):
        results = {}
        variant = {
            'rocm_path': '/opt/rocm/core-7.2',
            'tests_enabled': list(tb._TB_TEST_NAMES),
            'duration_seconds': 12.25,
        }
        tb.globals.error_list = []
        tb.record_transferbench_results(results, variant, 'a2a', {'nodeA': A2A_OUTPUT}, 1.0)
        tb.record_transferbench_results(results, variant, 'p2p', {'nodeA': P2P_MATRIX_OUTPUT}, 2.0)
        tb.record_transferbench_results(results, variant, 'healthcheck', {'nodeA': 'PASS'}, 3.0)
        tb.record_transferbench_results(results, variant, 'a2asweep', {'nodeA': A2A_SWEEP_OUTPUT}, 4.0)
        tb.record_transferbench_results(results, variant, 'scaling', {'nodeA': SCALING_OUTPUT}, 5.0)
        tb.record_transferbench_results(results, variant, 'schmoo', {'nodeA': SCHMOO_OUTPUT}, 6.0)

        profile = load_json_profile('transferbench_cvs')
        payload = build_rundeck_payload(
            profile=profile,
            store={'cvs_results_dict': results, 'variant_config': variant},
            cvs_version='1.0.0',
        )
        document = render_rundeck_html(payload)

        self.assertFalse(profile['interactive_viewer'])
        self.assertEqual(
            payload['results_table']['headers'],
            ['Test', 'Node', 'Metric', 'GPU / bytes / CUs', 'Value', 'Unit', 'Status', 'Duration (s)'],
        )
        self.assertIn('a2asweep_bw', payload['datasets']['series']['charts'])
        self.assertIn('TransferBench Run Deck', document)
        self.assertIn('/opt/rocm/core-7.2', document)
        self.assertIn('A2A sweep bandwidth by CU count', document)
        self.assertIn('healthcheck', document)


class TestScanTestResultsNumaAbort(unittest.TestCase):
    @patch('cvs.lib.utils_lib.fail_test')
    def test_cpuset_allocation_abort_is_a_failure(self, mock_fail_test):
        from cvs.lib import utils_lib

        utils_lib.scan_test_results({'nodeA': HELIOS_R_P2P_ABORT})
        mock_fail_test.assert_called()
        self.assertIn('allocate', mock_fail_test.call_args.args[0].lower())


if __name__ == '__main__':
    unittest.main()
