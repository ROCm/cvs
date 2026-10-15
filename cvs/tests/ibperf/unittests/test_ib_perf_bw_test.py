"""Unit tests for ibperf test collection."""

import json
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from cvs.lib import ibperf_lib
from cvs.lib import globals
from cvs.tests.ibperf import ib_perf_bw_test

_CONFIG = {
    'install_dir': '/opt/ibperf',
    'msg_size_list': [65536],
    'qp_count_list': [8],
    'gid_index': '3',
    'port_no': '1',
    'duration': '10',
    'verify_bw': 'False',
}
_PAIRING = {'pairs': [('n1', 'n2')], 'errors': []}


class TestIbPerfCollection(unittest.TestCase):
    def test_perf_tests_take_node_pairing(self):
        for test_function in (ib_perf_bw_test.test_ib_bw_perf, ib_perf_bw_test.test_ib_lat_perf):
            self.assertIn('node_pairing', inspect.signature(test_function).parameters)
        self.assertFalse(hasattr(ib_perf_bw_test, 'vpc_node_list'))

    def test_shipped_sample_config_pairing_defaults(self):
        config_path = Path(__file__).resolve().parents[3] / 'input' / 'config_file' / 'ibperf' / 'ibperf_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))['ibperf']
        self.assertEqual(config['pairing_mode'], 'sequential')
        self.assertEqual(config['vpod_source'], 'afm')
        self.assertEqual(config['afmctl_path'], 'afmctl')
        self.assertEqual(ibperf_lib.resolve_ibperf_node_pairing(['a', 'b'], config, {})['pairs'], [('a', 'b')])

    def _metafunc(self, config_path, fixturenames):
        return SimpleNamespace(
            config=SimpleNamespace(getoption=lambda _name: config_path),
            fixturenames=fixturenames,
            parametrize=Mock(),
        )

    def test_latency_chart_fails_cleanly_when_test_has_no_rows(self):
        results = {'ib_read_lat': {8192: {'server': {}, 'client': {}}}}
        with (
            patch.object(ib_perf_bw_test, 'ib_lat_dict', results),
            patch.object(globals, 'error_list', []),
        ):
            with self.assertRaisesRegex(BaseException, 'No latency results for ib_read_lat'):
                ib_perf_bw_test.test_build_ib_lat_perf_chart()

    def test_bw_and_lat_parametrize_from_config(self):
        config = {
            'ibperf': {
                'ib_bw_test_list': ['ib_write_bw'],
                'ib_lat_test_list': ['ib_write_lat', 'ib_read_lat'],
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'ibperf.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')

            bw_metafunc = self._metafunc(str(config_path), ['shdl', 'phdl', 'bw_test', 'config_dict'])
            ib_perf_bw_test.pytest_generate_tests(bw_metafunc)
            self.assertEqual(bw_metafunc.parametrize.call_args_list, [call('bw_test', ['ib_write_bw'])])

            lat_metafunc = self._metafunc(str(config_path), ['shdl', 'phdl', 'lat_test', 'config_dict'])
            ib_perf_bw_test.pytest_generate_tests(lat_metafunc)
            self.assertEqual(
                lat_metafunc.parametrize.call_args_list, [call('lat_test', ['ib_write_lat', 'ib_read_lat'])]
            )

    def test_missing_keys_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'ibperf.json'
            config_path.write_text(json.dumps({'ibperf': {}}), encoding='utf-8')
            for fixture_name, expected in (
                ('bw_test', ibperf_lib.DEFAULT_BW_TESTS),
                ('lat_test', ibperf_lib.DEFAULT_LAT_TESTS),
            ):
                with self.subTest(fixture_name=fixture_name):
                    metafunc = self._metafunc(str(config_path), [fixture_name])
                    ib_perf_bw_test.pytest_generate_tests(metafunc)
                    self.assertEqual(metafunc.parametrize.call_args_list, [call(fixture_name, expected)])

    def test_missing_config_file_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = str(Path(temp_dir) / 'missing.json')
            for fixture_name, expected in (
                ('bw_test', ibperf_lib.DEFAULT_BW_TESTS),
                ('lat_test', ibperf_lib.DEFAULT_LAT_TESTS),
            ):
                with self.subTest(fixture_name=fixture_name):
                    metafunc = self._metafunc(config_path, [fixture_name])
                    ib_perf_bw_test.pytest_generate_tests(metafunc)
                    self.assertEqual(metafunc.parametrize.call_args_list, [call(fixture_name, expected)])

    def test_unrelated_tests_not_parametrized(self):
        metafunc = self._metafunc(None, ['phdl'])
        ib_perf_bw_test.pytest_generate_tests(metafunc)
        metafunc.parametrize.assert_not_called()

    def test_invalid_name_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'ibperf.json'
            config_path.write_text(json.dumps({'ibperf': {'ib_bw_test_list': ['ib_bogus_bw']}}), encoding='utf-8')
            metafunc = self._metafunc(str(config_path), ['bw_test'])
            with self.assertRaisesRegex(ValueError, 'ib_bw_test_list.*ib_bogus_bw'):
                ib_perf_bw_test.pytest_generate_tests(metafunc)

    def test_shipped_sample_config_is_honored(self):
        config_path = Path(__file__).resolve().parents[3] / 'input' / 'config_file' / 'ibperf' / 'ibperf_config.json'
        ibperf_config = json.loads(config_path.read_text(encoding='utf-8'))['ibperf']

        bw_metafunc = self._metafunc(str(config_path), ['bw_test'])
        ib_perf_bw_test.pytest_generate_tests(bw_metafunc)
        self.assertEqual(bw_metafunc.parametrize.call_args_list, [call('bw_test', ibperf_config['ib_bw_test_list'])])
        self.assertNotIn('ib_read_bw', ibperf_config['ib_bw_test_list'])

        lat_metafunc = self._metafunc(str(config_path), ['lat_test'])
        ib_perf_bw_test.pytest_generate_tests(lat_metafunc)
        self.assertEqual(lat_metafunc.parametrize.call_args_list, [call('lat_test', ibperf_config['ib_lat_test_list'])])

    def test_no_hardcoded_parametrize_marks(self):
        for test_function in (ib_perf_bw_test.test_ib_bw_perf, ib_perf_bw_test.test_ib_lat_perf):
            with self.subTest(test_function=test_function.__name__):
                marks = [mark.name for mark in getattr(test_function, 'pytestmark', [])]
                self.assertNotIn('parametrize', marks)


class TestIbBwPerf(unittest.TestCase):
    @patch.object(ib_perf_bw_test, 'update_test_result')
    @patch.object(ib_perf_bw_test, 'verify_dmesg_for_errors')
    @patch.object(ib_perf_bw_test, 'ibperf_lib')
    @patch.object(ib_perf_bw_test, 'linux_utils')
    def test_runs_on_the_orch_all_handle(self, linux_utils, ibperf_lib, verify_dmesg, _update):
        orch = MagicMock()
        linux_utils.get_active_rdma_nic_dict.return_value = {}
        ib_perf_bw_test.test_ib_bw_perf(orch, 'ib_write_bw', _CONFIG, MagicMock(), {}, {}, _PAIRING)
        linux_utils.get_gpu_nic_mapping_dict.assert_called_once_with(orch.all, use_sudo=False)
        linux_utils.get_gpu_numa_dict.assert_called_once_with(orch.all, use_sudo=False)
        linux_utils.get_backend_nic_dict.assert_called_once_with(orch.all, use_sudo=False)
        self.assertIs(ibperf_lib.run_ib_perf_bw_test.call_args.args[0], orch.all)
        self.assertEqual(ibperf_lib.run_ib_perf_bw_test.call_args.kwargs['node_pairs'], _PAIRING['pairs'])
        self.assertIs(verify_dmesg.call_args.args[0], orch.all)

    @patch.object(ib_perf_bw_test, 'update_test_result')
    @patch.object(ib_perf_bw_test, 'get_passwordless_sudo_status', return_value={'n1': True, 'n2': False})
    @patch.object(ib_perf_bw_test, 'verify_dmesg_for_errors')
    @patch.object(ib_perf_bw_test, 'ibperf_lib')
    @patch.object(ib_perf_bw_test, 'linux_utils')
    def test_skips_dmesg_check_with_warning_without_sudo(self, linux_utils, _ibperf_lib, verify_dmesg, *_):
        linux_utils.get_active_rdma_nic_dict.return_value = {}
        orch = MagicMock()
        with self.assertLogs(ib_perf_bw_test.log, 'WARNING') as logs:
            ib_perf_bw_test.test_ib_bw_perf(orch, 'ib_write_bw', _CONFIG, MagicMock(), {}, {}, _PAIRING)
        verify_dmesg.assert_not_called()
        self.assertFalse(any('/dev/kmsg' in c.args[0] for c in orch.all.exec.call_args_list))
        self.assertIn("SKIPPING the dmesg error check", logs.output[0])

    @patch.object(ib_perf_bw_test, 'update_test_result')
    @patch.object(ib_perf_bw_test, 'verify_dmesg_for_errors')
    @patch.object(ib_perf_bw_test, 'ibperf_lib')
    @patch.object(ib_perf_bw_test, 'linux_utils')
    def test_records_each_run_for_the_run_deck(self, linux_utils, ibperf_lib, *_):
        linux_utils.get_active_rdma_nic_dict.return_value = {}
        linux_utils.get_gpu_nic_mapping_dict.return_value = {'n1': {'card0': {}, 'card1': {}}}
        ibperf_lib.check_perftest_dmabuf.return_value = True
        ibperf_lib.run_ib_perf_bw_test.return_value = {'n1': {0: {'bw': '390.5', 'pps': '0.7'}}}
        results, variant = {}, {}
        ib_perf_bw_test.test_ib_bw_perf(MagicMock(), 'ib_write_bw', _CONFIG, MagicMock(), results, variant, _PAIRING)
        self.assertEqual(results['_ibperf']['bw'], {'ib_write_bw|8|65536': {'n1': {'0': {'bw': 390.5}}}})
        self.assertEqual(variant, {'dmabuf': True, 'nic_count': 2})


class TestUnmappedGpuGuard(unittest.TestCase):
    @patch.object(ib_perf_bw_test, 'get_passwordless_sudo_status', return_value={'n1': True})
    @patch.object(ib_perf_bw_test, 'ibperf_lib')
    @patch.object(ib_perf_bw_test, 'linux_utils')
    def test_unmapped_card_fails_before_perftest(self, linux_utils, ibperf_lib, _sudo):
        ibperf_lib.check_perftest_dmabuf.return_value = True
        linux_utils.get_gpu_nic_mapping_dict.return_value = {
            'n1': {
                'card0': {'gpu_bdf': '0001:01:00.0', 'rdma_dev': 'rdma0'},
                'card1': {'gpu_bdf': '0002:01:00.0'},
            }
        }
        for test_function, test_name in (
            (ib_perf_bw_test.test_ib_bw_perf, 'ib_write_bw'),
            (ib_perf_bw_test.test_ib_lat_perf, 'ib_write_lat'),
        ):
            with self.subTest(test=test_name):
                orch = MagicMock()
                with self.assertRaisesRegex(BaseException, r'card1 \(GPU 0002:01:00.0\) on node n1'):
                    test_function(orch, test_name, _CONFIG, MagicMock(), {}, {}, _PAIRING)
                linux_utils.get_gpu_nic_mapping_dict.assert_called_with(orch.all, use_sudo=False)
        ibperf_lib.get_hip_device_dict.assert_not_called()
        ibperf_lib.run_ib_perf_bw_test.assert_not_called()
        ibperf_lib.run_ib_perf_lat_test.assert_not_called()


class TestPairingFixtures(unittest.TestCase):
    _CLUSTER = {'username': 'user', 'priv_key_file': '/keys/id', 'node_dict': {n: {} for n in ('n1', 'n2', 'n3')}}

    @patch.object(ib_perf_bw_test.OrchestratorFactory, 'create_orchestrator')
    def test_orch_covers_only_paired_nodes(self, create):
        gen = ib_perf_bw_test.orch.__wrapped__(self._CLUSTER, {'pairs': [('n3', 'n1')], 'errors': []})
        self.assertIs(next(gen), create.return_value)
        self.assertEqual(list(create.call_args.args[1].node_dict), ['n3', 'n1'])
        with self.assertRaises(StopIteration):
            next(gen)
        create.return_value.close.assert_called_once_with()

    @patch.object(ib_perf_bw_test.OrchestratorFactory, 'create_orchestrator')
    def test_node_pairing_discovers_on_a_closed_orch_handle(self, create):
        def resolve(node_list, _config, _cluster, phdl_factory):
            self.assertIs(phdl_factory(node_list), create.return_value.all)
            raise ValueError('no pairs')

        with patch.object(ib_perf_bw_test.ibperf_lib, 'resolve_ibperf_node_pairing', side_effect=resolve):
            with self.assertRaises(ValueError):
                ib_perf_bw_test.node_pairing.__wrapped__(self._CLUSTER, {})
        self.assertEqual(list(create.call_args.args[1].node_dict), ['n1', 'n2', 'n3'])
        create.return_value.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
