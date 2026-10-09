"""Unit tests for ibperf test collection."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from cvs.lib import ibperf_lib
from cvs.lib import globals
from cvs.tests.ibperf import ib_perf_bw_test


class TestIbPerfCollection(unittest.TestCase):
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
                ib_perf_bw_test.test_build_ib_lat_perf_chart(None)

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


if __name__ == '__main__':
    unittest.main()
