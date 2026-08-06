"""Unit tests for the per-vendor NIC driver version preflight checks."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.nic_driver_version import (
    AinicDriverVersionCheck,
    BroadcomDriverVersionCheck,
    MellanoxDriverVersionCheck,
    NicDriverVersionCheck,
)


EXPECTED_RE = "236.1.155.0"
EXPECTED_EN = "1.10.3-236.1.155.0"
EXPECTED_IONIC = "1.117.5-a-56"
EXPECTED_IONIC_RDMA = "1.117.5-a-56"
EXPECTED_MLX5 = "24.10.1000"
EXPECTED_OFED = "MLNX_OFED_LINUX-24.10-1.1.4.0"


def _broadcom_output(re_version, re_file, en_version, en_file):
    return "\n".join(
        [
            "VENDOR:BROADCOM",
            f"RE_VERSION:{re_version}",
            f"RE_FILE:{re_file}",
            f"EN_VERSION:{en_version}",
            f"EN_FILE:{en_file}",
        ]
    )


def _ainic_output(ionic_version, ionic_rdma_version):
    return "\n".join(
        [
            "VENDOR:AINIC",
            f"IONIC_VERSION:{ionic_version}",
            f"IONIC_RDMA_VERSION:{ionic_rdma_version}",
        ]
    )


def _mellanox_output(mlx5_version, ofed_version):
    return "\n".join(
        [
            "VENDOR:MELLANOX",
            f"MLX5_CORE_VERSION:{mlx5_version}",
            f"OFED_VERSION:{ofed_version}",
        ]
    )


class TestBroadcomDriverVersionCheck(unittest.TestCase):
    def test_non_broadcom_node_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:NOT_BROADCOM"}
        checker = BroadcomDriverVersionCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['vendor'], 'NOT_BROADCOM')
        self.assertEqual(results['node1']['errors'], [])

    def test_matching_dkms_versions_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _broadcom_output(
                EXPECTED_RE,
                "/lib/modules/5.15.0/updates/dkms/bnxt_re.ko",
                EXPECTED_EN,
                "/lib/modules/5.15.0/updates/dkms/bnxt_en.ko",
            ),
        }
        checker = BroadcomDriverVersionCheck(
            phdl, expected_bnxt_re_version=EXPECTED_RE, expected_bnxt_en_version=EXPECTED_EN
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'BROADCOM')
        self.assertTrue(results['node1']['bnxt_re']['dkms'])
        self.assertTrue(results['node1']['bnxt_en']['dkms'])
        self.assertEqual(results['node1']['errors'], [])

    def test_version_mismatch_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _broadcom_output(
                "999.0.0.0",
                "/lib/modules/5.15.0/updates/dkms/bnxt_re.ko",
                EXPECTED_EN,
                "/lib/modules/5.15.0/updates/dkms/bnxt_en.ko",
            ),
        }
        checker = BroadcomDriverVersionCheck(
            phdl, expected_bnxt_re_version=EXPECTED_RE, expected_bnxt_en_version=EXPECTED_EN
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('bnxt_re version=999.0.0.0' in e for e in results['node1']['errors']))

    def test_non_dkms_module_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _broadcom_output(
                EXPECTED_RE,
                "/lib/modules/5.15.0/kernel/drivers/infiniband/hw/bnxt_re/bnxt_re.ko",
                EXPECTED_EN,
                "/lib/modules/5.15.0/updates/dkms/bnxt_en.ko",
            ),
        }
        checker = BroadcomDriverVersionCheck(
            phdl, expected_bnxt_re_version=EXPECTED_RE, expected_bnxt_en_version=EXPECTED_EN
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('dkms=False' in e for e in results['node1']['errors']))

    def test_malformed_empty_output_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = BroadcomDriverVersionCheck(phdl)
        results = checker.run()

        # No VENDOR field at all -> not BROADCOM -> SKIPPED (mirrors non-broadcom nodes).
        self.assertEqual(results['node1']['status'], 'SKIPPED')

    def test_broadcom_vendor_with_missing_modinfo_fields_warns_unknown(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "VENDOR:BROADCOM\nRE_VERSION:\nRE_FILE:\nEN_VERSION:\nEN_FILE:",
        }
        checker = BroadcomDriverVersionCheck(
            phdl, expected_bnxt_re_version=EXPECTED_RE, expected_bnxt_en_version=EXPECTED_EN
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('UNKNOWN' in e for e in results['node1']['errors']))


class TestAinicDriverVersionCheck(unittest.TestCase):
    def test_non_ainic_node_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:NOT_AINIC"}
        checker = AinicDriverVersionCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['vendor'], 'NOT_AINIC')
        self.assertEqual(results['node1']['errors'], [])

    def test_matching_versions_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _ainic_output(EXPECTED_IONIC, EXPECTED_IONIC_RDMA)}
        checker = AinicDriverVersionCheck(
            phdl,
            expected_ionic_driver_version=EXPECTED_IONIC,
            expected_ionic_rdma_driver_version=EXPECTED_IONIC_RDMA,
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'AINIC')
        self.assertEqual(results['node1']['errors'], [])

    def test_version_mismatch_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _ainic_output("0.0.0-a-1", EXPECTED_IONIC_RDMA)}
        checker = AinicDriverVersionCheck(
            phdl,
            expected_ionic_driver_version=EXPECTED_IONIC,
            expected_ionic_rdma_driver_version=EXPECTED_IONIC_RDMA,
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('ionic version=0.0.0-a-1' in e for e in results['node1']['errors']))

    def test_malformed_empty_output_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = AinicDriverVersionCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')


class TestMellanoxDriverVersionCheck(unittest.TestCase):
    def test_non_mellanox_node_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:NOT_MELLANOX"}
        checker = MellanoxDriverVersionCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['vendor'], 'NOT_MELLANOX')
        self.assertEqual(results['node1']['errors'], [])

    def test_matching_versions_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _mellanox_output(EXPECTED_MLX5, EXPECTED_OFED)}
        checker = MellanoxDriverVersionCheck(
            phdl, expected_mlx5_core_version=EXPECTED_MLX5, expected_ofed_version=EXPECTED_OFED
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'MELLANOX')
        self.assertEqual(results['node1']['errors'], [])

    def test_version_mismatch_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _mellanox_output("0.0.0", EXPECTED_OFED)}
        checker = MellanoxDriverVersionCheck(
            phdl, expected_mlx5_core_version=EXPECTED_MLX5, expected_ofed_version=EXPECTED_OFED
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('mlx5_core version=0.0.0' in e for e in results['node1']['errors']))

    def test_malformed_empty_output_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = MellanoxDriverVersionCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')


class TestNicDriverVersionCheckDispatcher(unittest.TestCase):
    def test_single_vendor_matches_underlying_check(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _broadcom_output(
                EXPECTED_RE,
                "/lib/modules/5.15.0/updates/dkms/bnxt_re.ko",
                EXPECTED_EN,
                "/lib/modules/5.15.0/updates/dkms/bnxt_en.ko",
            ),
        }
        checker = NicDriverVersionCheck(
            phdl,
            nic_types=['broadcom'],
            vendor_configs={
                'broadcom': {'expected_bnxt_re_version': EXPECTED_RE, 'expected_bnxt_en_version': EXPECTED_EN}
            },
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['broadcom']['vendor'], 'BROADCOM')
        self.assertNotIn('ainic', results['node1'])
        self.assertEqual(results['node1']['errors'], [])

    def test_multi_vendor_merges_fail_over_warning(self):
        phdl = MagicMock()

        def fake_exec(cmd):
            if 'bnxt_re' in cmd:
                return {'node1': _broadcom_output("999.0.0.0", "dkms/bnxt_re.ko", EXPECTED_EN, "dkms/bnxt_en.ko")}
            if 'ionic' in cmd:
                return {'node1': "VENDOR:NOT_AINIC"}
            return {'node1': ''}

        phdl.exec.side_effect = fake_exec
        checker = NicDriverVersionCheck(phdl, nic_types=['ainic', 'broadcom'])
        results = checker.run()

        # broadcom WARNING, ainic SKIPPED (no hardware) -> merged WARNING, not all-SKIPPED.
        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['ainic']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['broadcom']['status'], 'WARNING')
        self.assertTrue(any('bnxt_re version=999.0.0.0' in e for e in results['node1']['errors']))

    def test_all_vendors_skipped_surfaces_skipped(self):
        phdl = MagicMock()
        phdl.exec.side_effect = lambda cmd: {'node1': "VENDOR:NOT_AINIC" if 'ionic' in cmd else "VENDOR:NOT_BROADCOM"}
        checker = NicDriverVersionCheck(phdl, nic_types=['ainic', 'broadcom'])
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['errors'], [])


class TestNicDriverVersionConfigVendorSubBlockValidation(unittest.TestCase):
    def _run_test_nic_driver_version(self, config):
        from cvs.tests.preflight import preflight_checks

        previous_results = dict(preflight_checks.preflight_results)
        try:
            with patch.object(preflight_checks, 'preflight_update_test_result'):
                preflight_checks.test_nic_driver_version(MagicMock(), config)
        finally:
            preflight_checks.preflight_results.clear()
            preflight_checks.preflight_results.update(previous_results)

    def test_non_dict_selected_vendor_subblock_raises_value_error(self):
        config = {
            'node_check': {
                'nic_driver_version': {'enabled': True, 'nic_type': ['broadcom'], 'broadcom': 999},
            },
        }
        with self.assertRaises(ValueError) as ctx:
            self._run_test_nic_driver_version(config)
        self.assertIn('preflight.node_check.nic_driver_version.broadcom must be an object', str(ctx.exception))

    def test_string_selected_vendor_subblock_raises_value_error(self):
        config = {
            'node_check': {
                'nic_driver_version': {'enabled': True, 'nic_type': ['broadcom'], 'broadcom': 'oops'},
            },
        }
        with self.assertRaises(ValueError) as ctx:
            self._run_test_nic_driver_version(config)
        self.assertIn('preflight.node_check.nic_driver_version.broadcom must be an object', str(ctx.exception))

    def test_non_selected_vendor_malformed_subblock_still_raises(self):
        config = {
            'node_check': {
                'nic_driver_version': {
                    'enabled': True,
                    'nic_type': ['broadcom'],
                    'broadcom': {'expected_bnxt_re_version': EXPECTED_RE},
                    'mellanox': 'oops',
                },
            },
        }
        with self.assertRaises(ValueError) as ctx:
            self._run_test_nic_driver_version(config)
        self.assertIn('preflight.node_check.nic_driver_version.mellanox must be an object', str(ctx.exception))


class TestInertNicVendorSkipPaths(unittest.TestCase):
    """_inert_nic_vendor_skip_paths() computes which vendor sub-blocks are safe to
    exempt from the <changeme>-placeholder check, since they're present but unused."""

    def _skip_paths(self, config_dict):
        from cvs.tests.preflight import preflight_checks

        return preflight_checks._inert_nic_vendor_skip_paths(config_dict)

    def test_unselected_present_vendor_is_skipped(self):
        config = {
            'node_check': {
                'nic_driver_version': {
                    'nic_type': ['broadcom'],
                    'broadcom': {'expected_bnxt_re_version': '236.1.155.0'},
                    'mellanox': {'expected_mlx5_core_version': '<changeme>'},
                },
            },
        }
        self.assertEqual(self._skip_paths(config), {'node_check.nic_driver_version.mellanox'})

    def test_selected_vendor_is_not_skipped(self):
        config = {
            'node_check': {
                'nic_driver_version': {
                    'nic_type': ['broadcom'],
                    'broadcom': {'expected_bnxt_re_version': '<changeme>'},
                },
            },
        }
        self.assertEqual(self._skip_paths(config), set())

    def test_absent_vendor_block_not_included(self):
        config = {
            'node_check': {
                'nic_driver_version': {'nic_type': ['broadcom'], 'broadcom': {}},
            },
        }
        self.assertEqual(self._skip_paths(config), set())

    def test_defaults_used_when_nic_type_absent(self):
        # nic_driver_version defaults nic_type to ['broadcom'] when absent, matching
        # _validate_nic_type's own default, so ainic/mellanox are still inert here.
        config = {
            'node_check': {
                'nic_driver_version': {
                    'ainic': {'expected_ionic_driver_version': '<changeme>'},
                },
            },
        }
        self.assertEqual(self._skip_paths(config), {'node_check.nic_driver_version.ainic'})

    def test_both_blocks_computed_independently(self):
        config = {
            'node_check': {
                'nic_driver_version': {
                    'nic_type': ['broadcom'],
                    'mellanox': {'expected_mlx5_core_version': '<changeme>'},
                },
            },
            'connectivity_check': {
                'ifoe': {
                    'nic_firmware': {
                        'nic_type': ['ainic'],
                        'broadcom': {'expected_fw_version': '<changeme>'},
                    },
                },
            },
        }
        self.assertEqual(
            self._skip_paths(config),
            {
                'node_check.nic_driver_version.mellanox',
                'connectivity_check.ifoe.nic_firmware.broadcom',
            },
        )

    def test_malformed_config_shape_does_not_raise(self):
        self.assertEqual(self._skip_paths({}), set())
        self.assertEqual(self._skip_paths({'node_check': 'not-a-dict'}), set())
        self.assertEqual(self._skip_paths({'node_check': {'nic_driver_version': {'nic_type': 'not-a-list'}}}), set())


if __name__ == '__main__':
    unittest.main()
