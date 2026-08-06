"""Unit tests for the per-vendor NIC firmware/host-software preflight checks."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.nic_firmware_check import (
    AinicFirmwareCheck,
    BroadcomFirmwareCheck,
    MellanoxFirmwareCheck,
    NicFirmwareCheck,
)


def _ainic_output(nic_count, fw_lines, host_line):
    lines = ["VENDOR:AINIC", f"NIC_COUNT:{nic_count}"]
    lines.extend(fw_lines)
    lines.append(host_line)
    return "\n".join(lines)


def _vendor_iface_output(vendor_line, nic_count, fw_lines):
    lines = [vendor_line, f"NIC_COUNT:{nic_count}"]
    lines.extend(fw_lines)
    return "\n".join(lines)


class TestAinicFirmwareCheck(unittest.TestCase):
    def test_non_ainic_node_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:NOT_AINIC"}
        checker = AinicFirmwareCheck(phdl, expected_nic_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['vendor'], 'NOT_AINIC')
        self.assertEqual(results['node1']['errors'], [])

    def test_all_matching_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _ainic_output(
                8,
                ["FW:0:1.117.5-a-56:1.117.5-a-56"],
                "HOST:1.117.5-a-56:1.117.5-a-56",
            ),
        }
        checker = AinicFirmwareCheck(phdl, expected_nic_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'AINIC')
        self.assertEqual(results['node1']['errors'], [])
        self.assertEqual(results['node1']['warnings'], [])

    def test_nic_count_mismatch_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _ainic_output(
                4,
                ["FW:0:1.117.5-a-56:1.117.5-a-56"],
                "HOST:1.117.5-a-56:1.117.5-a-56",
            ),
        }
        checker = AinicFirmwareCheck(phdl, expected_nic_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 8 AINIC device(s), found 4', results['node1']['errors'][0])

    def test_firmware_version_mismatch_warning(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _ainic_output(
                8,
                ["FW:0:1.100.0-a-1:1.100.0-a-1"],
                "HOST:1.117.5-a-56:1.117.5-a-56",
            ),
        }
        checker = AinicFirmwareCheck(phdl, expected_nic_count=8, expected_fw_version="1.117.5-a-56")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('uboot=1.100.0-a-1' in w for w in results['node1']['warnings']))

    def test_host_software_version_mismatch_warning(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _ainic_output(
                8,
                ["FW:0:1.117.5-a-56:1.117.5-a-56"],
                "HOST:1.100.0-a-1:1.100.0-a-1",
            ),
        }
        checker = AinicFirmwareCheck(phdl, expected_nic_count=8, expected_host_version="1.117.5-a-56")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('host-software version' in w for w in results['node1']['warnings']))

    def test_normalized_version_equivalence_no_warning(self):
        # '1.117.5-a-56' normalizes the same as '11175a56' -- differing punctuation
        # should not trigger a host-software warning.
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _ainic_output(
                8,
                ["FW:0:1.117.5-a-56:1.117.5-a-56"],
                "HOST:1117.5a56:1117.5a56",
            ),
        }
        checker = AinicFirmwareCheck(phdl, expected_nic_count=8, expected_host_version="1.117.5-a-56")
        results = checker.run()

        self.assertNotIn('host-software', ' '.join(results['node1']['warnings']))

    def test_malformed_empty_output_fails_unparseable(self):
        # No VENDOR line at all (distinct from an explicit VENDOR:NOT_AINIC) falls
        # through to normal parsing, where nic_count stays 0 -> FAIL.
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = AinicFirmwareCheck(phdl, expected_nic_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 8 AINIC device(s), found 0', results['node1']['errors'][0])
        self.assertTrue(any('Unable to parse firmware' in w for w in results['node1']['warnings']))
        self.assertTrue(any('Unable to parse host-software' in w for w in results['node1']['warnings']))


class TestBroadcomFirmwareCheck(unittest.TestCase):
    def test_non_broadcom_node_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:NOT_BROADCOM"}
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['vendor'], 'NOT_BROADCOM')
        self.assertEqual(results['node1']['errors'], [])

    def test_all_matching_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _vendor_iface_output("VENDOR:BROADCOM", 2, ["FW:eth0:1.2.3", "FW:eth1:1.2.3"]),
        }
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2, expected_fw_version="1.2.3")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'BROADCOM')
        self.assertEqual(results['node1']['errors'], [])

    def test_nic_count_mismatch_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _vendor_iface_output("VENDOR:BROADCOM", 1, ["FW:eth0:1.2.3"]),
        }
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2, expected_fw_version="1.2.3")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 2 Broadcom bnxt RDMA device(s), found 1', results['node1']['errors'][0])

    def test_firmware_version_mismatch_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _vendor_iface_output("VENDOR:BROADCOM", 2, ["FW:eth0:9.9.9", "FW:eth1:1.2.3"]),
        }
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2, expected_fw_version="1.2.3")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('eth0: firmware=9.9.9' in w for w in results['node1']['warnings']))

    def test_malformed_empty_output_fails_unparseable(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 2 Broadcom bnxt RDMA device(s), found 0', results['node1']['errors'][0])
        self.assertTrue(
            any("Unable to parse firmware version output from 'ethtool -i'" in w for w in results['node1']['warnings'])
        )


class TestMellanoxFirmwareCheck(unittest.TestCase):
    def test_non_mellanox_node_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:NOT_MELLANOX"}
        checker = MellanoxFirmwareCheck(phdl, expected_nic_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['vendor'], 'NOT_MELLANOX')
        self.assertEqual(results['node1']['errors'], [])

    def test_all_matching_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _vendor_iface_output("VENDOR:MELLANOX", 1, ["FW:eth0:28.40.1000"]),
        }
        checker = MellanoxFirmwareCheck(phdl, expected_nic_count=1, expected_fw_version="28.40.1000")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'MELLANOX')
        self.assertEqual(results['node1']['errors'], [])

    def test_nic_count_mismatch_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _vendor_iface_output("VENDOR:MELLANOX", 1, ["FW:eth0:28.40.1000"]),
        }
        checker = MellanoxFirmwareCheck(phdl, expected_nic_count=8, expected_fw_version="28.40.1000")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 8 Mellanox mlx5 RDMA device(s), found 1', results['node1']['errors'][0])

    def test_malformed_empty_output_fails_unparseable(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = MellanoxFirmwareCheck(phdl, expected_nic_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 8 Mellanox mlx5 RDMA device(s), found 0', results['node1']['errors'][0])


class TestNicFirmwareCheckDispatcher(unittest.TestCase):
    def test_single_vendor_matches_underlying_check(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _ainic_output(
                8,
                ["FW:0:1.117.5-a-56:1.117.5-a-56"],
                "HOST:1.117.5-a-56:1.117.5-a-56",
            ),
        }
        checker = NicFirmwareCheck(phdl, nic_types=['ainic'], vendor_configs={'ainic': {'expected_nic_count': 8}})
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['ainic']['vendor'], 'AINIC')
        self.assertNotIn('broadcom', results['node1'])
        self.assertEqual(results['node1']['errors'], [])

    def test_multi_vendor_merges_fail_over_warning(self):
        phdl = MagicMock()

        def fake_exec(cmd):
            if 'nicctl' in cmd:
                return {
                    'node1': _ainic_output(
                        8,
                        ["FW:0:1.100.0-a-1:1.100.0-a-1"],
                        "HOST:1.117.5-a-56:1.117.5-a-56",
                    )
                }
            return {'node1': "VENDOR:NOT_BROADCOM"}

        phdl.exec.side_effect = fake_exec
        checker = NicFirmwareCheck(
            phdl, nic_types=['ainic', 'broadcom'], vendor_configs={'ainic': {'expected_fw_version': '1.117.5-a-56'}}
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['ainic']['status'], 'WARNING')
        self.assertEqual(results['node1']['broadcom']['status'], 'SKIPPED')

    def test_fail_takes_precedence_over_warning(self):
        phdl = MagicMock()

        def fake_exec(cmd):
            if 'nicctl' in cmd:
                return {
                    'node1': _ainic_output(
                        4,
                        ["FW:0:1.100.0-a-1:1.100.0-a-1"],
                        "HOST:1.117.5-a-56:1.117.5-a-56",
                    )
                }
            return {'node1': "VENDOR:NOT_BROADCOM"}

        phdl.exec.side_effect = fake_exec
        checker = NicFirmwareCheck(
            phdl, nic_types=['ainic', 'broadcom'], vendor_configs={'ainic': {'expected_nic_count': 8}}
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')

    def test_all_vendors_skipped_surfaces_skipped(self):
        phdl = MagicMock()
        phdl.exec.side_effect = lambda cmd: {
            'node1': "VENDOR:NOT_BROADCOM" if 'bnxt_re' in cmd else "VENDOR:NOT_MELLANOX"
        }
        checker = NicFirmwareCheck(phdl, nic_types=['broadcom', 'mellanox'])
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['errors'], [])


class TestNicFirmwareConfigVendorSubBlockValidation(unittest.TestCase):
    def _run_test_nic_firmware(self, config):
        from cvs.tests.preflight import preflight_checks

        previous_results = dict(preflight_checks.preflight_results)
        try:
            with patch.object(preflight_checks, 'preflight_update_test_result'):
                preflight_checks.test_nic_firmware(MagicMock(), config)
        finally:
            preflight_checks.preflight_results.clear()
            preflight_checks.preflight_results.update(previous_results)

    def test_non_dict_selected_vendor_subblock_raises_value_error(self):
        config = {
            'connectivity_check': {
                'ifoe': {'nic_firmware': {'enabled': True, 'nic_type': ['broadcom'], 'broadcom': 999}},
            },
        }
        with self.assertRaises(ValueError) as ctx:
            self._run_test_nic_firmware(config)
        self.assertIn('preflight.connectivity_check.ifoe.nic_firmware.broadcom must be an object', str(ctx.exception))

    def test_string_selected_vendor_subblock_raises_value_error(self):
        config = {
            'connectivity_check': {
                'ifoe': {'nic_firmware': {'enabled': True, 'nic_type': ['broadcom'], 'broadcom': 'oops'}},
            },
        }
        with self.assertRaises(ValueError) as ctx:
            self._run_test_nic_firmware(config)
        self.assertIn('preflight.connectivity_check.ifoe.nic_firmware.broadcom must be an object', str(ctx.exception))

    def test_non_selected_vendor_malformed_subblock_still_raises(self):
        config = {
            'connectivity_check': {
                'ifoe': {
                    'nic_firmware': {
                        'enabled': True,
                        'nic_type': ['broadcom'],
                        'broadcom': {'expected_nic_count': 2},
                        'mellanox': 'oops',
                    },
                },
            },
        }
        with self.assertRaises(ValueError) as ctx:
            self._run_test_nic_firmware(config)
        self.assertIn('preflight.connectivity_check.ifoe.nic_firmware.mellanox must be an object', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
