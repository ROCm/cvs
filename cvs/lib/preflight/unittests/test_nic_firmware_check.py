"""Unit tests for the per-vendor NIC firmware/host-software preflight checks."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.nic_firmware_check import (
    AinicFirmwareCheck,
    BroadcomFirmwareCheck,
    MellanoxFirmwareCheck,
    NicFirmwareCheck,
)

# Real ``niccli --list`` output right-justifies the index column (e.g. "  1)", " 16)"),
# which caught out an earlier awk pattern anchored to '^[0-9]+\\)' -- it silently
# matched zero rows against real output while still passing against the
# hand-built ``_broadcom_output`` fixture below. This fixture is fed through the
# actual embedded shell/awk (via a stubbed niccli/lsmod/sudo on PATH) rather than
# mocking phdl.exec's return value, so a regression here fails the same way it
# did against real hardware.
_REAL_NICCLI_LIST_OUTPUT = """
     BoardId(Rev)    MAC Address        FwVersion    PCIAddr        Type   Mode
  1) BCM57608(B1)    22:D2:00:F3:1B:17  237.1.148.0  0000:06:00.0   NIC    PCI
  2) BCM57608(B1)    22:D2:00:F3:1B:17  237.1.148.0  0000:06:00.1   NIC    PCI
 10) BCM57608(B1)    DA:EB:78:11:87:A8  237.1.148.0  0000:86:00.1   NIC    PCI
 16) BCM57608(B1)    BE:DD:72:A3:1F:1A  237.1.148.0  0000:E6:00.1   NIC    PCI
"""


def _write_executable(path, contents):
    with open(path, 'w') as f:
        f.write(contents)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def _run_command_with_fake_niccli(command, niccli_list_output):
    """Execute ``command`` (a real shell/awk pipeline) with stubbed
    ``lsmod``/``sudo``/``niccli`` on PATH, so the actual parsing logic runs
    against realistic output instead of a hand-mocked parsed string."""
    with tempfile.TemporaryDirectory() as bindir:
        _write_executable(os.path.join(bindir, 'lsmod'), '#!/bin/bash\necho "bnxt_re 12345 0"\n')
        _write_executable(os.path.join(bindir, 'sudo'), '#!/bin/bash\nexec "$@"\n')
        _write_executable(
            os.path.join(bindir, 'niccli'),
            f'#!/bin/bash\nif [ "$1" = "--list" ]; then\ncat <<\'NICCLI_EOF\'\n{niccli_list_output}NICCLI_EOF\nfi\n',
        )
        env = dict(os.environ)
        env['PATH'] = f"{bindir}:{env['PATH']}"
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, env=env, check=False)
        return result.stdout


def _ainic_output(nic_count, fw_lines, host_line):
    lines = ["VENDOR:AINIC", f"NIC_COUNT:{nic_count}"]
    lines.extend(fw_lines)
    lines.append(host_line)
    return "\n".join(lines)


def _vendor_iface_output(vendor_line, nic_count, fw_lines):
    lines = [vendor_line, f"NIC_COUNT:{nic_count}"]
    lines.extend(fw_lines)
    return "\n".join(lines)


def _broadcom_output(fw_versions):
    lines = ["VENDOR:BROADCOM"]
    for idx, fw in enumerate(fw_versions):
        lines.append(f"FW:{idx}:{fw}:0000:0{idx}:00.0")
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
            'node1': _broadcom_output(["1.2.3", "1.2.3"]),
        }
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2, expected_fw_version="1.2.3")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'BROADCOM')
        self.assertEqual(results['node1']['errors'], [])

    def test_nic_count_mismatch_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _broadcom_output(["1.2.3"]),
        }
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2, expected_fw_version="1.2.3")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 2 Broadcom NIC(s), found 1', results['node1']['errors'][0])

    def test_firmware_version_mismatch_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': _broadcom_output(["9.9.9", "1.2.3"]),
        }
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2, expected_fw_version="1.2.3")
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('NIC 0' in w and 'firmware=9.9.9' in w for w in results['node1']['warnings']))

    def test_niccli_missing_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:BROADCOM\nNICCLI:MISSING"}
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('niccli not found' in w for w in results['node1']['warnings']))

    def test_malformed_empty_output_fails_unparseable(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:BROADCOM"}
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 2 Broadcom NIC(s), found 0', results['node1']['errors'][0])
        self.assertTrue(
            any(
                "Unable to parse firmware version output from 'niccli --list'" in w
                for w in results['node1']['warnings']
            )
        )

    def test_niccli_invoked_with_sudo_by_default(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _broadcom_output(["1.2.3", "1.2.3"])}
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2, expected_fw_version="1.2.3")
        checker.run()

        command = phdl.exec.call_args[0][0]
        self.assertIn('sudo niccli --list', command)

    def test_niccli_invoked_without_sudo_when_disabled(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _broadcom_output(["1.2.3", "1.2.3"])}
        checker = BroadcomFirmwareCheck(phdl, expected_nic_count=2, expected_fw_version="1.2.3", use_sudo=False)
        checker.run()

        command = phdl.exec.call_args[0][0]
        self.assertNotIn('sudo niccli --list', command)
        self.assertIn('niccli --list', command)

    def test_real_shell_command_parses_right_justified_niccli_index_column(self):
        checker = BroadcomFirmwareCheck(MagicMock(), expected_nic_count=4, expected_fw_version="237.1.148.0")
        output = _run_command_with_fake_niccli(checker._build_command(), _REAL_NICCLI_LIST_OUTPUT)
        result = checker._parse_and_evaluate(output)

        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['nic_count'], 4)
        self.assertEqual(result['errors'], [])


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
