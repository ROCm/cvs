"""Unit tests for the per-vendor NIC driver version preflight checks."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.nic_driver_version import (
    AinicDriverVersionCheck,
    BroadcomDriverVersionCheck,
    MellanoxDriverVersionCheck,
    NicDriverVersionCheck,
)


EXPECTED_PKG_VER = "233.0.150.0"
EXPECTED_FW_VER = "1.117.5-a-56"
EXPECTED_MLX5 = "24.10.1000"
EXPECTED_OFED = "MLNX_OFED_LINUX-24.10-1.1.4.0"

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


def _run_command_with_fake_niccli(command, niccli_list_output, pkg_version):
    """Execute ``command`` (the real for-loop/awk pipeline) with stubbed
    ``lsmod``/``sudo``/``niccli`` on PATH, so the actual parsing logic runs
    against realistic output instead of a hand-mocked parsed string. The fake
    ``niccli`` handles both ``--list`` (index enumeration) and
    ``-i <idx> show --pkg_ver`` (per-NIC package version) invocations."""
    with tempfile.TemporaryDirectory() as bindir:
        _write_executable(os.path.join(bindir, 'lsmod'), '#!/bin/bash\necho "bnxt_re 12345 0"\n')
        _write_executable(os.path.join(bindir, 'sudo'), '#!/bin/bash\nexec "$@"\n')
        _write_executable(
            os.path.join(bindir, 'niccli'),
            (
                '#!/bin/bash\n'
                'if [ "$1" = "--list" ]; then\n'
                f'cat <<\'NICCLI_EOF\'\n{niccli_list_output}NICCLI_EOF\n'
                'elif [ "$1" = "-i" ]; then\n'
                f'echo "Active Package Version : {pkg_version}"\n'
                'fi\n'
            ),
        )
        env = dict(os.environ)
        env['PATH'] = f"{bindir}:{env['PATH']}"
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, env=env, check=False)
        return result.stdout


def _broadcom_output(*pkg_versions):
    """Build fake ``niccli``-driven output: one ``PKG:<idx>:<version>`` line per NIC."""
    lines = ["VENDOR:BROADCOM"]
    lines.extend(f"PKG:{idx}:{version}" for idx, version in enumerate(pkg_versions, start=1))
    return "\n".join(lines)


def _ainic_output(*fw_versions):
    """Build fake ``nicctl show version firmware``-driven output: one
    ``FW:<nic>:<uboot>:<firmware>`` line per NIC (uboot/firmware always equal)."""
    lines = ["VENDOR:AINIC"]
    lines.extend(f"FW:ionic_{idx}:{version}:{version}" for idx, version in enumerate(fw_versions))
    return "\n".join(lines)


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

    def test_matching_package_version_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _broadcom_output(EXPECTED_PKG_VER, EXPECTED_PKG_VER)}
        checker = BroadcomDriverVersionCheck(phdl, expected_package_version=EXPECTED_PKG_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'BROADCOM')
        self.assertEqual(results['node1']['packages'], {'1': EXPECTED_PKG_VER, '2': EXPECTED_PKG_VER})
        self.assertEqual(results['node1']['errors'], [])

    def test_version_mismatch_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _broadcom_output("999.0.0.0")}
        checker = BroadcomDriverVersionCheck(phdl, expected_package_version=EXPECTED_PKG_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('NIC 1 package version=999.0.0.0' in e for e in results['node1']['errors']))

    def test_one_of_several_nics_mismatched_warns_only_that_nic(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _broadcom_output(EXPECTED_PKG_VER, "999.0.0.0")}
        checker = BroadcomDriverVersionCheck(phdl, expected_package_version=EXPECTED_PKG_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(len(results['node1']['errors']), 1)
        self.assertIn('NIC 2 package version=999.0.0.0', results['node1']['errors'][0])

    def test_malformed_empty_output_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = BroadcomDriverVersionCheck(phdl)
        results = checker.run()

        # No VENDOR field at all -> not BROADCOM -> SKIPPED (mirrors non-broadcom nodes).
        self.assertEqual(results['node1']['status'], 'SKIPPED')

    def test_niccli_missing_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:BROADCOM\nNICCLI:MISSING"}
        checker = BroadcomDriverVersionCheck(phdl, expected_package_version=EXPECTED_PKG_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['packages'], {})
        self.assertTrue(any('niccli not found' in e for e in results['node1']['errors']))

    def test_broadcom_vendor_with_no_packages_reported_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:BROADCOM"}
        checker = BroadcomDriverVersionCheck(phdl, expected_package_version=EXPECTED_PKG_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('niccli not found' in e for e in results['node1']['errors']))

    def test_niccli_invoked_with_sudo_by_default(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _broadcom_output(EXPECTED_PKG_VER)}
        checker = BroadcomDriverVersionCheck(phdl, expected_package_version=EXPECTED_PKG_VER)
        checker.run()

        command = phdl.exec.call_args[0][0]
        self.assertIn('sudo niccli --list', command)
        self.assertIn('sudo niccli -i', command)

    def test_niccli_invoked_without_sudo_when_disabled(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _broadcom_output(EXPECTED_PKG_VER)}
        checker = BroadcomDriverVersionCheck(phdl, expected_package_version=EXPECTED_PKG_VER, use_sudo=False)
        checker.run()

        command = phdl.exec.call_args[0][0]
        self.assertNotIn('sudo niccli --list', command)
        self.assertNotIn('sudo niccli -i', command)
        self.assertIn('niccli --list', command)
        self.assertIn('niccli -i', command)

    def test_real_shell_command_parses_right_justified_niccli_index_column(self):
        checker = BroadcomDriverVersionCheck(MagicMock(), expected_package_version=EXPECTED_PKG_VER)
        output = _run_command_with_fake_niccli(checker._build_command(), _REAL_NICCLI_LIST_OUTPUT, EXPECTED_PKG_VER)
        result = checker._parse_and_evaluate(output)

        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(
            result['packages'],
            {'1': EXPECTED_PKG_VER, '2': EXPECTED_PKG_VER, '10': EXPECTED_PKG_VER, '16': EXPECTED_PKG_VER},
        )
        self.assertEqual(result['errors'], [])


class TestAinicDriverVersionCheck(unittest.TestCase):
    def test_non_ainic_node_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:NOT_AINIC"}
        checker = AinicDriverVersionCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['vendor'], 'NOT_AINIC')
        self.assertEqual(results['node1']['errors'], [])

    def test_matching_fw_version_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _ainic_output(EXPECTED_FW_VER, EXPECTED_FW_VER)}
        checker = AinicDriverVersionCheck(phdl, expected_fw_version=EXPECTED_FW_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['vendor'], 'AINIC')
        self.assertEqual(
            results['node1']['fw_entries'],
            [
                {'nic': 'ionic_0', 'uboot': EXPECTED_FW_VER, 'firmware': EXPECTED_FW_VER},
                {'nic': 'ionic_1', 'uboot': EXPECTED_FW_VER, 'firmware': EXPECTED_FW_VER},
            ],
        )
        self.assertEqual(results['node1']['errors'], [])

    def test_version_mismatch_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _ainic_output("0.0.0-a-1")}
        checker = AinicDriverVersionCheck(phdl, expected_fw_version=EXPECTED_FW_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('NIC ionic_0' in e and 'uboot=0.0.0-a-1' in e for e in results['node1']['errors']))

    def test_one_of_several_nics_mismatched_warns_only_that_nic(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': _ainic_output(EXPECTED_FW_VER, "0.0.0-a-1")}
        checker = AinicDriverVersionCheck(phdl, expected_fw_version=EXPECTED_FW_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(len(results['node1']['errors']), 1)
        self.assertIn('NIC ionic_1', results['node1']['errors'][0])

    def test_malformed_empty_output_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = AinicDriverVersionCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'SKIPPED')

    def test_nicctl_missing_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:AINIC\nNICCTL:MISSING"}
        checker = AinicDriverVersionCheck(phdl, expected_fw_version=EXPECTED_FW_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['fw_entries'], [])
        self.assertTrue(any('nicctl not found' in e for e in results['node1']['errors']))

    def test_ainic_vendor_with_no_nics_reported_warns(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': "VENDOR:AINIC"}
        checker = AinicDriverVersionCheck(phdl, expected_fw_version=EXPECTED_FW_VER)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('nicctl not found' in e for e in results['node1']['errors']))


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
        phdl.exec.return_value = {'node1': _broadcom_output(EXPECTED_PKG_VER)}
        checker = NicDriverVersionCheck(
            phdl,
            nic_types=['broadcom'],
            vendor_configs={'broadcom': {'expected_package_version': EXPECTED_PKG_VER}},
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['broadcom']['vendor'], 'BROADCOM')
        self.assertNotIn('ainic', results['node1'])
        self.assertEqual(results['node1']['errors'], [])

    def test_multi_vendor_merges_fail_over_warning(self):
        phdl = MagicMock()

        def fake_exec(cmd):
            if 'niccli' in cmd:
                return {'node1': _broadcom_output("999.0.0.0")}
            if 'nicctl' in cmd:
                return {'node1': "VENDOR:NOT_AINIC"}
            return {'node1': ''}

        phdl.exec.side_effect = fake_exec
        checker = NicDriverVersionCheck(
            phdl,
            nic_types=['ainic', 'broadcom'],
            vendor_configs={'broadcom': {'expected_package_version': EXPECTED_PKG_VER}},
        )
        results = checker.run()

        # broadcom WARNING, ainic SKIPPED (no hardware) -> merged WARNING, not all-SKIPPED.
        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['ainic']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['broadcom']['status'], 'WARNING')
        self.assertTrue(any('NIC 1 package version=999.0.0.0' in e for e in results['node1']['errors']))

    def test_all_vendors_skipped_surfaces_skipped(self):
        phdl = MagicMock()
        phdl.exec.side_effect = lambda cmd: {'node1': "VENDOR:NOT_AINIC" if 'nicctl' in cmd else "VENDOR:NOT_BROADCOM"}
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
                    'broadcom': {'expected_package_version': EXPECTED_PKG_VER},
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
                    'broadcom': {'expected_package_version': '233.0.150.0'},
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
                    'broadcom': {'expected_package_version': '<changeme>'},
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
                    'ainic': {'expected_fw_version': '<changeme>'},
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
