'''Command shape for the rocm-smi/amd-smi readers used by host checks.'''

import json
import unittest
from unittest.mock import MagicMock

from cvs.lib import globals, rocm_plib


class TestAmdSmiJsonCommand(unittest.TestCase):
    def test_default_keeps_sudo_login_shell(self):
        cmd = rocm_plib._amd_smi_json_command('firmware')
        self.assertTrue(cmd.startswith('sudo bash -lc '))
        self.assertIn('firmware --json', cmd)

    def test_without_sudo_still_resolves_amd_smi(self):
        cmd = rocm_plib._amd_smi_json_command('firmware', use_sudo=False)
        self.assertTrue(cmd.startswith('bash -lc '))
        self.assertNotIn('sudo', cmd)
        self.assertIn('/opt/rocm/bin/amd-smi', cmd)

    def test_pcie_bus_map_prefers_rocm_smi_over_amd_smi(self):
        cmd = rocm_plib._gpu_pcie_bus_cmd(use_sudo=False)
        self.assertLess(cmd.index('rocm-smi'), cmd.index('amd-smi'))
        self.assertIn('rocm-smi --loglevel error --showbus --json', cmd)
        self.assertIn('amd-smi list --json', cmd)
        self.assertIn('/opt/rocm/bin/rocm-smi', cmd)
        self.assertIn('/opt/rocm/bin/amd-smi', cmd)

    def test_pcie_bus_map_uses_rocm_smi_payload_as_is(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': '{"card0": {"PCI Bus": "0000:03:00.0"}}'}
        result = rocm_plib.get_gpu_pcie_bus_dict(phdl, use_sudo=False)
        self.assertEqual(result['n1']['card0']['PCI Bus'], '0000:03:00.0')

    def test_pcie_bus_map_falls_back_to_amd_smi_list_when_rocm_smi_absent(self):
        # First two GPUs of `amd-smi list --json` from an MI325X node (ROCm 10.2.0rc0).
        captured = json.dumps(
            [
                {
                    "gpu": 0,
                    "bdf": "0000:05:00.0",
                    "uuid": "ab0074a5-0000-1000-8039-f74441a8bb09",
                    "kfd_id": 51110,
                    "node_id": 3,
                    "partition_id": 0,
                },
                {
                    "gpu": 1,
                    "bdf": "0000:15:00.0",
                    "uuid": "e70074a5-0000-1000-8004-9f08989f2cf4",
                    "kfd_id": 2987,
                    "node_id": 5,
                    "partition_id": 0,
                },
            ]
        )
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': '[]', 'n2': captured, 'n3': '[{"gpu": 0}]'}
        result = rocm_plib.get_gpu_pcie_bus_dict(phdl, use_sudo=False)
        self.assertIn('amd-smi list --json', phdl.exec.call_args.args[0])
        self.assertEqual(result['n1'], {})
        self.assertEqual(result['n3'], {})
        self.assertEqual(result['n2'], {'card0': {'PCI Bus': '0000:05:00.0'}, 'card1': {'PCI Bus': '0000:15:00.0'}})

    def test_privileged_pcie_bus_map_falls_back_to_opt_rocm_paths(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': '[]'}
        rocm_plib.get_gpu_pcie_bus_dict(phdl)
        cmd = phdl.exec.call_args.args[0]
        self.assertTrue(cmd.startswith('sudo bash -lc '))
        self.assertIn('/opt/rocm/bin/rocm-smi --loglevel error --showbus --json', cmd)
        self.assertIn('/opt/rocm/bin/amd-smi list --json', cmd)

    def test_privileged_pcie_bus_map_fails_when_neither_tool_is_present(self):
        globals.error_list = []
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': 'rocm-smi/amd-smi not found'}
        result = rocm_plib.get_gpu_pcie_bus_dict(phdl)
        self.assertNotIn('echo "[]"', phdl.exec.call_args.args[0])
        self.assertEqual(result['n1'], {})
        self.assertEqual(len(globals.error_list), 1)

    def test_firmware_reader_forwards_use_sudo(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': '[]'}
        rocm_plib.get_amd_smi_fw_dict(phdl, use_sudo=False)
        cmd = phdl.exec.call_args.args[0]
        self.assertNotIn('sudo', cmd)
        self.assertIn('firmware --json', cmd)
