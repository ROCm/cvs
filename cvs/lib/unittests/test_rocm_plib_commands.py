'''Command shape for the unprivileged amd-smi / rocm-smi readers used by host checks.'''

import unittest
from unittest.mock import MagicMock

from cvs.lib import rocm_plib


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

    def test_showbus_default_is_sudo_rocm_smi(self):
        self.assertEqual(
            rocm_plib._rocm_smi_showbus_cmd(),
            'sudo rocm-smi --loglevel error --showbus --json',
        )

    def test_showbus_without_sudo_resolves_binary(self):
        cmd = rocm_plib._rocm_smi_showbus_cmd(use_sudo=False)
        self.assertNotIn('sudo', cmd)
        self.assertIn('/opt/rocm/bin/rocm-smi', cmd)
        self.assertIn('echo "[]"', cmd)
        self.assertIn('exit 0', cmd)
        self.assertNotIn('exit 127', cmd)

    def test_showbus_empty_list_becomes_empty_card_map(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'n1': '[]',
            'n2': '{"card0": {"PCI Bus": "0000:03:00.0"}}',
        }
        result = rocm_plib.get_gpu_pcie_bus_dict(phdl, use_sudo=False)
        self.assertEqual(result['n1'], {})
        self.assertEqual(result['n2']['card0']['PCI Bus'], '0000:03:00.0')

    def test_firmware_reader_forwards_use_sudo(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': '[]'}
        rocm_plib.get_amd_smi_fw_dict(phdl, use_sudo=False)
        cmd = phdl.exec.call_args.args[0]
        self.assertNotIn('sudo', cmd)
        self.assertIn('firmware --json', cmd)
