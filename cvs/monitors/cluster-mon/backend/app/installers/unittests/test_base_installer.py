"""
Contract tests for BaseInstaller (synchronous exec + exec_cmd_list path).
"""

import unittest

from app.unittests.testing import FakeSshManager
from app.installers.base_installer import BaseInstaller


class _FakeInstaller(BaseInstaller):
    def get_package_name(self) -> str:
        return "mytool"

    def get_check_command(self) -> str:
        return "which mytool"


class TestBaseInstaller(unittest.TestCase):
    def test_detect_os_parses_per_node(self):
        ssh = FakeSshManager(
            ["n1", "n2"],
            command_map={"os-release": {"n1": "ubuntu\n", "n2": "rhel\n"}},
        )
        os_map = _FakeInstaller(ssh).detect_os()
        self.assertEqual(os_map, {"n1": "ubuntu", "n2": "rhel"})

    def test_check_installation_detects_presence(self):
        ssh = FakeSshManager(
            ["n1", "n2"],
            command_map={"which mytool": {"n1": "/usr/bin/mytool", "n2": "mytool not found"}},
        )
        installed = _FakeInstaller(ssh).check_installation()
        self.assertTrue(installed["n1"])
        self.assertFalse(installed["n2"])

    def test_install_package_uses_exec_cmd_list(self):
        ssh = FakeSshManager(
            ["n1"],
            command_map={
                "os-release": {"n1": "ubuntu\n"},
                "which mytool": {"n1": "mytool not found"},
            },
            cmd_list_response={"n1": "Setting up mytool ... done"},
        )
        result = _FakeInstaller(ssh).install_package()

        self.assertEqual(result["package"], "mytool")
        self.assertEqual(result["successful"], 1)
        self.assertTrue(result["results"]["n1"]["success"])
        # exec_cmd_list was the path used for the actual install.
        self.assertEqual(len(ssh.cmd_list_calls), 1)


if __name__ == "__main__":
    unittest.main()
