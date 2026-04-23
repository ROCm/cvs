"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import unittest
from unittest.mock import MagicMock

from cvs.core.runtimes.hostshell import HostShellRuntime


class TestHostShellRuntime(unittest.TestCase):
    def test_wrap_cmd_is_identity(self):
        rt = HostShellRuntime.parse_config(None)
        self.assertEqual(rt.wrap_cmd("echo hi"), "echo hi")
        self.assertEqual(rt.wrap_cmd("ls -la /opt/rocm"), "ls -la /opt/rocm")
        self.assertEqual(rt.wrap_cmd(""), "")

    def test_setup_and_teardown_are_no_ops(self):
        transport = MagicMock()
        rt = HostShellRuntime.parse_config(None)
        # Neither call should reach into the transport.
        self.assertIsNone(rt.setup(transport))
        self.assertIsNone(rt.teardown(transport))
        transport.exec.assert_not_called()
        transport.scp.assert_not_called()

    def test_workload_facts_are_baremetal_defaults(self):
        rt = HostShellRuntime.parse_config(None)
        self.assertEqual(rt.workload_ssh_port(), 22)
        self.assertEqual(rt.workload_hostfile_path(), "/tmp/mpi_hosts.txt")

    def test_default_capabilities_are_empty_so_in_namespace_phases_skip(self):
        rt = HostShellRuntime.parse_config(None)
        self.assertEqual(rt.capabilities, set())
        self.assertNotIn("in_namespace_sshd", rt.capabilities)

    def test_parse_config_accepts_overrides(self):
        rt = HostShellRuntime.parse_config(
            {
                "workload_ssh_port": 2200,
                "workload_hostfile_path": "/var/cvs/hosts",
                "capabilities": ["custom_tag"],
            }
        )
        self.assertEqual(rt.workload_ssh_port(), 2200)
        self.assertEqual(rt.workload_hostfile_path(), "/var/cvs/hosts")
        self.assertEqual(rt.capabilities, {"custom_tag"})

    def test_parse_config_accepts_empty_dict(self):
        rt = HostShellRuntime.parse_config({})
        self.assertEqual(rt.name, "hostshell")
        self.assertEqual(rt.workload_ssh_port(), 22)


if __name__ == "__main__":
    unittest.main()
