'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest

from cvs.lib.agfhc_cmd import agfhc_command, install_command, path_requires_root, rootless_install


class TestAgfhcCmd(unittest.TestCase):
    def test_container_is_not_rootless(self):
        self.assertFalse(rootless_install('container', ''))

    def test_baremetal_without_sudo_is_rootless(self):
        self.assertTrue(rootless_install('baremetal', ''))

    def test_baremetal_with_sudo_is_not_rootless(self):
        self.assertFalse(rootless_install('baremetal', 'sudo -n '))

    def test_opt_path_requires_root(self):
        self.assertTrue(path_requires_root('/opt/amd/agfhc'))
        self.assertFalse(path_requires_root('/home/user/INSTALL/agfhc/prefix/agfhc'))

    def test_sudo_install_uses_system_install(self):
        cmd = install_command('sudo -n ', 'baremetal', '/home/user/INSTALL/agfhc', '/opt/amd/agfhc')
        self.assertEqual(cmd, "sudo -n bash -c 'cd /home/user/INSTALL/agfhc && ./install --rocm-tar'")

    def test_container_install_omits_sudo_and_output_dir(self):
        cmd = install_command('', 'container', '/home/user/INSTALL/agfhc', '/opt/amd/agfhc')
        self.assertEqual(cmd, "bash -c 'cd /home/user/INSTALL/agfhc && ./install --rocm-tar'")
        self.assertNotIn(' -o ', cmd)

    def test_container_home_install_extracts_to_configured_prefix(self):
        cmd = install_command(
            '',
            'container',
            '/home/user/INSTALL/agfhc',
            '/home/user/INSTALL/agfhc/prefix/agfhc',
        )
        self.assertIn(
            './install --rocm-tar -o /home/user/INSTALL/agfhc/prefix',
            cmd,
        )

    def test_rootless_install_extracts_to_parent_of_path(self):
        cmd = install_command('', 'baremetal', '/home/user/INSTALL/agfhc/', '/home/user/INSTALL/agfhc/prefix/agfhc')
        self.assertNotIn('sudo', cmd)
        self.assertIn(
            './install --rocm-tar -o /home/user/INSTALL/agfhc/prefix',
            cmd,
        )

    def test_system_path_command_is_direct(self):
        self.assertEqual(
            agfhc_command('sudo -n ', '/opt/amd/agfhc', '-r hbm'),
            'sudo -n /opt/amd/agfhc/agfhc -r hbm',
        )
        self.assertEqual(
            agfhc_command('', '/opt/amd/agfhc', '-r hbm'),
            '/opt/amd/agfhc/agfhc -r hbm',
        )

    def test_rootless_command_exports_companion_tools(self):
        cmd = agfhc_command('', '/home/user/INSTALL/agfhc/prefix/agfhc', '-v')
        self.assertNotIn('sudo', cmd)
        self.assertIn('TXB_PATH="$root/transferbench"', cmd)
        self.assertIn('MINIHPL_PATH="$root/minihpl"', cmd)
        self.assertIn('MEMROCKER_PATH="$root/memrocker"', cmd)
        self.assertIn('AFIDSAG_PATH="/home/user/INSTALL/agfhc/prefix/agfhc/sag"', cmd)
        self.assertIn('firexs2-external', cmd)
        self.assertIn('exec /home/user/INSTALL/agfhc/prefix/agfhc/agfhc -v', cmd)
