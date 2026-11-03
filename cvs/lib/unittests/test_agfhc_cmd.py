'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.agfhc_cmd import (
    agfhc_command,
    companion_probe_command,
    install_command,
    probe_is_writable,
    resolve_agfhc_env,
    rootless_install,
    uses_output_dir,
)


class TestAgfhcCmd(unittest.TestCase):
    def test_container_is_not_rootless(self):
        self.assertFalse(rootless_install('container', ''))

    def test_baremetal_without_sudo_is_rootless(self):
        self.assertTrue(rootless_install('baremetal', ''))

    def test_baremetal_with_sudo_is_not_rootless(self):
        self.assertFalse(rootless_install('baremetal', 'sudo -n '))

    def test_non_opt_path_uses_output_dir_even_with_sudo(self):
        self.assertTrue(uses_output_dir('baremetal', 'sudo -n ', '/home/user/INSTALL/agfhc/prefix/agfhc'))
        self.assertFalse(uses_output_dir('baremetal', 'sudo -n ', '/opt/amd/agfhc'))
        self.assertTrue(uses_output_dir('container', '', '/home/user/INSTALL/agfhc/prefix/agfhc'))

    def test_sudo_install_to_opt_keeps_sudo_and_omits_output_dir(self):
        cmd = install_command('sudo -n ', 'baremetal', '/home/user/INSTALL/agfhc', '/opt/amd/agfhc')
        self.assertTrue(cmd.startswith('sudo -n bash -c '))
        self.assertIn('./install --rocm-tar', cmd)
        self.assertNotIn(' -o ', cmd)

    def test_sudo_install_to_home_drops_sudo(self):
        cmd = install_command(
            'sudo -n ', 'baremetal', '/home/user/INSTALL/agfhc', '/home/user/INSTALL/agfhc/prefix/agfhc'
        )
        self.assertNotIn('sudo', cmd)
        self.assertIn('-o /home/user/INSTALL/agfhc/prefix', cmd)
        self.assertTrue(cmd.startswith('bash -c '))

    def test_container_opt_install_omits_sudo_and_output_dir(self):
        cmd = install_command('', 'container', '/home/user/INSTALL/agfhc', '/opt/amd/agfhc')
        self.assertNotIn('sudo', cmd)
        self.assertNotIn(' -o ', cmd)
        self.assertIn('./install --rocm-tar', cmd)

    def test_rootless_install_extracts_to_parent_of_path(self):
        cmd = install_command('', 'baremetal', '/home/user/INSTALL/agfhc', '/home/user/INSTALL/agfhc/prefix/agfhc')
        self.assertIn('-o /home/user/INSTALL/agfhc/prefix', cmd)
        self.assertNotIn('sudo', cmd)

    def test_install_command_quotes_the_shell_script(self):
        cmd = install_command('', 'baremetal', '/home/user/INSTALL/agfhc', '/opt/amd/agfhc')
        self.assertIn("bash -c 'cd ", cmd)

    def test_opt_run_exports_derived_companion_env(self):
        cmd = agfhc_command('sudo -n ', '/opt/amd/agfhc', '-r hbm')
        self.assertTrue(cmd.startswith('sudo -n bash -c '))
        self.assertIn('TXB_PATH=/opt/amd/transferbench', cmd)
        self.assertIn('MINIHPL_PATH=/opt/amd/minihpl', cmd)
        self.assertIn('MEMROCKER_PATH=/opt/amd/memrocker', cmd)
        self.assertIn('FIREXS_PATH=/opt/amd/firexs2', cmd)
        self.assertIn('AFIDSAG_PATH=/opt/amd/agfhc/sag', cmd)
        self.assertIn('FIREXS2_APP_PATH=/opt/amd/firexs2/usr/lib/firexs2', cmd)
        self.assertIn('LD_LIBRARY_PATH=/opt/amd/firexs2/usr/lib', cmd)
        self.assertIn('exec /opt/amd/agfhc/agfhc -r hbm', cmd)

    def test_env_overrides_fall_back_to_derived_values(self):
        cmd = agfhc_command(
            '',
            '/home/user/INSTALL/agfhc/prefix/agfhc',
            '-r hbm',
            {'TXB_PATH': '/custom/transferbench', 'FIREXS_PATH': ''},
        )
        self.assertIn('TXB_PATH=/custom/transferbench', cmd)
        self.assertIn('FIREXS_PATH=/home/user/INSTALL/agfhc/prefix/firexs2', cmd)
        self.assertIn('FIREXS2_APP_PATH=/home/user/INSTALL/agfhc/prefix/firexs2/usr/lib/firexs2', cmd)
        self.assertIn('MINIHPL_PATH=/home/user/INSTALL/agfhc/prefix/minihpl', cmd)
        self.assertNotIn('sudo', cmd)

    def test_config_defaults_override_derived_afidsag(self):
        env = {
            'TXB_PATH': '/opt/amd/transferbench',
            'MINIHPL_PATH': '/opt/amd/minihpl',
            'MEMROCKER_PATH': '/opt/amd/memrocker',
            'FIREXS_PATH': '/opt/amd/firexs2',
            'AFIDSAG_PATH': '/opt/amd/afid',
        }
        resolved = resolve_agfhc_env('/opt/amd/agfhc', env)
        self.assertEqual(resolved['AFIDSAG_PATH'], '/opt/amd/afid')
        self.assertEqual(resolve_agfhc_env('/opt/amd/agfhc', {})['AFIDSAG_PATH'], '/opt/amd/agfhc/sag')

    def test_dotted_home_is_quoted(self):
        cmd = agfhc_command('', '/home/first.last/prefix/agfhc', '-v')
        self.assertIn("TXB_PATH=/home/first.last/prefix/transferbench", cmd)
        self.assertIn("exec /home/first.last/prefix/agfhc/agfhc -v", cmd)

    def test_probe_is_writable(self):
        self.assertTrue(probe_is_writable('writable'))
        self.assertFalse(probe_is_writable('not-writable'))
        self.assertFalse(probe_is_writable(''))

    def test_companion_probe_names_each_path(self):
        cmd = companion_probe_command(
            '/home/user/prefix/agfhc',
            {'TXB_PATH': '/custom/transferbench'},
        )
        self.assertIn('test -d /custom/transferbench || echo missing TXB_PATH /custom/transferbench', cmd)
        self.assertIn('missing FIREXS_PATH /home/user/prefix/firexs2', cmd)
        self.assertIn('missing AFIDSAG_PATH /home/user/prefix/agfhc/sag', cmd)


if __name__ == '__main__':
    unittest.main()
