# cvs/lib/unittests/test_linux_utils.py
import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cvs.lib.linux_utils as linux_utils


class TestGetRdmaNicDict(unittest.TestCase):
    def test_get_rdma_nic_dict_with_hyphenated_devices(self):
        """Test that get_rdma_nic_dict properly parses hyphenated device names like tw-eth0."""
        # Mock phdl object
        mock_phdl = MagicMock()

        # Simulate rdma link output with hyphenated device names (AMD AINICs)
        rdma_link_output = """link rdma0/1 state ACTIVE physical_state LINK_UP netdev tw-eth0 
link rdma1/1 state ACTIVE physical_state LINK_UP netdev tw-eth1"""

        mock_phdl.exec.return_value = {'node1': rdma_link_output}

        # Call the function
        result = linux_utils.get_rdma_nic_dict(mock_phdl)

        # Verify the function was called with correct command
        mock_phdl.exec.assert_called_once_with('rdma link')

        # Verify all RDMA devices are parsed
        self.assertIn('node1', result)
        self.assertEqual(len(result['node1']), 2)

        # Verify hyphenated device names are correctly captured
        self.assertIn('rdma0', result['node1'])
        self.assertEqual(result['node1']['rdma0']['eth_device'], 'tw-eth0')
        self.assertEqual(result['node1']['rdma0']['port'], '1')
        self.assertEqual(result['node1']['rdma0']['device_status'], 'ACTIVE')
        self.assertEqual(result['node1']['rdma0']['link_status'], 'LINK_UP')

        self.assertIn('rdma1', result['node1'])
        self.assertEqual(result['node1']['rdma1']['eth_device'], 'tw-eth1')

    def test_get_rdma_nic_dict_with_standard_devices(self):
        """Test that get_rdma_nic_dict also works with standard device names like ens26np0."""
        # Mock phdl object
        mock_phdl = MagicMock()

        # Simulate rdma link output with standard device names (Broadcom NICs)
        rdma_link_output = """link bnxt_re0/1 state ACTIVE physical_state LINK_UP netdev ens26np0 
link bnxt_re1/1 state ACTIVE physical_state LINK_UP netdev ens27np1"""

        mock_phdl.exec.return_value = {'node2': rdma_link_output}

        # Call the function
        result = linux_utils.get_rdma_nic_dict(mock_phdl)

        # Verify standard device names work too
        self.assertIn('node2', result)
        self.assertEqual(len(result['node2']), 2)
        self.assertEqual(result['node2']['bnxt_re0']['eth_device'], 'ens26np0')
        self.assertEqual(result['node2']['bnxt_re1']['eth_device'], 'ens27np1')


class TestGetActiveRdmaNicDict(unittest.TestCase):
    def test_get_active_rdma_nic_dict_with_hyphenated_devices(self):
        """Test that get_active_rdma_nic_dict filters only ACTIVE devices with hyphenated names."""
        # Mock phdl object
        mock_phdl = MagicMock()

        # Simulate rdma link output with mixed states and hyphenated device names
        rdma_link_output = """link rdma0/1 state ACTIVE physical_state LINK_UP netdev tw-eth0 
link rdma1/1 state DOWN physical_state LINK_DOWN netdev tw-eth1"""

        mock_phdl.exec.return_value = {'node1': rdma_link_output}

        # Call the function
        result = linux_utils.get_active_rdma_nic_dict(mock_phdl)

        # Verify only ACTIVE devices are included
        self.assertIn('node1', result)
        self.assertEqual(len(result['node1']), 1)  # Only rdma0 is ACTIVE

        # Verify ACTIVE device with hyphenated name is correctly captured
        self.assertIn('rdma0', result['node1'])
        self.assertEqual(result['node1']['rdma0']['eth_device'], 'tw-eth0')
        self.assertEqual(result['node1']['rdma0']['device_status'], 'ACTIVE')

        # Verify non-ACTIVE device is excluded
        self.assertNotIn('rdma1', result['node1'])  # DOWN state

    def test_get_active_rdma_nic_dict_with_standard_devices(self):
        """Test that get_active_rdma_nic_dict also works with standard device names like ens26np0."""
        # Mock phdl object
        mock_phdl = MagicMock()

        # Simulate rdma link output with standard device names (Broadcom NICs)
        rdma_link_output = """link bnxt_re0/1 state ACTIVE physical_state LINK_UP netdev ens26np0 
link bnxt_re1/1 state DOWN physical_state LINK_DOWN netdev ens27np1"""

        mock_phdl.exec.return_value = {'node1': rdma_link_output}

        # Call the function
        result = linux_utils.get_active_rdma_nic_dict(mock_phdl)

        # Verify only ACTIVE device is included
        self.assertIn('node1', result)
        self.assertEqual(len(result['node1']), 1)
        self.assertIn('bnxt_re0', result['node1'])
        self.assertEqual(result['node1']['bnxt_re0']['eth_device'], 'ens26np0')

        # Verify non-ACTIVE device is excluded
        self.assertNotIn('bnxt_re1', result['node1'])

    def test_get_active_rdma_nic_dict_with_underscore_devices(self):
        """Test that get_active_rdma_nic_dict also works with standard ACTIVE devices with underscore names."""
        # Mock phdl object
        mock_phdl = MagicMock()

        # Simulate rdma link output with mixed states and underscore device names
        rdma_link_output = """link rdma0/1 state ACTIVE physical_state LINK_UP netdev tw_eth0
link rdma1/1 state DOWN physical_state LINK_DOWN netdev tw_eth1"""

        mock_phdl.exec.return_value = {'node1': rdma_link_output}

        # Call the function
        result = linux_utils.get_active_rdma_nic_dict(mock_phdl)

        # Verify only ACTIVE devices are included
        self.assertIn('node1', result)
        self.assertEqual(len(result['node1']), 1)  # Only rdma0 is ACTIVE

        # Verify ACTIVE device with hyphenated name is correctly captured
        self.assertIn('rdma0', result['node1'])
        self.assertEqual(result['node1']['rdma0']['eth_device'], 'tw_eth0')
        self.assertEqual(result['node1']['rdma0']['device_status'], 'ACTIVE')

        # Verify non-ACTIVE device is excluded
        self.assertNotIn('rdma1', result['node1'])  # DOWN state


class TestGetNicEthtoolStatsDict(unittest.TestCase):
    def test_asymmetric_nic_counts_across_nodes(self):
        """Nodes with different backend RDMA NIC counts (e.g. a downed link) must not
        raise IndexError, and each node's stats must only contain its own interfaces."""
        mock_phdl = MagicMock()

        bck_nic_dict = {
            'node1': {
                'rdma0': {'eth_device': 'eth0'},
                'rdma1': {'eth_device': 'eth1'},
            },
            'node2': {
                'rdma0': {'eth_device': 'eth2'},
            },
        }

        # Batch 0: both nodes have an interface at index 0.
        # Batch 1: only node1 has a second interface; node2 ran the 'true' no-op.
        mock_phdl.exec_cmd_list.side_effect = [
            {'node1': 'rx_errors: 0', 'node2': 'rx_errors: 0'},
            {'node1': 'rx_errors: 1', 'node2': ''},
        ]

        with patch.object(linux_utils, 'get_backend_rdma_nic_dict', return_value=bck_nic_dict):
            result = linux_utils.get_nic_ethtool_stats_dict(mock_phdl)

        self.assertEqual(mock_phdl.exec_cmd_list.call_count, 2)

        self.assertEqual(set(result['node1'].keys()), {'eth0', 'eth1'})
        self.assertEqual(result['node1']['eth1']['rx_errors'], '1')

        # node2 must only have its single real interface, not a bogus entry
        # from the 'true' no-op run in the second batch.
        self.assertEqual(set(result['node2'].keys()), {'eth2'})

        second_batch_cmds = mock_phdl.exec_cmd_list.call_args_list[1].args[0]
        self.assertEqual(second_batch_cmds[1], 'true')


class TestSudoNOrDenied(unittest.TestCase):
    def _run(self, reader, filter_cmd):
        cmd = linux_utils._sudo_n_or_denied(reader, filter_cmd)
        return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)

    def test_failing_reader_emits_denial_token(self):
        result = self._run('ls /no/such/cvs-sudo-reader', 'grep ACSCtl')
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith('CVS_CMD_DENIED '))
        self.assertIn('cvs-sudo-reader', result.stdout)

    def test_filter_miss_is_empty_and_exits_zero(self):
        result = self._run("printf 'nothing to see\\n'", 'grep ACSCtl')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')

    def test_filter_hit_passes_matching_lines(self):
        result = self._run("printf 'ACSCtl: SrcValid+\\nnoise\\n'", 'grep ACSCtl')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, 'ACSCtl: SrcValid+\n')


class TestPcieLinkStatusCmd(unittest.TestCase):
    def test_wide_domain_is_kept(self):
        cmd = linux_utils.pcie_link_status_cmd('10000:e1:00.0')
        self.assertIn('dev=/sys/bus/pci/devices/10000:e1:00.0;', cmd)

    def test_short_bdf_is_prefixed_and_command_is_unprivileged(self):
        cmd = linux_utils.pcie_link_status_cmd('03:00.0')
        self.assertIn('dev=/sys/bus/pci/devices/0000:03:00.0;', cmd)
        self.assertIn('current_link_speed', cmd)
        self.assertIn('current_link_width', cmd)
        self.assertNotIn('sudo', cmd)
        self.assertNotIn('lspci', cmd)

    def test_prints_lnksta_and_downgrade(self):
        cmd = linux_utils.pcie_link_status_cmd('0000:03:00.0')
        with tempfile.TemporaryDirectory() as device:
            for name, text in (
                ('current_link_speed', '16.0 GT/s\n'),
                ('current_link_width', '8\n'),
                ('max_link_speed', '32.0 GT/s\n'),
                ('max_link_width', '16\n'),
            ):
                with open(os.path.join(device, name), 'w', encoding='utf-8') as handle:
                    handle.write(text)
            rendered = cmd.replace('/sys/bus/pci/devices/0000:03:00.0', device)
            result = subprocess.run(['bash', '-c', rendered], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, 'LnkSta: Speed 16GT/s, Width x8 (downgraded)\n')

    def test_rejects_unsafe_bdf(self):
        with self.assertRaises(ValueError):
            linux_utils.pcie_link_status_cmd('0000:03:00.0; rm -rf /')

    def test_lshw_default_still_uses_sudo(self):
        mock_phdl = MagicMock()
        mock_phdl.exec.return_value = {'node1': ''}
        linux_utils.get_lshw_network_dict(mock_phdl)
        self.assertEqual(mock_phdl.exec.call_args.args[0], 'sudo lshw -class network -businfo')

    def test_unprivileged_lshw_skips_sudo(self):
        mock_phdl = MagicMock()
        mock_phdl.exec.return_value = {'node1': ''}
        linux_utils.get_lshw_network_dict(mock_phdl, use_sudo=False)
        cmd = mock_phdl.exec.call_args.args[0]
        self.assertNotIn('sudo', cmd)
        self.assertIn('/usr/sbin/lshw', cmd)

    def test_gpu_nic_mapping_forwards_use_sudo(self):
        with (
            patch.object(linux_utils.rocm_plib, 'get_gpu_pcie_bus_dict', return_value={}) as bus,
            patch.object(linux_utils, 'get_lshw_backend_nic_dict', return_value={}) as lshw,
        ):
            linux_utils.get_gpu_nic_mapping_dict(MagicMock(), use_sudo=False)
        bus.assert_called_once()
        self.assertFalse(bus.call_args.kwargs['use_sudo'])
        lshw.assert_called_once()
        self.assertFalse(lshw.call_args.kwargs['use_sudo'])


if __name__ == '__main__':
    unittest.main()
