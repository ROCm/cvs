# cvs/lib/unittests/test_linux_utils.py
import unittest
from unittest.mock import MagicMock
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
        mock_phdl.exec.assert_called_once_with('sudo rdma link')

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


if __name__ == '__main__':
    unittest.main()
