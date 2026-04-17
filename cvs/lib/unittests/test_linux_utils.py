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


class TestGetRdmaNicDictMatchGuard(unittest.TestCase):
    """Regression tests for get_rdma_nic_dict missing-match guard."""

    def test_does_not_raise_on_malformed_link_line(self):
        """get_rdma_nic_dict must not crash on a line that starts with 'link'
        but does not conform to the strict netdev-bearing format.

        Previously `match.group(1)` was called unconditionally, raising
        AttributeError on any malformed line.
        """
        mock_phdl = MagicMock()
        rdma_link_output = (
            "link\n"
            "link rdma0/1 state ACTIVE physical_state LINK_UP netdev tw-eth0"
        )
        mock_phdl.exec.return_value = {'node1': rdma_link_output}

        try:
            result = linux_utils.get_rdma_nic_dict(mock_phdl)
        except AttributeError as exc:  # pragma: no cover - would fail pre-fix
            self.fail(
                "get_rdma_nic_dict raised AttributeError on a malformed "
                f"'link' line -- missing `if match:` guard: {exc}"
            )

        self.assertIn('rdma0', result['node1'])
        self.assertEqual(result['node1']['rdma0']['eth_device'], 'tw-eth0')
        self.assertEqual(result['node1']['rdma0']['device_status'], 'ACTIVE')

    def test_skips_rdma_link_down_line_without_netdev(self):
        """A realistic DOWN link from `rdma link` (no trailing netdev) must be
        skipped rather than raising. This is the live-observed case on banff:
        all non-active mlx5 devices emit subnet_prefix/lid/sm_lid/etc. but
        omit the `netdev <name>` clause the strict pattern requires.
        """
        mock_phdl = MagicMock()
        rdma_link_output = (
            "link mlx5_0/1 subnet_prefix fe80:0000:0000:0000 lid 65535 "
            "sm_lid 0 lmc 0 state DOWN physical_state DISABLED\n"
            "link mlx5_8/1 state ACTIVE physical_state LINK_UP netdev ens14np0"
        )
        mock_phdl.exec.return_value = {'node1': rdma_link_output}

        result = linux_utils.get_rdma_nic_dict(mock_phdl)

        self.assertNotIn('mlx5_0', result['node1'])
        self.assertIn('mlx5_8', result['node1'])
        self.assertEqual(result['node1']['mlx5_8']['eth_device'], 'ens14np0')


class TestGetDnsDict(unittest.TestCase):
    """Regression tests for get_dns_dict."""

    def test_populates_dict_from_resolvectl_output(self):
        """get_dns_dict must capture values into dns_dict[node].

        Previously every `elif` branch only called print('') and the
        function always returned {<node>: {}} regardless of input.
        """
        mock_phdl = MagicMock()
        resolvectl_output = (
            "Global\n"
            "       Protocols: -LLMNR -mDNS -DNSOverTLS DNSSEC=no/unsupported\n"
            "resolv.conf mode: stub\n"
            "\n"
            "Link 2 (ens14np0)\n"
            "    Current Scopes: DNS\n"
            "         Protocols: +DefaultRoute +LLMNR -mDNS -DNSOverTLS DNSSEC=no/unsupported\n"
        )
        mock_phdl.exec.return_value = {'node1': resolvectl_output}

        result = linux_utils.get_dns_dict(mock_phdl)

        self.assertIn('node1', result)
        self.assertTrue(result['node1'], "dns_dict[node] must not be empty")
        self.assertIn('protocols', result['node1'])
        # Global + Link -> two Protocols entries.
        self.assertEqual(len(result['node1']['protocols']), 2)

    def test_captures_current_and_list_dns_servers(self):
        """Verify current_dns_server, dns_servers list, and dns_domain fields."""
        mock_phdl = MagicMock()
        resolvectl_output = (
            "Current DNS Server: 10.0.0.1\n"
            "       DNS Servers: 10.0.0.1 10.0.0.2\n"
            "        DNS Domain: corp.example\n"
        )
        mock_phdl.exec.return_value = {'node1': resolvectl_output}

        result = linux_utils.get_dns_dict(mock_phdl)

        self.assertEqual(result['node1']['current_dns_server'], '10.0.0.1')
        self.assertEqual(result['node1']['dns_servers'], ['10.0.0.1', '10.0.0.2'])
        self.assertEqual(result['node1']['dns_domain'], 'corp.example')


class TestGetLinuxPerfTuningDict(unittest.TestCase):
    """Regression tests for get_linux_perf_tuning_dict."""

    def test_returns_populated_dict_not_none(self):
        """get_linux_perf_tuning_dict must return the dict it builds, not None.

        Previously the function built out_dict but ended without a `return`,
        so every caller received None.
        """
        mock_phdl = MagicMock()
        mock_phdl.exec.return_value = {'node1': 'stub output'}

        result = linux_utils.get_linux_perf_tuning_dict(mock_phdl)

        self.assertIsNotNone(result, "function returned None instead of out_dict")
        self.assertIsInstance(result, dict)
        for key in (
            'bios_version',
            'numa_balancing',
            'nmi_watchdog',
            'huge_pages',
            'cpu_power_profile',
        ):
            self.assertIn(key, result)


class TestGetIpAddrDictIntNamLeak(unittest.TestCase):
    """Regression tests for get_ip_addr_dict int_nam cross-node leak."""

    def test_no_keyerror_when_nodeB_stdout_starts_with_property_line(self):
        """int_nam must be scoped to each per-node iteration.

        Previously `int_nam = None` was set once outside the for-node loop,
        so after parsing nodeA the variable retained nodeA's last interface
        name. When nodeB's first matching line was a property line
        (mtu/state/mac/inet/inet6) rather than an interface header, the
        code did `ip_dict['nodeB']['<nodeA-iface>']['mtu'] = ...` and
        raised KeyError.
        """
        mock_phdl = MagicMock()
        out_dict = {
            'nodeA': (
                "2: tw-eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 state UP\n"
                "    link/ether aa:bb:cc:dd:ee:ff\n"
                "    inet 10.0.0.1/24 scope global\n"
            ),
            'nodeB': (
                "    inet 10.0.0.99/24 scope global\n"
                "3: tw-eth1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 state UP\n"
                "    inet 10.0.0.2/24 scope global\n"
            ),
        }
        mock_phdl.exec.return_value = out_dict

        try:
            result = linux_utils.get_ip_addr_dict(mock_phdl)
        except KeyError as exc:  # pragma: no cover - would fail pre-fix
            self.fail(
                "get_ip_addr_dict raised KeyError -- int_nam leaked from "
                f"nodeA into nodeB because it was initialized outside the "
                f"per-node loop: {exc}"
            )

        # nodeA's last interface must not be attributed to nodeB.
        self.assertNotIn('tw-eth0', result['nodeB'])
        # nodeA's own parsing should still be correct.
        self.assertIn('tw-eth0', result['nodeA'])
        self.assertEqual(result['nodeA']['tw-eth0']['mtu'], '9000')
        # nodeB's actual interface is still parsed after the stray line.
        self.assertIn('tw-eth1', result['nodeB'])


if __name__ == '__main__':
    unittest.main()
