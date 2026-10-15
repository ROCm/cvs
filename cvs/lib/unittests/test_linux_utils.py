# cvs/lib/unittests/test_linux_utils.py
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cvs.lib.linux_utils as linux_utils
from cvs.unittests.support import run_trusted


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

    def test_unparsable_link_line_is_skipped(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node': 'link weird/1 state\nlink rdma0/1 state ACTIVE physical_state LINK_UP netdev eth0'
        }
        self.assertEqual(list(linux_utils.get_rdma_nic_dict(phdl)['node']), ['rdma0'])

    def test_underscore_netdev(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node': 'link rdma0/1 state ACTIVE physical_state LINK_UP netdev tw_eth0'}
        self.assertEqual(linux_utils.get_rdma_nic_dict(phdl)['node']['rdma0']['eth_device'], 'tw_eth0')

    def test_down_devices_included(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node': 'link rdma0/1 state ACTIVE physical_state LINK_UP netdev eth0\n'
            'link rdma1/1 state DOWN physical_state DISABLED netdev eth1'
        }
        result = linux_utils.get_rdma_nic_dict(phdl)['node']
        self.assertEqual(result['rdma0']['device_status'], 'ACTIVE')
        self.assertEqual(result['rdma1']['device_status'], 'DOWN')


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


class TestReadOrDenied(unittest.TestCase):
    def _run(self, reader):
        cmd = linux_utils._read_or_denied(reader)
        return run_trusted(['bash', '-c', cmd], capture_output=True, text=True)

    def test_missing_file_emits_denial_token(self):
        result = self._run('cat /no/such/cvs-bios-version')
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith('CVS_CMD_DENIED '))
        self.assertIn('cvs-bios-version', result.stdout)

    def test_success_prints_reader_output(self):
        result = self._run("printf '20171212\\n'")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '20171212\n')


class TestSudoNOrDenied(unittest.TestCase):
    def _run(self, reader, filter_cmd):
        cmd = linux_utils._sudo_n_or_denied(reader, filter_cmd)
        return run_trusted(['bash', '-c', cmd], capture_output=True, text=True)

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
            result = run_trusted(['bash', '-c', rendered], capture_output=True, text=True, check=True)
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
        self.assertIs(lshw.call_args.kwargs['active_only'], False)


class TestGetRdmaCapableDevicesDict(unittest.TestCase):
    def test_lists_all_netdevs_in_one_exec(self):
        mock_phdl = MagicMock()
        mock_phdl.exec.return_value = {
            'node1': 'enp1s0np0\nenp2s0np0\nenp1s0np0\n',
            'node2': '',
        }

        result = linux_utils.get_rdma_capable_devices_dict(mock_phdl)

        mock_phdl.exec.assert_called_once()
        self.assertIn('/sys/class/infiniband/*/device/net/*', mock_phdl.exec.call_args.args[0])
        self.assertEqual(result, {'node1': ['enp1s0np0', 'enp2s0np0'], 'node2': []})

    def test_backend_nic_detection_uses_two_execs_for_any_cluster_size(self):
        nodes = [f'node{i}' for i in range(16)]
        lshw = '\n'.join(f'pci@0000:{i:02x}:00.0  eth{i}  network  BCM57608 Ethernet' for i in range(8))
        netdevs = '\n'.join(f'eth{i}' for i in range(8))

        def fake_exec(cmd):
            return {node: lshw if 'lshw' in cmd else netdevs for node in nodes}

        mock_phdl = MagicMock()
        mock_phdl.exec.side_effect = fake_exec

        result = linux_utils.get_backend_nic_dict(mock_phdl)

        self.assertEqual(mock_phdl.exec.call_count, 2)
        self.assertEqual(result['node15'], [f'eth{i}' for i in range(8)])


class TestParsePciBdf(unittest.TestCase):
    def test_valid_addresses(self):
        cases = {
            '0001:44:00.0': (1, 0x44, 0, 0),
            '44:00.1': (0, 0x44, 0, 1),
            '10000:e1:00.0': (0x10000, 0xE1, 0, 0),
            '0000:E5:00.0': (0, 0xE5, 0, 0),
        }
        for bdf, expected in cases.items():
            with self.subTest(bdf=bdf):
                self.assertEqual(linux_utils.parse_pci_bdf(bdf), expected)

    def test_invalid_addresses(self):
        for bdf in ('garbage', '', '0000:03:00.0; rm -rf /'):
            with self.subTest(bdf=bdf):
                self.assertIsNone(linux_utils.parse_pci_bdf(bdf))


class TestGetBackendRdmaNicDict(unittest.TestCase):
    def test_active_filter_and_backend_membership(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'n1': 'link rdma0/1 state ACTIVE physical_state LINK_UP netdev eth0\n'
            'link rdma1/1 state DOWN physical_state DISABLED netdev eth1\n'
            'link rdma9/1 state ACTIVE physical_state LINK_UP netdev eth9',
            'n2': 'link rdma2/1 state ACTIVE physical_state LINK_UP netdev eth2',
        }
        with patch.object(linux_utils, 'get_backend_nic_dict', return_value={'n1': ['eth0', 'eth1']}):
            active = linux_utils.get_backend_rdma_nic_dict(phdl)
            all_states = linux_utils.get_backend_rdma_nic_dict(phdl, active_only=False)
        self.assertEqual(list(active['n1']), ['rdma0'])
        self.assertEqual(set(all_states['n1']), {'rdma0', 'rdma1'})
        self.assertEqual(active['n2'], {})
        self.assertEqual(all_states['n2'], {})


class TestGetLshwBackendNicDict(unittest.TestCase):
    def test_forwards_active_only_and_copies_status(self):
        lshw_info = {'n': {'eth0': {'pci_bus': '0001:44:00.0', 'description': 'DSC Ethernet Controller'}}}
        rdma_info = {'n': {'rdma0': {'eth_device': 'eth0', 'device_status': 'DOWN'}}}
        with (
            patch.object(linux_utils, 'get_lshw_network_dict', return_value=lshw_info),
            patch.object(linux_utils, 'get_backend_rdma_nic_dict', return_value=rdma_info) as backend,
        ):
            for active_only in (True, False):
                result = linux_utils.get_lshw_backend_nic_dict(MagicMock(), active_only=active_only)
                self.assertIs(backend.call_args.kwargs['active_only'], active_only)
                self.assertEqual(result['n']['eth0']['device_status'], 'DOWN')
                self.assertEqual(result['n']['eth0']['rdma_dev'], 'rdma0')


class TestMapGpusToNics(unittest.TestCase):
    def _helios_layout(self):
        gpus = {f'card{i - 1}': f'000{i}:01:00.0' for i in range(1, 5)}
        nics = {
            'enp4s0np0': {
                'pci_bus': '0000:04:00.0',
                'rdma_dev': 'rocep4s0',
                'device_status': 'ACTIVE',
            }
        }
        for i in range(1, 5):
            for port, bus in ((68, 0x44), (72, 0x48)):
                nics[f'enP{i}p{port}s0'] = {
                    'pci_bus': f'000{i}:{bus:02x}:00.0',
                    'rdma_dev': f'roceP{i}p{port}s0',
                    'device_status': 'DOWN',
                }
        return gpus, nics

    def _domain0_layout(self):
        gpu_buses = (0x05, 0x15, 0x65, 0x75, 0x85, 0x95, 0xE5, 0xF5)
        nic_buses = (0x06, 0x16, 0x66, 0x76, 0x86, 0x96, 0xE6, 0xF6)
        gpus = {f'card{i}': f'0000:{bus:02x}:00.0' for i, bus in enumerate(gpu_buses)}
        nics = {
            f'eth{i}': {'pci_bus': f'0000:{bus:02x}:00.0', 'rdma_dev': f'rdma{i}', 'device_status': 'ACTIVE'}
            for i, bus in enumerate(nic_buses)
        }
        return gpus, nics

    def test_mi450_never_selects_management_nic(self):
        gpus, nics = self._helios_layout()
        with self.assertLogs(linux_utils.log, 'WARNING') as captured:
            active = linux_utils.map_gpus_to_nics(gpus, nics)
        for i in range(1, 5):
            card = f'card{i - 1}'
            self.assertEqual(active[card], {'gpu_bdf': gpus[card]})
            self.assertTrue(
                any(
                    card in warning and f'domain 000{i}' in warning and gpus[card] in warning
                    for warning in captured.output
                )
            )
        self.assertNotIn('rocep4s0', str(active))

        topology = linux_utils.map_gpus_to_nics(gpus, nics, active_only=False)
        for i in range(1, 5):
            card = f'card{i - 1}'
            self.assertEqual(topology[card]['rdma_dev'], f'roceP{i}p68s0')
            self.assertEqual(topology[card]['nic_bdf'], f'000{i}:44:00.0')
        self.assertNotIn('rocep4s0', str(topology))

    def test_mi450_links_up_maps_same_domain(self):
        gpus, nics = self._helios_layout()
        for info in nics.values():
            info['device_status'] = 'ACTIVE'
        result = linux_utils.map_gpus_to_nics(gpus, nics)
        for i in range(1, 5):
            self.assertEqual(result[f'card{i - 1}']['rdma_dev'], f'roceP{i}p68s0')
        self.assertNotIn('rocep4s0', str(result))

    def test_domain0_mapping_unchanged(self):
        gpus, nics = self._domain0_layout()
        result = linux_utils.map_gpus_to_nics(gpus, nics)
        for i in range(8):
            self.assertEqual(result[f'card{i}']['rdma_dev'], f'rdma{i}')

    def test_domain0_down_nic_falls_back_to_nearest_active(self):
        gpus, nics = self._domain0_layout()
        nics['eth0']['device_status'] = 'DOWN'
        result = linux_utils.map_gpus_to_nics(gpus, nics)
        self.assertEqual(result['card0']['rdma_dev'], 'rdma1')

    def test_tie_break_prefers_smaller_bus(self):
        gpus = {'card0': '0000:10:00.0'}
        nics = {
            'high': {'pci_bus': '0000:12:00.0', 'rdma_dev': 'high', 'device_status': 'ACTIVE'},
            'low': {'pci_bus': '0000:0e:00.0', 'rdma_dev': 'low', 'device_status': 'ACTIVE'},
        }
        self.assertEqual(linux_utils.map_gpus_to_nics(gpus, nics)['card0']['rdma_dev'], 'low')

    def test_no_shared_domain_uses_all_nics(self):
        gpus = {'card0': '0000:10:00.0'}
        nics = {'eth0': {'pci_bus': '0001:11:00.0', 'rdma_dev': 'rdma0', 'device_status': 'ACTIVE'}}
        self.assertEqual(linux_utils.map_gpus_to_nics(gpus, nics)['card0']['rdma_dev'], 'rdma0')

    def test_unparsable_bdfs(self):
        gpus = {'card0': '0000:10:00.0', 'card1': 'N/A'}
        nics = {
            'bad': {'pci_bus': 'invalid', 'rdma_dev': 'bad', 'device_status': 'ACTIVE'},
            'good': {'pci_bus': '0000:11:00.0', 'rdma_dev': 'good', 'device_status': 'ACTIVE'},
        }
        with self.assertLogs(linux_utils.log, 'WARNING') as captured:
            result = linux_utils.map_gpus_to_nics(gpus, nics)
        self.assertEqual(result['card0']['rdma_dev'], 'good')
        self.assertEqual(result['card1'], {'gpu_bdf': 'N/A'})
        self.assertTrue(any('bad' in warning and 'invalid' in warning for warning in captured.output))
        self.assertTrue(any('card1' in warning and 'N/A' in warning for warning in captured.output))


class TestGetGpuNicMappingDict(unittest.TestCase):
    def test_helios_capture_end_to_end(self):
        lshw_lines = [
            'Bus info          Device           Class          Description',
            '=============================================================',
            'pci@0000:04:00.0  enp4s0np0        network        DSC Ethernet Controller',
        ]
        rdma_lines = ['link rocep4s0/1 state ACTIVE physical_state LINK_UP netdev enp4s0np0']
        netdevs = ['enp4s0np0']
        for i in range(1, 5):
            lshw_lines.append(f'pci@000{i}:01:00.1                   network        Advanced Micro Devices, Inc. [AMD]')
            for port, bus in ((68, 0x44), (72, 0x48)):
                netdev = f'enP{i}p{port}s0'
                lshw_lines.append(f'pci@000{i}:{bus:02x}:00.0  {netdev}        network        DSC Ethernet Controller')
                rdma_lines.append(f'link roceP{i}p{port}s0/1 state DOWN physical_state DISABLED netdev {netdev}')
                netdevs.append(netdev)
        lshw_lines.append('usb@1:1.1         enxaa5bd15c64f9  network        Ethernet interface')
        gpu_json = '{' + ','.join(f'"card{i - 1}": {{"PCI Bus": "000{i}:01:00.0"}}' for i in range(1, 5)) + '}'

        def fake_exec(cmd, *args, **kwargs):
            if 'rocm-smi' in cmd:
                return {'tray': gpu_json}
            if 'lshw' in cmd:
                return {'tray': '\n'.join(lshw_lines)}
            if '/sys/class/infiniband' in cmd:
                return {'tray': '\n'.join(netdevs)}
            if cmd == 'rdma link':
                return {'tray': '\n'.join(rdma_lines)}
            raise AssertionError(cmd)

        phdl = MagicMock()
        phdl.exec.side_effect = fake_exec
        with self.assertLogs(linux_utils.log, 'WARNING'):
            active = linux_utils.get_gpu_nic_mapping_dict(phdl)
        self.assertEqual(len(active['tray']), 4)
        self.assertTrue(all('rdma_dev' not in info for info in active['tray'].values()))
        self.assertNotIn('rocep4s0', str(active))

        topology = linux_utils.get_gpu_nic_mapping_dict(phdl, active_only=False)
        self.assertEqual(topology['tray']['card0']['rdma_dev'], 'roceP1p68s0')
        self.assertEqual(topology['tray']['card0']['nic_bdf'], '0001:44:00.0')
        self.assertEqual(topology['tray']['card3']['rdma_dev'], 'roceP4p68s0')
        self.assertNotIn('rocep4s0', str(topology))


class TestGetUcxNetDevices(unittest.TestCase):
    def test_unmapped_card_raises(self):
        with patch.object(
            linux_utils, 'get_gpu_nic_mapping_dict', return_value={'n': {'card0': {'gpu_bdf': '0001:01:00.0'}}}
        ):
            with self.assertRaisesRegex(ValueError, 'PCI domain'):
                linux_utils.get_ucx_net_devices(MagicMock())

    def test_multiple_cards(self):
        mapping = {'n': {'card0': {'rdma_dev': 'a'}, 'card1': {'rdma_dev': 'b'}}}
        with patch.object(linux_utils, 'get_gpu_nic_mapping_dict', return_value=mapping):
            self.assertEqual(linux_utils.get_ucx_net_devices(MagicMock()), 'a:1,b:1')


if __name__ == '__main__':
    unittest.main()
