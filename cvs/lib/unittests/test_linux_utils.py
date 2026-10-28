# cvs/lib/unittests/test_linux_utils.py
import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cvs.lib.linux_utils as linux_utils
from cvs.unittests.support import run_trusted


class TestGetGpuNumaDict(unittest.TestCase):
    def test_each_node_uses_its_own_cards(self):
        for short_node in ('a', 'b'):
            with self.subTest(short_node=short_node):
                gpu_pcie = {
                    node: {
                        card: {'PCI Bus': f'0000:{index + 1:02x}:00.0'}
                        for index, card in enumerate(('card0',) if node == short_node else ('card0', 'card1'))
                    }
                    for node in ('a', 'b')
                }
                phdl = MagicMock()

                def exec_commands(commands):
                    return {
                        node: ('0-3\n' if 'local_cpulist' in command else '0\n') if command != 'true' else ''
                        for node, command in zip(('a', 'b'), commands)
                    }

                phdl.exec_cmd_list.side_effect = exec_commands
                with patch.object(linux_utils.rocm_plib, 'get_gpu_pcie_bus_dict', return_value=gpu_pcie):
                    result = linux_utils.get_gpu_numa_dict(phdl)

                self.assertEqual(list(result[short_node]), ['card0'])
                self.assertEqual(list(result['b' if short_node == 'a' else 'a']), ['card0', 'card1'])
                self.assertEqual(result[short_node]['card0'], {'local_cpulist': '0-3', 'numa_node': '0'})
                self.assertEqual(phdl.exec_cmd_list.call_count, 4)
                for call in (phdl.exec_cmd_list.call_args_list[1], phdl.exec_cmd_list.call_args_list[3]):
                    self.assertEqual(call.args[0][0 if short_node == 'a' else 1], 'true')


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
        mock_phdl.exec.assert_called_once_with(linux_utils.RDMA_LINK_CMD)

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


class TestRdmaLookup(unittest.TestCase):
    def test_rdma_link_runs_the_binary_found_on_path(self):
        with tempfile.TemporaryDirectory() as bindir:
            stub = os.path.join(bindir, 'rdma')
            with open(stub, 'w') as fp:
                fp.write('#!/bin/sh\necho "link stub0/1 state ACTIVE physical_state LINK_UP netdev eth9 args:$*"\n')
            os.chmod(stub, 0o700)

            def exec_locally(cmd):
                result = run_trusted(['/bin/bash', '-c', cmd], capture_output=True, text=True, env={'PATH': bindir})
                return {'n1': result.stdout}

            phdl = MagicMock()
            phdl.exec.side_effect = exec_locally
            self.assertEqual(linux_utils.get_active_rdma_nic_dict(phdl)['n1']['stub0']['eth_device'], 'eth9')
            self.assertIn('args:link', exec_locally(linux_utils.RDMA_LINK_CMD)['n1'])

    def test_rdma_lookup_falls_back_to_usr_sbin_when_not_on_path(self):
        with tempfile.TemporaryDirectory() as empty_path:
            result = run_trusted(
                ['/bin/bash', '-c', f'printf %s {linux_utils.RDMA_BIN}'],
                capture_output=True,
                text=True,
                env={'PATH': empty_path},
            )
        self.assertEqual(result.stdout, '/usr/sbin/rdma')

    def test_both_rdma_link_readers_use_the_lookup(self):
        for reader in (linux_utils.get_rdma_nic_dict, linux_utils.get_active_rdma_nic_dict):
            with self.subTest(reader=reader.__name__):
                phdl = MagicMock()
                phdl.exec.return_value = {'n1': ''}
                reader(phdl)
                phdl.exec.assert_called_once_with(linux_utils.RDMA_LINK_CMD)


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

    def test_gpu_numa_forwards_use_sudo(self):
        with patch.object(linux_utils.rocm_plib, 'get_gpu_pcie_bus_dict', return_value={'node1': {}}) as bus:
            linux_utils.get_gpu_numa_dict(MagicMock(), use_sudo=False)
        bus.assert_called_once()
        self.assertFalse(bus.call_args.kwargs['use_sudo'])


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
    def test_parses_domain_bus_device_function(self):
        self.assertEqual(linux_utils._parse_pci_bdf('0002:e1:1f.7'), (2, 0xE1, 0x1F, 7))

    def test_parses_vmd_wide_domain(self):
        self.assertEqual(linux_utils._parse_pci_bdf('10000:e1:00.0'), (0x10000, 0xE1, 0, 0))

    def test_missing_domain_defaults_to_zero(self):
        self.assertEqual(linux_utils._parse_pci_bdf('05:00.0'), (0, 5, 0, 0))

    def test_garbage_raises_value_error_naming_input(self):
        with self.assertRaisesRegex(ValueError, 'not-a-bdf'):
            linux_utils._parse_pci_bdf('not-a-bdf')

    def test_surrounding_junk_is_rejected(self):
        for bdf in ('0000:05:00.0:extra', 'junk 0000:05:00.0 junk'):
            with self.assertRaises(ValueError):
                linux_utils._parse_pci_bdf(bdf)


class TestGetGpuNicMappingDict(unittest.TestCase):
    GPU_BUSES = (0x05, 0x15, 0x65, 0x75, 0x85, 0x95, 0xE5, 0xF5)
    SINGLE_DOMAIN_GPUS = [f'0000:{bus:02x}:00.0' for bus in GPU_BUSES]
    SINGLE_DOMAIN_NICS = [(f'eth{i}', f'0000:{bus + 1:02x}:00.0', f'rdma{i}') for i, bus in enumerate(GPU_BUSES)]

    def _map(self, gpu_bdfs, nics):
        gpu_dict = {'node1': {f'card{i}': {'PCI Bus': bdf} for i, bdf in enumerate(gpu_bdfs)}}
        nic_dict = {'node1': {eth: {'pci_bus': bdf, 'rdma_dev': rdma} for eth, bdf, rdma in nics}}
        with (
            patch.object(linux_utils.rocm_plib, 'get_gpu_pcie_bus_dict', return_value=gpu_dict),
            patch.object(linux_utils, 'get_lshw_backend_nic_dict', return_value=nic_dict),
        ):
            return linux_utils.get_gpu_nic_mapping_dict(MagicMock())['node1']

    def _rdma_devs(self, gpu_bdfs, nics):
        result = self._map(gpu_bdfs, nics)
        return [result[f'card{i}']['rdma_dev'] for i in range(len(gpu_bdfs))]

    def test_single_domain_pairs_each_gpu_with_next_bus_nic(self):
        result = self._map(self.SINGLE_DOMAIN_GPUS, self.SINGLE_DOMAIN_NICS)
        self.assertEqual([result[f'card{i}']['rdma_dev'] for i in range(8)], [f'rdma{i}' for i in range(8)])
        self.assertEqual(
            result['card3'],
            {'gpu_bdf': '0000:75:00.0', 'eth_dev': 'eth3', 'rdma_dev': 'rdma3', 'nic_bdf': '0000:76:00.0'},
        )

    def test_node_with_no_gpus_or_nics_maps_to_empty(self):
        gpu_dict = {'n1': {'card0': {'PCI Bus': '0000:05:00.0'}}, 'n2': {}}
        nic_dict = {'n1': {'eth0': {'pci_bus': '0000:06:00.0', 'rdma_dev': 'rdma0'}}, 'n2': {}}
        with (
            patch.object(linux_utils.rocm_plib, 'get_gpu_pcie_bus_dict', return_value=gpu_dict),
            patch.object(linux_utils, 'get_lshw_backend_nic_dict', return_value=nic_dict),
        ):
            result = linux_utils.get_gpu_nic_mapping_dict(MagicMock())
        self.assertEqual(result['n1']['card0']['rdma_dev'], 'rdma0')
        self.assertEqual(result['n2'], {})

    def test_map_is_logged_per_node(self):
        with self.assertLogs(level='INFO') as logs:
            self._map(self.SINGLE_DOMAIN_GPUS, self.SINGLE_DOMAIN_NICS)
        self.assertEqual(len(logs.records), 1)
        self.assertIn('node node1: card0: 0000:05:00.0 -> rdma0 (0000:06:00.0); card1:', logs.output[0])

    def test_two_domains_on_bus_zero_pair_within_domain_in_device_order(self):
        domains = ('0002', '0003')
        gpus = [f'{domain}:00:{dev:02x}.0' for domain in domains for dev in range(1, 5)]
        nic_slots = [(domain, dev) for domain in domains for dev in range(9, 13)]
        nics = [(f'eth{i}', f'{domain}:00:{dev:02x}.0', f'ionic_{i}') for i, (domain, dev) in enumerate(nic_slots)]
        self.assertEqual(self._rdma_devs(gpus, nics), [f'ionic_{i}' for i in range(8)])

    def test_same_domain_nic_beats_nearer_bus_in_other_domain(self):
        nics = [('eth0', '0002:10:00.0', 'rdma0'), ('eth1', '0001:40:00.0', 'rdma1')]
        self.assertEqual(self._rdma_devs(['0001:10:00.0'], nics), ['rdma1'])

    def test_gpu_shares_nic_only_when_none_is_free(self):
        nics = [nic for nic in self.SINGLE_DOMAIN_NICS if nic[2] != 'rdma3']
        with self.assertLogs(level='WARNING') as logs:
            rdma_devs = self._rdma_devs(self.SINGLE_DOMAIN_GPUS, nics)
        self.assertEqual(rdma_devs, ['rdma0', 'rdma1', 'rdma2', 'rdma2', 'rdma4', 'rdma5', 'rdma6', 'rdma7'])
        self.assertEqual(len(logs.records), 1)
        self.assertIn('0000:75:00.0', logs.output[0])
        self.assertIn('node1', logs.output[0])

    def test_uneven_domains_share_within_domain(self):
        gpus = [f'{domain}:00:{dev:02x}.0' for domain in ('0002', '0003') for dev in range(1, 5)]
        nic_slots = [('0002', dev) for dev in range(9, 12)] + [('0003', dev) for dev in range(9, 14)]
        nics = [(f'eth{i}', f'{domain}:00:{dev:02x}.0', f'ionic_{i}') for i, (domain, dev) in enumerate(nic_slots)]
        with self.assertLogs(level='WARNING') as logs:
            result = self._map(gpus, nics)
        self.assertEqual(result['card3']['nic_bdf'], '0002:00:09.0')
        self.assertNotIn('0003:00:0d.0', str(result))
        self.assertEqual(len(logs.records), 1)
        for text in ('0002:00:04.0', 'shares NIC 0002:00:09.0', 'node1'):
            self.assertIn(text, logs.output[0])


class TestGetRdmaNicDictAllStates(unittest.TestCase):
    def _parse(self, output):
        phdl = MagicMock()
        phdl.exec.return_value = {'n1': output}
        return linux_utils.get_rdma_nic_dict(phdl)['n1']

    def test_unparsable_link_line_is_skipped(self):
        result = self._parse('link weird/1 state\nlink rdma0/1 state ACTIVE physical_state LINK_UP netdev eth0')
        self.assertEqual(list(result), ['rdma0'])

    def test_ib_link_without_netdev_is_parsed(self):
        output = (
            'link mlx5_0/1 subnet_prefix fe80:0000:0000:0000 lid 1 sm_lid 1 lmc 0 '
            'state ACTIVE physical_state LINK_UP\n'
            'link rdma0/1 state ACTIVE physical_state LINK_UP netdev eth0'
        )
        result = self._parse(output)
        self.assertEqual(list(result), ['mlx5_0', 'rdma0'])
        self.assertEqual(
            result['mlx5_0'],
            {'port': '1', 'device_status': 'ACTIVE', 'link_status': 'LINK_UP', 'eth_device': None},
        )

    def test_ib_link_active_only(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'n1': 'link mlx5_0/1 subnet_prefix fe80:0000:0000:0000 lid 1 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP'
        }
        result = linux_utils.get_active_rdma_nic_dict(phdl)['n1']
        self.assertIsNone(result['mlx5_0']['eth_device'])

    def test_underscore_netdev(self):
        result = self._parse('link rdma0/1 state ACTIVE physical_state LINK_UP netdev tw_eth0')
        self.assertEqual(result['rdma0']['eth_device'], 'tw_eth0')

    def test_down_port_does_not_hide_active_port(self):
        active = 'link mlx4_0/1 state ACTIVE physical_state LINK_UP netdev ib0'
        down = 'link mlx4_0/2 state DOWN physical_state DISABLED netdev ib1'
        for output in (f'{active}\n{down}', f'{down}\n{active}'):
            result = self._parse(output)
            self.assertEqual(result['mlx4_0']['eth_device'], 'ib0')
            self.assertEqual(result['mlx4_0']['device_status'], 'ACTIVE')

    def test_down_devices_included(self):
        output = (
            'link rdma0/1 state ACTIVE physical_state LINK_UP netdev eth0\n'
            'link rdma1/1 state DOWN physical_state DISABLED netdev eth1'
        )
        result = self._parse(output)
        self.assertEqual(result['rdma0']['device_status'], 'ACTIVE')
        self.assertEqual(result['rdma1']['device_status'], 'DOWN')


class TestBackendNicLinkStateFilter(unittest.TestCase):
    def test_get_backend_rdma_nic_dict_active_only(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'n1': '\n'.join(
                (
                    'link rdma0/1 state ACTIVE physical_state LINK_UP netdev eth0',
                    'link rdma1/1 state DOWN physical_state DISABLED netdev eth1',
                    'link rdma9/1 state ACTIVE physical_state LINK_UP netdev eth9',
                )
            )
        }
        with patch.object(linux_utils, 'get_backend_nic_dict', return_value={'n1': ['eth0', 'eth1']}):
            active = linux_utils.get_backend_rdma_nic_dict(phdl)
            all_states = linux_utils.get_backend_rdma_nic_dict(phdl, active_only=False)
        self.assertEqual(list(active['n1']), ['rdma0'])
        self.assertEqual(set(all_states['n1']), {'rdma0', 'rdma1'})

    def test_lshw_backend_forwards_active_only_and_status(self):
        lshw = {'n1': {'eth0': {'pci_bus': '0000:06:00.0', 'description': 'Backend NIC'}}}
        rdma = {
            'n1': {
                'rdma0': {'eth_device': 'eth0', 'device_status': 'ACTIVE'},
            }
        }
        for active_only in (True, False):
            with (
                patch.object(linux_utils, 'get_lshw_network_dict', return_value=lshw),
                patch.object(linux_utils, 'get_backend_rdma_nic_dict', return_value=rdma) as backend,
            ):
                result = linux_utils.get_lshw_backend_nic_dict(MagicMock(), active_only=active_only)
            self.assertEqual(backend.call_args.kwargs['active_only'], active_only)
            self.assertEqual(result['n1']['eth0']['rdma_dev'], 'rdma0')
            self.assertEqual(result['n1']['eth0']['device_status'], 'ACTIVE')


class TestGpuNicMappingPciDomain(unittest.TestCase):
    def _map(self, gpu_bdfs, nics, active_only=True):
        gpu_dict = {'node1': {f'card{i}': {'PCI Bus': bdf} for i, bdf in enumerate(gpu_bdfs)}}
        nic_dict = {
            'node1': {
                eth: {'pci_bus': bdf, 'rdma_dev': rdma, 'device_status': status} for eth, bdf, rdma, status in nics
            }
        }
        with (
            patch.object(linux_utils.rocm_plib, 'get_gpu_pcie_bus_dict', return_value=gpu_dict),
            patch.object(linux_utils, 'get_lshw_backend_nic_dict', return_value=nic_dict) as lshw,
        ):
            result = linux_utils.get_gpu_nic_mapping_dict(MagicMock(), active_only=active_only)
        return result['node1'], lshw

    def _helios(self):
        gpus = [f'000{i}:01:00.0' for i in range(1, 5)]
        nics = [('enp4s0np0', '0000:04:00.0', 'rocep4s0', 'ACTIVE')]
        for i in range(1, 5):
            for port, bus in ((68, 0x44), (72, 0x48)):
                nics.append((f'enP{i}p{port}s0', f'000{i}:{bus:02x}:00.0', f'roceP{i}p{port}s0', 'DOWN'))
        return gpus, nics

    def test_requests_backend_nics_in_every_link_state(self):
        gpus, nics = self._helios()
        for active_only in (True, False):
            _, lshw = self._map(gpus, nics, active_only=active_only)
            lshw.assert_called_once()
            self.assertFalse(lshw.call_args.kwargs['active_only'])

    def test_mi450_default_leaves_every_gpu_unmapped(self):
        gpus, nics = self._helios()
        with self.assertLogs(linux_utils.log, 'WARNING') as logs:
            result, _ = self._map(gpus, nics)
        for i, bdf in enumerate(gpus):
            self.assertEqual(result[f'card{i}'], {'gpu_bdf': bdf})
            for fragment in (f'card{i}', bdf, 'node1', f'PCI domain 000{i + 1}'):
                self.assertIn(fragment, logs.output[i])
        self.assertEqual(len(logs.records), 4)
        self.assertNotIn('rocep4s0', str(result))
        self.assertNotIn('0000:04:00.0', str(result))

    def test_mi450_topology_maps_each_gpu_to_its_p68_port(self):
        gpus, nics = self._helios()
        result, _ = self._map(gpus, nics, active_only=False)
        for i in range(1, 5):
            info = result[f'card{i - 1}']
            self.assertEqual(info['rdma_dev'], f'roceP{i}p68s0')
            self.assertEqual(info['eth_dev'], f'enP{i}p68s0')
            self.assertEqual(info['nic_bdf'], f'000{i}:44:00.0')
        self.assertNotIn('rocep4s0', str(result))

    def test_mi450_links_up_maps_same_domain(self):
        gpus, nics = self._helios()
        nics = [(eth, bdf, rdma, 'ACTIVE') for eth, bdf, rdma, _ in nics]
        result, _ = self._map(gpus, nics)
        for i in range(1, 5):
            self.assertEqual(result[f'card{i - 1}']['rdma_dev'], f'roceP{i}p68s0')
        self.assertNotIn('rocep4s0', str(result))

    def test_partial_overlap_leaves_gpu_without_active_local_nic_unmapped(self):
        gpus = ['0000:05:00.0', '0001:05:00.0']
        nics = [('eth0', '0000:06:00.0', 'rdma0', 'ACTIVE'), ('eth1', '0001:06:00.0', 'rdma1', 'DOWN')]
        with self.assertLogs(linux_utils.log, 'WARNING') as logs:
            result, _ = self._map(gpus, nics)
        self.assertEqual(
            result['card0'],
            {'gpu_bdf': gpus[0], 'eth_dev': 'eth0', 'rdma_dev': 'rdma0', 'nic_bdf': '0000:06:00.0'},
        )
        self.assertEqual(result['card1'], {'gpu_bdf': gpus[1]})
        self.assertEqual(len(logs.records), 1)
        self.assertIn(gpus[1], logs.output[0])
        self.assertIn('node1', logs.output[0])
        topology, _ = self._map(gpus, nics, active_only=False)
        self.assertEqual(topology['card1']['rdma_dev'], 'rdma1')

    def test_partial_overlap_gpu_domain_without_any_backend_nic_is_unmapped(self):
        gpus = ['0000:05:00.0', '0001:05:00.0']
        nics = [('eth0', '0000:06:00.0', 'rdma0', 'ACTIVE')]
        with self.assertLogs(linux_utils.log, 'WARNING'):
            result, _ = self._map(gpus, nics)
        self.assertEqual(result['card0']['rdma_dev'], 'rdma0')
        self.assertEqual(result['card1'], {'gpu_bdf': gpus[1]})

    def test_no_domain_overlap_keeps_cross_domain_pairing(self):
        with self.assertLogs(linux_utils.log, 'WARNING') as logs:
            result, _ = self._map(['0000:05:00.0'], [('eth0', '10000:e1:00.0', 'rdma0', 'ACTIVE')])
        self.assertEqual(result['card0']['rdma_dev'], 'rdma0')
        self.assertIn('outside its PCI domain', logs.output[0])

    def test_domain0_down_nic_shares_within_domain(self):
        gpus = TestGetGpuNicMappingDict.SINGLE_DOMAIN_GPUS
        nics = [
            (eth, bdf, rdma, 'DOWN' if rdma == 'rdma3' else 'ACTIVE')
            for eth, bdf, rdma in TestGetGpuNicMappingDict.SINGLE_DOMAIN_NICS
        ]
        with self.assertLogs(linux_utils.log, 'WARNING'):
            result, _ = self._map(gpus, nics)
        self.assertEqual(
            [result[f'card{i}']['rdma_dev'] for i in range(8)],
            ['rdma0', 'rdma1', 'rdma2', 'rdma2', 'rdma4', 'rdma5', 'rdma6', 'rdma7'],
        )

    def test_domain0_all_links_down_leaves_cards_unmapped_without_raising(self):
        gpus = TestGetGpuNicMappingDict.SINGLE_DOMAIN_GPUS
        nics = [(eth, bdf, rdma, 'DOWN') for eth, bdf, rdma in TestGetGpuNicMappingDict.SINGLE_DOMAIN_NICS]
        with self.assertLogs(linux_utils.log, 'WARNING'):
            result, _ = self._map(gpus, nics)
        for i, bdf in enumerate(gpus):
            self.assertEqual(result[f'card{i}'], {'gpu_bdf': bdf})

    def test_unmapped_card_in_info_line(self):
        gpus = ['0000:05:00.0', '0001:05:00.0']
        nics = [('eth0', '0000:06:00.0', 'rdma0', 'ACTIVE'), ('eth1', '0001:06:00.0', 'rdma1', 'DOWN')]
        with self.assertLogs(linux_utils.log, 'INFO') as logs:
            self._map(gpus, nics)
        self.assertIn('card1: 0001:05:00.0 -> unmapped', logs.output[-1])
        self.assertIn('card0: 0000:05:00.0 -> rdma0 (0000:06:00.0)', logs.output[-1])

    def test_down_local_nic_shares_in_domain_instead_of_crossing(self):
        gpus = ['0001:01:00.0', '0001:02:00.0']
        nics = [
            ('enp4s0np0', '0000:04:00.0', 'rocep4s0', 'ACTIVE'),
            ('enP1p68s0', '0001:44:00.0', 'roceP1p68s0', 'ACTIVE'),
            ('enP1p72s0', '0001:48:00.0', 'roceP1p72s0', 'DOWN'),
        ]
        with self.assertLogs(linux_utils.log, 'WARNING') as logs:
            result, _ = self._map(gpus, nics)
        self.assertEqual([result[f'card{i}']['rdma_dev'] for i in range(2)], ['roceP1p68s0', 'roceP1p68s0'])
        self.assertNotIn('rocep4s0', str(result))
        self.assertEqual(len(logs.records), 1)
        self.assertIn('shares NIC 0001:44:00.0', logs.output[0])

    def test_no_domain_overlap_without_active_nic_leaves_cards_unmapped(self):
        gpus = ['0000:05:00.0', '0000:15:00.0']
        nics = [('eth0', '10000:e1:00.0', 'rdma0', 'DOWN')]
        with self.assertLogs(linux_utils.log, 'WARNING') as logs:
            result, _ = self._map(gpus, nics)
        for i, bdf in enumerate(gpus):
            self.assertEqual(result[f'card{i}'], {'gpu_bdf': bdf})
        self.assertEqual(len(logs.records), 2)
        self.assertIn('no active backend NIC', logs.output[0])
        self.assertNotIn('PCI domain', logs.output[0])

    def test_node_with_gpus_and_no_backend_nics_leaves_cards_unmapped(self):
        gpu_dict = {
            'n1': {'card0': {'PCI Bus': '0000:05:00.0'}},
            'n2': {'card0': {'PCI Bus': '0000:05:00.0'}},
        }
        nic_dict = {
            'n1': {'eth0': {'pci_bus': '0000:06:00.0', 'rdma_dev': 'rdma0', 'device_status': 'ACTIVE'}},
            'n2': {},
        }
        with (
            patch.object(linux_utils.rocm_plib, 'get_gpu_pcie_bus_dict', return_value=gpu_dict),
            patch.object(linux_utils, 'get_lshw_backend_nic_dict', return_value=nic_dict),
            self.assertLogs(linux_utils.log, 'WARNING') as logs,
        ):
            result = linux_utils.get_gpu_nic_mapping_dict(MagicMock())
        self.assertEqual(result['n1']['card0']['rdma_dev'], 'rdma0')
        self.assertEqual(result['n2']['card0'], {'gpu_bdf': '0000:05:00.0'})
        self.assertIn('n2', logs.output[0])


class TestHeliosCaptureEndToEnd(unittest.TestCase):
    def setUp(self):
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
        output = {
            'rocm-smi': json.dumps({f'card{i - 1}': {'PCI Bus': f'000{i}:01:00.0'} for i in range(1, 5)}),
            'lshw': '\n'.join(lshw_lines),
            '/sys/class/infiniband': '\n'.join(netdevs),
            linux_utils.RDMA_LINK_CMD: '\n'.join(rdma_lines),
        }

        def fake_exec(cmd, *args, **kwargs):
            for key, value in output.items():
                if key in cmd:
                    return {'tray': value}
            raise AssertionError(cmd)

        self.phdl = MagicMock()
        self.phdl.exec.side_effect = fake_exec

    def test_default_mapping_never_selects_management_nic(self):
        with self.assertLogs(linux_utils.log, 'WARNING') as logs:
            mapping = linux_utils.get_gpu_nic_mapping_dict(self.phdl)['tray']
        self.assertEqual(len(mapping), 4)
        self.assertEqual(len(logs.records), 4)
        for i in range(1, 5):
            self.assertEqual(mapping[f'card{i - 1}'], {'gpu_bdf': f'000{i}:01:00.0'})
        self.assertNotIn('rocep4s0', str(mapping))

    def test_topology_mapping(self):
        mapping = linux_utils.get_gpu_nic_mapping_dict(self.phdl, active_only=False)['tray']
        for i in range(1, 5):
            self.assertEqual(mapping[f'card{i - 1}']['rdma_dev'], f'roceP{i}p68s0')
            self.assertEqual(mapping[f'card{i - 1}']['nic_bdf'], f'000{i}:44:00.0')
        self.assertNotIn('rocep4s0', str(mapping))

    def test_ucx_net_devices_refuses_unmapped_tray(self):
        with self.assertLogs(linux_utils.log, 'WARNING'):
            with self.assertRaisesRegex(ValueError, 'PCI domain'):
                linux_utils.get_ucx_net_devices(self.phdl)


if __name__ == '__main__':
    unittest.main()
