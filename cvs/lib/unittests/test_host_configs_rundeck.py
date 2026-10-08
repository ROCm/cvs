'''Unit tests for cvs.lib.host_configs_rundeck (Run Deck capture for host config checks).'''

import shlex
import unittest

from cvs.lib import host_configs_rundeck
from cvs.lib.report.profile import load_json_profile

_OS_OLD = 'NAME="Ubuntu"\nVERSION="22.04.1 LTS"\n'
_OS_NEW = 'NAME="Ubuntu"\nVERSION="24.04.1 LTS"\n'
_KERNEL_OLD = 'Linux n1 5.15.0-179-generic #179-Ubuntu SMP Fri Jan 1 00:00:00 UTC 2026 x86_64 GNU/Linux\n'
_KERNEL_NEW = 'Linux n1 6.8.0-60-generic #60-Ubuntu SMP Fri Jan 1 00:00:00 UTC 2026 x86_64 GNU/Linux\n'
_LNK_OK = 'LnkSta: Speed 32GT/s, Width x16\n'
_LNK_SLOW = 'LnkSta: Speed 16GT/s (downgraded), Width x8\n'


class TestMakeMeta(unittest.TestCase):
    def test_pulls_cluster_name(self):
        meta = host_configs_rundeck.make_meta({'cluster_name': 'helios'}, 'host_configs_cvs', '7.0.2')
        self.assertEqual(meta['cluster'], 'helios')
        self.assertEqual(meta['version'], '7.0.2')
        self.assertEqual(meta['version_label'], 'ROCm version')
        self.assertEqual(meta['suite'], 'host_configs_cvs')

    def test_defaults_when_missing(self):
        meta = host_configs_rundeck.make_meta({}, '', None)
        self.assertEqual(meta['cluster'], '—')
        self.assertEqual(meta['version'], '—')
        self.assertEqual(meta['suite'], 'host_configs_cvs')

    def test_profile_labels_match_check_ids(self):
        profile = load_json_profile('host_configs_cvs')
        self.assertEqual(profile['lifecycle']['session_labels'], list(host_configs_rundeck.CHECK_IDS))


class TestVersionChecks(unittest.TestCase):
    def test_os_release_pass_and_fail(self):
        records, messages = host_configs_rundeck.eval_os_release(
            {'n1': _OS_NEW, 'n2': _OS_OLD},
            '24.04',
        )
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(records['n1']['items_summary'], '24.04')
        self.assertEqual(records['n1']['items'], [])
        self.assertEqual(records['n2']['status'], 'fail')
        self.assertEqual(records['n2']['items_summary'], '22.04.1')
        self.assertEqual(
            messages,
            ['Installed OS Version 22.04.1  not matching expected version 24.04 on node n2'],
        )

    def test_os_release_match_is_case_insensitive(self):
        text = 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\n' + _OS_NEW
        records, messages = host_configs_rundeck.eval_os_release({'n1': text}, 'ubuntu 24.04')
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(messages, [])

    def test_kernel_version_extracts_generic_build(self):
        records, messages = host_configs_rundeck.eval_kernel_version(
            {'n1': _KERNEL_OLD},
            '6.8.0-60-generic',
        )
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(
            messages,
            ['Installed Kernel Version 5.15.0-179-generic not matching expected version 6.8.0-60-generic on node n1'],
        )

    def test_kernel_version_pass(self):
        records, messages = host_configs_rundeck.eval_kernel_version({'n1': _KERNEL_NEW}, '6.8.0-60-generic')
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(messages, [])

    def test_bios_version(self):
        records, messages = host_configs_rundeck.eval_bios_version({'n1': '20171212\n', 'n2': 'ABC-99\n'}, '20171212')
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(records['n2']['status'], 'fail')
        self.assertEqual(
            messages,
            ['Installed BIOS Version ABC-99 not matching expected version 20171212 on node n2'],
        )

    def test_rocm_version_detects_banner_and_fails_mismatch(self):
        text = 'ROCm version: 6.3.1\n'
        records, messages, detected = host_configs_rundeck.eval_rocm_version({'node-a': text}, '7.0.2')
        self.assertEqual(detected, '6.3.1')
        self.assertEqual(records['node-a']['status'], 'fail')
        self.assertEqual(
            messages,
            ['Installed rocm version 6.3.1 not matching expected version 7.0.2 on node node-a'],
        )

    def test_rocm_version_pass_uses_expected_when_banner_regex_misses(self):
        records, messages, detected = host_configs_rundeck.eval_rocm_version(
            {'n1': 'package 7.0.2 installed\n'}, '7.0.2'
        )
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(messages, [])
        self.assertEqual(detected, '7.0.2')


class TestHostFlags(unittest.TestCase):
    def test_firmware_reports_each_mismatch(self):
        out = {
            'n1': [
                {
                    'gpu': 0,
                    'fw_list': [
                        {'fw_id': 'RLC', 'fw_version': '65'},
                        {'fw_id': 'PM', 'fw_version': '1.0'},
                    ],
                },
                {'gpu': 1, 'fw_list': [{'fw_id': 'RLC', 'fw_version': '65'}]},
            ]
        }
        records, messages = host_configs_rundeck.eval_firmware(out, {'RLC': '65', 'PM': '2.0'})
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(records['n1']['items_summary'], '1 firmware mismatch(es)')
        self.assertEqual(len(messages), 1)
        self.assertIn('For Firmware PM actual FW version 1.0 for gpu 0 on node n1', messages[0])
        self.assertIn('expected FW version 2.0', messages[0])

    def test_firmware_pass(self):
        out = {'n1': [{'gpu': 0, 'fw_list': [{'fw_id': 'RLC', 'fw_version': '65'}]}]}
        records, messages = host_configs_rundeck.eval_firmware(out, {'RLC': '65'})
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(records['n1']['items_summary'], 'firmware matched')
        self.assertEqual(messages, [])

    def test_pci_realloc_and_iommu(self):
        cmdline = {'n1': 'pci=realloc=off iommu=pt\n', 'n2': 'quiet\n'}
        realloc_records, realloc_messages = host_configs_rundeck.eval_pci_realloc(cmdline, 'off')
        iommu_records, iommu_messages = host_configs_rundeck.eval_iommu_pt(cmdline)
        self.assertEqual(realloc_records['n1']['status'], 'pass')
        self.assertEqual(realloc_records['n2']['status'], 'fail')
        self.assertEqual(realloc_messages, ['PCI realloc flag not set to off on node n2'])
        self.assertEqual(iommu_records['n2']['status'], 'fail')
        self.assertEqual(iommu_messages, ['IOMMU not set to pt on node n2'])

    def test_numa_accepts_either_spacing(self):
        spaced, spaced_messages = host_configs_rundeck.eval_numa_balancing({'n1': 'kernel.numa_balancing = 0\n'})
        tight, tight_messages = host_configs_rundeck.eval_numa_balancing({'n1': 'kernel.numa_balancing=0\n'})
        enabled, enabled_messages = host_configs_rundeck.eval_numa_balancing({'n1': 'kernel.numa_balancing = 1\n'})
        self.assertEqual(spaced['n1']['status'], 'pass')
        self.assertEqual(tight['n1']['status'], 'pass')
        self.assertEqual(spaced_messages, [])
        self.assertEqual(tight_messages, [])
        self.assertEqual(enabled['n1']['status'], 'fail')
        self.assertEqual(enabled_messages, ['NUMA balancing not disabled on node n1'])

    def test_online_memory(self):
        text = 'RANGE SIZE  STATE\nTotal online memory: 512G\n'
        records, messages = host_configs_rundeck.eval_online_memory({'n1': text}, '1.3T')
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(records['n1']['items_summary'], '512G')
        self.assertEqual(messages, ['Total online memory 512G not matching expected online mem 1.3T on node n1'])

    def test_pci_accelerators_counts_advanced_lines(self):
        text = '01:00.0 Processing accelerators: Advanced Micro Devices\n03:00.0 Processing accelerators: Advanced\n'
        records, messages = host_configs_rundeck.eval_pci_accelerators({'n1': text}, '8')
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(
            messages,
            ['Expected GPU count in PCI 8 not matching actual GPU count 2 on node n1'],
        )

    def test_pci_acs_fails_only_when_acsctl_is_present(self):
        clean, clean_messages = host_configs_rundeck.eval_pci_acs({'n1': ''})
        dirty, dirty_messages = host_configs_rundeck.eval_pci_acs({'n1': 'ACSCtl: SrcValid+\n'})
        self.assertEqual(clean['n1']['status'], 'pass')
        self.assertEqual(clean_messages, [])
        self.assertEqual(dirty['n1']['status'], 'fail')
        self.assertEqual(dirty_messages, ['PCIe ACS not disabled on node n1'])

    def test_dmesg_driver_and_reset_use_different_patterns(self):
        driver, driver_messages = host_configs_rundeck.eval_dmesg_driver({'n1': 'amdgpu: ring timeout error\n'})
        reset, reset_messages = host_configs_rundeck.eval_dmesg_reset({'n1': 'amdgpu: Traceback\n'})
        hung, hung_messages = host_configs_rundeck.eval_dmesg_reset({'n1': 'amdgpu: GPU reset\n'})
        self.assertEqual(driver['n1']['status'], 'fail')
        self.assertEqual(driver_messages, ['Dmesg has amdgpu driver errors on node n1'])
        self.assertEqual(reset['n1']['status'], 'pass')
        self.assertEqual(reset_messages, [])
        self.assertEqual(hung['n1']['status'], 'fail')
        self.assertEqual(hung_messages, ['Dmesg has amdgpu reset/hang errors on node n1'])


class TestPcieLinks(unittest.TestCase):
    def test_gpu_link_accumulates_across_cards(self):
        bus = {
            'n1': {
                0: {'PCI Bus': '0000:03:00.0'},
                1: {'PCI Bus': '0000:04:00.0'},
            }
        }
        records = {}
        first = host_configs_rundeck.absorb_gpu_pcie(records, {'n1': _LNK_OK}, bus, 0, '32', '16')
        second = host_configs_rundeck.absorb_gpu_pcie(records, {'n1': _LNK_SLOW}, bus, 1, '32', '16')
        self.assertEqual(first, [])
        self.assertEqual(len(second), 3)
        self.assertIn('expected 32GT/s but got', second[0])
        self.assertIn('expected 16 but got', second[1])
        self.assertIn('downgraded state for bus 0000:04:00.0 on node n1', second[2])
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(records['n1']['items_summary'], '3 link check(s) failed')
        self.assertEqual(
            [item['name'] for item in records['n1']['items']],
            ['0000:04:00.0 speed', '0000:04:00.0 width', '0000:04:00.0 downgrade'],
        )

    def test_gpu_link_pass_summary(self):
        bus = {'n1': {0: {'PCI Bus': '0000:03:00.0'}}}
        records = {}
        messages = host_configs_rundeck.absorb_gpu_pcie(records, {'n1': _LNK_OK}, bus, 0, '32', '16')
        self.assertEqual(messages, [])
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(records['n1']['items_summary'], 'Speed 32GT Width x16')

    def test_nic_link_messages(self):
        bus = {'n1': {0: {'nic_bdf': '0000:41:00.0'}}}
        records = {}
        messages = host_configs_rundeck.absorb_nic_pcie(records, {'n1': _LNK_SLOW}, bus, 0, '32', '16')
        self.assertEqual(
            messages[0],
            f'NIC PCIe speed mismatch for 0000:41:00.0 on n1: expected 32GT/s, got {_LNK_SLOW}',
        )
        self.assertIn('expected x16', messages[1])
        self.assertEqual(messages[2], 'NIC PCIe in downgraded state for 0000:41:00.0 on n1')


class TestNicLinkSpeed(unittest.TestCase):
    def test_setting_defaults_and_parses(self):
        self.assertEqual(host_configs_rundeck.parse_nic_link_speed_setting(None), 400000)
        for value, expected in [('400000', 400000), (400000, 400000), (' 200000 ', 200000)]:
            with self.subTest(value=value):
                self.assertEqual(host_configs_rundeck.parse_nic_link_speed_setting(value), expected)

    def test_setting_rejects_invalid(self):
        for value in ('400G', '0', '-1', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nic_link_speed'):
                host_configs_rundeck.parse_nic_link_speed_setting(value)

    def test_cmd_dedups_sorts_and_quotes(self):
        self.assertEqual(
            host_configs_rundeck.nic_link_speed_cmd(['eth1', 'eth0', 'eth1']),
            'for i in eth0 eth1; do echo "$i $(cat "/sys/class/net/$i/speed" 2>/dev/null)"; done',
        )
        name = 'eth0;touch /tmp/unwanted'
        self.assertIn(shlex.quote(name), host_configs_rundeck.nic_link_speed_cmd([name]))

    def test_cmd_empty_is_noop(self):
        self.assertEqual(host_configs_rundeck.nic_link_speed_cmd([]), 'true')

    def test_parse_skips_unreadable_lines(self):
        self.assertEqual(
            host_configs_rundeck.parse_nic_link_speeds('eth0 400000\neth1 \neth2 -1\n'),
            {'eth0': 400000, 'eth2': -1},
        )

    def test_pass(self):
        records, messages = host_configs_rundeck.eval_nic_link_speed(
            {'n1': 'eth0 400000\neth1 400000\n'}, {'n1': ['eth0', 'eth1']}, 400000
        )
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(records['n1']['items_summary'], '2 x 400000 Mb/s')
        self.assertEqual(records['n1']['items'], [])
        self.assertEqual(messages, [])

    def test_lower_speed_fails_with_node_iface_actual(self):
        records, messages = host_configs_rundeck.eval_nic_link_speed(
            {'n1': 'eth0 400000\neth1 200000\n'}, {'n1': ['eth0', 'eth1']}, 400000
        )
        message = 'Backend NIC eth1 link speed 200000 Mb/s not matching expected 400000 Mb/s on node n1'
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(records['n1']['items_summary'], '1 of 2 NIC(s) not at 400000 Mb/s')
        self.assertEqual(records['n1']['items'], [{'name': 'eth1 speed', 'status': 'fail', 'message': message}])
        self.assertEqual(messages, [message])

    def test_unreadable_speed_fails(self):
        records, messages = host_configs_rundeck.eval_nic_link_speed(
            {'n1': 'eth0 \n', 'n2': 'SSH connection failed'},
            {'n1': ['eth0', 'eth1'], 'n2': ['eth0', 'eth1']},
            400000,
        )
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(records['n2']['status'], 'fail')
        self.assertEqual(len(messages), 4)
        for node in ('n1', 'n2'):
            for iface in ('eth0', 'eth1'):
                self.assertIn(
                    f'Backend NIC {iface} link speed unreadable from /sys/class/net/{iface}/speed, '
                    f'expected 400000 Mb/s on node {node}',
                    messages,
                )

    def test_unknown_speed_reports_link_down(self):
        records, messages = host_configs_rundeck.eval_nic_link_speed(
            {'n1': 'eth0 -1\neth1 4294967295\n'}, {'n1': ['eth0', 'eth1']}, 400000
        )
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(
            messages,
            [
                'Backend NIC eth0 link speed unknown (-1), link may be down, expected 400000 Mb/s on node n1',
                'Backend NIC eth1 link speed unknown (4294967295), link may be down, expected 400000 Mb/s on node n1',
            ],
        )

    def test_per_node_lists_ignore_other_nodes_interfaces(self):
        records, messages = host_configs_rundeck.eval_nic_link_speed(
            {'n1': 'ethA 400000\nethB 200000\n', 'n2': 'ethA 200000\nethB 400000\n'},
            {'n1': ['ethA'], 'n2': ['ethB']},
            400000,
        )
        self.assertEqual(records['n1']['status'], 'pass')
        self.assertEqual(records['n2']['status'], 'pass')
        self.assertEqual(messages, [])

    def test_node_without_nics_fails(self):
        records, messages = host_configs_rundeck.eval_nic_link_speed({'n1': ''}, {'n1': [], 'n2': ['eth0']}, 400000)
        self.assertEqual(records['n1']['status'], 'fail')
        self.assertEqual(records['n1']['items_summary'], 'no NICs')
        self.assertEqual(records['n2']['status'], 'fail')
        self.assertEqual(
            messages,
            [
                'No backend NICs found on node n1 to check link speed',
                'Backend NIC eth0 link speed unreadable from /sys/class/net/eth0/speed, '
                'expected 400000 Mb/s on node n2',
            ],
        )

    def test_custom_expected_speed(self):
        slow, slow_messages = host_configs_rundeck.eval_nic_link_speed(
            {'n1': 'eth0 400000\n'}, {'n1': ['eth0']}, 800000
        )
        matching, matching_messages = host_configs_rundeck.eval_nic_link_speed(
            {'n1': 'eth0 200000\n'}, {'n1': ['eth0']}, 200000
        )
        self.assertEqual(slow['n1']['status'], 'fail')
        self.assertEqual(
            slow_messages,
            ['Backend NIC eth0 link speed 400000 Mb/s not matching expected 800000 Mb/s on node n1'],
        )
        self.assertEqual(matching['n1']['status'], 'pass')
        self.assertEqual(matching_messages, [])


class TestRecordGroup(unittest.TestCase):
    def test_version_placeholder_is_replaced_by_later_real_version(self):
        results = {}
        host_configs_rundeck.record_group(
            results,
            'os_release',
            {'n1': host_configs_rundeck.build_node_record('pass', [], '24.04')},
            meta=host_configs_rundeck.make_meta({'name': 'lab'}, 'host_configs_cvs'),
        )
        host_configs_rundeck.record_group(
            results,
            'rocm_version',
            {'n1': host_configs_rundeck.build_node_record('pass', [], '7.0.2')},
            meta=host_configs_rundeck.make_meta({'name': 'lab'}, 'host_configs_cvs', '7.0.2'),
        )
        self.assertEqual(results['_meta']['version'], '7.0.2')
        self.assertEqual(results['_meta']['cluster'], 'lab')
        self.assertEqual(results['groups']['os_release']['nodes']['n1']['status'], 'pass')

    def test_rerun_overwrites_same_node(self):
        results = {}
        host_configs_rundeck.record_group(
            results,
            'iommu_pt',
            {'n1': host_configs_rundeck.build_node_record('fail', [], 'missing')},
        )
        host_configs_rundeck.record_group(
            results,
            'iommu_pt',
            {'n1': host_configs_rundeck.build_node_record('pass', [], 'iommu=pt')},
        )
        self.assertEqual(results['groups']['iommu_pt']['nodes']['n1']['status'], 'pass')


if __name__ == '__main__':
    unittest.main()
