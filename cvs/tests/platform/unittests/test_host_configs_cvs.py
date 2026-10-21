'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest
from unittest.mock import MagicMock, patch

from _pytest.outcomes import Failed

from cvs.lib import globals
from cvs.tests.platform import host_configs_cvs as suite


def _link(iface, speed='400000'):
    return f'{iface} speed={speed} operstate=up flags=0x1003\n'


def _orch(outputs):
    orch = MagicMock()
    orch.hosts = list(outputs)
    orch.all.exec.return_value = outputs
    return orch


class TestCheckDmesgDriverErrors(unittest.TestCase):
    def setUp(self):
        globals.error_list = []

    def test_dmesg_reads_disable_console_echo(self):
        orch = _orch({'n1': '\n', 'n2': '\n'})
        suite.test_check_dmesg_driver_errors(orch, {}, None, {}, None)
        orch.all.exec.assert_called_once()
        self.assertIs(orch.all.exec.call_args.kwargs['print_console'], False)
        self.assertIn('--time-format iso -x', orch.all.exec.call_args.args[0])

    def test_dmesg_driver_error_fails(self):
        line = 'kern  :err   : 2026-09-30T16:38:51,000000+00:00 amdgpu 0000:05:00.0: amdgpu: [drm] *ERROR* ring gfx timeout'
        orch = _orch({'n1': line + '\n', 'n2': '\n'})
        host_res_dict = {}
        with self.assertRaises(Failed):
            suite.test_check_dmesg_driver_errors(orch, {}, host_res_dict, {}, None)
        self.assertTrue(any('n1' in message for message in globals.error_list))


class TestCheckBeNicLinkSpeed(unittest.TestCase):
    def setUp(self):
        globals.error_list = []

    def _run(self, orch, config):
        host_res_dict = {}
        with self.assertRaises(Failed) as ctx:
            suite.test_check_be_nic_link_speed(orch, config, host_res_dict, {}, None)
        return host_res_dict['groups']['nic_link']['nodes'], str(ctx.exception)

    def test_invalid_speed_records_fail_for_every_node(self):
        orch = _orch({'n1': '', 'n2': ''})
        nodes, failure = self._run(orch, {'nic_link_speed': '400G'})
        self.assertIn('nic_link_speed must be a positive integer', failure)
        self.assertEqual({node: record['status'] for node, record in nodes.items()}, {'n1': 'fail', 'n2': 'fail'})
        self.assertEqual(nodes['n1']['items_summary'], 'config error')
        orch.all.exec.assert_not_called()

    def test_auto_detect_fails_node_with_fewer_nics(self):
        orch = _orch({'n1': _link('eth0') + _link('eth1'), 'n2': _link('eth0')})
        detected = {'n1': ['eth0', 'eth1'], 'n2': ['eth0']}
        with patch.object(suite.linux_utils, 'get_backend_nic_dict', return_value=detected):
            nodes, failure = self._run(orch, {'nic_link_interfaces': []})
        self.assertEqual(nodes['n1']['status'], 'pass')
        self.assertEqual(nodes['n2']['status'], 'fail')
        self.assertIn('Only 1 backend NIC(s) detected on node n2', failure)
        orch.all.exec.assert_called_once()

    def test_configured_interfaces_skip_count_comparison(self):
        orch = _orch({'n1': _link('eth0'), 'n2': _link('eth0') + _link('eth1', '200000')})
        suite.test_check_be_nic_link_speed(orch, {'nic_link_interfaces': ['eth0']}, {}, {}, None)
        self.assertEqual(globals.error_list, [])


class TestCheckBeNicPcieSpeedWidth(unittest.TestCase):
    def setUp(self):
        globals.error_list = []

    def test_maps_nics_in_every_link_state(self):
        orch = MagicMock()
        orch.all.exec_cmd_list.return_value = {'n1': 'LnkSta: Speed 32GT/s, Width x16'}
        mapping = {'n1': {'card0': {'gpu_bdf': '0001:05:00.0', 'nic_bdf': '0001:06:00.0'}}}
        config = {'nic_pcie_speed': '32', 'nic_pcie_width': '16'}
        with patch.object(suite.linux_utils, 'get_gpu_nic_mapping_dict', return_value=mapping) as nic_map:
            suite.test_check_be_nic_pcie_speed_width(orch, config, {}, {}, None)
        nic_map.assert_called_once_with(orch.all, use_sudo=False, active_only=False)
        self.assertIn('0001:06:00.0', orch.all.exec_cmd_list.call_args.args[0][0])
        self.assertEqual(globals.error_list, [])


if __name__ == '__main__':
    unittest.main()
