"""Unit tests for preflight backend RDMA interface checks."""

import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.preflight.interface_consistency import InterfaceConsistencyCheck


def _rdma_devices(disabled=None):
    return {
        f'mlx5_{i}': {
            'device_status': 'Disabled' if i == disabled else 'ACTIVE',
            'link_status': 'LinkUp',
        }
        for i in range(9)
    }


class TestInterfaceConsistencyCheck(unittest.TestCase):
    def setUp(self):
        self.orch = MagicMock()
        self.orch.all.reachable_hosts = ['n1', 'n2']
        self.names = [f'mlx5_{i}' for i in range(1, 9)]
        self.config = {'connectivity_check': {'rdma': {'expected_nic_count': 8}}}

    def _run(self, config=None, devices=None, names=None):
        check = InterfaceConsistencyCheck(self.orch, names or self.names, config if config is not None else self.config)
        with patch('cvs.lib.preflight.interface_consistency.linux_utils.get_rdma_nic_dict') as rdma:
            rdma.return_value = {node: devices if devices is not None else _rdma_devices() for node in ('n1', 'n2')}
            results = check.run()
        rdma.assert_called_once_with(self.orch.all)
        return results

    def test_unlisted_management_device_does_not_fail(self):
        results = self._run()
        for result in results.values():
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['unexpected_interfaces'], ['mlx5_0'])
            self.assertEqual(result['active_expected_count'], 8)
            self.assertEqual(result['expected_nic_count'], 8)

    def test_disabled_backend_device_fails(self):
        result = self._run(devices=_rdma_devices(disabled=4))['n1']
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('7 of 8 backend RDMA NICs present and active', result['errors'])

    def test_short_configured_list_fails_every_node(self):
        results = self._run(names=self.names[:7])
        for result in results.values():
            self.assertEqual(result['status'], 'FAIL')
            self.assertIn(
                'connectivity_check.rdma.expected_nic_count is 8 but connectivity_check.rdma.interfaces lists 7 device(s); list every backend RDMA device',
                result['errors'],
            )

    def test_missing_count_uses_interface_list_length(self):
        results = self._run(config={}, names=self.names[:7])
        for result in results.values():
            self.assertEqual(result['expected_nic_count'], 7)
            self.assertEqual(result['status'], 'PASS')

    def test_invalid_count_fails_every_node(self):
        config = {'connectivity_check': {'rdma': {'expected_nic_count': 'x'}}}
        results = self._run(config=config)
        for result in results.values():
            self.assertEqual(result['status'], 'FAIL')
            self.assertIn('connectivity_check.rdma.expected_nic_count must be a positive integer', result['errors'][0])

    def test_no_rdma_devices_reports_count(self):
        result = self._run(devices={})['n1']
        self.assertEqual(result['active_expected_count'], 0)
        self.assertIn('0 of 8 backend RDMA NICs present and active', result['errors'])
