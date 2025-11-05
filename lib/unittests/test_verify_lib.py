import unittest
from unittest.mock import MagicMock, patch


import lib.verify_lib as verify_lib

class TestVerifyGpuPcieBusWidth(unittest.TestCase):

    @patch('lib.verify_lib.get_gpu_pcie_bus_dict')
    @patch('lib.verify_lib.fail_test')
    def test_valid_bus_width(self, mock_fail_test, mock_get_bus_dict):
        mock_get_bus_dict.return_value = {
            'node1': {
                'card0': {'PCI Bus': '0000:01:00.0'},
                'card1': {'PCI Bus': '0000:02:00.0'}
            },
            'node2': {
                'card0': {'PCI Bus': '0000:03:00.0'},
                'card1': {'PCI Bus': '0000:04:00.0'}
            }
        }

        phdl = MagicMock()
        phdl.exec_cmd_list.return_value = {
            'node1': 'LnkSta: Speed 32GT/s, Width x16',
            'node2': 'LnkSta: Speed 32GT/s, Width x16'
        }

        result = verify_lib.verify_gpu_pcie_bus_width(phdl, expected_cards=2)
        self.assertEqual(result, {'node1': [], 'node2': []})
        mock_fail_test.assert_not_called()

    @patch('lib.verify_lib.get_gpu_pcie_bus_dict')
    @patch('lib.verify_lib.fail_test')
    def test_invalid_bus_speed(self, mock_fail_test, mock_get_bus_dict):
        mock_get_bus_dict.return_value = {
            'node1': {
                'card0': {'PCI Bus': '0000:01:00.0'}
            }
        }

        phdl = MagicMock()
        phdl.exec_cmd_list.return_value = {
            'node1': 'LnkSta: Speed 16GT/s, Width x16'
        }

        verify_lib.verify_gpu_pcie_bus_width(phdl, expected_cards=1)
        mock_fail_test.assert_called()


class TestVerifyGpuPcieErrors(unittest.TestCase):

    @patch('lib.verify_lib.get_gpu_metrics_dict')
    @patch('lib.verify_lib.fail_test')
    def test_valid_error_metrics(self, mock_fail_test, mock_get_metrics):
        mock_get_metrics.return_value = {
            'node1': {
                'card0': {
                    'pcie_l0_to_recov_count_acc (Count)': '10',
                    'pcie_nak_sent_count_acc (Count)': '20',
                    'pcie_nak_rcvd_count_acc (Count)': '30'
                }
            }
        }

        phdl = MagicMock()
        result = verify_lib.verify_gpu_pcie_errors(phdl)
        self.assertEqual(result, {'node1': []})
        mock_fail_test.assert_not_called()

    @patch('lib.verify_lib.get_gpu_metrics_dict')
    @patch('lib.verify_lib.fail_test')
    def test_threshold_exceeded(self, mock_fail_test, mock_get_metrics):
        mock_get_metrics.return_value = {
            'node1': {
                'card0': {
                    'pcie_l0_to_recov_count_acc (Count)': '101',
                    'pcie_nak_sent_count_acc (Count)': '150',
                    'pcie_nak_rcvd_count_acc (Count)': '200'
                }
            }
        }

        phdl = MagicMock()
        result = verify_lib.verify_gpu_pcie_errors(phdl)
        self.assertEqual(len(result['node1']), 3)
        mock_fail_test.assert_called()


if __name__ == '__main__':
    unittest.main()