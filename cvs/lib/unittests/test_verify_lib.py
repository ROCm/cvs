import os
import unittest
from unittest.mock import MagicMock, patch


# Import the module under test
import cvs.lib.verify_lib as verify_lib


class TestVerifyGpuPcieBusWidth(unittest.TestCase):
    @patch("cvs.lib.verify_lib.get_gpu_pcie_bus_dict")
    @patch("cvs.lib.verify_lib.fail_test")
    def test_valid_bus_width(self, mock_fail_test, mock_get_bus_dict):
        mock_get_bus_dict.return_value = {
            "node1": {"card0": {"PCI Bus": "0000:01:00.0"}, "card1": {"PCI Bus": "0000:02:00.0"}},
            "node2": {"card0": {"PCI Bus": "0000:03:00.0"}, "card1": {"PCI Bus": "0000:04:00.0"}},
        }

        phdl = MagicMock()
        phdl.exec_cmd_list.return_value = {
            "node1": "LnkSta: Speed 32GT/s, Width x16",
            "node2": "LnkSta: Speed 32GT/s, Width x16",
        }

        result = verify_lib.verify_gpu_pcie_bus_width(phdl, expected_cards=2)
        self.assertEqual(result, {"node1": [], "node2": []})
        mock_fail_test.assert_not_called()

    @patch("cvs.lib.verify_lib.get_gpu_pcie_bus_dict")
    @patch("cvs.lib.verify_lib.fail_test")
    def test_invalid_bus_speed(self, mock_fail_test, mock_get_bus_dict):
        mock_get_bus_dict.return_value = {"node1": {"card0": {"PCI Bus": "0000:01:00.0"}}}

        phdl = MagicMock()
        phdl.exec_cmd_list.return_value = {"node1": "LnkSta: Speed 16GT/s, Width x16"}

        verify_lib.verify_gpu_pcie_bus_width(phdl, expected_cards=1)
        mock_fail_test.assert_called()


class TestVerifyGpuPcieErrors(unittest.TestCase):
    @patch("cvs.lib.verify_lib.get_gpu_metrics_dict")
    @patch("cvs.lib.verify_lib.fail_test")
    def test_valid_error_metrics(self, mock_fail_test, mock_get_metrics):
        mock_get_metrics.return_value = {
            "node1": {
                "card0": {
                    "pcie_l0_to_recov_count_acc (Count)": "10",
                    "pcie_nak_sent_count_acc (Count)": "20",
                    "pcie_nak_rcvd_count_acc (Count)": "30",
                }
            }
        }

        phdl = MagicMock()
        result = verify_lib.verify_gpu_pcie_errors(phdl)
        self.assertEqual(result, {"node1": []})
        mock_fail_test.assert_not_called()

    @patch("cvs.lib.verify_lib.get_gpu_metrics_dict")
    @patch("cvs.lib.verify_lib.fail_test")
    def test_threshold_exceeded(self, mock_fail_test, mock_get_metrics):
        mock_get_metrics.return_value = {
            "node1": {
                "card0": {
                    "pcie_l0_to_recov_count_acc (Count)": "101",
                    "pcie_nak_sent_count_acc (Count)": "150",
                    "pcie_nak_rcvd_count_acc (Count)": "200",
                }
            }
        }

        phdl = MagicMock()
        result = verify_lib.verify_gpu_pcie_errors(phdl)
        self.assertEqual(len(result["node1"]), 3)
        mock_fail_test.assert_called()


class TestFullDmesgScan(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(verify_lib.DMESG_PARSER_ENV, None)

    @patch("cvs.lib.verify_lib.fail_test")
    def test_legacy_path_matches_err_patterns(self, mock_fail_test):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node1": "Mar 1 00:00:00 host kernel: amdgpu page fault segfault at 0",
        }

        result = verify_lib.full_dmesg_scan(phdl)

        # legacy path collects with human-readable `dmesg -T`
        self.assertIn("dmesg -T", phdl.exec.call_args[0][0])
        self.assertTrue(result["node1"])
        mock_fail_test.assert_called()

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "parse_dmesg")
    @patch.object(verify_lib.node_scraper_adapter, "is_available", return_value=True)
    def test_node_scraper_path_uses_adapter(self, mock_avail, mock_parse, mock_fail_test):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        mock_parse.return_value = [
            {
                "priority": "ERROR",
                "category": "SW_DRIVER",
                "description": "Out of memory error",
                "match_content": "Out of memory: Killed process 123 (foo)",
                "count": 1,
                "timestamps": [],
                "source": "dmesg",
            }
        ]
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "raw dmesg text"}

        result = verify_lib.full_dmesg_scan(phdl)

        # node-scraper path collects with ISO timestamps + decoded prefix
        self.assertIn("--time-format iso -x", phdl.exec.call_args[0][0])
        mock_parse.assert_called_once()
        self.assertEqual(len(result["node1"]), 1)
        self.assertIn("Out of memory error", result["node1"][0])
        mock_fail_test.assert_called()

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "is_available", return_value=False)
    def test_falls_back_to_legacy_when_unavailable(self, mock_avail, mock_fail_test):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "nothing interesting here"}

        verify_lib.full_dmesg_scan(phdl)

        # even though node-scraper is requested, missing dep -> legacy `dmesg -T`
        self.assertIn("dmesg -T", phdl.exec.call_args[0][0])


class TestVerifyHostLspci(unittest.TestCase):
    def setUp(self):
        self.mock_phdl = MagicMock()

    @patch('cvs.lib.verify_lib.fail_test')
    def test_verify_host_lspci_failure(self, mock_fail_test):
        # Mock failing output
        self.mock_phdl.exec.return_value = {'node1': 'BDF: 0000:01:00.0'}
        self.mock_phdl.exec_cmd_list.return_value = {'node1': 'LnkSta: Speed 16GT/s, Width x8'}
        verify_lib.verify_host_lspci(self.mock_phdl, 32, 16)
        mock_fail_test.assert_called()


if __name__ == "__main__":
    unittest.main()
