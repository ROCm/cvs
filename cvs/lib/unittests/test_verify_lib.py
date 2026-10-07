import datetime
import os
import re
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

    @patch("cvs.lib.verify_lib.get_gpu_pcie_bus_dict")
    @patch("cvs.lib.verify_lib.fail_test")
    def test_auto_detects_expected_cards_when_omitted(self, mock_fail_test, mock_get_bus_dict):
        """Platforms with fewer than 8 GPUs per node (e.g. a 4-GPU Helios-R tray)
        must not be judged against a hardcoded expectation when none is given."""
        mock_get_bus_dict.return_value = {
            "node1": {"card0": {"PCI Bus": "0000:01:00.0"}, "card1": {"PCI Bus": "0000:02:00.0"}},
            "node2": {"card0": {"PCI Bus": "0000:03:00.0"}, "card1": {"PCI Bus": "0000:04:00.0"}},
        }

        phdl = MagicMock()
        phdl.exec_cmd_list.return_value = {
            "node1": "LnkSta: Speed 32GT/s, Width x16",
            "node2": "LnkSta: Speed 32GT/s, Width x16",
        }

        result = verify_lib.verify_gpu_pcie_bus_width(phdl)
        self.assertEqual(result, {"node1": [], "node2": []})
        mock_fail_test.assert_not_called()

    @patch("cvs.lib.verify_lib.get_gpu_pcie_bus_dict")
    @patch("cvs.lib.verify_lib.fail_test")
    def test_flags_node_disagreeing_with_auto_detected_count(self, mock_fail_test, mock_get_bus_dict):
        mock_get_bus_dict.return_value = {
            "node1": {"card0": {"PCI Bus": "0000:01:00.0"}, "card1": {"PCI Bus": "0000:02:00.0"}},
            "node2": {"card0": {"PCI Bus": "0000:03:00.0"}},
        }

        phdl = MagicMock()
        phdl.exec_cmd_list.return_value = {
            "node1": "LnkSta: Speed 32GT/s, Width x16",
            "node2": "LnkSta: Speed 32GT/s, Width x16",
        }

        verify_lib.verify_gpu_pcie_bus_width(phdl)
        mock_fail_test.assert_any_call('ERROR !! Number of cards not matching expected no 2 on node node2')


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
    def test_node_scraper_path_uses_adapter(self, mock_parse, mock_fail_test):
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


class TestDmesgMigrations(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(verify_lib.DMESG_PARSER_ENV, None)

    def test_parse_cvs_time(self):
        dt = verify_lib._parse_cvs_time("Mon Jun  5 08:53")
        self.assertIsNotNone(dt)
        self.assertEqual((dt.month, dt.day, dt.hour, dt.minute, dt.second), (6, 5, 8, 53, 0))
        self.assertIsNotNone(dt.tzinfo)
        self.assertIsNone(verify_lib._parse_cvs_time(""))
        self.assertIsNone(verify_lib._parse_cvs_time("garbage"))

    def test_parse_cvs_time_with_seconds(self):
        dt = verify_lib._parse_cvs_time("Mon Jun  5 08:53:27")
        self.assertIsNotNone(dt)
        self.assertEqual((dt.month, dt.day, dt.hour, dt.minute, dt.second), (6, 5, 8, 53, 27))

    def test_cvs_dmesg_error_regex_shape(self):
        regexes = verify_lib.cvs_dmesg_error_regex()
        self.assertTrue(regexes)
        for item in regexes:
            self.assertIn("regex", item)
            self.assertIn("message", item)
            self.assertIn("event_category", item)
            self.assertTrue(item["regex"].startswith("(?i)"))

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "parse_dmesg")
    def test_verify_dmesg_for_errors_uses_time_range(self, mock_parse, mock_fail):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        mock_parse.return_value = [{"description": "GPU Reset", "match_content": "GPU reset begin", "category": "RAS"}]
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "raw"}
        start = {"node1": "Mon Jun  5 08:00"}
        end = {"node1": "Mon Jun  5 09:00"}

        result = verify_lib.verify_dmesg_for_errors(phdl, start, end, till_end_flag=False)

        self.assertIn("--time-format iso -x", phdl.exec.call_args[0][0])
        passed_args = mock_parse.call_args.kwargs["analysis_args"]
        self.assertIn("analysis_range_start", passed_args)
        self.assertIn("analysis_range_end", passed_args)
        self.assertTrue(result["node1"])
        mock_fail.assert_called()

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "parse_dmesg")
    def test_verify_dmesg_for_errors_till_end_omits_end(self, mock_parse, mock_fail):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        mock_parse.return_value = []
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "raw"}
        start = {"node1": "Mon Jun  5 08:00"}
        end = {"node1": "Mon Jun  5 09:00"}

        verify_lib.verify_dmesg_for_errors(phdl, start, end, till_end_flag=True)

        passed_args = mock_parse.call_args.kwargs["analysis_args"]
        self.assertIn("analysis_range_start", passed_args)
        self.assertNotIn("analysis_range_end", passed_args)

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "parse_dmesg")
    def test_full_journalctl_scan_node_scraper(self, mock_parse, mock_fail):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        mock_parse.return_value = [
            {"description": "Out of memory error", "match_content": "Out of memory: killed", "category": "OS"}
        ]
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "raw"}

        result = verify_lib.full_journalctl_scan(phdl)

        self.assertIn("journalctl -k -o short-iso", phdl.exec.call_args[0][0])
        self.assertNotIn("--since", phdl.exec.call_args[0][0])
        self.assertTrue(result["node1"])
        mock_fail.assert_called()

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "parse_dmesg")
    def test_full_journalctl_scan_bounds_with_since(self, mock_parse, mock_fail):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        mock_parse.return_value = []
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "raw"}

        verify_lib.full_journalctl_scan(phdl, start_time_dict={"node1": "Mon Jun  5 08:53:27"})

        cmd = phdl.exec.call_args[0][0]
        expected_year = datetime.datetime.now().astimezone().year
        self.assertIn("journalctl -k -o short-iso", cmd)
        self.assertIn(f'--since="{expected_year}-06-05 08:53:27"', cmd)

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "parse_dmesg")
    def test_verify_driver_errors_filters_to_driver(self, mock_parse, mock_fail):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        mock_parse.return_value = [
            {
                "description": "amdgpu Page Fault",
                "match_content": "amdgpu 0000:01:00.0 page fault",
                "category": "SW_DRIVER",
            },
            {
                "description": "Filesystem corrupted!",
                "match_content": "EXT4-fs error (device sda1):",
                "category": "OS",
            },
        ]
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "raw"}

        result = verify_lib.verify_driver_errors(phdl)

        self.assertEqual(len(result["node1"]), 1)
        self.assertIn("amdgpu", result["node1"][0].lower())
        mock_fail.assert_called_once()


class TestNodeScraperTimeRangeFiltering(unittest.TestCase):
    """Exercises the real node-scraper analyzer (no mocking of parse_dmesg) to
    guard against analysis_range_end silently dropping events that occurred
    before a test's true end time but after that time got truncated to whole
    minutes.
    """

    def test_analysis_range_end_keeps_events_up_to_the_real_second(self):
        dmesg = "2026-07-17T10:16:30,000000+00:00 kern  :err   : [1.0] GPU reset begin on card0\n"

        end_with_seconds = datetime.datetime(2026, 7, 17, 10, 16, 45, tzinfo=datetime.timezone.utc)
        events = verify_lib.node_scraper_adapter.parse_dmesg(
            dmesg, analysis_args={"analysis_range_end": end_with_seconds}
        )
        self.assertEqual(len(events), 1, "event before the real (second-precision) end time must be kept")

        end_truncated_to_minute = datetime.datetime(2026, 7, 17, 10, 16, 0, tzinfo=datetime.timezone.utc)
        events = verify_lib.node_scraper_adapter.parse_dmesg(
            dmesg, analysis_args={"analysis_range_end": end_truncated_to_minute}
        )
        self.assertEqual(
            len(events),
            0,
            "minute-truncated analysis_range_end reproduces the historical bug "
            "(demonstrates why _parse_cvs_time must preserve seconds)",
        )


BENIGN_DMESG_LINES = [
    "infiniband rdma0: Changing to default roce traffic class DSCP 26 and SL 3",
    "PCI: CLS 64 bytes, default 64",
    "ast 0000:54:00.0: Using default configuration",
    "mpt3sas_cm0: CurrentHostPageSize is 0: Setting default host page size to 4k",
    "RAS: Correctable Errors collector initialized.",
    "RAS: Uncorrectable Errors collector initialized.",
]

REAL_FAULT_DMESG_LINES = [
    "python[3215696]: segfault at 75b800000034 ip 000075b86b22d5aa sp 00007ffef62526f0 error 4 "
    "in libmori_application.so[75b86b200000+c7000] likely on CPU 124 (core 84, socket 1)",
    "traps: python[4021] general protection fault ip:7f3a5c0b12e4 sp:7ffd2a8c1e50 error:0 in libc.so.6",
    "amdgpu 0000:05:00.0: amdgpu: [gfxhub] page fault (src_id:0 ring:153 vmid:8 pasid:32770)",
    "amdgpu 0000:05:00.0: amdgpu: [mmhub0] retry page fault (src_id:0 ring:0 vmid:0 pasid:0)",
    "amdgpu 0000:05:00.0: GPU fault detected: 146 0x0c680401",
    "BUG: unable to handle page fault for address: ffffc90000a3f000",
    "pcieport 0000:00:01.1: AER: Correctable error message received from 0000:01:00.0",
    "pcieport 0000:00:01.1: AER: Uncorrectable (Non-Fatal) error message received from 0000:01:00.0",
    "amdgpu 0000:05:00.0: amdgpu: Uncorrectable error detected in UMC inst: 0, chan_idx: 3",
]


class TestErrPatterns(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(verify_lib.DMESG_PARSER_ENV, None)

    def _matching_keys(self, line):
        return [key for key, pattern in verify_lib.err_patterns_dict.items() if re.search(pattern, line, re.I)]

    def test_benign_lines_match_no_error_pattern(self):
        for line in BENIGN_DMESG_LINES:
            with self.subTest(line=line):
                self.assertEqual(self._matching_keys(line), [])

    def test_real_fault_lines_match_an_error_pattern(self):
        for line in REAL_FAULT_DMESG_LINES:
            with self.subTest(line=line):
                self.assertTrue(self._matching_keys(line))

    def test_standalone_fault_word_still_flags_crash(self):
        for line in (
            "traps: python[4021] general protection fault ip:7f3a5c0b12e4 sp:7ffd2a8c1e50 error:0 in libc.so.6",
            "amdgpu 0000:05:00.0: GPU fault detected: 146 0x0c680401",
        ):
            with self.subTest(line=line):
                self.assertEqual(self._matching_keys(line), ["crash"])

    def test_segfault_flags_crash(self):
        self.assertIn("crash", self._matching_keys(REAL_FAULT_DMESG_LINES[0]))

    def test_correctable_error_report_still_flags_hardware(self):
        self.assertEqual(
            self._matching_keys("pcieport 0000:00:01.1: AER: Correctable error message received from 0000:01:00.0"),
            ["hardware"],
        )

    @patch("cvs.lib.verify_lib.fail_test")
    def test_legacy_verify_dmesg_ignores_benign_lines(self, mock_fail_test):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "\n".join(BENIGN_DMESG_LINES)}

        result = verify_lib.verify_dmesg_for_errors(
            phdl, {"node1": "Wed Sep 30 18:05:22"}, {"node1": "Wed Sep 30 18:26:07"}, till_end_flag=False
        )

        self.assertEqual(result, {"node1": []})
        mock_fail_test.assert_not_called()

    @patch("cvs.lib.verify_lib.fail_test")
    def test_legacy_verify_dmesg_flags_real_faults(self, mock_fail_test):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "\n".join(BENIGN_DMESG_LINES + REAL_FAULT_DMESG_LINES)}

        result = verify_lib.verify_dmesg_for_errors(
            phdl, {"node1": "Wed Sep 30 18:05:22"}, {"node1": "Wed Sep 30 18:26:07"}, till_end_flag=False
        )

        self.assertEqual(set(result["node1"]), set(REAL_FAULT_DMESG_LINES))
        mock_fail_test.assert_called()

    def _cvs_events(self, lines, level):
        dmesg = "".join(
            f"2026-09-30T16:38:51,{i:06d}+00:00 kern  :{level:<6}: {line}\n" for i, line in enumerate(lines)
        )
        events = verify_lib.node_scraper_adapter.parse_dmesg(
            dmesg, analysis_args={"error_regex": verify_lib.cvs_dmesg_error_regex()}
        )
        return [e for e in events if (e["description"] or "").startswith("CVS ")]

    def test_node_scraper_cvs_patterns_ignore_benign_lines(self):
        self.assertEqual(self._cvs_events(BENIGN_DMESG_LINES, "warn"), [])

    def test_node_scraper_cvs_patterns_flag_bare_fault(self):
        events = self._cvs_events(["amdgpu 0000:05:00.0: GPU fault detected: 146 0x0c680401"], "err")
        self.assertEqual([e["description"] for e in events], ["CVS crash pattern"])


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
