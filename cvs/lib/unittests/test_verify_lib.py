import datetime
import os
import shutil
import subprocess
import unittest
from unittest.mock import MagicMock, patch


# Import the module under test
import cvs.lib.verify_lib as verify_lib

_HAS_BASH_AND_AWK = shutil.which("bash") and shutil.which("awk")


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
        self.assertIn("grep -E -v", phdl.exec.call_args[0][0])
        self.assertNotIn("egrep", phdl.exec.call_args[0][0])
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


class TestVerifyDmesgDuringTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(verify_lib.DMESG_PARSER_ENV, None)

    @staticmethod
    def _run_marker_awk_slice(dmesg_text, start_marker, end_marker):
        """Run the same awk slice used remotely against in-memory dmesg text."""
        full_cmd = verify_lib._dmesg_slice_by_markers_cmd(start_marker, end_marker)
        awk_tail = full_cmd.rsplit(" | awk ", 1)[1]
        completed = subprocess.run(
            ["bash", "-c", f"awk {awk_tail}"],
            input=dmesg_text,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout

    @patch("cvs.lib.verify_lib.log")
    def test_nonempty_slice_without_start_marker_is_skipped(self, mock_log):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        phdl = MagicMock()
        phdl.exec.return_value = {"node1": "unrelated kernel line\n"}

        with patch("cvs.lib.verify_lib.fail_test") as mock_fail:
            result = verify_lib.verify_dmesg_during_test(
                phdl, "Starting Test all_reduce_perf", "End of Test all_reduce_perf"
            )

        self.assertEqual(result, {"node1": []})
        mock_fail.assert_not_called()
        mock_log.warning.assert_called()
        warn_msg = mock_log.warning.call_args[0][0]
        self.assertIn("start marker", warn_msg)
        self.assertIn("not found in slice", warn_msg)

    @patch("cvs.lib.verify_lib.fail_test")
    def test_legacy_slice_command_uses_markers(self, mock_fail_test):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node1": (
                "line before\n"
                "Starting Test all_reduce_perf\n"
                "Mar 1 00:00:00 host kernel: amdgpu page fault segfault at 0\n"
                "End of Test all_reduce_perf\n"
                "line after\n"
            )
        }

        result = verify_lib.verify_dmesg_during_test(
            phdl, "Starting Test all_reduce_perf", "End of Test all_reduce_perf"
        )

        cmd = phdl.exec.call_args[0][0]
        self.assertIn("dmesg -T", cmd)
        self.assertIn("grep -E -v", cmd)
        self.assertNotIn("egrep", cmd)
        self.assertIn("-v s='Starting Test all_reduce_perf'", cmd)
        self.assertIn("-v e='End of Test all_reduce_perf'", cmd)
        self.assertIn("last=buf", cmd)
        self.assertIn('printf "%s",last', cmd)
        self.assertTrue(result["node1"])
        mock_fail_test.assert_called()
        fail_msg = mock_fail_test.call_args[0][0]
        self.assertIn("Failure pattern ***", fail_msg)
        self.assertIn("on node node1", fail_msg)
        self.assertNotIn("Failue", fail_msg)

    def test_trim_trailing_blank_dmesg_lines(self):
        self.assertEqual(
            verify_lib._trim_trailing_blank_dmesg_lines("a\nb\n\n"),
            "a\nb\n",
        )
        self.assertEqual(
            verify_lib._trim_trailing_blank_dmesg_lines("a\n\nb\n\n"),
            "a\n\nb\n",
        )

    @unittest.skipUnless(_HAS_BASH_AND_AWK, "requires bash and awk")
    def test_awk_slice_has_no_trailing_blank_line(self):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        dmesg_text = "Starting Test all_reduce_perf\nEnd of Test all_reduce_perf\n"
        sliced = self._run_marker_awk_slice(dmesg_text, "Starting Test all_reduce_perf", "End of Test all_reduce_perf")
        self.assertEqual(len(sliced.splitlines()), 2)

    @patch("cvs.lib.verify_lib.log")
    def test_log_line_count_excludes_trailing_blank(self, mock_log):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node1": ("Starting Test all_reduce_perf\nEnd of Test all_reduce_perf\n\n"),
        }

        with patch("cvs.lib.verify_lib.fail_test"):
            verify_lib.verify_dmesg_during_test(phdl, "Starting Test all_reduce_perf", "End of Test all_reduce_perf")

        info_call = mock_log.info.call_args_list[-1]
        self.assertEqual(info_call[0][2], 2)

    @unittest.skipUnless(_HAS_BASH_AND_AWK, "requires bash and awk")
    def test_awk_slice_returns_last_marker_pair_only(self):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        dmesg_text = (
            "noise before\n"
            "Starting Test all_reduce_perf\n"
            "old run segfault\n"
            "End of Test all_reduce_perf\n"
            "Starting Test all_reduce_perf\n"
            "current run line\n"
            "End of Test all_reduce_perf\n"
            "noise after\n"
        )
        sliced = self._run_marker_awk_slice(dmesg_text, "Starting Test all_reduce_perf", "End of Test all_reduce_perf")
        self.assertIn("current run line", sliced)
        self.assertNotIn("old run segfault", sliced)
        self.assertNotIn("noise before", sliced)
        self.assertNotIn("noise after", sliced)

    @unittest.skipUnless(_HAS_BASH_AND_AWK, "requires bash and awk")
    def test_awk_slice_returns_partial_when_end_marker_missing(self):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        dmesg_text = (
            "Starting Test all_reduce_perf\n"
            "old complete\n"
            "End of Test all_reduce_perf\n"
            "Starting Test all_reduce_perf\n"
            "crash line still in buffer\n"
        )
        sliced = self._run_marker_awk_slice(dmesg_text, "Starting Test all_reduce_perf", "End of Test all_reduce_perf")
        self.assertIn("crash line still in buffer", sliced)
        self.assertNotIn("old complete", sliced)

    @patch("cvs.lib.verify_lib.log")
    def test_crash_partial_slice_warns_missing_end(self, mock_log):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "legacy"
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node1": "Starting Test all_reduce_perf\ncrash line still in buffer\n",
        }

        with patch("cvs.lib.verify_lib.fail_test") as mock_fail:
            result = verify_lib.verify_dmesg_during_test(
                phdl, "Starting Test all_reduce_perf", "End of Test all_reduce_perf"
            )

        self.assertEqual(result, {"node1": []})
        mock_fail.assert_not_called()
        mock_log.warning.assert_called()

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "parse_dmesg")
    def test_node_scraper_scans_bounded_slice(self, mock_parse, mock_fail_test):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        mock_parse.return_value = [
            {
                "priority": "ERROR",
                "category": "SW_DRIVER",
                "description": "GPU reset",
                "match_content": "GPU reset begin",
                "count": 1,
                "timestamps": [],
                "source": "dmesg",
            }
        ]
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node1": "Starting Test all_reduce_perf\nGPU reset begin\nEnd of Test all_reduce_perf\n"
        }

        result = verify_lib.verify_dmesg_during_test(
            phdl, "Starting Test all_reduce_perf", "End of Test all_reduce_perf"
        )

        self.assertIn("--time-format iso -x", phdl.exec.call_args[0][0])
        mock_parse.assert_called_once()
        self.assertEqual(len(result["node1"]), 1)
        mock_fail_test.assert_called()

    @patch("cvs.lib.verify_lib.fail_test")
    @patch.object(verify_lib.node_scraper_adapter, "parse_dmesg")
    @patch("cvs.lib.verify_lib.log")
    def test_node_scraper_warning_does_not_fail(self, mock_log, mock_parse, mock_fail_test):
        os.environ[verify_lib.DMESG_PARSER_ENV] = "node-scraper"
        mock_parse.return_value = [
            {
                "priority": "WARNING",
                "category": "SW_DRIVER",
                "description": "Runlist oversubscribed",
                "match_content": "Runlist is getting oversubscribed",
                "count": 1,
                "timestamps": [],
                "source": "dmesg",
            }
        ]
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node1": ("Starting Test all_reduce_perf\nRunlist is getting oversubscribed\nEnd of Test all_reduce_perf\n")
        }

        result = verify_lib.verify_dmesg_during_test(
            phdl, "Starting Test all_reduce_perf", "End of Test all_reduce_perf"
        )

        self.assertEqual(result, {"node1": []})
        mock_fail_test.assert_not_called()
        mock_log.warning.assert_called()


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
