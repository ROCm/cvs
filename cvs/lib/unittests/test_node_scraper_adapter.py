'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest
from datetime import datetime, timezone

from cvs.lib import node_scraper_adapter
from cvs.lib.unittests import mce_dmesg_fixtures as mce


# Small, self-contained dmesg sample in node-scraper's `dmesg --time-format iso -x`
# format (decoded facility/level prefix + ISO timestamp). Covers an OOM kill, a
# segfault, and a RAS correctable error so the adapter exercises several of
# node-scraper's built-in patterns without depending on a large external file.
SAMPLE_DMESG = (
    "kern  :info  : 2026-02-09T03:08:45,029495-05:00 [2731443] gnome-session-binary\n"
    "kern  :err   : 2026-02-09T03:08:45,500000-05:00 Out of memory: Killed process "
    "2746553 (dbus-daemon) total-vm:130169kB\n"
    "kern  :info  : 2026-02-09T03:09:01,000000-05:00 gnome-shell[1234]: segfault at 0 "
    "ip 00007f0000000000 sp 00007ffd error 4 in libc.so.6\n"
    "kern  :warn  : 2026-02-09T03:10:00,000000-05:00 amdgpu: 5 correctable hardware "
    "errors detected in total in gfx block\n"
    "kern  :info  : 2026-02-09T03:11:00,000000-05:00 healthy line, nothing to report\n"
)


class TestParseDmesg(unittest.TestCase):
    def test_detects_known_errors(self):
        events = node_scraper_adapter.parse_dmesg(SAMPLE_DMESG, node_name="node1")
        descriptions = {event["description"] for event in events}
        self.assertIn("Out of memory error", descriptions)
        self.assertIn("Segmentation fault", descriptions)

    def test_event_shape(self):
        events = node_scraper_adapter.parse_dmesg(SAMPLE_DMESG)
        self.assertTrue(events, "expected at least one event from the sample dmesg")
        for event in events:
            for key in node_scraper_adapter.EVENT_KEYS:
                self.assertIn(key, event)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(node_scraper_adapter.parse_dmesg(""), [])

    def test_custom_error_regex_is_applied(self):
        events = node_scraper_adapter.parse_dmesg(
            SAMPLE_DMESG,
            analysis_args={
                "check_unknown_dmesg_errors": False,
                "error_regex": [
                    {
                        "regex": r"nothing to report",
                        "message": "CVS custom marker",
                        "event_category": "OS",
                    }
                ],
            },
        )
        descriptions = {event["description"] for event in events}
        self.assertIn("CVS custom marker", descriptions)

    def test_event_match_lines_flatten(self):
        events = node_scraper_adapter.parse_dmesg(SAMPLE_DMESG)
        lines = node_scraper_adapter.event_match_lines(events)
        self.assertTrue(any("Out of memory" in line for line in lines))

    def test_event_match_text_omits_description(self):
        event = {"description": "Unknown dmesg error", "match_content": "amdgpu: WALKER_ERROR: 0x0"}
        self.assertEqual(node_scraper_adapter.event_match_text(event), "amdgpu: WALKER_ERROR: 0x0")
        self.assertEqual(
            node_scraper_adapter.event_match_text({"description": "d", "match_content": ["a", "", "b"]}), "a b"
        )
        self.assertEqual(node_scraper_adapter.event_match_text({"description": "d", "match_content": None}), "")
        self.assertEqual(
            node_scraper_adapter.event_match_lines([event]), ["Unknown dmesg error: amdgpu: WALKER_ERROR: 0x0"]
        )


class TestFilterDmesgRange(unittest.TestCase):
    def test_no_bounds_preserve_text(self):
        source = mce.text(mce.CE_LINES)
        self.assertIs(node_scraper_adapter.filter_dmesg_range(source), source)
        self.assertEqual(node_scraper_adapter.filter_dmesg_range(None), '')

    def test_real_analyzer_filters_to_window(self):
        source = mce.text(mce.CE_LINES_EARLIER + mce.CE_LINES)
        start = datetime(2026, 10, 15, 6, 24, tzinfo=timezone.utc)
        end = datetime(2026, 10, 15, 6, 25, tzinfo=timezone.utc)
        kept = node_scraper_adapter.filter_dmesg_range(source, start, end)
        self.assertIn('06:24:34', kept)
        self.assertNotIn('06:18:47', kept)
        self.assertNotIn('\n\n', kept)


if __name__ == "__main__":
    unittest.main()
