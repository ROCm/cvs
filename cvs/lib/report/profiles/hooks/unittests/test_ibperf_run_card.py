'''Unit tests for the ibperf Run Deck run card.'''

import unittest

from cvs.lib.ibperf_reporting import OUTLIER_RULE
from cvs.lib.report.profiles.hooks.ibperf_run_card import ibperf_run_card_display


class TestIbperfRunCard(unittest.TestCase):
    def test_shows_ibperf_run_fields(self):
        rows = ibperf_run_card_display(
            {
                "node_count": 2,
                "hosts": ["host-a", "host-b"],
                "nic_count": 8,
                "ib_bw_test_list": ["ib_write_bw", "ib_send_bw"],
                "ib_lat_test_list": ["ib_write_lat"],
                "msg_size_list": [8388608, 2, 65536],
                "qp_count_list": ["8", "16"],
                "duration": "30",
                "dmabuf": False,
                "expected_results": {"ib_write_bw": {"8388608": {"8": "280"}, "8192": {"8": "180"}}},
            },
            {"pytest_html_href": "ibperf.html"},
        )

        display = {label: value for label, value, _is_link in rows}
        self.assertEqual(display["Nodes"], "2 (1 server/client pair)")
        self.assertEqual(display["Node names"], "node00 = host-a, node01 = host-b")
        self.assertEqual(display["BW tests"], "ib_write_bw, ib_send_bw")
        self.assertEqual(display["Message sizes"], "2 B – 8 MiB · 3 sizes")
        self.assertEqual(display["dmabuf"], "off")
        self.assertEqual(display["BW thresholds"], "ib_write_bw at 8 KiB, 8 MiB")
        self.assertEqual(display["Outlier rule"], OUTLIER_RULE)
        self.assertEqual(display["Pytest report"], "ibperf.html")


if __name__ == "__main__":
    unittest.main()
