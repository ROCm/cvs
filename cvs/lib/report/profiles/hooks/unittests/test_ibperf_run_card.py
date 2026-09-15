'''Unit tests for the ibperf Run Deck run card.'''

import unittest

from cvs.lib.report.profiles.hooks.ibperf_run_card import ibperf_run_card_display


class TestIbperfRunCard(unittest.TestCase):
    def test_includes_only_whitelisted_ibperf_settings(self):
        rows = ibperf_run_card_display(
            {
                "gid_index": "3",
                "duration": "30",
                "msg_size_list": [2, 4],
                "qp_count_list": ["8", "16"],
                "install_dir": "/secret/path",
                "port_no": "1516",
            },
            {},
        )

        self.assertEqual(
            rows,
            [
                ("GID index", "3", False),
                ("Duration (s)", "30", False),
                ("Message sizes", "2, 4", False),
                ("QP counts", "8, 16", False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
