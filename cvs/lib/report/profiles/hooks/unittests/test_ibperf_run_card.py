'''Unit tests for the ibperf Run Deck run card.'''

import unittest

from cvs.lib.report.profiles.hooks.ibperf_run_card import ibperf_run_card_display


class TestIbperfRunCard(unittest.TestCase):
    def test_shows_ibperf_run_fields(self):
        rows = ibperf_run_card_display(
            {
                "node_count": 2,
                "nic_count": 8,
                "msg_size_list": [2, 65536],
                "qp_count_list": ["8", "16"],
                "duration": "30",
                "dmabuf": False,
                "orchestrator": "baremetal",
            },
            {"pytest_html_href": "ibperf.html"},
        )

        display = {label: value for label, value, _is_link in rows}
        self.assertEqual(display["Nodes"], "2")
        self.assertEqual(display["NICs per node"], "8")
        self.assertEqual(display["Message sizes (bytes)"], "2, 65536")
        self.assertEqual(display["QP counts"], "8, 16")
        self.assertEqual(display["Duration per test"], "30 s")
        self.assertEqual(display["dmabuf"], "off")
        self.assertEqual(display["Orchestrator"], "baremetal")
        self.assertEqual(display["Pytest report"], "ibperf.html")
        self.assertNotIn("Model", display)

    def test_missing_variant_renders_dashes(self):
        display = {label: value for label, value, _is_link in ibperf_run_card_display(None, {})}
        self.assertEqual(set(display.values()), {"—"})


if __name__ == "__main__":
    unittest.main()
