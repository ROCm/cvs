'''Unit tests for the Mori Run Deck run card.'''

import unittest

from cvs.lib.report.profiles.hooks.mori_run_card import mori_run_card_display


class TestMoriRunCard(unittest.TestCase):
    def test_uses_mori_runtime_fields_without_inference_placeholders(self):
        rows = mori_run_card_display(
            {
                "gpu_type": "MI355X",
                "node_count": 2,
                "nic_type": "thor2",
                "mori_device_list": "rdma0,rdma1",
                "container_image": "example/mori:latest",
            },
            {"pytest_html_href": "mori.html"},
        )

        display = {label: value for label, value, _is_link in rows}
        self.assertEqual(display["GPU"], "MI355X")
        self.assertEqual(display["Nodes"], "2")
        self.assertEqual(display["NIC type"], "thor2")
        self.assertEqual(display["Pytest report"], "mori.html")
        self.assertNotIn("Model", display)
        self.assertNotIn("TP", display)


if __name__ == "__main__":
    unittest.main()
