'''Unit tests for the Mori Run Deck run card.'''

import unittest

from cvs.lib.report.profiles.hooks.mori_run_card import mori_run_card_display


class TestMoriRunCard(unittest.TestCase):
    def test_uses_mori_runtime_fields_without_inference_placeholders(self):
        rows = mori_run_card_display(
            {
                "gpu_name": "MI355X",
                "node_count": 2,
                "mori_device_list": "rdma0,rdma1",
                "container_image": "example/mori:latest",
            },
            {"pytest_html_href": "mori.html"},
        )

        display = {label: value for label, value, _is_link in rows}
        self.assertEqual(display["GPU"], "MI355X")
        self.assertEqual(display["Nodes"], "2")
        self.assertEqual(display["Mori devices"], "rdma0,rdma1")
        self.assertEqual(display["Container image"], "example/mori:latest")
        self.assertEqual(display["Pytest report"], "mori.html")
        self.assertNotIn("Model", display)
        self.assertNotIn("TP", display)

    def test_missing_fields_render_as_dash(self):
        # Baremetal runs have no container image, and gpu_name is optional in the config.
        for variant in ({}, {"gpu_name": "", "container_image": None}, None):
            with self.subTest(variant=variant):
                display = {label: value for label, value, _is_link in mori_run_card_display(variant, {})}
                self.assertEqual(display["GPU"], "—")
                self.assertEqual(display["Container image"], "—")
                self.assertEqual(display["Nodes"], "—")


if __name__ == "__main__":
    unittest.main()
