'''Unit tests for the host configuration Run Deck run-card hook.'''

import unittest

from cvs.lib.report.profiles.hooks.host_configs_run_card import host_configs_run_card_display


class TestHostConfigsRunCard(unittest.TestCase):
    def test_rows_count_nodes_and_collected_facts(self):
        rows = host_configs_run_card_display(
            {
                "nodes": {"n0": {"bios": "A1", "kernel": "k1"}, "n1": {"bios": "A1"}},
                "firmware": {},
            },
            {},
        )
        values = {label: value for label, value, _link in rows}
        self.assertEqual(values["Nodes sampled"], "2")
        self.assertEqual(values["Fact categories"], "2")
        self.assertIn("BIOS", values["Facts collected"])
        self.assertNotIn("GPU firmware", values["Facts collected"])


if __name__ == "__main__":
    unittest.main()
