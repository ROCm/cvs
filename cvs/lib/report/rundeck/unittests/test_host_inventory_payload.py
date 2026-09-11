'''Unit tests for the host configuration Run Deck.'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def sample_results():
    return {
        "nodes": {
            "node-a": {
                "os_release": "Ubuntu 24.04.1 LTS",
                "kernel": "6.8.0-60-generic",
                "bios": "A1",
                "rocm": "7.0.2",
                "gpu_count": "8",
                "online_memory": "1.3T",
                "gpu_pcie": {"card0": "32GT/s \u00b7 x16"},
            },
            "node-b": {
                "os_release": "Ubuntu 24.04.1 LTS",
                "kernel": "6.8.0-61-generic",
                "bios": "A1",
                "rocm": "7.0.2",
                "gpu_count": "8",
                "online_memory": "1.3T",
                "gpu_pcie": {"card0": "32GT/s \u00b7 x16"},
            },
        },
        "firmware": {
            "node-a": {"0": {"CP_MEC1": "32945"}},
            "node-b": {"0": {"CP_MEC1": "32946"}},
        },
    }


class TestHostInventoryPayload(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile("host_configs_cvs")

    def test_profile_builds_inventory_and_drift_tables(self):
        datasets = build_datasets("host_inventory", {"results": sample_results()}, self.profile)

        self.assertEqual(datasets["node_count"], 2)
        self.assertEqual(datasets["inventory_table"]["headers"][0], "Node")
        self.assertEqual(len(datasets["inventory_table"]["rows"]), 2)

        rows = {row["key"]: row for row in datasets["drift_matrix"]["rows"]}
        self.assertTrue(rows["kernel"]["mismatch"])
        self.assertFalse(rows["bios"]["mismatch"])
        self.assertTrue(rows["firmware.0.CP_MEC1"]["mismatch"])
        self.assertEqual(datasets["drift_matrix"]["mismatch_count"], 2)
        self.assertEqual(len(datasets["firmware_table"]["rows"]), 2)

    def test_payload_renders_inventory_without_performance_charts_or_viewer(self):
        results = sample_results()
        payload = build_rundeck_payload(
            profile=self.profile,
            store={"cvs_results_dict": results, "variant_config": results},
            cvs_version="1.0.0",
        )
        doc = render_rundeck_html(payload)

        self.assertEqual(payload["overall_status"], "record")
        self.assertNotIn("viewer_config", payload)
        self.assertIn("Nodes sampled", doc)
        self.assertIn("Cross-node drift matrix", doc)
        self.assertIn("drift-cell-mismatch", doc)
        self.assertIn("Per-node inventory", doc)
        self.assertIn("GPU firmware inventory", doc)
        self.assertNotIn("Sweep analytics", doc)
        self.assertNotIn("tok/s", doc)

    def test_empty_firmware_card_is_hidden_and_outliers_are_marked(self):
        results = {
            "nodes": {
                "node-a": {"bios": "A1", "kernel": "k1"},
                "node-b": {"bios": "A1", "kernel": "k1"},
                "node-c": {"bios": "B9", "kernel": "k1"},
            }
        }
        payload = build_rundeck_payload(
            profile=self.profile,
            store={"cvs_results_dict": results, "variant_config": results},
            cvs_version="1.0.0",
        )
        self.assertEqual(payload["datasets"]["host_inventory"]["firmware_table"], {})
        doc = render_rundeck_html(payload)
        self.assertNotIn("GPU firmware inventory", doc)
        self.assertIn("class='drift-cell-mismatch'>B9</td>", doc)
        self.assertNotIn("class='drift-cell-mismatch'>A1</td>", doc)
        self.assertIn("B9", doc)

    def test_hostile_host_facts_are_escaped(self):
        results = {
            "nodes": {
                "node-a": {"bios": "<script>alert(1)</script>"},
                "node-b": {"bios": "<script>alert(1)</script>"},
            }
        }
        doc = render_rundeck_html(
            build_rundeck_payload(
                profile=self.profile,
                store={"cvs_results_dict": results, "variant_config": results},
                cvs_version="1.0.0",
            )
        )
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", doc)
        self.assertNotIn("<script>alert(1)</script>", doc)


if __name__ == "__main__":
    unittest.main()
