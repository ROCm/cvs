'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the RVS Run Deck dataset builder.
'''

import unittest

from cvs.lib.report.rundeck.dataset_builders import rvs  # noqa: F401
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets


def _sample_records():
    return [
        {
            "node": "node1",
            "gpu": "42583",
            "module": "gst",
            "action": "gst-Tflops-8K-trig-fp64",
            "metric": "gflops",
            "value": 6478.0,
            "unit": "GFLOPS",
            "target": 5000.0,
            "passed": True,
        },
        {
            "node": "node1",
            "gpu": "42583",
            "module": "pebb",
            "action": "pcie_h2d_bandwidth",
            "metric": "pcie_gbps",
            "value": 57.678,
            "unit": "GB/s",
            "passed": None,
        },
        {
            "node": "node1",
            "gpu": "11806",
            "module": "pbqt",
            "action": "xgmi_d2d_unidir_bandwidth",
            "metric": "p2p_gbps",
            "value": 61.37,
            "unit": "GB/s",
            "passed": None,
        },
        {
            "node": "node1",
            "gpu": "42583",
            "module": "babel",
            "action": "Triad",
            "metric": "babel_mbytes_s",
            "value": 5950804.403,
            "unit": "MB/s",
            "passed": None,
        },
        {
            "node": "node1",
            "gpu": "42583",
            "module": "iet",
            "action": "iet_stress",
            "metric": "power_w",
            "value": 245.8,
            "unit": "W",
            "passed": None,
        },
        {
            "node": "node1",
            "gpu": "42583",
            "module": "iet",
            "action": "iet_stress",
            "metric": "status",
            "value": None,
            "unit": "",
            "passed": False,
        },
    ]


class TestRvsDatasetBuilder(unittest.TestCase):
    def test_build_datasets_table_gate_matrix_and_charts(self):
        sources = {"results": {"records": _sample_records()}}
        profile = {"tier_order": ["result"]}
        datasets = build_datasets("rvs", sources, profile)

        table = datasets["results_table"]
        self.assertEqual(
            table["headers"],
            ["Node", "GPU", "Module", "Action", "Metric", "Value", "Unit", "Target", "Pass"],
        )
        self.assertEqual(len(table["rows"]), len(_sample_records()))

        tiers = {row["label"]: row["tiers"]["result"] for row in datasets["gate_matrix"]}
        self.assertEqual(tiers["node1 · gst · 42583"], "pass")
        self.assertEqual(tiers["node1 · iet · 42583"], "fail")

        charts = datasets["charts"]
        self.assertIn("gst_gflops", charts)
        self.assertIn("pebb_gbps", charts)
        self.assertIn("pbqt_gbps", charts)
        self.assertIn("babel_mbytes_s", charts)
        self.assertIn("iet_power_w", charts)
        self.assertTrue(charts["gst_gflops"])
        self.assertTrue(charts["pebb_gbps"])
        self.assertEqual(datasets["metric_tier_order"], ("result",))


if __name__ == "__main__":
    unittest.main()
