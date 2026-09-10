'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Integration tests for the RVS Run Deck profile payload and HTML render path.
'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import (
    ProfileConfigResolver,
    build_inference_config_from_profile,
    resolve_report_config,
)
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


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
    ]


class TestRvsRundeckPayload(unittest.TestCase):
    def test_rvs_profile_resolves_without_sweep_hooks(self):
        profile = load_json_profile("rvs_cvs")
        self.assertIsNotNone(profile)
        config = resolve_report_config(profile)
        self.assertEqual(config.suite_id, "rvs")
        self.assertEqual(ProfileConfigResolver.from_profile_dict(profile).suite_id, "rvs")
        self.assertEqual(build_inference_config_from_profile(profile).suite_id, "rvs")

    def test_rvs_payload_renders_core_panels(self):
        from cvs.lib.report.rundeck.dataset_builders import rvs  # noqa: F401

        profile = load_json_profile("rvs_cvs")
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": {"records": _sample_records()},
                "variant_config": {"rvs_version": "2.3.1", "rvs_test_level": 1},
                "lifecycle_report": {},
            },
            cvs_version="1.0.0",
        )
        self.assertEqual(payload["suite_id"], "rvs")
        self.assertIn("rvs", payload["datasets"])
        doc = render_rundeck_html(payload)
        self.assertIn("RVS Run Deck", doc)
        self.assertTrue("GST GFLOPS" in doc or "Full results" in doc)


if __name__ == "__main__":
    unittest.main()
