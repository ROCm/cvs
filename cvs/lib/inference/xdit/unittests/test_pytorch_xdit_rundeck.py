"""Unit tests for xDiT Run Deck result normalization."""

import unittest

from cvs.lib.inference.xdit.pytorch_xdit_rundeck import (
    build_xdit_result_record,
    xdit_run_card_display,
)
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


class TestPytorchXditRundeck(unittest.TestCase):
    def test_build_flux_record(self):
        record = build_xdit_result_record(
            workload="FLUX text-to-image",
            label="node-a",
            inference_config={"model_repo": "black-forest-labs/FLUX.1-dev"},
            benchmark_params={"height": 1024, "width": 1024, "num_inference_steps": 25},
            gpu="mi300x",
            nnodes=1,
            sample_times=[1.0, 2.0],
            average_time=1.5,
            sample_kind="repetition",
            passed=True,
        )

        self.assertEqual(record["resolution"], "1024×1024")
        self.assertAlmostEqual(record["resolution_mp"], 1.048576)
        self.assertAlmostEqual(record["output_throughput_per_s"], 2 / 3)
        self.assertEqual(record["sample_count"], 2)
        self.assertEqual(record["status"], "PASS")

    def test_build_wan_record_accepts_size_and_explicit_count(self):
        record = build_xdit_result_record(
            workload="WAN 2.2 image-to-video",
            label="node-b",
            inference_config={"model_repo": "/models/Wan2.2-I2V-A14B"},
            benchmark_params={"size": "720*1280", "frame_num": 81},
            gpu="mi350",
            nnodes=2,
            sample_times=[],
            average_time=90,
            sample_kind="benchmark step",
            passed=False,
            sample_count=5,
        )

        self.assertEqual(record["resolution"], "720×1280")
        self.assertAlmostEqual(record["resolution_mp"], 0.9216)
        self.assertEqual(record["frame_count"], 81)
        self.assertEqual(record["sample_count"], 5)
        self.assertEqual(record["status"], "FAIL")

    def test_run_card_uses_diffusion_metadata(self):
        records = [{"model": "model-id", "gpu": "mi300x", "nnodes": 4, "resolution": "1024×1024"}]
        rows = xdit_run_card_display(records, {"pytest_html_href": "report.html"})

        self.assertIn(("Model", "model-id", False), rows)
        self.assertIn(("GPU", "mi300x", False), rows)
        self.assertIn(("Nodes", "4", False), rows)
        self.assertIn(("Pytest report", "report.html", True), rows)

    def test_profile_builds_static_diffusion_deck(self):
        record = build_xdit_result_record(
            workload="FLUX text-to-image",
            label="node-a",
            inference_config={"model_repo": "FLUX.1-dev"},
            benchmark_params={"height": 1024, "width": 1024},
            gpu="mi300x",
            nnodes=1,
            sample_times=[1.2, 1.3],
            average_time=1.25,
            sample_kind="repetition",
            passed=True,
        )
        profile = load_json_profile("pytorch_xdit_flux_dev_single")
        payload = build_rundeck_payload(
            profile=profile,
            store={"cvs_results_dict": [record], "variant_config": [record]},
        )
        document = render_rundeck_html(payload)

        self.assertFalse(profile["interactive_viewer"])
        self.assertEqual(payload["run_card_display"][0], ("Model", "FLUX.1-dev", False))
        self.assertEqual(len(payload["results_table"]["rows"]), 1)
        self.assertIn("Latency vs resolution", document)
        self.assertIn("Throughput vs resolution", document)
        self.assertIn("Step and repetition time", document)
        self.assertNotIn(">Viewer<", document)


if __name__ == "__main__":
    unittest.main()
