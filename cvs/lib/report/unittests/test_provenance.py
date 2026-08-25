'''Unit tests for report provenance helpers.'''

import unittest
from types import SimpleNamespace

from cvs.lib.image_display import format_image_display
from cvs.lib.report.inference import build_inference_report_payload
from cvs.lib.report.provenance import (
    build_inference_report_provenance,
    extend_run_card_display,
    provenance_run_card_rows,
)
from cvs.lib.report.testing.fixtures import (
    generic_inference_report_config,
    generic_variant,
    two_cell_inf_res,
)


class TestProvenance(unittest.TestCase):
    def test_format_image_display_shortens_digest_with_tag(self):
        display = format_image_display(
            image_tag="rocm/cvs:test",
            image_digest="rocm/atom-dev@sha256:abc123deadbeefcafebabe",
        )
        self.assertTrue(display.startswith("rocm/cvs:test @ sha256:abc123deadbe"))

    def test_build_inference_report_provenance_collects_paths(self):
        config = SimpleNamespace(
            option=SimpleNamespace(
                cluster_file="/cluster.json",
                config_file="/variant.json",
            )
        )
        prov = build_inference_report_provenance(
            config,
            cvs_version="9.9.9",
            pytest_html_path="/out/report.html",
            log_file_path="/out/run.log",
        )
        self.assertEqual(prov["cvs_version"], "9.9.9")
        self.assertEqual(prov["cluster_file"], "/cluster.json")
        self.assertEqual(prov["config_file"], "/variant.json")
        self.assertEqual(prov["pytest_html_path"], "/out/report.html")
        self.assertEqual(prov["log_file_path"], "/out/run.log")

    def test_provenance_run_card_rows_includes_standard_fields(self):
        rows = provenance_run_card_rows(
            {
                "cvs_version": "1.0.0",
                "git_ref": "abc1234 @ feature/x",
                "image_display": "rocm/atom-dev:latest @ sha256:deadbeef…",
                "launch_summary": "atom · TP=8 · max_model_len=10240",
                "cluster_file": "/cluster.json",
                "config_file": "/config.json",
            }
        )
        labels = [r[0] for r in rows]
        self.assertEqual(
            labels,
            [
                "CVS version",
                "Git ref",
                "Image",
                "Launch",
                "Cluster file",
                "Config file",
            ],
        )

    def test_extend_run_card_display_skips_duplicate_labels(self):
        base = [("Model", "m1", False)]
        extended = extend_run_card_display(
            base,
            {"cvs_version": "1.0.0", "cluster_file": "/cluster.json"},
        )
        labels = [r[0] for r in extended]
        self.assertIn("Model", labels)
        self.assertIn("CVS version", labels)
        self.assertEqual(labels.count("Model"), 1)

    def test_payload_includes_provenance_and_run_card_fields(self):
        payload = build_inference_report_payload(
            config=generic_inference_report_config(),
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
            cvs_version="1.0.0",
            provenance={
                "cluster_file": "/cluster.json",
                "config_file": "/config.json",
                "git_ref": "deadbeef @ main",
                "launch_summary": "atom · TP=8",
                "launch_server_cmd": "python -m atom.entrypoints.openai_server --model demo",
                "launch_bench_cmd": "python -m atom.benchmarks.benchmark_serving --model demo",
            },
        )
        self.assertEqual(payload["provenance"]["cluster_file"], "/cluster.json")
        labels = [r[0] for r in payload["run_card_display"]]
        self.assertIn("Cluster file", labels)
        self.assertIn("Config file", labels)
        self.assertIn("Git ref", labels)
        self.assertIn("Launch", labels)
        self.assertTrue(payload["panels"]["launch"]["server_cmd"].startswith("python -m atom"))


if __name__ == "__main__":
    unittest.main()
