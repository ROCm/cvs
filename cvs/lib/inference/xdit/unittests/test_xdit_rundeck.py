import unittest
from types import SimpleNamespace

from cvs.lib.report.inference import build_inference_report_payload
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile


class TestXditRundeck(unittest.TestCase):
    def test_payload_contains_timing_gate_and_results_row(self):
        cell_id = "ISL=1024x1024,OSL=25,C=8"
        variant = SimpleNamespace(
            enforce_thresholds=True,
            thresholds={cell_id: {"avg_pipe_time_s": {"kind": "max", "value": 3.0}}},
            model=SimpleNamespace(id="FLUX.1-dev"),
            gpu_arch="mi300x",
            topology="single",
            inference={"benchmark_serv_node": "node-a"},
            benchmark_params={"flux1_dev_t2i": {"torchrun_nproc": 8}},
        )
        key = ("FLUX.1-dev", "mi300x", "1024x1024", 25, cell_id, "8")
        results = {
            key: {
                "node-a": {
                    "avg_pipe_time_s": 2.5,
                    "sample_count": 25,
                    "backend": "diffusers",
                    "topology": "single",
                    "output_dir": "/outputs/flux",
                }
            }
        }
        config = build_inference_config_from_profile(load_json_profile("xdit"))

        payload = build_inference_report_payload(
            config=config,
            variant_config=variant,
            inf_res_dict=results,
            lifecycle_report={},
        )

        self.assertEqual(payload["overall_status"], "pass")
        self.assertEqual(payload["cells"][0]["tiers"]["latency"], "pass")
        self.assertEqual(payload["results_table"]["rows"][0][8], 2.5)


if __name__ == "__main__":
    unittest.main()
