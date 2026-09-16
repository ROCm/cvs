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
                    "ulysses_degree": 8,
                    "ring_degree": 1,
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
        self.assertEqual(payload["results_table"]["rows"][0][10], 2.5)

    def test_run_card_shows_ulysses_ring_and_total_workers(self):
        from cvs.lib.report.profiles.hooks.xdit_run_card import xdit_run_card_display

        flux = SimpleNamespace(
            enforce_thresholds=True,
            gpu_arch="mi325",
            topology="distributed",
            model=SimpleNamespace(id="/data/models/FLUX.2-dev"),
            inference={"_execution_hosts": ["10.32.80.110", "10.32.80.111"]},
            benchmark_params={"flux1_dev_t2i": {"torchrun_nproc": 8, "ulysses_degree": 8, "ring_degree": 1}},
        )
        flux_rows = {label: value for label, value, _ in xdit_run_card_display(flux, {})}
        self.assertEqual(flux_rows["GPUs/node"], "8")
        self.assertEqual(flux_rows["Workers"], "16")
        self.assertEqual(flux_rows["Ulysses"], "8")
        self.assertEqual(flux_rows["Ring"], "1")

        wan = SimpleNamespace(
            enforce_thresholds=True,
            gpu_arch="mi325",
            topology="distributed",
            model=SimpleNamespace(id="Wan-AI/Wan2.2-I2V-A14B"),
            inference={"nnodes": 2, "_execution_hosts": ["n1", "n2"]},
            benchmark_params={
                "wan22_i2v_a14b": {
                    "torchrun_nproc": 8,
                    "ulysses_size": 8,
                    "ring_size": 2,
                    "model_format": "diffusers",
                }
            },
        )
        wan_rows = {label: value for label, value, _ in xdit_run_card_display(wan, {})}
        self.assertEqual(wan_rows["Workload"], "WAN (Diffusers)")
        self.assertEqual(wan_rows["Workers"], "16")
        self.assertEqual(wan_rows["Ulysses"], "8")
        self.assertEqual(wan_rows["Ring"], "2")


if __name__ == "__main__":
    unittest.main()
