import unittest
from types import SimpleNamespace

from cvs.lib.report.inference import build_inference_report_payload
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.render.gate_matrix import GateMatrixRenderer
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile


class TestXditRundeck(unittest.TestCase):
    def test_payload_contains_timing_gate_and_results_row(self):
        cell_id = "SIZE=1024x1024,STEPS=50,BENCH=25"
        variant = SimpleNamespace(
            enforce_thresholds=True,
            thresholds={cell_id: {"avg_pipe_time_s": {"kind": "max", "value": 3.0}}},
            model=SimpleNamespace(id="FLUX.1-dev"),
            gpu_arch="mi300x",
            topology="single",
            inference={"benchmark_serv_node": "node-a"},
            benchmark_params={"flux1_dev_t2i": {"torchrun_nproc": 8}},
        )
        key = ("FLUX.1-dev", "mi300x", "1024x1024", 50, cell_id, "1")
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
        self.assertEqual(payload["results_table"]["headers"][5], "nnodes")
        self.assertEqual(payload["results_table"]["headers"][6], "Host")
        self.assertNotIn("Workers", payload["results_table"]["headers"])
        self.assertNotIn("Topology", payload["results_table"]["headers"])
        self.assertEqual(payload["results_table"]["rows"][0][9], 2.5)
        self.assertEqual(
            GateMatrixRenderer.cell_label(payload["cells"][0]),
            "SIZE=1024x1024,STEPS=50,BENCH=25 \u00b7 NNODES=1",
        )

    def test_run_card_shows_nnodes_and_benchmark_node(self):
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
        self.assertEqual(flux_rows["Server nodes"], "10.32.80.110, 10.32.80.111")
        self.assertEqual(flux_rows["nnodes"], "2")
        self.assertEqual(flux_rows["Benchmark node"], "10.32.80.110")
        self.assertNotIn("GPUs/node", flux_rows)
        self.assertNotIn("Workers", flux_rows)
        self.assertNotIn("Workload", flux_rows)
        self.assertNotIn("Topology", flux_rows)
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
        self.assertEqual(wan_rows["Server nodes"], "n1, n2")
        self.assertEqual(wan_rows["nnodes"], "2")
        self.assertEqual(wan_rows["Benchmark node"], "n1")
        self.assertEqual(wan_rows["Ulysses"], "8")
        self.assertEqual(wan_rows["Ring"], "2")

        single = SimpleNamespace(
            enforce_thresholds=True,
            gpu_arch="mi325",
            topology="single",
            model=SimpleNamespace(id="FLUX.1-dev"),
            inference={
                "nnodes": 1,
                "benchmark_serv_node": "node-a",
                "_execution_hosts": ["10.32.80.110", "10.32.80.111"],
            },
            benchmark_params={"flux1_dev_t2i": {"torchrun_nproc": 8, "ulysses_degree": 8, "ring_degree": 1}},
        )
        single_rows = {label: value for label, value, _ in xdit_run_card_display(single, {})}
        self.assertEqual(single_rows["Server nodes"], "10.32.80.110, 10.32.80.111")
        self.assertEqual(single_rows["nnodes"], "1")
        self.assertEqual(single_rows["Benchmark node"], "10.32.80.110, 10.32.80.111")


if __name__ == "__main__":
    unittest.main()
