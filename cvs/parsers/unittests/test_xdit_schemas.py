import json
import unittest
from pathlib import Path

from cvs.parsers.schemas import (
    PytorchXditFluxConfigFile,
    PytorchXditThresholdFile,
    PytorchXditUnifiedConfigFile,
    PytorchXditWanConfigFile,
    is_pytorch_xdit_unified_config,
    validate_config_file,
)


def _wan_benchmark_params():
    return {
        "wan22_i2v_a14b": {
            "prompt": "test prompt",
            "size": "720*1280",
            "frame_num": 81,
            "num_benchmark_steps": 5,
            "compile": True,
            "torchrun_nproc": 8,
            "ulysses_size": 8,
            "ring_size": 2,
            "expected_results": {"auto": {"max_avg_total_time_s": 15.0}},
        }
    }


def _flux_benchmark_params():
    return {
        "flux1_dev_t2i": {
            "prompt": "A cat",
            "seed": 42,
            "num_inference_steps": 25,
            "max_sequence_length": 256,
            "no_use_resolution_binning": True,
            "warmup_steps": 1,
            "warmup_calls": 5,
            "num_repetitions": 25,
            "height": 1024,
            "width": 1024,
            "ulysses_degree": 8,
            "ring_degree": 2,
            "use_torch_compile": True,
            "torchrun_nproc": 8,
            "expected_results": {"auto": {"max_avg_pipe_time_s": 12.0}},
        }
    }


def _unified_flux_config():
    return {
        "schema_version": 1,
        "framework": "xdit",
        "gpu_arch": "mi3xx",
        "topology": "single",
        "enforce_thresholds": False,
        "threshold_json": "mi3xx_xdit_flux1_dev_single_threshold.json",
        "paths": {
            "shared_fs": "{home}",
            "models_dir": "{home}/.cache/huggingface",
            "log_dir": "{home}/cvs_flux_output",
            "hf_token_file": "{home}/.hf_token",
        },
        "model": {"id": "black-forest-labs/FLUX.1-dev", "remote": 0},
        "container": {
            "lifetime": "per_run",
            "name": "flux-benchmark",
            "image": "<changeme>",
            "runtime": {
                "name": "docker",
                "args": {
                    "network": "host",
                    "ipc": "host",
                    "privileged": True,
                    "shm_size": "128G",
                    "volumes": ["{paths.models_dir}:/hf_home", "{paths.log_dir}:/outputs"],
                    "devices": ["/dev/dri", "/dev/kfd"],
                    "env": {"NCCL_DEBUG": "ERROR"},
                },
            },
        },
        "params": {
            "flux1_dev_t2i": {
                "prompt": "A small cat",
                "seed": 42,
                "num_inference_steps": 25,
                "max_sequence_length": 256,
                "no_use_resolution_binning": True,
                "warmup_steps": 1,
                "warmup_calls": 5,
                "num_repetitions": 25,
                "height": 1024,
                "width": 1024,
                "ulysses_degree": 8,
                "ring_degree": 1,
                "use_torch_compile": True,
                "torchrun_nproc": 8,
            }
        },
    }


class TestPytorchXditDistributedSchemas(unittest.TestCase):
    def test_wan_config_accepts_example_nccl_hints(self):
        raw = {
            "config": {
                "hf_home": "/home/user",
                "output_base_dir": "/home/user/out",
                "nnodes": 2,
                "master_addr": "10.0.0.1",
                "master_port": 29500,
                "_example_nccl_ib_hca": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
                "nccl_ib_hca": "rdma0",
                "_example_nccl_socket_ifname": "eno0",
                "nccl_socket_ifname": "eno0",
                "_example_gloo_socket_ifname": "eno0",
                "gloo_socket_ifname": "eno0",
            },
            "benchmark_params": _wan_benchmark_params(),
        }

        validated = PytorchXditWanConfigFile.model_validate(raw)

        self.assertEqual(
            validated.config.example_nccl_ib_hca,
            "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        )
        self.assertEqual(validated.config.example_nccl_socket_ifname, "eno0")
        self.assertEqual(validated.config.example_gloo_socket_ifname, "eno0")

    def test_flux_config_accepts_example_nccl_hints(self):
        raw = {
            "config": {
                "hf_home": "/home/user",
                "output_base_dir": "/home/user/out",
                "nnodes": 2,
                "master_addr": "10.0.0.1",
                "master_port": 29500,
                "_example_nccl_ib_hca": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
                "nccl_ib_hca": "rdma0",
                "_example_nccl_socket_ifname": "eno0",
                "nccl_socket_ifname": "eno0",
                "_example_gloo_socket_ifname": "eno0",
                "gloo_socket_ifname": "eno0",
            },
            "benchmark_params": _flux_benchmark_params(),
        }

        validated = PytorchXditFluxConfigFile.model_validate(raw)

        self.assertEqual(
            validated.config.example_nccl_ib_hca,
            "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        )


class TestPytorchXditUnifiedSchemas(unittest.TestCase):
    def test_is_pytorch_xdit_unified_config(self):
        self.assertTrue(is_pytorch_xdit_unified_config(_unified_flux_config()))
        self.assertFalse(is_pytorch_xdit_unified_config({"config": {}, "benchmark_params": {}}))

    def test_unified_flux_config_validates_without_embedded_thresholds(self):
        validated = PytorchXditUnifiedConfigFile.model_validate(_unified_flux_config())

        self.assertEqual(validated.framework, "xdit")
        self.assertEqual(validated.topology, "single")
        self.assertIsNone(validated.params.flux1_dev_t2i.expected_results)
        self.assertEqual(
            validated.threshold_json,
            "mi3xx_xdit_flux1_dev_single_threshold.json",
        )

    def test_unified_distributed_requires_nnodes(self):
        raw = dict(_unified_flux_config())
        raw["topology"] = "distributed"
        with self.assertRaisesRegex(ValueError, "nnodes >= 2"):
            PytorchXditUnifiedConfigFile.model_validate(raw)

    def test_threshold_file_preserves_gpu_metric_keys(self):
        validated = PytorchXditThresholdFile.model_validate(
            {
                "auto": {"max_avg_pipe_time_s": 10.0},
                "mi300x": {"max_avg_pipe_time_s": 3.0},
            }
        )

        self.assertEqual(validated.thresholds["mi300x"].max_avg_pipe_time_s, 3.0)

    def test_packaged_unified_configs_validate(self):
        config_dir = Path(__file__).resolve().parents[2] / "input" / "config_file" / "inference" / "xdit"
        workload_files = sorted(path for path in config_dir.glob("mi3xx_*.json") if "threshold" not in path.name)
        self.assertEqual(len(workload_files), 7)
        for path in workload_files:
            validated = validate_config_file(path)
            self.assertIsInstance(validated, PytorchXditUnifiedConfigFile)
            self.assertIsNotNone(validated.server_params)
            self.assertEqual(validated.server_params.backend, "xdit")
            self.assertTrue(validated.benchmark_params)
            if path.name.endswith("_single.json"):
                self.assertIsNone(
                    validated.server_params.benchmark_serv_node,
                    msg=f"{path.name} runs on every node and must not pin benchmark_serv_node",
                )
            if path.name.endswith("_distributed.json"):
                self.assertFalse(
                    validated.server_params.server_node_list,
                    msg=f"{path.name} must take hosts from the cluster, not server_node_list",
                )
                self.assertFalse(
                    validated.server_params.master_addr,
                    msg=f"{path.name} must resolve master_addr from the first cluster node",
                )
                self.assertFalse(
                    validated.server_params.benchmark_serv_node,
                    msg=f"{path.name} must use the first cluster node as the benchmark host",
                )
                self.assertNotIn(
                    "master_port",
                    json.loads(path.read_text(encoding="utf-8")).get("server_params") or {},
                    msg=f"{path.name} must use the default torchrun master_port",
                )
            threshold_path = config_dir / validated.threshold_json
            self.assertTrue(threshold_path.is_file(), msg=f"missing threshold for {path.name}")
            PytorchXditThresholdFile.model_validate(json.loads(threshold_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
