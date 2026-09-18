import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.inference.xdit.xdit_config_loader import (
    load_variant,
    orchestrator_container_from_variant,
)


class TestXditConfigLoader(unittest.TestCase):
    def _write_config(self, payload, thresholds=None):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        if thresholds is not None:
            threshold_path = Path(directory.name) / "threshold.json"
            threshold_path.write_text(json.dumps(thresholds), encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return str(path)

    def test_loads_legacy_flux_config(self):
        path = self._write_config(
            {
                "config": {
                    "container_image": "xdit:test",
                    "container_name": "xdit-job",
                    "hf_token_file": "/home/test/.hf_token",
                    "hf_home": "/models",
                    "output_base_dir": "/results",
                    "model_repo": "/models/FLUX.1-dev",
                    "container_config": {
                        "device_list": ["/dev/kfd"],
                        "volume_dict": {},
                        "env_dict": {"CUSTOM": "1"},
                    },
                },
                "benchmark_params": {
                    "flux1_dev_t2i": {
                        "torchrun_nproc": 8,
                        "ulysses_degree": 8,
                        "ring_degree": 1,
                    }
                },
            }
        )

        variant = load_variant(path, {})

        self.assertEqual(variant.framework, "xdit")
        self.assertEqual(variant.topology, "single")
        self.assertEqual(variant.benchmark_params["flux1_dev_t2i"]["torchrun_nproc"], 8)
        self.assertEqual(variant.inference["_resolved_model_path_container"], "/model")
        self.assertEqual(variant.inference["output_base_dir_container"], "/outputs")
        container = orchestrator_container_from_variant(variant)
        self.assertIn("/results:/outputs", container["runtime"]["args"]["volumes"])
        self.assertEqual(container["env"]["CUSTOM"], "1")

    def test_loads_unified_wan_config(self):
        path = self._write_config(
            {
                "schema_version": 1,
                "framework": "xdit",
                "gpu_arch": "mi300x",
                "topology": "distributed",
                "threshold_json": "threshold.json",
                "paths": {
                    "shared_fs": "/shared",
                    "models_dir": "/shared/models",
                    "log_dir": "/shared/results",
                    "hf_token_file": "/shared/token",
                },
                "model": {"id": "/shared/models/wan", "remote": 0},
                "container": {
                    "lifetime": "per_run",
                    "name": "wan-job",
                    "image": "xdit:test",
                    "env": {"NCCL_DEBUG": "INFO"},
                    "runtime": {
                        "name": "docker",
                        "args": {
                            "volumes": [
                                "/shared/models:/models",
                                "/shared/results:/outputs",
                            ],
                            "devices": ["/dev/kfd"],
                        },
                    },
                },
                "params": {
                    "wan22_i2v_a14b": {
                        "torchrun_nproc": 8,
                        "ulysses_size": 8,
                        "ring_size": 2,
                    }
                },
                "nnodes": 2,
                "master_addr": "10.0.0.1",
            },
            {"auto": {"max_avg_pipe_time_s": 300.0}},
        )

        variant = load_variant(path, {})

        self.assertEqual(variant.topology, "distributed")
        self.assertEqual(variant.inference["output_base_dir_container"], "/outputs")
        self.assertEqual(variant.inference["hf_home"], "/shared/models")
        self.assertEqual(variant.inference["hf_home_container"], "/models")
        self.assertEqual(variant.inference["_resolved_ckpt_dir_container"], "/models/wan")
        self.assertEqual(variant.inference["nnodes"], 2)
        self.assertEqual(variant.params.wan22_i2v_a14b["ring_size"], 2)
        self.assertEqual(
            variant.benchmark_params["wan22_i2v_a14b"]["expected_results"],
            {"auto": {"max_avg_pipe_time_s": 300.0}},
        )

    def test_runtime_args_env_flattens_into_orchestrator_and_inference(self):
        path = self._write_config(
            {
                "schema_version": 1,
                "framework": "xdit",
                "gpu_arch": "mi300x",
                "topology": "distributed",
                "threshold_json": "threshold.json",
                "paths": {
                    "shared_fs": "/shared",
                    "models_dir": "/shared/models",
                    "log_dir": "/shared/results",
                    "hf_token_file": "/shared/token",
                },
                "model": {"id": "/shared/models/wan", "remote": 0},
                "container": {
                    "lifetime": "per_run",
                    "name": "wan-job",
                    "image": "xdit:test",
                    "runtime": {
                        "name": "docker",
                        "args": {
                            "network": "host",
                            "ipc": "host",
                            "privileged": True,
                            "shm_size": "128G",
                            "volumes": ["/shared/models:/models", "/shared/results:/outputs"],
                            "devices": ["/dev/dri", "/dev/kfd", "/dev/infiniband/rdma_cm"],
                            "env": {
                                "NCCL_IB_HCA": "rdma0,rdma1",
                                "NCCL_SOCKET_IFNAME": "eno0",
                                "GLOO_SOCKET_IFNAME": "eno0",
                                "NCCL_DEBUG": "ERROR",
                            },
                        },
                    },
                },
                "params": {"wan22_i2v_a14b": {"torchrun_nproc": 8}},
                "nnodes": 2,
                "master_addr": "10.0.0.1",
            },
            {"auto": {"max_avg_pipe_time_s": 300.0}},
        )

        variant = load_variant(path, {})
        container = orchestrator_container_from_variant(variant)

        self.assertEqual(variant.inference["nccl_ib_hca"], "rdma0,rdma1")
        self.assertEqual(variant.inference["nccl_socket_ifname"], "eno0")
        self.assertEqual(container["env"]["NCCL_IB_HCA"], "rdma0,rdma1")
        self.assertNotIn("env", container["runtime"]["args"])
        self.assertEqual(container["runtime"]["args"]["shm_size"], "128G")

    def test_loads_sglang_style_flat_benchmark_params(self):
        path = self._write_config(
            {
                "gpu_name": "mi325",
                "enforce_thresholds": True,
                "threshold_json": "threshold.json",
                "paths": {
                    "shared_fs": "/shared",
                    "models_dir": "/shared/models",
                    "log_dir": "/shared/results",
                    "hf_token_file": "/shared/token",
                },
                "container": {
                    "lifetime": "per_run",
                    "name": "flux-benchmark_single",
                    "image": "xdit:test",
                    "runtime": {
                        "name": "docker",
                        "args": {
                            "network": "host",
                            "ipc": "host",
                            "privileged": True,
                            "volumes": ["/shared/models:/hf_home", "/shared/results:/outputs"],
                            "devices": ["/dev/dri", "/dev/kfd"],
                            "env": {"NCCL_DEBUG": "ERROR"},
                        },
                    },
                },
                "server_params": {
                    "backend": "xdit",
                    "nnodes": "1",
                    "model": "black-forest-labs/FLUX.1-dev",
                    "benchmark_serv_node": "node-a",
                },
                "benchmark_params": {
                    "prompt": "A small cat",
                    "height": 1024,
                    "width": 1024,
                    "ulysses_degree": 8,
                    "torchrun_nproc": 8,
                },
            },
            {"auto": {"max_avg_pipe_time_s": 10.0}},
        )

        variant = load_variant(path, {"node_dict": {"node-a": {}}})

        self.assertEqual(variant.topology, "single")
        self.assertEqual(variant.model.id, "black-forest-labs/FLUX.1-dev")
        container = orchestrator_container_from_variant(variant)
        self.assertEqual(container["env"]["HF_HOME"], "/hf_home")
        self.assertEqual(variant.benchmark_params["flux1_dev_t2i"]["torchrun_nproc"], 8)
        self.assertEqual(
            variant.benchmark_params["flux1_dev_t2i"]["expected_results"],
            {"auto": {"max_avg_pipe_time_s": 10.0}},
        )

    def test_local_model_path_is_mounted_into_container(self):
        path = self._write_config(
            {
                "gpu_name": "mi325",
                "paths": {
                    "shared_fs": "/shared",
                    "models_dir": "/shared/models",
                    "log_dir": "/shared/results",
                    "hf_token_file": "/shared/token",
                },
                "container": {
                    "lifetime": "per_run",
                    "name": "flux-benchmark_single",
                    "image": "xdit:test",
                    "runtime": {
                        "name": "docker",
                        "args": {
                            "volumes": ["/shared/models:/hf_home", "/shared/results:/outputs"],
                            "devices": ["/dev/kfd"],
                        },
                    },
                },
                "server_params": {
                    "backend": "xdit",
                    "nnodes": "1",
                    "model": "/data/models/FLUX.1-dev",
                    "benchmark_serv_node": "node-a",
                },
                "benchmark_params": {
                    "height": 1024,
                    "ulysses_degree": 8,
                    "torchrun_nproc": 8,
                },
            }
        )

        variant = load_variant(path, {"node_dict": {"node-a": {}}})

        self.assertEqual(variant.inference["_resolved_model_mount_host"], "/data/models/FLUX.1-dev")
        self.assertEqual(variant.inference["_resolved_model_path_container"], "/model")
        container = orchestrator_container_from_variant(variant)
        self.assertIn("/data/models/FLUX.1-dev:/model", container["runtime"]["args"]["volumes"])

    def test_local_model_path_under_existing_mount_is_reused(self):
        path = self._write_config(
            {
                "gpu_name": "mi325",
                "paths": {
                    "shared_fs": "/shared",
                    "models_dir": "/data/models",
                    "log_dir": "/shared/results",
                    "hf_token_file": "/shared/token",
                },
                "container": {
                    "lifetime": "per_run",
                    "name": "flux-benchmark_single",
                    "image": "xdit:test",
                    "runtime": {
                        "name": "docker",
                        "args": {
                            "volumes": ["/data/models:/hf_home", "/shared/results:/outputs"],
                            "devices": ["/dev/kfd"],
                        },
                    },
                },
                "server_params": {
                    "backend": "xdit",
                    "nnodes": "1",
                    "model": "/data/models/FLUX.1-dev",
                    "benchmark_serv_node": "node-a",
                },
                "benchmark_params": {
                    "height": 1024,
                    "ulysses_degree": 8,
                    "torchrun_nproc": 8,
                },
            }
        )

        variant = load_variant(path, {"node_dict": {"node-a": {}}})

        self.assertEqual(variant.inference["_resolved_model_path_container"], "/hf_home/FLUX.1-dev")
        container = orchestrator_container_from_variant(variant)
        self.assertNotIn(
            "/data/models/FLUX.1-dev:/model",
            container["runtime"]["args"]["volumes"],
        )

    def test_rejects_other_unified_framework(self):
        path = self._write_config(
            {
                "schema_version": 1,
                "framework": "vllm",
                "paths": {},
            }
        )
        with self.assertRaisesRegex(ValueError, "unsupported framework"):
            load_variant(path, {})

    def test_all_packaged_configs_load_thresholds_into_workload(self):
        config_dir = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "xdit"
        workload_paths = sorted(path for path in config_dir.glob("mi3xx_*.json") if "threshold" not in path.name)
        self.assertEqual(len(workload_paths), 7)

        for workload_path in workload_paths:
            raw = json.loads(workload_path.read_text(encoding="utf-8"))
            threshold_path = config_dir / raw["threshold_json"]
            expected = {
                key: value
                for key, value in json.loads(threshold_path.read_text(encoding="utf-8")).items()
                if not key.startswith("_")
            }
            with self.subTest(config=workload_path.name), tempfile.TemporaryDirectory() as directory:
                copied_config = Path(directory) / workload_path.name
                copied_threshold = Path(directory) / threshold_path.name
                copied_config.write_text(
                    json.dumps(raw).replace("<changeme>", "configured"),
                    encoding="utf-8",
                )
                copied_threshold.write_text(
                    threshold_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

                variant = load_variant(
                    copied_config,
                    {
                        "username": "tester",
                        "node_dict": {"node-a": {}, "node-b": {}},
                    },
                )
                workload = variant.benchmark_params.get("flux1_dev_t2i") or variant.benchmark_params.get(
                    "wan22_i2v_a14b"
                )
                self.assertEqual(workload["expected_results"], expected)
                self.assertEqual(variant.container.runtime.name, "docker")


if __name__ == "__main__":
    unittest.main()
