'''Unit tests for llm-d vLLM configuration and command generation.'''

import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.orchestrator.llm_d.llm_d_common import (
    envoy_run_command,
    epp_run_command,
    render_endpoints,
    render_envoy_config,
    render_epp_config,
    scripts_dir,
)
from cvs.lib.orchestrator.llm_d.vllm.config import ConfigSection, load_config
from cvs.lib.orchestrator.llm_d.vllm.llm_d_vllm_lib import worker_run_command
from cvs.lib.orchestrator.llm_d.vllm.topology import LlmdTopology, scope_cluster


def _raw_config():
    return {
        "paths": {
            "shared_fs": "/home/{user-id}",
            "models_dir": "/models/llama",
            "log_dir": "{shared_fs}/logs",
            "hf_token_file": "{shared_fs}/.hf_token",
        },
        "container": {
            "image": "vllm:test",
            "env": {"HF_HUB_OFFLINE": "1"},
            "runtime": {
                "name": "docker",
                "args": {
                    "network": "host",
                    "ipc": "host",
                    "privileged": True,
                    "shm_size": "128g",
                    "volumes": ["/models/llama:/models/llama:ro"],
                    "devices": ["/dev/dri", "/dev/kfd"],
                    "group_add": ["video"],
                    "cap_add": ["SYS_PTRACE"],
                    "security_opt": ["seccomp=unconfined"],
                },
            },
        },
        "server_params": {
            "backend": "vllm",
            "model": "/models/llama",
            "served_model_name": "llama",
            "tensor_parallel_size": 8,
            "hip_visible_devices": "0,1,2,3,4,5,6,7",
            "add_flags": ["--disable-access-log-for-endpoints=/health,/metrics"],
        },
        "gateway": {
            "node": "node1",
            "listen_port": 8081,
            "admin_port": 19000,
            "epp_grpc_port": 9002,
            "epp_grpc_health_port": 9003,
            "epp_metrics_port": 9090,
            "config_dir": "/etc/epp",
            "envoy_config_dir": "/etc/envoy",
            "envoy_image": "envoy:test",
            "epp_image": "epp",
            "epp_version": "v1",
        },
        "workers": [
            {"name": "vllm-0", "node": "node1", "port": 8000},
            {"name": "vllm-1", "node": "node2", "port": 8001},
        ],
        "smoke": {"timeout_s": 120},
    }


def _cluster():
    return {
        "orchestrator": "container",
        "username": "tester",
        "priv_key_file": "/home/tester/.ssh/id_rsa",
        "node_dict": {
            "node1": {"vpc_ip": "10.0.0.1"},
            "node2": {"vpc_ip": "10.0.0.2"},
            "unused": {"vpc_ip": "10.0.0.3"},
        },
    }


class TestConfig(unittest.TestCase):
    def test_load_resolves_user_and_self_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(_raw_config()), encoding="utf-8")
            config = load_config(path, _cluster())
        self.assertEqual(config.paths.shared_fs, "/home/tester")
        self.assertEqual(config.paths.log_dir, "/home/tester/logs")
        self.assertEqual(config.paths.hf_token_file, "/home/tester/.hf_token")

    def test_load_rejects_unknown_worker_node(self):
        raw = _raw_config()
        raw["workers"][1]["node"] = "missing"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent from cluster"):
                load_config(path, _cluster())


class TestTopology(unittest.TestCase):
    def setUp(self):
        self.config = ConfigSection(_raw_config())
        self.topology = LlmdTopology(self.config, _cluster())

    def test_gateway_worker_uses_loopback_and_remote_uses_vpc(self):
        self.assertEqual(self.topology.workers[0]["address"], "127.0.0.1")
        self.assertEqual(self.topology.workers[1]["address"], "10.0.0.2")

    def test_scope_uses_baremetal_and_excludes_unreferenced_nodes(self):
        scoped = scope_cluster(_cluster(), self.topology)
        self.assertEqual(scoped["orchestrator"], "baremetal")
        self.assertEqual(list(scoped["node_dict"]), ["node1", "node2"])


class TestRenderers(unittest.TestCase):
    def setUp(self):
        self.config = ConfigSection(_raw_config())
        self.topology = LlmdTopology(self.config, _cluster())

    def test_endpoints_contain_both_worker_addresses(self):
        rendered = render_endpoints(self.config, self.topology)
        self.assertIn("address: 127.0.0.1", rendered)
        self.assertIn("address: 10.0.0.2", rendered)
        self.assertIn('port: "8001"', rendered)

    def test_epp_and_envoy_ports_are_parametrized(self):
        self.assertIn("path: /etc/epp/endpoints.yaml", render_epp_config(self.config))
        envoy = render_envoy_config(self.config)
        self.assertIn("port_value: 8081", envoy)
        self.assertIn("port_value: 9002", envoy)
        self.assertIn("name: vllm", envoy)

    def test_gateway_yaml_lives_in_shared_scripts(self):
        names = {path.name for path in scripts_dir().iterdir()}
        self.assertTrue({"epp_config.yaml", "envoy.yaml", "endpoints.yaml", "endpoint_entry.yaml"} <= names)


class TestDockerCommands(unittest.TestCase):
    def setUp(self):
        self.config = ConfigSection(_raw_config())

    def test_worker_command_uses_named_container_without_token_value(self):
        command = worker_run_command(self.config, _raw_config()["workers"][0], include_hf_token=True)
        self.assertIn("docker rm -f vllm-0", command)
        self.assertIn("--tensor-parallel-size 8", command)
        self.assertIn("-e HF_TOKEN", command)
        self.assertIn(".hf_token", command)

    def test_worker_command_omits_hf_token_when_file_was_missing(self):
        command = worker_run_command(self.config, _raw_config()["workers"][0])
        self.assertNotIn("HF_TOKEN", command)

    def test_gateway_commands_use_configured_images(self):
        self.assertIn("epp:v1", epp_run_command(self.config))
        self.assertIn("envoy:test", envoy_run_command(self.config))


if __name__ == "__main__":
    unittest.main()
