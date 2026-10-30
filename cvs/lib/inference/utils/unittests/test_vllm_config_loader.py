'''Unit tests for loading shipped and user vLLM configs through load_variant.'''

import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.inference.utils.vllm_config_loader import VariantConfig, load_variant
from cvs.lib.utils.config_loader import ConfigError, load_config
from cvs.unittests.support import write_filled_config

CONFIG_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"
DISTRIBUTED_TEMPLATE = CONFIG_DIR / "mi3xx_vllm_llama33-70b_fp8_distributed.json"
CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"


def _templates():
    return sorted(path for path in CONFIG_DIR.glob("*.json") if not path.name.endswith("threshold.json"))


class TestShippedTemplates(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_every_template_loads_once_its_placeholders_are_filled(self):
        templates = _templates()
        self.assertTrue(templates)
        for template in templates:
            with self.subTest(config=template.name):
                load_variant(write_filled_config(template, self.tmp), {"username": "test"})

    def test_unfilled_distributed_template_names_the_placeholders_it_checks(self):
        with self.assertRaises(ConfigError) as ctx:
            load_variant(DISTRIBUTED_TEMPLATE, {"username": "test"})
        self.assertEqual(
            sorted(problem.split(":", 1)[0] for problem in ctx.exception.problems),
            [
                "container.env.GLOO_SOCKET_IFNAME",
                "container.env.NCCL_IB_GID_INDEX",
                "container.env.NCCL_SOCKET_IFNAME",
                "container.env.TP_SOCKET_IFNAME",
                "container.image",
                "container.runtime.args.volumes.1",
                "container.runtime.args.volumes.3",
                "server_params.model",
            ],
        )

    def test_placeholder_in_an_option_value_fails_the_load(self):
        filled = write_filled_config(DISTRIBUTED_TEMPLATE, self.tmp)
        config = json.loads(filled.read_text())
        config["server_params"]["served_model_name"] = "<changeme-name>"
        (cell,) = config["runs"]
        config["sweeps"][cell]["tokenizer"] = "<changeme>"
        filled.write_text(json.dumps(config))
        with self.assertRaises(ConfigError) as ctx:
            load_variant(filled, {"username": "test"})
        self.assertEqual(
            sorted(problem.split(":", 1)[0] for problem in ctx.exception.problems),
            ["server_params.served_model_name", f"sweeps.{cell}.tokenizer"],
        )


class TestContainerForOrchestrator(unittest.TestCase):
    def test_cluster_env_volumes_and_shm_size_merge_without_aliasing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "threshold.json").write_text("{}")
            config = {
                "enforce_thresholds": False,
                "threshold_json": str(tmp / "threshold.json"),
                "paths": {"shared_fs": "/home/x", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": "/t"},
                "container": {
                    "name": "vllm",
                    "image": "rocm/vllm:1",
                    "env": {"B": "config"},
                    "runtime": {"name": "docker", "args": {"volumes": ["/config:/config"]}},
                },
                "server_params": {"model": "/models/m", "tensor_parallel_size": 8},
                "sweeps": {CELL: {}},
                "runs": [CELL],
            }
            (tmp / "config.json").write_text(json.dumps(config))
            cluster = {
                "container": {
                    "name": "from-cluster",
                    "env": {"A": "cluster", "B": "cluster"},
                    "runtime": {
                        "name": "docker",
                        "args": {
                            "volumes": ["/cluster:/cluster"],
                            "shm_size": "64g",
                            "registry": {"username": "user", "password_file": "/home/x/registry"},
                        },
                    },
                }
            }
            loaded = load_config(VariantConfig, tmp / "config.json", cluster)
        self.assertEqual(loaded.container["name"], "vllm")
        self.assertEqual(loaded.container["env"], {"A": "cluster", "B": "config"})
        self.assertEqual(loaded.container["runtime"]["args"]["volumes"], ["/config:/config"])
        self.assertEqual(loaded.container["runtime"]["args"]["shm_size"], "64g")
        loaded.container["env"]["A"] = "mutated"
        loaded.container["runtime"]["args"]["volumes"].append("/extra")
        loaded.container["runtime"]["args"]["shm_size"] = "1g"
        loaded.container["runtime"]["args"]["registry"]["username"] = "mutated"
        cluster_container = loaded.cluster["container"]
        self.assertEqual(cluster_container["env"]["A"], "cluster")
        self.assertEqual(cluster_container["runtime"]["args"]["volumes"], ["/cluster:/cluster"])
        self.assertEqual(cluster_container["runtime"]["args"]["shm_size"], "64g")
        self.assertEqual(cluster_container["runtime"]["args"]["registry"]["username"], "user")


if __name__ == "__main__":
    unittest.main()
