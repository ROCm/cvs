'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.conf_lib.loader.
'''

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvs.core.run_layout import RunLayout
from cvs.lib.conf_lib.config_blocks import Paths, _Forbid
from cvs.lib.conf_lib.loader import ConfigError, load_config
from cvs.lib.conf_lib.schemas.vllm import VariantConfig
from cvs.lib.utils.config_loader import BaseVariantConfig
from cvs.unittests.support import FAKE_CRED


def _paths_dict():
    return {
        "shared_fs": "/data/shared",
        "models_dir": "/data/models",
        "log_dir": "/data/logs",
        "hf_token_file": "/data/.hf_token",  # nosec B105 - a file path, not a secret
    }


def _base_config_dict():
    return {
        "schema_version": 1,
        "paths": _paths_dict(),
        "model": {"id": "amd/Llama-3.1-70B-Instruct", "remote": 0},
        "container": {"name": "my-container", "image": "rocm/vllm-dev:nightly", "runtime": {"name": "docker"}},
    }


class _PathsOnlyConfig(_Forbid):
    threshold_json: str = ""
    paths: Paths


class _InlineThresholdsConfig(_Forbid):
    log_dir: str
    expected_results: dict


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.tmp / "threshold.json").write_text(json.dumps({"_comment": "notes", "cell": {"m": {}}}))

    def _write_config(self, config):
        config = dict(config, threshold_json=str(self.tmp / "threshold.json"))
        path = self.tmp / "config.json"
        path.write_text(json.dumps(config))
        return path

    def test_cluster_file_and_cluster_dict_load_the_same_config(self):
        config_path = self._write_config(_base_config_dict())
        cluster = {"username": "jdoe", "priv_key_file": "/home/{user-id}/.ssh/id_rsa"}
        cluster_path = self.tmp / "cluster.json"
        cluster_path.write_text(json.dumps(cluster))
        from_dict = load_config(BaseVariantConfig, config_path, cluster)
        from_path = load_config(BaseVariantConfig, config_path, str(cluster_path))
        self.assertEqual(from_path.cluster, from_dict.cluster)
        self.assertNotIn("{user-id}", from_path.cluster["priv_key_file"])
        self.assertEqual(from_path.config.thresholds, {"cell": {"m": {}}})
        self.assertEqual(from_path.container, from_dict.container)

    def test_changeme_in_the_cluster_file_still_exits(self):
        config_path = self._write_config(_base_config_dict())
        with self.assertRaises(SystemExit):
            load_config(BaseVariantConfig, config_path, {"username": "<changeme>"})

    def test_model_without_a_container_returns_none(self):
        loaded = load_config(_PathsOnlyConfig, self._write_config({"paths": _paths_dict()}), {})
        self.assertIsNone(loaded.container)
        self.assertEqual(loaded.config.paths.log_dir, "/data/logs")

    def _write_config_without_threshold_file(self, config):
        empty = Path(tempfile.mkdtemp(dir=self.tmp))
        path = empty / "config.json"
        path.write_text(json.dumps(config))
        return path

    def test_model_without_thresholds_or_paths_loads_with_no_threshold_file(self):
        config = {"log_dir": "/home/{user-id}/logs", "expected_results": {"io_read": {}}}
        loaded = load_config(
            _InlineThresholdsConfig, self._write_config_without_threshold_file(config), {"username": "jdoe"}
        )
        self.assertEqual(loaded.config.log_dir, "/home/jdoe/logs")
        self.assertEqual(loaded.config.expected_results, {"io_read": {}})

    def test_vllm_model_still_needs_a_threshold_file(self):
        config_path = self._write_config_without_threshold_file(_base_config_dict())
        with self.assertRaises(FileNotFoundError):
            load_config(VariantConfig, config_path, {})

    def test_config_container_is_merged_onto_the_cluster_container(self):
        config = _base_config_dict()
        config["container"]["env"] = {"B": "config", "C": "config"}
        config["container"]["runtime"]["args"] = {"volumes": ["/config:/config"]}
        cluster_path = self.tmp / "cluster.json"
        cluster_path.write_text(
            json.dumps(
                {
                    "container": {
                        "name": "cluster-container",
                        "image": "cluster/image:1",
                        "env": {"A": "cluster", "B": "cluster"},
                        "runtime": {
                            "name": "docker",
                            "args": {
                                "shm_size": "64g",
                                "volumes": ["/cluster:/cluster"],
                                "registry": {"username": "jdoe"},
                            },
                        },
                    }
                }
            )
        )
        loaded = load_config(BaseVariantConfig, self._write_config(config), str(cluster_path))
        container = loaded.container
        self.assertEqual((container["name"], container["image"]), ("my-container", "rocm/vllm-dev:nightly"))
        self.assertEqual(container["env"], {"A": "cluster", "B": "config", "C": "config"})
        self.assertEqual(container["runtime"]["args"]["volumes"], ["/config:/config"])
        self.assertEqual(container["runtime"]["args"]["shm_size"], "64g")
        self.assertEqual(container["runtime"]["args"]["registry"], {"username": "jdoe"})
        container["runtime"]["args"]["registry"]["username"] = "changed"
        self.assertEqual(loaded.cluster["container"]["runtime"]["args"]["registry"], {"username": "jdoe"})

    def test_configstr_fields_with_placeholders_are_named_in_one_error(self):
        config = _base_config_dict()
        config["paths"]["models_dir"] = "<changeme-models-mount>"
        config["container"]["image"] = "<CHANGEME>"
        config["container"]["env"] = {"NCCL_SOCKET_IFNAME": "<changeme>", "NCCL_DEBUG": "INFO"}
        with self.assertRaises(ConfigError) as ctx:
            load_config(BaseVariantConfig, self._write_config(config), {})
        named = sorted(problem.split(":", 1)[0] for problem in ctx.exception.problems)
        self.assertEqual(named, ["container.env.NCCL_SOCKET_IFNAME", "container.image", "paths.models_dir"])

    def test_errors_name_fields_without_echoing_their_values(self):
        config = _base_config_dict()
        config["paths"]["token"] = FAKE_CRED
        config["container"]["env"] = {"HF_HOME": ["not", FAKE_CRED]}
        with self.assertRaises(ConfigError) as ctx:
            load_config(BaseVariantConfig, self._write_config(config), {})
        self.assertIn("paths.token", str(ctx.exception))
        self.assertIn("container.env.HF_HOME", str(ctx.exception))
        self.assertNotIn(FAKE_CRED, str(ctx.exception))

    def test_every_test_config_placeholder_expands(self):
        RunLayout._reset()
        self.addCleanup(RunLayout._reset)
        layout = RunLayout.get(str(self.tmp / "workspace"))
        config = _base_config_dict()
        config["paths"] = {
            "shared_fs": "/{home-mount-dir}/{user}",
            "models_dir": "{shared_fs}/{node-dir-name}/models",
            "log_dir": "{run_dir}/logs",
            "hf_token_file": "{home}/.hf_token",
        }
        config["container"]["name"] = "vllm-{user-id}"
        config["container"]["env"] = {"LOG_DIR": "{paths.log_dir}"}
        cluster = {"username": "jdoe", "home_mount_dir_name": "mnt", "node_dir_name": "node7"}
        with patch("cvs.lib.utils_lib.os.path.expanduser", side_effect=lambda p: "/nfs/jdoe" if p == "~jdoe" else p):
            loaded = load_config(BaseVariantConfig, self._write_config(config), cluster)
        paths = loaded.config.paths
        self.assertEqual(
            (paths.shared_fs, paths.models_dir, paths.log_dir, paths.hf_token_file),
            ("/mnt/jdoe", "/mnt/jdoe/node7/models", f"{layout.run_dir}/logs", "/nfs/jdoe/.hf_token"),
        )
        self.assertEqual(loaded.container["name"], "vllm-jdoe")
        self.assertEqual(loaded.container["env"], {"LOG_DIR": f"{layout.run_dir}/logs"})

    def test_config_without_run_dir_creates_no_run_layout(self):
        RunLayout._reset()
        self.addCleanup(RunLayout._reset)
        config = _base_config_dict()
        config["paths"]["log_dir"] = "/{home-mount-dir}/{user}/logs"
        loaded = load_config(BaseVariantConfig, self._write_config(config), {"username": "jdoe"})
        self.assertEqual(loaded.config.paths.log_dir, "/home/jdoe/logs")
        self.assertIsNone(RunLayout._instance)

    def test_blank_cluster_username_resolves_to_the_user_env(self):
        config = _base_config_dict()
        config["paths"]["shared_fs"] = "/home/{user-id}"
        with patch.dict(os.environ, {"USER": "envuser"}):
            loaded = load_config(BaseVariantConfig, self._write_config(config), {"username": ""})
        self.assertEqual(loaded.config.paths.shared_fs, "/home/envuser")
        self.assertEqual(loaded.cluster["username"], "")


if __name__ == "__main__":
    unittest.main()
