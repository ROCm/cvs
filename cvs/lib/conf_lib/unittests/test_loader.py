'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.conf_lib.loader.
'''

import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.conf_lib.config_blocks import Paths, _Forbid
from cvs.lib.conf_lib.loader import ConfigError, load_config
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


if __name__ == "__main__":
    unittest.main()
