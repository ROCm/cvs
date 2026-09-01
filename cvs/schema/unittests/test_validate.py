"""Unit tests for cvs/schema/validate.py file-loading helper."""

import json
import tempfile
import unittest
from pathlib import Path


from cvs.schema.cluster_file.cluster import ClusterConfigFile
from cvs.schema.config_file.aorta.benchmark import AortaBenchmarkConfigFile
from cvs.schema.config_file.inference.pytorch_xdit.config import PytorchXditWanConfigFile
from cvs.schema.config_file.preflight.config import PreflightConfigFile
from cvs.schema.config_file.training.megatron.variant import MegatronVariantConfig
from cvs.schema.validate import validate_config_file

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class TestValidateConfigFile(unittest.TestCase):
    def test_auto_detect_cluster_json(self):
        sample = _PACKAGE_ROOT / "input" / "cluster_file" / "cluster.json"
        config = validate_config_file(sample, config_type="auto")
        self.assertIsInstance(config, ClusterConfigFile)

    def test_preflight_unwraps_top_level_key(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(
                {"preflight": {"node_check": {"enabled": True, "gpus_per_node": 4}}},
                handle,
            )
            path = handle.name

        try:
            config = validate_config_file(path, config_type="preflight")
            self.assertIsInstance(config, PreflightConfigFile)
            self.assertTrue(config.node_check.enabled)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_aorta_yaml_load(self):
        sample = _PACKAGE_ROOT / "input" / "config_file" / "aorta" / "aorta_benchmark.yaml"
        config = validate_config_file(sample, config_type="aorta")
        self.assertIsInstance(config, AortaBenchmarkConfigFile)

    def test_auto_detect_megatron_variant(self):
        sample = (
            _PACKAGE_ROOT
            / "input"
            / "config_file"
            / "training"
            / "megatron"
            / "mi300x_megatron_llama-3.1-8b_single.json"
        )
        config = validate_config_file(sample, config_type="auto")
        self.assertIsInstance(config, MegatronVariantConfig)

    def test_auto_detect_pytorch_xdit_wan(self):
        sample = (
            _PACKAGE_ROOT / "input" / "config_file" / "inference" / "xdit" / "mi3xx_pytorch_xdit_wan22_14b_single.json"
        )
        config = validate_config_file(sample, config_type="auto")
        self.assertIsInstance(config, PytorchXditWanConfigFile)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            validate_config_file("/nonexistent/config.json")

    def test_unknown_auto_detect_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump({"unknown_top_level": True}, handle)
            path = handle.name

        try:
            with self.assertRaisesRegex(ValueError, "Cannot auto-detect config type"):
                validate_config_file(path, config_type="auto")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_yaml_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            handle.write("")
            path = handle.name

        try:
            with self.assertRaisesRegex(ValueError, "empty"):
                validate_config_file(path, config_type="aorta")
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
