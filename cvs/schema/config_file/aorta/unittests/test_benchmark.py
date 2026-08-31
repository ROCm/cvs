"""Unit tests for Aorta benchmark config schema (config_file/aorta/benchmark.py)."""

import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from cvs.schema.config_file.aorta.benchmark import AortaBenchmarkConfigFile

_PACKAGE_ROOT = Path(__file__).resolve().parents[4]
_SAMPLE_CONFIG = _PACKAGE_ROOT / "input" / "config_file" / "aorta" / "aorta_benchmark.yaml"


class TestAortaBenchmarkConfigFile(unittest.TestCase):
    def test_sample_yaml_validates(self):
        raw = yaml.safe_load(_SAMPLE_CONFIG.read_text())
        config = AortaBenchmarkConfigFile.model_validate(raw)
        self.assertIn("aorta", config.aorta_path)
        self.assertFalse(config.analysis.enable_tracelens)

    def test_aorta_path_changeme_rejected(self):
        with self.assertRaisesRegex(ValidationError, "placeholder '<changeme>'"):
            AortaBenchmarkConfigFile.model_validate({"aorta_path": "/path/<changeme>/aorta"})

    def test_nested_defaults_applied(self):
        config = AortaBenchmarkConfigFile.model_validate({"aorta_path": "/opt/aorta"})
        self.assertEqual(config.docker.container_name, "aorta-benchmark")
        self.assertEqual(config.environment.NCCL_DEBUG, "VERSION")


if __name__ == "__main__":
    unittest.main()
