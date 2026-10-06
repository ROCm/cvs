"""Configuration validation entry-point compatibility."""

import json
import getpass
import tempfile
import unittest
from pathlib import Path

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.parsers.schemas import validate_config_file


class TestAortaConfigDispatch(unittest.TestCase):
    def test_generic_validator_delegates_to_shared_variant_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = variant_dict()
            thresholds = raw.pop("thresholds")
            (root / "test_threshold.json").write_text(json.dumps(thresholds))
            path = root / "aorta.json"
            path.write_text(json.dumps(raw))
            for mode in ("auto", "aorta"):
                with self.subTest(mode=mode):
                    config = validate_config_file(path, config_type=mode)
                    self.assertIsInstance(config, AortaVariantConfig)
                    self.assertEqual(config.thresholds, thresholds)

    def test_generic_validator_uses_local_user_without_cluster_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = variant_dict()
            thresholds = raw.pop("thresholds")
            raw["container"]["name"] = "{user-id}_aorta"
            (root / "test_threshold.json").write_text(json.dumps(thresholds))
            path = root / "aorta.json"
            path.write_text(json.dumps(raw))
            config = validate_config_file(path)
            self.assertEqual(config.container.name, f"{getpass.getuser()}_aorta")
