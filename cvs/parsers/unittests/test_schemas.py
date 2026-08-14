"""
Unit tests for the config-file Pydantic schemas in ``cvs/parsers/schemas.py``.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.parsers.schemas import AortaBenchmarkConfigFile


class TestAortaMultiNodeBlock(unittest.TestCase):
    def test_default_multi_node_block_has_auto_mode(self):
        cfg = AortaBenchmarkConfigFile.model_validate({"aorta_path": "/tmp/aorta"})
        self.assertEqual(cfg.multi_node.master_launch_mode, "auto")
        self.assertTrue(cfg.multi_node.collect_traces)
        self.assertEqual(cfg.multi_node.train_script, "train.py")

    def test_yaml_without_multi_node_block_still_validates(self):
        # Backward compatibility: configs written before this block existed.
        cfg = AortaBenchmarkConfigFile.model_validate({"aorta_path": "/tmp/aorta"})
        self.assertIsNotNone(cfg.multi_node)

    def test_extra_keys_under_multi_node_are_rejected(self):
        raw = {"aorta_path": "/tmp/aorta", "multi_node": {"bogus_key": "value"}}
        with self.assertRaises(ValidationError):
            AortaBenchmarkConfigFile.model_validate(raw)

    def test_invalid_master_launch_mode_rejected(self):
        raw = {"aorta_path": "/tmp/aorta", "multi_node": {"master_launch_mode": "magic"}}
        with self.assertRaises(ValidationError):
            AortaBenchmarkConfigFile.model_validate(raw)

    def test_out_of_range_master_port_rejected(self):
        raw = {"aorta_path": "/tmp/aorta", "multi_node": {"master_port": 80}}
        with self.assertRaises(ValidationError):
            AortaBenchmarkConfigFile.model_validate(raw)


class TestAortaTrainScriptPathCheck(unittest.TestCase):
    """``validate_paths_exist`` should only demand train_script in torchrun mode."""

    def _config(self, root: Path, mode: str) -> AortaBenchmarkConfigFile:
        for rel in ("config/distributed.yaml", "scripts/build_rccl.sh", "scripts/rccl_exp.sh"):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return AortaBenchmarkConfigFile.model_validate(
            {
                "aorta_path": str(root),
                "build_script": "scripts/build_rccl.sh",
                "experiment_script": "scripts/rccl_exp.sh",
                "analysis": {"enable_tracelens": False, "enable_gemm_analysis": False},
                "multi_node": {"master_launch_mode": mode},
            }
        )

    def test_torchrun_mode_reports_missing_train_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors = self._config(Path(tmp), "torchrun").validate_paths_exist()
            self.assertTrue(any("train_script" in e for e in errors), errors)

    def test_torchrun_mode_passes_when_train_script_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), "torchrun")
            (Path(tmp) / "train.py").touch()
            self.assertEqual(cfg.validate_paths_exist(), [])

    def test_script_mode_does_not_require_train_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._config(Path(tmp), "script").validate_paths_exist(), [])


if __name__ == "__main__":
    unittest.main()
