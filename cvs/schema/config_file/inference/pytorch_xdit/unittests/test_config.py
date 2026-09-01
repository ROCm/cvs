"""Unit tests for PyTorch xDiT config schemas (inference/pytorch_xdit/config.py)."""

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.schema.config_file.inference.pytorch_xdit.config import (
    PytorchXditFluxConfigFile,
    PytorchXditWanConfigFile,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[5]
_WAN_SAMPLE = (
    _PACKAGE_ROOT / "input" / "config_file" / "inference" / "xdit" / "mi3xx_pytorch_xdit_wan22_14b_single.json"
)
_FLUX_SAMPLE = (
    _PACKAGE_ROOT / "input" / "config_file" / "inference" / "xdit" / "mi3xx_pytorch_xdit_flux1_dev_single.json"
)


class TestPytorchXditWanConfigFile(unittest.TestCase):
    def test_sample_wan_json_validates(self):
        raw = json.loads(_WAN_SAMPLE.read_text())
        config = PytorchXditWanConfigFile.model_validate(raw)
        self.assertIsNotNone(config.benchmark_params.wan22_i2v_a14b)

    def test_wan_requires_wan_benchmark_block(self):
        with self.assertRaisesRegex(ValidationError, "wan22_i2v_a14b"):
            PytorchXditWanConfigFile.model_validate(
                {
                    "config": {
                        "hf_home": "/hf",
                        "output_base_dir": "/out",
                    },
                    "benchmark_params": {},
                }
            )

    def test_hf_home_changeme_rejected(self):
        with self.assertRaisesRegex(ValidationError, "placeholder '<changeme>'"):
            PytorchXditWanConfigFile.model_validate(
                {
                    "config": {
                        "hf_home": "/home/<changeme>/hf",
                        "output_base_dir": "/out",
                    },
                    "benchmark_params": {
                        "wan22_i2v_a14b": {
                            "prompt": "test",
                            "expected_results": {"auto": {"max_avg_total_time_s": 100.0}},
                        }
                    },
                }
            )


class TestPytorchXditFluxConfigFile(unittest.TestCase):
    def test_sample_flux_json_validates(self):
        raw = json.loads(_FLUX_SAMPLE.read_text())
        config = PytorchXditFluxConfigFile.model_validate(raw)
        self.assertIsNotNone(config.benchmark_params.flux1_dev_t2i)

    def test_flux_requires_flux_benchmark_block(self):
        with self.assertRaisesRegex(ValidationError, "flux1_dev_t2i"):
            PytorchXditFluxConfigFile.model_validate(
                {
                    "config": {
                        "hf_home": "/hf",
                        "output_base_dir": "/out",
                    },
                    "benchmark_params": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
