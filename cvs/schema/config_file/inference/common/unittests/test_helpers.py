"""Unit tests for shared inference helper schemas (inference/common/)."""

import unittest

from cvs.schema.config_file.inference.common.functional import FunctionalConfig
from cvs.schema.config_file.inference.common.long_context_accuracy import (
    LongContextAccCell,
    LongContextAccuracyConfig,
)
from cvs.schema.config_file.inference.common.platform import PlatformConfig


class TestFunctionalConfig(unittest.TestCase):
    def test_defaults_api_smoke_false(self):
        cfg = FunctionalConfig()
        self.assertFalse(cfg.api_smoke)


class TestPlatformConfig(unittest.TestCase):
    def test_defaults_dmesg_scan_false(self):
        cfg = PlatformConfig()
        self.assertFalse(cfg.dmesg_scan)

    def test_defaults_gpu_metrics_poll_false(self):
        cfg = PlatformConfig()
        self.assertFalse(cfg.gpu_metrics_poll)


class TestLongContextAccuracyConfig(unittest.TestCase):
    def test_rejects_duplicate_cell_ids(self):
        with self.assertRaises(ValueError):
            LongContextAccuracyConfig(
                cells=[
                    LongContextAccCell(id="a", isl=1024),
                    LongContextAccCell(id="a", isl=2048),
                ]
            )


if __name__ == "__main__":
    unittest.main()
