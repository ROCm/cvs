'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for long_context_accuracy_config and functional_config schemas.
'''

import unittest

from cvs.lib.inference.utils.functional_config import FunctionalConfig
from cvs.lib.inference.utils.long_context_accuracy_config import (
    LongContextAccuracyConfig,
    LongContextAccCell,
)


class TestFunctionalConfig(unittest.TestCase):
    def test_defaults_api_smoke_false(self):
        cfg = FunctionalConfig()
        self.assertFalse(cfg.api_smoke)


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
