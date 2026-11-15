'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.migrate.maps.
'''

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvs.config_validation.migrate import maps


class TestSuiteNames(unittest.TestCase):
    def test_lists_each_public_module_sorted_without_importing_any(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("zeta", "alpha", "_helpers"):
                (root / f"{name}.py").write_text("raise AssertionError('listing a suite imports its map')\n")
            (root / "unittests").mkdir()
            (root / "unittests" / "__init__.py").write_text("")
            with mock.patch.object(maps, "__path__", [str(root)]):
                self.assertEqual(maps.suite_names(), ["alpha", "zeta"])


if __name__ == "__main__":
    unittest.main()
