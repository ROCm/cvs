"""
A4 -- cvs run / list suite-name resolution for DTNI tier directories.

The pre-DTNI surface (ListPlugin._find_test) maps a name to ONE test_*.py
stem. DTNI tier trees live as DIRECTORIES under cvs/tests/dtni/<name>/, so
the resolver must (a) recognize a DTNI suite name and return the directory
path, (b) fail-closed when the name does not resolve -- never silently run
zero tests, (c) preserve the existing flat-file lookup for legacy suites.
"""

from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path

import cvs
from cvs.cli_plugins.list_plugin import ListPlugin


CVS_PKG_DIR = Path(cvs.__file__).resolve().parent
DTNI_TESTS_DIR = CVS_PKG_DIR / "tests" / "dtni"


class TestA4Resolution(unittest.TestCase):
    """A4: DTNI suite name -> tests/dtni/<name>/ directory; fail-closed on miss."""

    def setUp(self) -> None:
        # Use a real suite subdir under the actual package so ListPlugin's
        # importlib.resources walk picks it up. Clean up in tearDown.
        self.suite_name = "barebones_a4_probe"
        self.suite_dir = DTNI_TESTS_DIR / self.suite_name
        self.suite_dir.mkdir(parents=True, exist_ok=True)
        (self.suite_dir / "__init__.py").write_text("")
        (self.suite_dir / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")
        # Force re-discovery
        self.plugin = ListPlugin()

    def tearDown(self) -> None:
        if self.suite_dir.exists():
            shutil.rmtree(self.suite_dir)

    def test_known_dtni_suite_resolves_to_directory(self) -> None:
        """A DTNI suite name resolves to the cvs/tests/dtni/<name>/ directory."""
        resolved = self.plugin._find_test(self.suite_name)
        self.assertIsNotNone(
            resolved,
            f"Expected DTNI suite {self.suite_name!r} to resolve, got None",
        )
        # The resolver returns a path/module identifier; either form must point
        # at the DTNI suite directory, not a single test_*.py stem.
        resolved_path = str(resolved)
        self.assertIn(
            os.path.join("tests", "dtni", self.suite_name),
            resolved_path.replace(".", os.sep),
            f"Resolved path {resolved_path!r} does not target the dtni/{self.suite_name}/ dir",
        )

    def test_unknown_dtni_suite_fails_closed(self) -> None:
        """An unknown suite name returns None (NOT a stub that silently runs zero tests)."""
        resolved = self.plugin._find_test("definitely_not_a_real_dtni_suite_zzz")
        self.assertIsNone(
            resolved,
            "Unknown suite must fail-closed (None), never silently match an empty tree",
        )

    def test_legacy_flat_test_lookup_preserved(self) -> None:
        """Pre-existing flat cvs/tests/<category>/test_<name>.py lookup must still work
        (regression guard -- A4 must extend, not replace)."""
        # Collect every non-dtni stem upfront so an empty set FAILS the test,
        # not silently skips it (the prior implementation could pass-as-skip
        # on a hypothetical dtni-only checkout, which would mask a real regression).
        non_dtni = [
            (stem, modpath)
            for tests in self.plugin.test_map.values()
            for stem, modpath in tests.items()
            if "tests.dtni." not in modpath
        ]
        self.assertTrue(
            non_dtni,
            "regression guard requires at least one non-dtni test stem to validate",
        )
        stem, _modpath = non_dtni[0]
        resolved = self.plugin._find_test(stem)
        self.assertIsNotNone(resolved, f"Legacy stem {stem!r} regressed")


if __name__ == "__main__":
    unittest.main()
