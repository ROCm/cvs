"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cvs.lib.failure_pattern_scanner import FailurePatternScanner
from cvs.lib.failure_taxonomy import FailureCategory


def _write_yaml(dir_path: Path, body: str) -> Path:
    p = dir_path / "patterns.yaml"
    p.write_text(body)
    return p


class TestFailurePatternScanner(unittest.TestCase):
    def test_default_catalog_loads_and_every_pattern_has_valid_category(self):
        """G6b row #1: ship the seed catalog; each entry binds to a FailureCategory."""
        scanner = FailurePatternScanner()
        self.assertGreater(len(scanner.patterns), 0, "default catalog must not be empty")
        valid = {c for c in FailureCategory}
        for pat in scanner.patterns:
            self.assertIn(
                pat.category,
                valid,
                f"pattern {pat.id!r} has category {pat.category!r} not in FailureCategory",
            )

    def test_duplicate_pattern_id_raises_value_error(self):
        """G6b row #2: duplicate id at load time is a fail-closed ValueError."""
        body = (
            "patterns:\n"
            "  - id: oom_killer\n"
            "    source: dmesg\n"
            "    pattern: \"Out of memory\"\n"
            "    category: failure_pattern_matched\n"
            "    severity: fatal\n"
            "    hint: \"x\"\n"
            "  - id: oom_killer\n"
            "    source: dmesg\n"
            "    pattern: \"invoked oom-killer\"\n"
            "    category: failure_pattern_matched\n"
            "    severity: fatal\n"
            "    hint: \"y\"\n"
        )
        with TemporaryDirectory() as d:
            path = _write_yaml(Path(d), body)
            with self.assertRaises(ValueError):
                FailurePatternScanner(patterns_file=path)

    def test_unknown_category_in_yaml_raises_value_error(self):
        """G6b row #3: unknown FailureCategory at load is a fail-closed ValueError."""
        body = (
            "patterns:\n"
            "  - id: bogus\n"
            "    source: dmesg\n"
            "    pattern: \"whatever\"\n"
            "    category: not_a_real_category\n"
            "    severity: fatal\n"
            "    hint: \"z\"\n"
        )
        with TemporaryDirectory() as d:
            path = _write_yaml(Path(d), body)
            with self.assertRaises(ValueError):
                FailurePatternScanner(patterns_file=path)


if __name__ == "__main__":
    unittest.main()
