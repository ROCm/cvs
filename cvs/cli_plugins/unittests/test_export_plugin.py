"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from cvs.cli_plugins.export_plugin import ExportPlugin
from cvs.lib.manifest.export import FACT_COLUMNS
from cvs.lib.manifest.schema import (
    ConfigInputs,
    Identity,
    Manifest,
    Verdicts,
)
from cvs.main import discover_plugins


def _make_manifest(run_id: str, cell_id: str = "cell-a", test_id: str = "suite") -> Manifest:
    """Build a minimal valid manifest with scalars={} via the public schema."""
    return Manifest(
        identity=Identity(
            run_id=run_id,
            test_id=test_id,
            cell_id=cell_id,
            config_hash="ch",
            workload_hash="wh",
            verification_hash="vh",
            cvs_git_sha="abc123",
            started_at="2025-01-01T00:00:00+00:00",
            finished_at="2025-01-01T00:01:00+00:00",
        ),
        config=ConfigInputs(model="llama-3.1-70b"),
        verdicts=Verdicts(overall_status="complete", scalars={}),
    )


def _write_tree(root: Path, run_ids):
    """Write one manifest per run_id under root/<test_id>/<cell_id>/hrun-X/<run_id>/manifest.json."""
    for i, rid in enumerate(run_ids, start=1):
        m = _make_manifest(rid)
        path = root / m.identity.test_id / m.identity.cell_id / f"hrun-{i}" / rid / "manifest.json"
        m.write(path)


class TestExportPluginDiscovery(unittest.TestCase):
    """Plugin must be discoverable by the CLI."""

    def test_export_is_discovered(self):
        plugins = discover_plugins()
        names = [p.get_name() for p in plugins]
        self.assertIn("export", names)


class TestExportPluginParser(unittest.TestCase):
    """--artifact-dir and -o/--out are required; no other surface."""

    def _build_parser(self):
        plugin = ExportPlugin()
        root = argparse.ArgumentParser(prog="cvs")
        sub = root.add_subparsers(dest="command")
        plugin.get_parser(sub)
        return root

    def test_artifact_dir_and_out_required(self):
        parser = self._build_parser()
        # Missing --artifact-dir AND --out -> argparse calls sys.exit (SystemExit).
        with self.assertRaises(SystemExit):
            parser.parse_args(["export"])
        # Missing -o -> still fails.
        with self.assertRaises(SystemExit):
            parser.parse_args(["export", "--artifact-dir", "/tmp/x"])
        # Missing --artifact-dir -> still fails.
        with self.assertRaises(SystemExit):
            parser.parse_args(["export", "-o", "/tmp/x.parquet"])

    def test_parses_minimal_valid_args(self):
        parser = self._build_parser()
        args = parser.parse_args(["export", "--artifact-dir", "/tmp/a", "-o", "/tmp/b.parquet"])
        self.assertEqual(args.artifact_dir, "/tmp/a")
        self.assertEqual(args.out, "/tmp/b.parquet")


class TestExportPluginRun(unittest.TestCase):
    """End-to-end run() behavior against the real engine."""

    def test_empty_dir_yields_zero_row_fact_table_with_exact_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "empty"
            root.mkdir()
            out = Path(td) / "empty.parquet"
            plugin = ExportPlugin()
            args = argparse.Namespace(artifact_dir=str(root), out=str(out))
            rc = plugin.run(args)
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            schema_names = set(pq.read_schema(out).names)
            self.assertEqual(schema_names, set(FACT_COLUMNS))
            # Zero rows.
            self.assertEqual(pq.read_table(out).num_rows, 0)

    def test_fixture_tree_yields_n_rows_with_exact_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "runs"
            _write_tree(root, ["run-1", "run-2", "run-3"])
            out = Path(td) / "fact.parquet"
            plugin = ExportPlugin()
            args = argparse.Namespace(artifact_dir=str(root), out=str(out))
            rc = plugin.run(args)
            self.assertEqual(rc, 0)
            table = pq.read_table(out)
            self.assertEqual(table.num_rows, 3)
            # scalars={} on every row -> NO scalar_* columns; schema is exactly FACT_COLUMNS.
            self.assertEqual(set(table.schema.names), set(FACT_COLUMNS))

    def test_corrupt_manifest_is_skipped_visibly_good_rows_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "runs"
            _write_tree(root, ["run-1", "run-2"])
            # Drop a corrupt manifest into the tree.
            bad = root / "suite" / "cell-a" / "hrun-bad" / "run-bad" / "manifest.json"
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_text("{ this is not valid json ")
            out = Path(td) / "fact.parquet"
            plugin = ExportPlugin()
            args = argparse.Namespace(artifact_dir=str(root), out=str(out))
            with self.assertLogs("cvs.lib.manifest.export", level="WARNING") as cm:
                rc = plugin.run(args)
            self.assertEqual(rc, 0)
            # Visible skip: warning was emitted, naming the corrupt path.
            self.assertTrue(any("skipping unreadable manifest" in line for line in cm.output))
            # Good rows preserved.
            table = pq.read_table(out)
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(set(table.schema.names), set(FACT_COLUMNS))


if __name__ == "__main__":
    unittest.main()
