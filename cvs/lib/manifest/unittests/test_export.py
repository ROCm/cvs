"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/manifest/export.py: the manifest-tree fan-in to a flat
# fact table.
#
# Pinned invariants:
#   - N manifests collapse to one fact table; a corrupt/forward-incompatible
#     manifest.json is skipped (logged, not crashed or silently dropped).
#   - An empty/all-unreadable tree still yields the fixed FACT_COLUMNS schema, so
#     column access never KeyErrors.
#   - FACT_COLUMNS stays in parity with the static (non-scalar) keys emitted by
#     _flatten_manifest.

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cvs.lib.manifest import RunLayout
from cvs.lib.manifest.export import FACT_COLUMNS, _flatten_manifest, collect_manifests, export_runs

from ._fixtures import _full_manifest


class TestExport(unittest.TestCase):
    def test_fan_in_skips_unreadable(self):
        """Flow 11: N manifests -> one fact table; junk manifest.json skipped."""
        tmp = Path(tempfile.mkdtemp())
        for rid, thr in [("run-1", 1000), ("run-2", 2000)]:
            layout = RunLayout(tmp, "suite", "cell-a", "h123", rid).ensure()
            _full_manifest(run_id=rid, extra_metric=thr).write(layout.manifest_path)
        # A partial/corrupt manifest.json must be skipped, not crash the export.
        junk = RunLayout(tmp, "suite", "cell-b", "h999", "run-junk").ensure()
        junk.manifest_path.write_text("{ not valid json")

        manifests = collect_manifests(tmp)
        self.assertEqual(len(manifests), 2)

        out = export_runs(tmp, tmp / "fact.parquet")
        fact = pd.read_parquet(out)
        self.assertEqual(len(fact), 2)
        for col in (
            "run_id",
            "test_id",
            "workload_hash",
            "overall_status",
            "scalar_total_throughput",
            "scalar_extra_metric",
        ):
            self.assertIn(col, fact.columns)
        self.assertEqual(set(fact["run_id"]), {"run-1", "run-2"})

    def test_empty_export_has_fact_columns(self):
        """Exporting an empty tree yields the fixed-schema fact table, not column-less."""
        tmp = Path(tempfile.mkdtemp())
        fact = pd.read_parquet(export_runs(tmp, tmp / "fact.parquet"))
        self.assertEqual(len(fact), 0)
        self.assertEqual(list(fact.columns), FACT_COLUMNS)
        self.assertEqual(list(fact["run_id"]), [])  # column access must not KeyError

    def test_fact_columns_match_flatten(self):
        """Parity guard: FACT_COLUMNS must equal the static (non-scalar) keys of _flatten_manifest."""
        static = [k for k in _flatten_manifest(_full_manifest()) if not k.startswith("scalar_")]
        self.assertEqual(static, FACT_COLUMNS)

    def test_collect_logs_unreadable_skip(self):
        """A skipped corrupt/forward-incompatible manifest is logged, not silently dropped."""
        tmp = Path(tempfile.mkdtemp())
        good = RunLayout(tmp, "s", "c", "h", "good").ensure()
        _full_manifest(run_id="good").write(good.manifest_path)
        bad = RunLayout(tmp, "s", "c", "h", "bad").ensure()
        payload = json.loads(good.manifest_path.read_text())
        payload["unknown_future_field"] = 1  # rejected by extra="forbid"
        bad.manifest_path.write_text(json.dumps(payload))
        with self.assertLogs("cvs.lib.manifest.export", level="WARNING") as cm:
            manifests = collect_manifests(tmp)
        self.assertEqual([m.identity.run_id for m in manifests], ["good"])
        self.assertTrue(any("bad" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()
