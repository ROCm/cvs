"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/manifest/sidecars.py: the parquet/YAML sidecar writers.
#
# Pinned invariants:
#   - Wide per-sample rows round-trip with their named columns.
#   - Trajectory rows are long-format (step/metric/value/role/host); an
#     off-schema (wide) row is rejected.
#   - An empty write preserves the declared columns instead of a column-less
#     frame.
#   - write_resolved_config dumps YAML that reloads to the same dict.

import tempfile
import unittest
from pathlib import Path

import yaml

from cvs.lib.manifest import (
    read_samples,
    read_trajectory,
    write_resolved_config,
    write_samples,
    write_trajectory,
)


class TestSidecars(unittest.TestCase):
    def test_samples_roundtrip_wide(self):
        """Flow 8: per-sample wide rows round-trip with named columns."""
        tmp = Path(tempfile.mkdtemp())
        rows = [{"request_id": i, "ttft_ms": 10 + i, "role": "server", "host": "n0"} for i in range(5)]
        write_samples(tmp / "samples.parquet", rows)
        frame = read_samples(tmp / "samples.parquet")
        self.assertEqual(len(frame), 5)
        self.assertIn("ttft_ms", frame.columns)

    def test_trajectory_roundtrip_long_format(self):
        """Flow 9: long-format trajectory rows round-trip with stable columns."""
        tmp = Path(tempfile.mkdtemp())
        rows = [
            {"step": i, "metric": "loss", "value": 2.0 - i * 0.1, "role": "trainer", "host": "n0"} for i in range(4)
        ]
        write_trajectory(tmp / "trajectory.parquet", rows)
        frame = read_trajectory(tmp / "trajectory.parquet")
        self.assertEqual(len(frame), 4)
        self.assertEqual(sorted(frame.columns), sorted(["step", "metric", "value", "role", "host"]))

    def test_write_resolved_config_yaml(self):
        """Flow 10: resolved config dumped as YAML, reloads to the same dict."""
        tmp = Path(tempfile.mkdtemp())
        dump = {"framework": "vllm", "params": {"model": "llama", "tensor_parallelism": 8}}
        out = write_resolved_config(tmp / "config.resolved.yaml", dump)
        self.assertEqual(yaml.safe_load(out.read_text()), dump)

    def test_trajectory_rejects_wide_row(self):
        """Long-format invariant: a row with an off-schema column is rejected."""
        tmp = Path(tempfile.mkdtemp())
        wide = [{"step": 0, "loss": 2.0, "role": "trainer", "host": "n0"}]
        with self.assertRaisesRegex(ValueError, "long-format"):
            write_trajectory(tmp / "trajectory.parquet", wide)

    def test_empty_rows_preserve_declared_columns(self):
        """Empty input keeps the declared schema instead of a column-less frame."""
        tmp = Path(tempfile.mkdtemp())
        write_trajectory(tmp / "trajectory.parquet", [])
        traj = read_trajectory(tmp / "trajectory.parquet")
        self.assertEqual(len(traj), 0)
        self.assertEqual(sorted(traj.columns), sorted(["step", "metric", "value", "role", "host"]))
        write_samples(tmp / "samples.parquet", [])
        samples = read_samples(tmp / "samples.parquet")
        self.assertEqual(len(samples), 0)
        self.assertIn("request_id", samples.columns)


if __name__ == "__main__":
    unittest.main()
