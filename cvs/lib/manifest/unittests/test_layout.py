"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/manifest/layout.py: the content-addressable RunLayout.
#
# Pinned invariants:
#   - The local root is <artifact_dir>/<test_id>/<cell_id>/<hash>/<run_id> and
#     ensure() materializes it (plus logs_dir).
#   - A1: a remote_artifact_dir mirrors the same suffix; to_remote re-bases a
#     local path onto the remote root.
#   - Fail-closed: to_remote raises with a *distinct* message for "no remote
#     configured" vs "path not under root"; id segments that bear a path / are
#     empty / contain ".." are rejected so the mirror cannot escape its root.

import tempfile
import unittest
from pathlib import Path

from cvs.lib.manifest import RunLayout


class TestLayout(unittest.TestCase):
    def test_local_paths_and_ensure(self):
        """Flow 4: content-addressable root + ensure() creates dirs."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "h123", "run-1")
        self.assertEqual(layout.root, tmp / "suite" / "cell-a" / "h123" / "run-1")
        self.assertEqual(layout.manifest_path.name, RunLayout.MANIFEST)
        self.assertEqual(layout.samples_path.name, RunLayout.SAMPLES)
        self.assertEqual(layout.events_path.name, RunLayout.EVENTS)
        self.assertFalse(layout.root.exists())
        layout.ensure()
        self.assertTrue(layout.root.is_dir())
        self.assertTrue(layout.logs_dir.is_dir())

    def test_remote_root_mirror(self):
        """Flow 5: A1 remote run-root mirrors the suffix; to_remote re-bases."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "h123", "run-1", remote_artifact_dir="/remote/artifacts")
        self.assertEqual(layout.remote_root, Path("/remote/artifacts") / "suite" / "cell-a" / "h123" / "run-1")
        self.assertEqual(layout.to_remote(layout.samples_path), layout.remote_root / RunLayout.SAMPLES)
        self.assertEqual(
            layout.to_remote(layout.logs_dir / "container.log"),
            layout.remote_root / "logs" / "container.log",
        )

    def test_remote_unset_is_none_and_fails_closed(self):
        """Flow 5: no remote dir -> remote_root None, to_remote raises."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "h123", "run-1")
        self.assertIsNone(layout.remote_root)
        with self.assertRaisesRegex(ValueError, "no remote_artifact_dir"):
            layout.to_remote(layout.samples_path)

    def test_to_remote_rejects_path_not_under_root(self):
        """Flow 5: a path outside root raises a *distinct* error from the no-remote case."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "h123", "run-1", remote_artifact_dir="/remote")
        with self.assertRaisesRegex(ValueError, "not under RunLayout.root"):
            layout.to_remote(Path("/etc/passwd"))

    def test_rejects_escaping_segment(self):
        """A path-bearing or absolute component must not silently escape the root."""
        tmp = Path(tempfile.mkdtemp())
        for bad in ("/etc/cron.d", "..", "a/b", ""):
            with self.assertRaisesRegex(ValueError, "single path segment"):
                RunLayout(tmp, "suite", bad, "h123", "run-1")


if __name__ == "__main__":
    unittest.main()
