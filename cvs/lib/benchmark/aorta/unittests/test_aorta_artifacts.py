"""Real filesystem tests for container-side packing and local extraction."""

import contextlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from cvs.lib.benchmark.aorta.aorta_artifacts import extract_artifacts, main, pack_reports, pack_traces


class TestAortaArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        self.archive = Path(self.tmp.name) / "traces.tar.gz"

    def write_trace(self, path, mtime):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"traceEvents": []}')
        os.utime(target, (mtime, mtime))
        return target

    def test_per_file_freshness_excludes_old_and_collected_traces(self):
        self.write_trace("output/torch_profiler/rank0/fresh.json", 2000)
        self.write_trace("output/torch_profiler/rank0/stale.json", 1000)
        self.write_trace("combined_traces/node_9/torch_profiler/old.json", 3000)
        self.write_trace(".cvs-aorta/old/torch_profiler/old.json", 3000)
        trees = pack_traces(self.root, self.archive, 1500)
        self.assertEqual(trees, ["output/torch_profiler"])
        dest = Path(self.tmp.name) / "collected"
        extract_artifacts(self.archive, dest)
        self.assertTrue((dest / "output/torch_profiler/rank0/fresh.json").exists())
        self.assertEqual(len(list(dest.rglob("*.json"))), 1)

    def test_latest_tree_for_single_node(self):
        self.write_trace("old/torch_profiler/rank0.json", 2000)
        self.write_trace("new with spaces/torch_profiler/rank0.json", 3000)
        self.assertEqual(
            pack_traces(self.root, self.archive, 1500, all_traces=False), ["new with spaces/torch_profiler"]
        )

    def test_empty_trace_archive(self):
        self.assertEqual(pack_traces(self.root, self.archive, 0), [])
        with tarfile.open(self.archive) as bundle:
            self.assertEqual(bundle.getnames(), [])

    def test_reports_and_trace_symlinks_are_excluded(self):
        source = self.write_trace("torch_profiler/rank0.json", 2000)
        (source.parent / "link.json").symlink_to(source)
        pack_reports(self.root, self.archive, 0)
        with tarfile.open(self.archive) as bundle:
            self.assertEqual(bundle.getnames(), ["torch_profiler/rank0.json"])
        pack_traces(self.root, self.archive, 0)
        with tarfile.open(self.archive) as bundle:
            self.assertEqual(bundle.getnames(), ["torch_profiler/rank0.json"])

    def test_reports_respect_min_mtime_filter(self):
        self.write_trace("old_report.json", 1000)
        self.write_trace("new_report.json", 2000)
        pack_reports(self.root, self.archive, 1500)
        with tarfile.open(self.archive) as bundle:
            self.assertEqual(bundle.getnames(), ["new_report.json"])

    def test_archive_escape_and_links_rejected(self):
        for name, kind in (("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
            with self.subTest(name=name):
                with tarfile.open(self.archive, "w:gz") as bundle:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "/tmp/escape"
                    bundle.addfile(member, io.BytesIO())
                with self.assertRaises(ValueError):
                    extract_artifacts(self.archive, Path(self.tmp.name) / "dest")

    def test_existing_destination_symlink_cannot_escape(self):
        self.write_trace("torch_profiler/rank0.json", 2000)
        pack_traces(self.root, self.archive, 0)
        dest = Path(self.tmp.name) / "dest"
        outside = Path(self.tmp.name) / "outside"
        dest.mkdir()
        outside.mkdir()
        (dest / "torch_profiler").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            extract_artifacts(self.archive, dest)

    def test_remote_entry_point(self):
        self.write_trace("output/torch_profiler/rank0.json", 2000)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["traces", str(self.root), str(self.archive), "--min-mtime", "1500", "--latest"])
        self.assertEqual(json.loads(output.getvalue()), ["output/torch_profiler"])
        main(["reports", str(self.root), str(self.archive)])
        self.assertTrue(self.archive.is_file())
