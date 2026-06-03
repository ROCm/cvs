"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/cli_plugins/export_plugin.py: the `cvs export` CLI shell
# over G3's `export_runs` fact-table engine.
#
# Pinned invariants (from g3.1-export-cli.md "Bake-in"):
#   - Output Parquet column SET equals FACT_COLUMNS (set equality, not subset),
#     so dashboards/notebooks have a stable schema even on empty/all-skipped
#     trees.
#   - Empty input directory yields a zero-row Parquet with the FACT_COLUMNS
#     schema; never crashes, never writes a column-less file.
#   - Corrupt / malformed manifest.json logs a warning but does NOT abort the
#     export (G3 visible-skip discipline).
#
# Surface (brief):
#   cvs export --artifact-dir <dir> [--since <duration>] -o <out.parquet>
#
# Tests are self-contained: manifests are constructed via the public
# cvs.lib.manifest schema models, not pulled from the spike oracle.

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.parquet as pq

from cvs.lib.manifest import (
    ConfigInputs,
    HostFingerprint,
    Identity,
    Manifest,
    PhaseTiming,
    RunLayout,
    SystemFingerprint,
    Verdicts,
)
from cvs.lib.manifest.export import FACT_COLUMNS


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _manifest(run_id: str = "run-1", started_at: str | None = None, **scalars) -> Manifest:
    """Self-contained manifest builder: every required submodel populated, no oracle import."""
    return Manifest(
        identity=Identity(
            run_id=run_id,
            test_id="suite",
            cell_id="cell-a",
            config_hash="ch",
            workload_hash="wh",
            verification_hash="vh",
            cvs_git_sha="abc123",
            started_at=started_at or _iso(datetime(2025, 1, 1, tzinfo=timezone.utc)),
            finished_at=_iso(datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc)),
            invoker="tester",
        ),
        system=SystemFingerprint(
            hosts=[HostFingerprint(hostname="n0", gpus=["mi300x"] * 8, nics=["mlx5_0"])],
            topology_hash="th",
        ),
        config=ConfigInputs(
            resolved_config_path="/run/config.resolved.yaml",
            model="llama-3.1-70b",
            commands=["docker run ..."],
            seed=7,
        ),
        phases=[PhaseTiming(phase="prepare", duration_s=1.0, status="complete")],
        verdicts=Verdicts(
            overall_status="complete",
            scalars={"total_throughput": 1500.0, **{k: float(v) for k, v in scalars.items()}},
        ),
    )


def _write_manifest(root: Path, run_id: str, **kwargs) -> Path:
    layout = RunLayout(root, "suite", "cell-a", "h" + run_id, run_id).ensure()
    _manifest(run_id=run_id, **kwargs).write(layout.manifest_path)
    return layout.manifest_path


class TestExportPluginSurface(unittest.TestCase):
    """The plugin must register as a `cvs export` subcommand with the brief's flags."""

    def test_plugin_class_importable(self):
        """ExportPlugin class must exist at cvs.cli_plugins.export_plugin.ExportPlugin."""
        from cvs.cli_plugins.export_plugin import ExportPlugin  # noqa: F401

    def test_get_name_is_export(self):
        from cvs.cli_plugins.export_plugin import ExportPlugin

        self.assertEqual(ExportPlugin().get_name(), "export")

    def test_subclasses_subcommand_plugin(self):
        from cvs.cli_plugins.base import SubcommandPlugin
        from cvs.cli_plugins.export_plugin import ExportPlugin

        self.assertTrue(issubclass(ExportPlugin, SubcommandPlugin))

    def test_parser_accepts_brief_flags(self):
        """--artifact-dir and -o must parse; --since must be optional."""
        from cvs.cli_plugins.export_plugin import ExportPlugin

        plugin = ExportPlugin()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        plugin.get_parser(subparsers)
        ns = parser.parse_args(["export", "--artifact-dir", "/tmp/r", "-o", "/tmp/out.parquet"])
        self.assertEqual(ns.command, "export")
        # The plugin must be dispatchable via the `_plugin` attribute the main CLI uses.
        self.assertTrue(hasattr(ns, "_plugin"))

    def test_parser_accepts_since(self):
        from cvs.cli_plugins.export_plugin import ExportPlugin

        plugin = ExportPlugin()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        plugin.get_parser(subparsers)
        # Must not raise.
        parser.parse_args(["export", "--artifact-dir", "/tmp/r", "--since", "1h", "-o", "/tmp/out.parquet"])

    def test_missing_required_flag_exits_nonzero(self):
        """Adversarial: omitting -o (required) must argparse-exit, not silently succeed."""
        from cvs.cli_plugins.export_plugin import ExportPlugin

        plugin = ExportPlugin()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        plugin.get_parser(subparsers)
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["export", "--artifact-dir", "/tmp/r"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_missing_artifact_dir_exits_nonzero(self):
        from cvs.cli_plugins.export_plugin import ExportPlugin

        plugin = ExportPlugin()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        plugin.get_parser(subparsers)
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["export", "-o", "/tmp/out.parquet"])
        self.assertNotEqual(cm.exception.code, 0)


class TestExportPluginRun(unittest.TestCase):
    """The `run` method must drive G3's export_runs and honor the bake-in invariants."""

    def _run_plugin(self, artifact_dir: Path, out_path: Path, since: str | None = None):
        from cvs.cli_plugins.export_plugin import ExportPlugin

        plugin = ExportPlugin()
        args = MagicMock(spec=["artifact_dir", "out", "since", "o"])
        args.artifact_dir = str(artifact_dir)
        args.out = str(out_path)
        args.o = str(out_path)
        args.since = since
        plugin.run(args)

    def test_end_to_end_fact_table_columns_exact(self):
        """Bake-in #1: column SET equals FACT_COLUMNS (equality, not subset)."""
        tmp = Path(tempfile.mkdtemp())
        for rid in ("run-1", "run-2", "run-3"):
            _write_manifest(tmp, rid)
        out = tmp / "fact.parquet"
        self._run_plugin(tmp, out)
        self.assertTrue(out.exists(), "plugin must write the Parquet at -o path")
        schema_names = set(pq.read_schema(out).names)
        # Equality on the static FACT_COLUMNS set; scalar_* columns are additive
        # and not part of the fixed schema contract.
        static = schema_names - {n for n in schema_names if n.startswith("scalar_")}
        self.assertEqual(
            static,
            set(FACT_COLUMNS),
            f"static columns must equal FACT_COLUMNS, diff={sorted(static ^ set(FACT_COLUMNS))}",
        )

    def test_empty_dir_yields_fact_columns_schema(self):
        """Bake-in #2: empty input -> zero-row Parquet with the FACT_COLUMNS schema."""
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "fact.parquet"
        self._run_plugin(tmp, out)
        self.assertTrue(out.exists(), "empty input must still write a Parquet, not crash")
        table = pq.read_table(out)
        self.assertEqual(table.num_rows, 0, "empty input must yield zero rows")
        self.assertEqual(
            set(table.schema.names),
            set(FACT_COLUMNS),
            "empty input must yield exactly FACT_COLUMNS columns -- never a column-less Parquet",
        )

    def test_corrupt_manifest_does_not_abort_export(self):
        """Bake-in #3: malformed manifest.json logs a warning but does NOT abort."""
        tmp = Path(tempfile.mkdtemp())
        _write_manifest(tmp, "good-1")
        # Plant a corrupt manifest.json next to a real run dir.
        bad_layout = RunLayout(tmp, "suite", "cell-b", "hbad", "bad-1").ensure()
        bad_layout.manifest_path.write_text("{ this is not valid json")
        out = tmp / "fact.parquet"
        # The export must complete and skip the bad one, not raise.
        with self.assertLogs(level=logging.WARNING):
            self._run_plugin(tmp, out)
        self.assertTrue(out.exists())
        table = pq.read_table(out)
        self.assertEqual(table.num_rows, 1, "corrupt manifest must be skipped, good ones kept")
        self.assertEqual(list(table.column("run_id").to_pylist()), ["good-1"])

    def test_malformed_yaml_in_manifest_skipped(self):
        """Adversarial: a JSON-shaped but schema-invalid manifest (extra forbidden field) is skipped."""
        tmp = Path(tempfile.mkdtemp())
        good_path = _write_manifest(tmp, "good-1")
        bad_layout = RunLayout(tmp, "suite", "cell-c", "hbad2", "bad-2").ensure()
        payload = json.loads(good_path.read_text())
        payload["unknown_future_field"] = "rejected_by_extra_forbid"
        bad_layout.manifest_path.write_text(json.dumps(payload))
        out = tmp / "fact.parquet"
        with self.assertLogs(level=logging.WARNING):
            self._run_plugin(tmp, out)
        table = pq.read_table(out)
        self.assertEqual(list(table.column("run_id").to_pylist()), ["good-1"])

    def test_since_filters_old_runs(self):
        """--since <duration> excludes runs whose started_at is older than the cutoff."""
        tmp = Path(tempfile.mkdtemp())
        now = datetime.now(timezone.utc)
        # One ancient run, one fresh run.
        _write_manifest(tmp, "old", started_at=_iso(now - timedelta(days=30)))
        _write_manifest(tmp, "new", started_at=_iso(now - timedelta(minutes=5)))
        out = tmp / "fact.parquet"
        self._run_plugin(tmp, out, since="1h")
        table = pq.read_table(out)
        run_ids = set(table.column("run_id").to_pylist())
        self.assertEqual(run_ids, {"new"}, "--since 1h must drop the 30-day-old run")

    def test_since_none_includes_all_runs(self):
        """Omitting --since must not filter anything."""
        tmp = Path(tempfile.mkdtemp())
        now = datetime.now(timezone.utc)
        _write_manifest(tmp, "old", started_at=_iso(now - timedelta(days=30)))
        _write_manifest(tmp, "new", started_at=_iso(now - timedelta(minutes=5)))
        out = tmp / "fact.parquet"
        self._run_plugin(tmp, out, since=None)
        table = pq.read_table(out)
        self.assertEqual(set(table.column("run_id").to_pylist()), {"old", "new"})

    def test_creates_parent_directory_of_output(self):
        """-o into a not-yet-existing directory must create it (engine contract surfaced via CLI)."""
        tmp = Path(tempfile.mkdtemp())
        _write_manifest(tmp, "run-1")
        out = tmp / "nested" / "dir" / "fact.parquet"
        self._run_plugin(tmp, out)
        self.assertTrue(out.exists())


class TestExportPluginRunHardening(unittest.TestCase):
    """Adversarial gap-closers added by test-adversary: scalar round-trip, row counts,
    --since unit handling, real-parser dispatch path, and failure_category null-passthrough.

    These tests build `args` via the plugin's *real* argparse parser rather than a
    MagicMock, so a future implementer who renames the argparse `dest` (e.g.
    --artifact-dir -> dest='artifacts') gets caught here instead of silently
    sliding past a permissive MagicMock spec.
    """

    def _build_args(self, artifact_dir: Path, out_path: Path, since: str | None = None):
        from cvs.cli_plugins.export_plugin import ExportPlugin

        plugin = ExportPlugin()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        plugin.get_parser(subparsers)
        argv = ["export", "--artifact-dir", str(artifact_dir), "-o", str(out_path)]
        if since is not None:
            argv += ["--since", since]
        return plugin, parser.parse_args(argv)

    def _run_via_real_parser(self, artifact_dir: Path, out_path: Path, since: str | None = None):
        plugin, ns = self._build_args(artifact_dir, out_path, since)
        plugin.run(ns)

    def test_scalar_value_round_trips_to_scalar_column(self):
        """Sabotage guard: a manifest scalar must land in `scalar_<name>` with the right value.

        Catches a mutant that drops the scalar-flatten loop or relabels columns.
        """
        tmp = Path(tempfile.mkdtemp())
        _write_manifest(tmp, "run-1")  # _manifest seeds scalars={"total_throughput": 1500.0}
        out = tmp / "fact.parquet"
        self._run_via_real_parser(tmp, out)
        table = pq.read_table(out)
        self.assertIn(
            "scalar_total_throughput",
            table.schema.names,
            "manifest scalars must round-trip to scalar_<name> columns",
        )
        values = table.column("scalar_total_throughput").to_pylist()
        self.assertEqual(values, [1500.0], "scalar value must round-trip unchanged")

    def test_multi_run_row_count_and_ids(self):
        """Catches a mutant that writes only the first manifest or de-duplicates rows."""
        tmp = Path(tempfile.mkdtemp())
        for rid in ("run-1", "run-2", "run-3"):
            _write_manifest(tmp, rid)
        out = tmp / "fact.parquet"
        self._run_via_real_parser(tmp, out)
        table = pq.read_table(out)
        self.assertEqual(table.num_rows, 3, "must write one row per manifest")
        self.assertEqual(
            set(table.column("run_id").to_pylist()),
            {"run-1", "run-2", "run-3"},
            "all run_ids must appear (no truncation, no dedup)",
        )

    def test_failure_category_column_present_and_null_when_unset(self):
        """`failure_category` is Optional[str]=None on Verdicts. Schema must carry it
        regardless and the row value must be null, not the literal string 'None'."""
        tmp = Path(tempfile.mkdtemp())
        _write_manifest(tmp, "run-1")
        out = tmp / "fact.parquet"
        self._run_via_real_parser(tmp, out)
        table = pq.read_table(out)
        self.assertIn("failure_category", table.schema.names)
        vals = table.column("failure_category").to_pylist()
        self.assertEqual(
            vals,
            [None],
            "unset failure_category must serialize as null, not the string 'None' or empty string",
        )

    def test_since_rejects_garbage_value(self):
        """`--since banana` is uninterpretable; the plugin must surface an error,
        not silently treat it as 'include everything' (which would mask config bugs).

        Accepts either argparse rejection (preferred) or a run-time exception."""
        tmp = Path(tempfile.mkdtemp())
        _write_manifest(tmp, "run-1")
        out = tmp / "fact.parquet"
        plugin, ns_or_exit = None, None
        try:
            plugin, ns_or_exit = self._build_args(tmp, out, since="banana")
        except SystemExit as e:
            self.assertNotEqual(e.code, 0)
            return
        # Parser accepted; the run() call must reject.
        with self.assertRaises((ValueError, SystemExit, Exception)):
            plugin.run(ns_or_exit)

    def test_since_honors_day_unit(self):
        """`--since 7d` must keep a 5-day-old run and drop a 30-day-old run.

        Catches a mutant that hardcodes hours regardless of the unit suffix."""
        tmp = Path(tempfile.mkdtemp())
        now = datetime.now(timezone.utc)
        _write_manifest(tmp, "ancient", started_at=_iso(now - timedelta(days=30)))
        _write_manifest(tmp, "recent", started_at=_iso(now - timedelta(days=5)))
        out = tmp / "fact.parquet"
        self._run_via_real_parser(tmp, out, since="7d")
        table = pq.read_table(out)
        self.assertEqual(
            set(table.column("run_id").to_pylist()),
            {"recent"},
            "--since 7d must keep the 5-day-old run and drop the 30-day-old run",
        )

    def test_dest_name_artifact_dir_present_on_namespace(self):
        """The argparse `dest` for --artifact-dir must be `artifact_dir` (the
        standard hyphen->underscore translation). Pinning this prevents an
        impl that renames the dest from passing the MagicMock-based tests
        above while breaking the real CLI dispatch path."""
        from cvs.cli_plugins.export_plugin import ExportPlugin

        plugin = ExportPlugin()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        plugin.get_parser(subparsers)
        ns = parser.parse_args(["export", "--artifact-dir", "/tmp/r", "-o", "/tmp/out.parquet"])
        self.assertTrue(hasattr(ns, "artifact_dir"), "argparse dest for --artifact-dir must be 'artifact_dir'")
        self.assertEqual(ns.artifact_dir, "/tmp/r")
        # `-o` may use either dest='out' or dest='o'; require one of them is set
        # to the right value so the impl can't ship a third spelling.
        out_attrs = [getattr(ns, name, None) for name in ("out", "o", "output")]
        self.assertIn("/tmp/out.parquet", out_attrs, "-o must expose its value as out/o/output on the namespace")

    def test_corrupt_manifest_does_not_drop_good_run_id(self):
        """Strengthens the existing corrupt-manifest test: assert the *bad* run_id
        is NOT silently leaking into the fact table (e.g. as a row of nulls)."""
        tmp = Path(tempfile.mkdtemp())
        _write_manifest(tmp, "good-1")
        bad_layout = RunLayout(tmp, "suite", "cell-x", "hbadx", "should-not-appear").ensure()
        bad_layout.manifest_path.write_text("{not json")
        out = tmp / "fact.parquet"
        with self.assertLogs(level=logging.WARNING):
            self._run_via_real_parser(tmp, out)
        table = pq.read_table(out)
        run_ids = table.column("run_id").to_pylist()
        self.assertNotIn(
            "should-not-appear",
            run_ids,
            "a skipped (corrupt) manifest must NOT leak its run_id into the fact table",
        )
        # And no all-null row was emitted in its place.
        self.assertNotIn(None, run_ids, "no null-row placeholder for skipped manifests")


class TestExportPluginAdversarialGapClosers(unittest.TestCase):
    """sabotage-gate additions: exit-code 0 on success, all-skipped tree,
    warning message specificity. These close mutants the existing tests would
    miss."""

    def _build_args_and_plugin(self, artifact_dir: Path, out_path: Path, since: str | None = None):
        from cvs.cli_plugins.export_plugin import ExportPlugin

        plugin = ExportPlugin()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        plugin.get_parser(subparsers)
        argv = ["export", "--artifact-dir", str(artifact_dir), "-o", str(out_path)]
        if since is not None:
            argv += ["--since", since]
        return plugin, parser.parse_args(argv)

    def test_run_returns_without_raising_on_success(self):
        """Sabotage guard: a mutant that sys.exit(1) after writing the parquet
        must be caught. plugin.run() on a happy-path tree must not raise."""
        tmp = Path(tempfile.mkdtemp())
        _write_manifest(tmp, "run-1")
        out = tmp / "fact.parquet"
        plugin, ns = self._build_args_and_plugin(tmp, out)
        try:
            rc = plugin.run(ns)
        except SystemExit as e:
            # Allowed only if exit code is 0 (CLI shells sometimes sys.exit(0)).
            self.assertEqual(e.code, 0, "plugin.run on success must not exit non-zero")
            rc = 0
        # If a return code is returned (modern plugin convention), it must be 0/None.
        self.assertIn(rc, (None, 0), "plugin.run on success must return None or 0")

    def test_all_skipped_tree_yields_empty_fact_columns_parquet(self):
        """Brief explicitly says 'empty/all-skipped' both yield FACT_COLUMNS schema.
        A tree of *only* corrupt manifests must still produce a zero-row, schema-bearing
        Parquet -- never a column-less file, never an abort."""
        tmp = Path(tempfile.mkdtemp())
        for i, rid in enumerate(("bad-1", "bad-2", "bad-3")):
            bad_layout = RunLayout(tmp, "suite", f"cell-{i}", f"h{rid}", rid).ensure()
            bad_layout.manifest_path.write_text("{not valid json at all")
        out = tmp / "fact.parquet"
        plugin, ns = self._build_args_and_plugin(tmp, out)
        with self.assertLogs(level=logging.WARNING):
            plugin.run(ns)
        self.assertTrue(out.exists(), "all-skipped tree must still write a Parquet")
        table = pq.read_table(out)
        self.assertEqual(table.num_rows, 0, "all-skipped tree must yield zero rows")
        self.assertEqual(
            set(table.schema.names),
            set(FACT_COLUMNS),
            "all-skipped tree must yield exactly FACT_COLUMNS columns",
        )

    def test_corrupt_manifest_warning_mentions_path(self):
        """Sabotage guard: a mutant that downgrades real errors to WARNING-level
        no-ops (e.g. logger.warning('ok')) would slip past assertLogs alone. Pin
        that the warning record actually mentions the offending manifest path."""
        tmp = Path(tempfile.mkdtemp())
        _write_manifest(tmp, "good-1")
        bad_layout = RunLayout(tmp, "suite", "cell-bad", "hbadW", "bad-W").ensure()
        bad_layout.manifest_path.write_text("not json")
        out = tmp / "fact.parquet"
        plugin, ns = self._build_args_and_plugin(tmp, out)
        with self.assertLogs(level=logging.WARNING) as captured:
            plugin.run(ns)
        joined = chr(10).join(captured.output)
        self.assertIn(
            str(bad_layout.manifest_path),
            joined,
            "WARNING for corrupt manifest must include the path of the offending file",
        )

    def test_since_boundary_keeps_run_inside_window(self):
        """A run started 30 minutes ago must be kept under --since 1h.
        Catches a mutant that flips the inequality (< vs >) on the cutoff."""
        tmp = Path(tempfile.mkdtemp())
        now = datetime.now(timezone.utc)
        _write_manifest(tmp, "inside", started_at=_iso(now - timedelta(minutes=30)))
        out = tmp / "fact.parquet"
        plugin, ns = self._build_args_and_plugin(tmp, out, since="1h")
        plugin.run(ns)
        table = pq.read_table(out)
        self.assertEqual(
            set(table.column("run_id").to_pylist()),
            {"inside"},
            "a 30-min-old run must be inside the 1h window (not dropped by an inverted comparison)",
        )


class TestExportPluginDiscoverable(unittest.TestCase):
    """The plugin must be discoverable by cvs.main.discover_plugins."""

    def test_discoverable_by_main(self):
        from cvs.main import discover_plugins

        names = {p.get_name() for p in discover_plugins()}
        self.assertIn("export", names, "export plugin must be auto-discovered by cvs.main")


if __name__ == "__main__":
    unittest.main()
