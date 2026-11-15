import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import cvs.main as main
from cvs.cli_plugins.config_plugin import ConfigPlugin
from cvs.config_validation.migrate import maps
from cvs.unittests.support import TMP

DEMO_MAP = '''
from pydantic import create_model

from cvs.config_validation.schemas.base import SuiteConfig

SCHEMA = create_model("DemoConfig", __base__=SuiteConfig, tp=(int, ...), port=(int, 8000))
CONFIG = {"tensor_parallel_size": "tp", "backend": None}
THRESHOLDS = {}
'''
BROKEN_MAP = "raise RuntimeError('broken map')\n"
OLD = {"_comment": "Demo config.", "backend": "demo", "tensor_parallel_size": 8, "port": 8010}
NEW = {"_comment": "Demo config.", "tp": 8, "port": 8010}


class TestConfigPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cvs

        cvs_dir = os.path.dirname(cvs.__file__)
        cls.expected_roots = [
            os.path.join(cvs_dir, "input", "config_file"),
            os.path.join(cvs_dir, "input", "cluster_file"),
            os.path.join(cvs_dir, "input", "env_file"),
        ]

    def setUp(self):
        self.plugin = ConfigPlugin()
        subparsers = argparse.ArgumentParser().add_subparsers()
        self.plugin.get_parser(subparsers)

    def test_bare_config_shows_catalog(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.plugin.run(argparse.Namespace(config_command=None))
        output = captured.getvalue()
        self.assertIn("Available config commands:", output)
        self.assertIn("list - List config files grouped by directory", output)
        self.assertIn("list-dirs - List config directories grouped by category", output)
        self.assertIn("copy - Copy bundled config file(s) to --output", output)
        self.assertIn("migrate - Convert an old suite config or threshold file to the new format", output)

    def test_bare_config_copy_requires_output(self):
        parser = main.build_arg_parser(main.discover_plugins())
        with self.assertRaises(SystemExit) as exc:
            with redirect_stdout(io.StringIO()):
                parser.parse_args(["config", "copy"])
        self.assertEqual(exc.exception.code, 2)

    def test_config_copy_missing_path_shows_help(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            with self.assertRaises(SystemExit) as exc:
                self.plugin.run(
                    argparse.Namespace(
                        config_command="copy",
                        path=None,
                        output=f"{TMP}/out",
                        all=False,
                        force=False,
                    )
                )
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("usage: cvs config copy", captured.getvalue())
        self.assertIn("--output OUTPUT", captured.getvalue())

    def test_list_dirs_unscoped(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.plugin.run(argparse.Namespace(config_command="list-dirs", path=""))
        output = captured.getvalue()
        self.assertIn("config_file_dirs:", output)
        self.assertNotIn("config_file:\n", output)
        self.assertIn("training/jaxmaxtext/", output)
        self.assertIn("inference/atom/", output)
        self.assertIn("cluster_file_dirs:", output)
        self.assertIn("cluster.json", output)

    def test_list_dirs_training_scoped(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.plugin.run(argparse.Namespace(config_command="list-dirs", path="training"))
        output = captured.getvalue()
        self.assertIn("training/jaxmaxtext/", output)
        self.assertIn("training/torchtitan/", output)
        self.assertNotIn("inference/atom/", output)
        self.assertNotIn("config_file_dirs:", output)

    def test_list_dirs_rccl_shows_both_roots(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.plugin.run(argparse.Namespace(config_command="list-dirs", path="rccl"))
        output = captured.getvalue()
        self.assertIn("config_file_dirs:", output)
        self.assertIn("env_file_dirs:", output)
        self.assertEqual(output.count("rccl/"), 2)

    def test_list_torchtitan_scoped(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.plugin.run(argparse.Namespace(config_command="list", path="training/torchtitan"))
        output = captured.getvalue()
        self.assertIn("training/torchtitan:", output)
        self.assertIn("training/torchtitan/mi355x_torchtitan_llama-3.1-8b_single.json", output)
        self.assertNotIn("inference/atom/", output)
        self.assertNotIn("config_file:", output)

    def test_copy_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "host_config.json")
            captured = io.StringIO()
            with redirect_stdout(captured):
                self.plugin.run(
                    argparse.Namespace(
                        config_command="copy",
                        path="platform/host_config.json",
                        output=dest,
                        all=False,
                        force=False,
                    )
                )
            self.assertTrue(os.path.exists(dest))
            self.assertIn("Copied", captured.getvalue())

    def test_copy_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            captured = io.StringIO()
            with redirect_stdout(captured):
                self.plugin.run(
                    argparse.Namespace(
                        config_command="copy",
                        path=None,
                        output=tmpdir,
                        all=True,
                        force=False,
                    )
                )
            for expected_root in self.expected_roots:
                label = os.path.basename(expected_root)
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, label)))
            self.assertIn("Copied", captured.getvalue())

    def test_copy_all_without_output_errors(self):
        parser = main.build_arg_parser(main.discover_plugins())
        with self.assertRaises(SystemExit) as exc:
            with redirect_stdout(io.StringIO()):
                parser.parse_args(["config", "copy", "--all"])
        self.assertEqual(exc.exception.code, 2)

    def test_copy_overwrite_requires_force(self):
        test_config_path = "training/jaxmaxtext/mi3xx_jaxmaxtext_llama-3.3-70b_distributed.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "cfg.json")
            self.plugin.run(
                argparse.Namespace(
                    config_command="copy",
                    path=test_config_path,
                    output=dest,
                    all=False,
                    force=False,
                )
            )
            with open(dest, "w", encoding="utf-8") as handle:
                handle.write('{"modified": true}')

            captured = io.StringIO()
            with redirect_stdout(captured):
                self.plugin.run(
                    argparse.Namespace(
                        config_command="copy",
                        path=test_config_path,
                        output=dest,
                        all=False,
                        force=False,
                    )
                )
            self.assertIn("already exists", captured.getvalue())

            with redirect_stdout(io.StringIO()):
                self.plugin.run(
                    argparse.Namespace(
                        config_command="copy",
                        path=test_config_path,
                        output=dest,
                        all=False,
                        force=True,
                    )
                )
            with open(dest, encoding="utf-8") as handle:
                content = handle.read()
            self.assertNotIn('"modified": true', content)


class TestConfigMigrate(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self._use_maps(demo=DEMO_MAP)
        self.old = self.tmp / "old.json"
        self.old.write_text(json.dumps(OLD))
        self.new = self.tmp / "new.json"

    def _use_maps(self, **sources):
        """Point map discovery at a new folder holding one map module per entry of ``sources``."""
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        for name, source in sources.items():
            (root / f"{name}.py").write_text(source)
            self.addCleanup(sys.modules.pop, f"{maps.__name__}.{name}", None)
        found = mock.patch.object(maps, "__path__", [str(root)])
        found.start()
        self.addCleanup(found.stop)

    def _cvs(self, *argv):
        """Run ``cvs <argv>`` through ``cvs.main``; return the exit code, stdout and stderr."""
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", ["cvs", *argv]), redirect_stdout(out), redirect_stderr(err):
            try:
                main.main([ConfigPlugin()])
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def _migrate(self, *extra):
        return self._cvs("config", "migrate", "demo", str(self.old), str(self.new), *extra)

    def test_writes_the_converted_config_to_output_and_exits_0(self):
        before = self.old.read_bytes()
        code, out, err = self._migrate()
        self.assertEqual((code, out, err), (0, f"wrote {self.new}\n", ""))
        self.assertEqual(json.loads(self.new.read_text()), NEW)
        self.assertEqual(self.old.read_bytes(), before)

    def test_exits_1_naming_the_unmapped_and_changeme_fields_it_wrote(self):
        self.old.write_text(json.dumps({"backend": "demo", "knob": 1}))
        code, out, _ = self._migrate()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(self.new.read_text()), {"_unmapped": {"knob": 1}, "tp": "<changeme>"})
        self.assertIn("_unmapped.knob:", out)
        self.assertIn("tp: required; written as <changeme>", out)

    def test_replaces_an_existing_output_only_with_force(self):
        self.new.write_text("{}")
        code, out, _ = self._migrate()
        self.assertEqual(code, 2)
        self.assertIn(f"{self.new} already exists; use --force to overwrite it", out)
        self.assertEqual(self.new.read_text(), "{}")

        code, _, _ = self._migrate("--force")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.new.read_text()), NEW)

    def test_stray_arguments_exit_2_naming_them(self):
        code, out, err = self._migrate("--froce", "extra.json")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("usage: cvs config migrate", err)
        self.assertIn("error: unrecognized arguments: --froce extra.json\n", err)
        self.assertFalse(self.new.exists())

    def test_a_map_that_fails_to_import_breaks_only_migrate_for_its_suite(self):
        self._use_maps(demo=DEMO_MAP, broken=BROKEN_MAP)
        code, out, _ = self._cvs("--help")
        self.assertEqual(code, 0)
        self.assertIn("cvs config migrate <suite> <input.json> <output>", out)
        code, out, _ = self._cvs("config", "list")
        self.assertEqual(code, 0)
        self.assertIn("cluster.json", out)
        code, out, _ = self._cvs("config", "migrate", "--help")
        self.assertEqual(code, 0)
        self.assertIn("(available: broken, demo)", " ".join(out.split()))

        with self.assertRaisesRegex(RuntimeError, "broken map"):
            self._cvs("config", "migrate", "broken", str(self.old), str(self.new))
        self.assertFalse(self.new.exists())
        self.assertEqual(self._migrate()[0], 0)
        self.assertEqual(json.loads(self.new.read_text()), NEW)

    def test_an_unknown_suite_exits_2_listing_the_suites_or_none(self):
        cases = {
            "demo": {"demo": DEMO_MAP},
            "demo, other": {"demo": "", "other": ""},
            "none": {},
        }
        for listed, sources in cases.items():
            with self.subTest(listed=listed):
                self._use_maps(**sources)
                code, out, err = self._cvs("config", "migrate", "vllm", str(self.old), str(self.new))
                self.assertEqual((code, out), (2, ""))
                self.assertIn("usage: cvs config migrate", err)
                self.assertIn(f"error: argument suite: unknown suite vllm; available suites: {listed}\n", err)
        self.assertFalse(self.new.exists())

    def test_requires_suite_input_and_output(self):
        code, _, err = self._cvs("config", "migrate", "demo", str(self.old))
        self.assertEqual(code, 2)
        self.assertIn("the following arguments are required: output", err)
        self.assertFalse(self.new.exists())

    def test_help_lists_the_suites_force_and_exit_codes(self):
        for listed, sources in {"demo": {"demo": DEMO_MAP}, "none yet": {}}.items():
            with self.subTest(listed=listed):
                self._use_maps(**sources)
                code, out, _ = self._cvs("config", "migrate", "--help")
                self.assertEqual(code, 0)
                text = " ".join(out.split())
                self.assertIn("usage: cvs config migrate [-h] [--force] suite input.json output", text)
                self.assertIn("The input file is never changed.", text)
                self.assertIn(f"(available: {listed})", text)
                self.assertIn("--force Force overwrite of an existing output file", text)
                self.assertIn("1 when the new file is written with fields to fix; 2 when nothing is written", text)

    def test_cvs_help_lists_the_migrate_command(self):
        code, out, _ = self._cvs("--help")
        self.assertEqual(code, 0)
        self.assertIn("cvs config migrate <suite> <input.json> <output>", out)


if __name__ == "__main__":
    unittest.main()
