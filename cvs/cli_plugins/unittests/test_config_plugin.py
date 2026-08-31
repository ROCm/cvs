import argparse
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import cvs.main as main
from cvs.cli_plugins.config_plugin import ConfigPlugin


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
                        output="/tmp/out",
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
        test_config_path = "training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_distributed.json"
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


if __name__ == "__main__":
    unittest.main()
