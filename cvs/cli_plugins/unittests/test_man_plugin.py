import argparse
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout

from cvs.cli_plugins.man_plugin import ManPlugin
from cvs.parsers.config_registry import resolve_sample_path


def make_args(test=None, parameter=None, as_json=False, extra_pytest_args=None):
    args = argparse.Namespace()
    args.test = test
    args.parameter = parameter
    args.as_json = as_json
    args.extra_pytest_args = extra_pytest_args if extra_pytest_args is not None else []
    return args


def run_plugin(plugin, args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        plugin.run(args)
    return buf.getvalue()


class TestManPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = ManPlugin()

    def test_get_name(self):
        self.assertEqual("man", self.plugin.get_name())

    def test_registers_itself_for_dispatch(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        self.plugin.get_parser(subparsers)

        parsed = parser.parse_args(["man", "rccl_perf", "nic_model", "--json"])
        self.assertIs(self.plugin, parsed._plugin)
        self.assertEqual("rccl_perf", parsed.test)
        self.assertEqual("nic_model", parsed.parameter)
        self.assertTrue(parsed.as_json)

    def test_test_argument_is_optional(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        self.plugin.get_parser(subparsers)

        parsed = parser.parse_args(["man"])
        self.assertIsNone(parsed.test)

    def test_lists_documented_tests(self):
        output = run_plugin(self.plugin, make_args())
        self.assertIn("rccl_perf", output)
        self.assertIn("preflight_checks", output)
        self.assertIn("Total:", output)

    def test_explains_a_test(self):
        output = run_plugin(self.plugin, make_args(test="rccl_perf"))
        self.assertIn("rccl.mpi_params", output)
        self.assertIn("no_of_nodes", output)

        sample_path = resolve_sample_path("input/config_file/rccl/rccl_config.json")
        self.assertIn(sample_path, output)
        self.assertTrue(os.path.isfile(sample_path), f"printed sample path {sample_path} does not exist")

    def test_documents_every_registered_section(self):
        output = run_plugin(self.plugin, make_args(test="megatron_llama3_1_8b_single"))
        self.assertIn("config", output)
        self.assertIn("model_params", output)

    def test_explains_a_single_parameter(self):
        output = run_plugin(self.plugin, make_args(test="rccl_perf", parameter="nic_model"))
        self.assertIn("nic_model", output)
        self.assertNotIn("no_of_local_ranks", output)

    def test_documents_the_code_default_not_the_sample_value(self):
        # rccl_config.json ships nic_model "thor"; rccl_lib.py defaults to "ainic".
        output = run_plugin(self.plugin, make_args(test="rccl_perf", parameter="nic_model"))
        self.assertIn("ainic", output)

    def test_json_output_is_parseable(self):
        output = run_plugin(self.plugin, make_args(test="rccl_perf", parameter="nic_model", as_json=True))
        payload = json.loads(output)
        self.assertEqual("rccl_perf", payload["test"])
        self.assertEqual(1, len(payload["parameters"]))
        self.assertEqual("rccl.cvs_params.nic_model", payload["parameters"][0]["path"])

    def test_json_config_files_are_real_paths(self):
        output = run_plugin(self.plugin, make_args(test="rccl_perf", as_json=True))
        payload = json.loads(output)
        self.assertEqual([resolve_sample_path("input/config_file/rccl/rccl_config.json")], payload["config_files"])
        for config_file in payload["config_files"]:
            self.assertTrue(os.path.isfile(config_file), f"{config_file} does not exist")

    def test_unknown_test_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            run_plugin(self.plugin, make_args(test="no_such_test_anywhere"))
        self.assertEqual(1, ctx.exception.code)

    def test_undocumented_but_real_test_is_distinguished(self):
        buf = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(buf):
            self.plugin.run(make_args(test="ib_perf_bw_test"))
        self.assertIn("not documented yet", buf.getvalue())

    def test_unknown_parameter_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            run_plugin(self.plugin, make_args(test="rccl_perf", parameter="no_such_parameter"))
        self.assertEqual(1, ctx.exception.code)

    def test_unknown_test_error_goes_to_stderr_in_json_mode(self):
        # In --json mode, error text must not land on stdout, or piping the
        # output to a JSON parser would fail on the human-readable message.
        out, err = io.StringIO(), io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(out), redirect_stderr(err):
            self.plugin.run(make_args(test="no_such_test_anywhere", as_json=True))
        self.assertEqual("", out.getvalue())
        self.assertIn("no config parameter reference", err.getvalue())

    def test_unknown_parameter_error_goes_to_stderr_in_json_mode(self):
        out, err = io.StringIO(), io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(out), redirect_stderr(err):
            self.plugin.run(make_args(test="rccl_perf", parameter="no_such_parameter", as_json=True))
        self.assertEqual("", out.getvalue())
        self.assertIn("no parameter matching", err.getvalue())

    def test_unrecognized_flag_is_rejected(self):
        # main() parses with parse_known_args, so stray flags arrive here
        # silently rather than being caught by argparse.
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, redirect_stdout(buf):
            self.plugin.run(make_args(test="rccl_perf", extra_pytest_args=["--bogus"]))
        self.assertEqual(1, ctx.exception.code)
        self.assertIn("unrecognized arguments", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
