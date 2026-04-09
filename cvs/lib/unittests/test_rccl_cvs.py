import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault(
    "pytest", types.SimpleNamespace(fail=lambda message: (_ for _ in ()).throw(AssertionError(message)))
)

from cvs.lib.rccl_cvs import (  # noqa: E402
    RcclConfig,
    _build_summary,
    _ensure_unique_case_id,
    _matching_rows,
    _no_matrix_case_id,
    _resolved_case_payload,
    _scan_rccl_stdout,
    _slug,
    build_collective_command,
    load_rccl_config,
    parse_and_validate_results,
)


class _DummySshHandle:
    def __init__(self):
        self.commands = []

    def exec(self, command, timeout=None):  # noqa: ARG002
        self.commands.append(command)
        return {"node0": ""}


def _minimal_rccl(
    *,
    num_ranks=16,
    ranks_per_node=8,
    env_script="/tmp/env.sh",
    profile="smoke",
    thresholds=None,
    thresholds_file=None,
    collectives=None,
):
    validation = {"profile": profile}
    if thresholds is not None:
        validation["thresholds"] = thresholds
    if thresholds_file is not None:
        validation["thresholds_file"] = thresholds_file
    run = {
        "env_script": env_script,
        "num_ranks": num_ranks,
        "ranks_per_node": ranks_per_node,
        "collectives": collectives or ["all_reduce_perf"],
        "datatype": "float",
        "start_size": "1024",
        "end_size": "16g",
        "step_factor": "2",
        "warmups": "10",
        "iterations": "20",
        "cycles": "1",
    }
    return {"rccl": {"run": run, "validation": validation, "artifacts": {"output_dir": "/tmp/rccl_cvs_out", "export_raw": False}}}


def _rccl_cfg(**kwargs):
    base = {
        "required_nodes": 1,
        "collectives": ["all_reduce_perf"],
        "datatype": "float",
        "num_ranks": 8,
        "ranks_per_node": 8,
        "env_script": "/tmp/env.sh",
        "artifacts_output_dir": "/tmp/out",
        "artifacts_export_raw": False,
        "config_echo": {},
        "start_size": "1024",
        "end_size": "16g",
        "step_factor": "2",
        "warmups": "10",
        "iterations": "20",
        "cycles": "1",
        "validation_profile": "smoke",
        "thresholds": {},
    }
    base.update(kwargs)
    return RcclConfig(**base)


class TestRcclCvs(unittest.TestCase):
    def test_load_rccl_config_nested_multi_node(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}, "node1": {}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(_minimal_rccl()))
            loaded = load_rccl_config(str(config_path), cluster_dict)

        self.assertEqual(loaded.required_nodes, 2)
        self.assertFalse(loaded.is_single_node)
        self.assertEqual(loaded.num_ranks, 16)
        self.assertEqual(loaded.ranks_per_node, 8)
        self.assertEqual(loaded.collectives, ["all_reduce_perf"])
        self.assertIn("run", loaded.config_echo)
        self.assertEqual(loaded.config_echo["artifacts"]["output_dir"], "/tmp/rccl_cvs_out")

    def test_load_rccl_config_single_node(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            loaded = load_rccl_config(str(config_path), cluster_dict)

        self.assertEqual(loaded.required_nodes, 1)
        self.assertTrue(loaded.is_single_node)
        self.assertEqual(loaded.env_script, "/tmp/env.sh")

    def test_load_rccl_config_rejects_missing_env_script(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
        body["rccl"]["run"].pop("env_script", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("env_script", str(ctx.exception).lower())

    def test_load_rccl_config_multi_node_requires_env_script_path(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}, "node1": {}}}
        body = _minimal_rccl(env_script="/site/env.sh", profile="smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            loaded = load_rccl_config(str(config_path), cluster_dict)
        self.assertFalse(loaded.is_single_node)
        self.assertEqual(loaded.env_script, "/site/env.sh")

    def test_load_rccl_config_rejects_flat_legacy_shape(self):
        config = {
            "rccl": {
                "mode": "multi_node",
                "rccl_tests_dir": "/opt/rccl-tests/build",
                "num_ranks": 16,
                "ranks_per_node": 8,
            }
        }
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_rccl_config(str(config_path), cluster_dict)

    def test_load_rccl_config_rejects_both_threshold_sources(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8,
            ranks_per_node=8,
            profile="thresholds",
            thresholds={"all_reduce_perf": {"bus_bw": {"1024": 1.0}}},
        )
        body["rccl"]["validation"]["thresholds_file"] = "/tmp/x.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("at most one", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_empty_inline_thresholds_object_for_strict(self):
        """{} is not a valid threshold source for thresholds/strict (schema + profile rules)."""
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8,
            ranks_per_node=8,
            profile="strict",
            thresholds={},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("non-empty", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_inline_thresholds_missing_bus_bw(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8,
            ranks_per_node=8,
            profile="thresholds",
            thresholds={"all_reduce_perf": {"not_bus_bw": {"1024": 1.0}}},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("extra", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_empty_bus_bw_map(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8,
            ranks_per_node=8,
            profile="thresholds",
            thresholds={"all_reduce_perf": {"bus_bw": {}}},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("bus_bw", str(ctx.exception).lower())

    def test_load_rccl_config_thresholds_file_same_shape_as_inline(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        payload = {"all_reduce_perf": {"bus_bw": {"1024": 1.0}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            thresh_path = Path(tmpdir) / "t.json"
            thresh_path.write_text(json.dumps(payload))
            body = _minimal_rccl(
                num_ranks=8,
                ranks_per_node=8,
                profile="thresholds",
                thresholds_file=str(thresh_path),
            )
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            loaded = load_rccl_config(str(config_path), cluster_dict)
        self.assertEqual(loaded.thresholds["all_reduce_perf"]["bus_bw"]["1024"], 1.0)

    def test_load_rccl_config_rejects_empty_thresholds_file_object(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            thresh_path = Path(tmpdir) / "t.json"
            thresh_path.write_text("{}")
            body = _minimal_rccl(
                num_ranks=8,
                ranks_per_node=8,
                profile="thresholds",
                thresholds_file=str(thresh_path),
            )
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("non-empty", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_unknown_rccl_key(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
        body["rccl"]["legacy_alias"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError):
                load_rccl_config(str(config_path), cluster_dict)

    def test_load_rccl_config_rejects_extra_top_level_keys(self):
        """Exactly one top-level key ``rccl``."""
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
        body["comment"] = "not allowed"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("exactly one top-level", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_extra_key_in_run(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
        body["rccl"]["run"]["legacy_mode"] = "single_node"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("extra", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_extra_key_in_validation(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
        body["rccl"]["validation"]["results_file"] = "/tmp/x.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("extra", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_extra_key_in_artifacts(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
        body["rccl"]["artifacts"]["summary_json"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("extra", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_threshold_key_not_in_collectives(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8,
            ranks_per_node=8,
            profile="thresholds",
            thresholds={
                "all_reduce_perf": {"bus_bw": {"1024": 1.0}},
                "broadcast_perf": {"bus_bw": {"1024": 1.0}},
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("broadcast_perf", str(ctx.exception))
        self.assertIn("collectives", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_smoke_profile_with_thresholds(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8,
            ranks_per_node=8,
            profile="smoke",
            thresholds={"all_reduce_perf": {"bus_bw": {"1024": 1.0}}},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("profile", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_num_ranks_not_divisible(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=9, ranks_per_node=8, profile="smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("divisible", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_collective_with_path_separator(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8, ranks_per_node=8, profile="smoke", collectives=["bin/all_reduce_perf"]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("basename", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_nonempty_matrix(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
        body["rccl"]["matrix"] = {"kind": "variants", "cases": [{"name": "a", "env": {}}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("matrix", str(ctx.exception).lower())

    def test_load_rccl_config_allows_omitted_or_empty_matrix(self):
        """Schema permits omitted matrix, ``null``, or ``{}`` until expansion is implemented."""
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}

        def assert_loads(matrix_payload):
            body = _minimal_rccl(num_ranks=8, ranks_per_node=8, profile="smoke")
            if matrix_payload is not ...:
                body["rccl"]["matrix"] = matrix_payload
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "rccl.json"
                config_path.write_text(json.dumps(body))
                loaded = load_rccl_config(str(config_path), cluster_dict)
            self.assertEqual(loaded.required_nodes, 1)

        assert_loads(...)  # key omitted
        assert_loads({})
        assert_loads(None)

    def test_load_rccl_config_rejects_thresholds_file_not_json_object(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            thresh_path = Path(tmpdir) / "t.json"
            thresh_path.write_text("[]")
            body = _minimal_rccl(
                num_ranks=8,
                ranks_per_node=8,
                profile="thresholds",
                thresholds_file=str(thresh_path),
            )
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("object", str(ctx.exception).lower())

    def test_load_rccl_config_raises_valueerror_for_schema_errors(self):
        """Pydantic validation failures are wrapped as ValueError for a stable caller contract."""
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = {"rccl": {"run": {}, "artifacts": {"output_dir": "/tmp/x", "export_raw": False}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("validation", str(ctx.exception).lower())

    def test_no_matrix_case_id_slug(self):
        self.assertEqual(_slug("all_reduce_perf"), "all_reduce_perf")
        self.assertEqual(_no_matrix_case_id(0, "all_reduce_perf"), "c0_all_reduce_perf")
        self.assertEqual(_no_matrix_case_id(2, "a/b"), "c2_a_b")

    def test_ensure_unique_case_id_appends_dup_on_collision(self):
        used: set[str] = set()
        r1 = {"collective": "x", "k": 1}
        self.assertEqual(_ensure_unique_case_id("cid", used, r1), "cid")
        r2 = {"collective": "x", "k": 2}
        cid2 = _ensure_unique_case_id("cid", used, r2)
        self.assertTrue(cid2.startswith("cid__dup"))
        self.assertIn(cid2, used)

    def test_parse_and_validate_results_strict_profile(self):
        config = _rccl_cfg(
            validation_profile="strict",
            thresholds={"all_reduce_perf": {"bus_bw": {"1024": "90.0"}}},
        )

        raw_results = [
            {
                "numCycle": 1,
                "name": "allreduce",
                "size": 1024,
                "type": "float",
                "redop": "sum",
                "inPlace": 1,
                "time": 1.0,
                "algBw": 100.0,
                "busBw": 95.0,
                "wrong": 0,
            }
        ]

        rows, summary = parse_and_validate_results(config, "all_reduce_perf", raw_results)
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["schema"], "passed")
        self.assertEqual(summary["thresholds_bus_bw"], "passed")
        self.assertEqual(summary["bw_dip"], "passed")
        self.assertEqual(summary["lat_dip"], "passed")

    def test_parse_and_validate_results_wrong_nonzero_maps_to_corruption_runtimeerror(self):
        config = _rccl_cfg()
        raw_results = [
            {
                "numCycle": 1,
                "name": "allreduce",
                "size": 1024,
                "type": "float",
                "redop": "sum",
                "inPlace": 1,
                "time": 1.0,
                "algBw": 100.0,
                "busBw": 95.0,
                "wrong": 1,
            }
        ]
        with self.assertRaises(RuntimeError) as ctx:
            parse_and_validate_results(config, "all_reduce_perf", raw_results)
        self.assertIn("corrupted", str(ctx.exception).lower())
        self.assertIn("#wrong", str(ctx.exception))

    def test_parse_and_validate_results_none_profile_skips_schema(self):
        config = _rccl_cfg(validation_profile="none")
        raw = [{"size": 1024, "busBw": 1.0}]
        rows, summary = parse_and_validate_results(config, "all_reduce_perf", raw)
        self.assertEqual(rows, raw)
        self.assertEqual(summary["schema"], "skipped")

    def test_build_collective_command_multi_node_wraps_mpirun_with_env_script(self):
        config = _rccl_cfg(
            required_nodes=2,
            num_ranks=16,
            ranks_per_node=8,
            env_script="/tmp/env.sh",
        )
        shdl = _DummySshHandle()
        command = build_collective_command(
            config,
            "all_reduce_perf",
            "/tmp/all_reduce.json",
            ["10.0.0.1", "10.0.0.2"],
            shdl,
        )
        self.assertIn("bash -lc", command)
        self.assertIn("source /tmp/env.sh", command)
        self.assertIn('exec "${MPI_HOME}/bin/mpirun"', command)
        self.assertIn("--np 16", command)
        self.assertTrue(shdl.commands)

    def test_build_shell_requires_rccl_tests_build_dir_from_env(self):
        config = _rccl_cfg(env_script="/site/env.sh")
        shdl = _DummySshHandle()
        command = build_collective_command(config, "all_reduce_perf", "/tmp/x.json", ["10.0.0.1"], shdl)
        self.assertIn("RCCL_TESTS_BUILD_DIR", command)
        self.assertIn("all_reduce_perf", command)
        self.assertIn(': "${RCCL_TESTS_BUILD_DIR:?', command)

    def test_load_rccl_config_rejects_removed_path_field_rccl_tests_dir(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(profile="smoke")
        body["rccl"]["run"]["rccl_tests_dir"] = "/opt/rccl-tests/build"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("extra", str(ctx.exception).lower())

    def test_scan_rccl_stdout_is_a_directory_is_actionable(self):
        out = "bash: line 1: /opt/ompi/bin: Is a directory\n"
        with self.assertRaises(RuntimeError) as ctx:
            _scan_rccl_stdout(out)
        self.assertIn("RCCL launch failed", str(ctx.exception))
        self.assertIn("directory", str(ctx.exception).lower())

    def test_scan_rccl_stdout_missing_bandwidth_hint(self):
        with self.assertRaises(RuntimeError) as ctx:
            _scan_rccl_stdout("some noise\n")
        msg = str(ctx.exception)
        self.assertIn("Avg bus bandwidth", msg)
        self.assertIn("MPI_HOME", msg)
        self.assertIn("RCCL_TESTS_BUILD_DIR", msg)

    def test_matching_rows_uses_out_of_place_rows_for_alltoall_variants(self):
        rows = [
            {"inPlace": 0, "size": 1024},
            {"inPlace": 1, "size": 1024},
        ]
        self.assertEqual(_matching_rows("alltoall_perf", rows), [rows[0]])
        self.assertEqual(_matching_rows("all_to_allv_perf", rows), [rows[0]])

    def test_matching_rows_uses_in_place_rows_for_other_collectives(self):
        rows = [
            {"inPlace": 0, "size": 1024},
            {"inPlace": 1, "size": 1024},
        ]
        self.assertEqual(_matching_rows("all_reduce_perf", rows), [rows[1]])

    def test_build_collective_command_shell_only_when_single_node(self):
        config = _rccl_cfg()
        shdl = _DummySshHandle()
        command = build_collective_command(config, "all_reduce_perf", "/tmp/x.json", ["10.0.0.1"], shdl)
        self.assertNotIn("mpirun", command)
        self.assertIn("bash -lc", command)
        self.assertFalse(shdl.commands)

    def test_build_summary_matches_run_json_roll_up(self):
        """Zero cases ⇒ failed; any failure ⇒ overall failed."""
        self.assertEqual(_build_summary([])["overall_status"], "failed")
        self.assertEqual(_build_summary([])["cases_total"], 0)

        mixed = _build_summary(
            [
                {"status": "passed"},
                {"status": "failed"},
                {"status": "skipped"},
            ]
        )
        self.assertEqual(mixed["cases_total"], 3)
        self.assertEqual(mixed["cases_passed"], 1)
        self.assertEqual(mixed["cases_failed"], 1)
        self.assertEqual(mixed["cases_skipped"], 1)
        self.assertEqual(mixed["overall_status"], "failed")

        all_pass = _build_summary([{"status": "passed"}, {"status": "passed"}])
        self.assertEqual(all_pass["overall_status"], "passed")

    def test_resolved_case_payload_includes_required_run_json_keys(self):
        """``resolved`` minimum keys; no-matrix path uses empty ``env``."""
        cfg = _rccl_cfg()
        resolved = _resolved_case_payload(cfg, "all_reduce_perf")
        self.assertEqual(
            set(resolved.keys()),
            {
                "collective",
                "datatype",
                "start_size",
                "end_size",
                "step_factor",
                "warmups",
                "iterations",
                "cycles",
                "env",
            },
        )
        self.assertEqual(resolved["env"], {})
        self.assertEqual(resolved["collective"], "all_reduce_perf")

    def test_load_rccl_config_rejects_non_numeric_bus_bw_threshold(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8,
            ranks_per_node=8,
            profile="thresholds",
            thresholds={"all_reduce_perf": {"bus_bw": {"1024": "not-a-number"}}},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("number", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
