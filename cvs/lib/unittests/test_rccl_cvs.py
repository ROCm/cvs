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
    _no_matrix_case_id,
    _resolved_case_payload,
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
    mpi_root="/usr",
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
    return {
        "rccl": {
            "run": {
                "rccl_tests_dir": "/opt/rccl-tests/build",
                "mpi_root": mpi_root,
                "rocm_path": "/opt/rocm",
                "env_script": "/tmp/env.sh",
                "rccl_library_path": None,
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
            },
            "validation": validation,
            "artifacts": {"output_dir": "/tmp/rccl_cvs_out", "export_raw": False},
        }
    }


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

    def test_load_rccl_config_single_node_allows_missing_mpi(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            loaded = load_rccl_config(str(config_path), cluster_dict)

        self.assertEqual(loaded.required_nodes, 1)
        self.assertTrue(loaded.is_single_node)
        self.assertIsNone(loaded.mpirun_path)

    def test_load_rccl_config_multi_node_requires_mpi(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}, "node1": {}}}
        body = _minimal_rccl(mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("mpi_root", str(ctx.exception).lower())

    def test_load_rccl_config_mpirun_path_wins_over_mpi_root(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}, "node1": {}}}
        body = _minimal_rccl()
        body["rccl"]["run"]["mpirun_path"] = "/custom/bin/mpirun"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            loaded = load_rccl_config(str(config_path), cluster_dict)
        self.assertEqual(loaded.mpirun_path, "/custom/bin/mpirun")

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
            mpi_root=None,
            profile="thresholds",
            thresholds={"all_reduce_perf": {"bus_bw": {"1024": 1.0}}},
        )
        body["rccl"]["run"].pop("mpi_root", None)
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
            mpi_root=None,
            profile="strict",
            thresholds={},
        )
        body["rccl"]["run"].pop("mpi_root", None)
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
            mpi_root=None,
            profile="thresholds",
            thresholds={"all_reduce_perf": {"not_bus_bw": {"1024": 1.0}}},
        )
        body["rccl"]["run"].pop("mpi_root", None)
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
            mpi_root=None,
            profile="thresholds",
            thresholds={"all_reduce_perf": {"bus_bw": {}}},
        )
        body["rccl"]["run"].pop("mpi_root", None)
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
                mpi_root=None,
                profile="thresholds",
                thresholds_file=str(thresh_path),
            )
            body["rccl"]["run"].pop("mpi_root", None)
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
                mpi_root=None,
                profile="thresholds",
                thresholds_file=str(thresh_path),
            )
            body["rccl"]["run"].pop("mpi_root", None)
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("non-empty", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_unknown_rccl_key(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
        body["rccl"]["legacy_alias"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError):
                load_rccl_config(str(config_path), cluster_dict)

    def test_load_rccl_config_rejects_extra_top_level_keys(self):
        """Exactly one top-level key ``rccl``."""
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
        body["comment"] = "not allowed"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("exactly one top-level", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_extra_key_in_run(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
        body["rccl"]["run"]["legacy_mode"] = "single_node"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("extra", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_extra_key_in_validation(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
        body["rccl"]["validation"]["results_file"] = "/tmp/x.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("extra", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_extra_key_in_artifacts(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
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
            mpi_root=None,
            profile="thresholds",
            thresholds={
                "all_reduce_perf": {"bus_bw": {"1024": 1.0}},
                "broadcast_perf": {"bus_bw": {"1024": 1.0}},
            },
        )
        body["rccl"]["run"].pop("mpi_root", None)
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
            mpi_root=None,
            profile="smoke",
            thresholds={"all_reduce_perf": {"bus_bw": {"1024": 1.0}}},
        )
        body["rccl"]["run"].pop("mpi_root", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("profile", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_num_ranks_not_divisible(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=9, ranks_per_node=8, mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("divisible", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_collective_with_path_separator(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(
            num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke", collectives=["bin/all_reduce_perf"]
        )
        body["rccl"]["run"].pop("mpi_root", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("basename", str(ctx.exception).lower())

    def test_load_rccl_config_rejects_nonempty_matrix(self):
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}
        body = _minimal_rccl(num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke")
        body["rccl"]["run"].pop("mpi_root", None)
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
            body = _minimal_rccl(num_ranks=8, ranks_per_node=8, mpi_root=None, profile="smoke")
            body["rccl"]["run"].pop("mpi_root", None)
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
                mpi_root=None,
                profile="thresholds",
                thresholds_file=str(thresh_path),
            )
            body["rccl"]["run"].pop("mpi_root", None)
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
        config = RcclConfig(
            required_nodes=1,
            collectives=["all_reduce_perf"],
            datatype="float",
            num_ranks=8,
            ranks_per_node=8,
            rccl_tests_dir="/opt/rccl-tests/build",
            rocm_path="/opt/rocm",
            mpi_root=None,
            mpirun_path=None,
            env_script=None,
            artifacts_output_dir="/tmp/out",
            artifacts_export_raw=False,
            config_echo={},
            start_size="1024",
            end_size="16g",
            step_factor="2",
            warmups="10",
            iterations="20",
            cycles="1",
            validation_profile="strict",
            thresholds={"all_reduce_perf": {"bus_bw": {"1024": "90.0"}}},
            rccl_library_path=None,
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
        config = RcclConfig(
            required_nodes=1,
            collectives=["all_reduce_perf"],
            datatype="float",
            num_ranks=8,
            ranks_per_node=8,
            rccl_tests_dir="/opt/rccl-tests/build",
            rocm_path="/opt/rocm",
            mpi_root=None,
            mpirun_path=None,
            env_script=None,
            artifacts_output_dir="/tmp/out",
            artifacts_export_raw=False,
            config_echo={},
            start_size="1024",
            end_size="16g",
            step_factor="2",
            warmups="10",
            iterations="20",
            cycles="1",
            validation_profile="smoke",
            thresholds={},
            rccl_library_path=None,
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
                "wrong": 1,
            }
        ]
        with self.assertRaises(RuntimeError) as ctx:
            parse_and_validate_results(config, "all_reduce_perf", raw_results)
        self.assertIn("corrupted", str(ctx.exception).lower())
        self.assertIn("#wrong", str(ctx.exception))

    def test_parse_and_validate_results_none_profile_skips_schema(self):
        config = RcclConfig(
            required_nodes=1,
            collectives=["all_reduce_perf"],
            datatype="float",
            num_ranks=8,
            ranks_per_node=8,
            rccl_tests_dir="/opt/rccl-tests/build",
            rocm_path="/opt/rocm",
            mpi_root=None,
            mpirun_path=None,
            env_script=None,
            artifacts_output_dir="/tmp/out",
            artifacts_export_raw=False,
            config_echo={},
            start_size="1024",
            end_size="16g",
            step_factor="2",
            warmups="10",
            iterations="20",
            cycles="1",
            validation_profile="none",
            thresholds={},
            rccl_library_path=None,
        )
        raw = [{"size": 1024, "busBw": 1.0}]
        rows, summary = parse_and_validate_results(config, "all_reduce_perf", raw)
        self.assertEqual(rows, raw)
        self.assertEqual(summary["schema"], "skipped")

    def test_build_collective_command_uses_mpirun_when_multi_node(self):
        config = RcclConfig(
            required_nodes=2,
            collectives=["all_reduce_perf"],
            datatype="float",
            num_ranks=16,
            ranks_per_node=8,
            rccl_tests_dir="/opt/rccl-tests/build",
            rocm_path="/opt/rocm",
            mpi_root="/usr",
            mpirun_path="/usr/bin/mpirun",
            env_script="/tmp/env.sh",
            artifacts_output_dir="/tmp/out",
            artifacts_export_raw=False,
            config_echo={},
            start_size="1024",
            end_size="16g",
            step_factor="2",
            warmups="10",
            iterations="20",
            cycles="1",
            validation_profile="smoke",
            thresholds={},
            rccl_library_path=None,
        )
        shdl = _DummySshHandle()

        command = build_collective_command(
            config,
            "all_reduce_perf",
            "/tmp/all_reduce.json",
            ["10.0.0.1", "10.0.0.2"],
            shdl,
        )

        self.assertIn("/usr/bin/mpirun", command)
        self.assertIn("--np 16", command)
        self.assertTrue(shdl.commands)

    def test_build_collective_command_shell_only_when_single_node(self):
        config = RcclConfig(
            required_nodes=1,
            collectives=["all_reduce_perf"],
            datatype="float",
            num_ranks=8,
            ranks_per_node=8,
            rccl_tests_dir="/opt/rccl-tests/build",
            rocm_path="/opt/rocm",
            mpi_root=None,
            mpirun_path=None,
            env_script=None,
            artifacts_output_dir="/tmp/out",
            artifacts_export_raw=False,
            config_echo={},
            start_size="1024",
            end_size="16g",
            step_factor="2",
            warmups="10",
            iterations="20",
            cycles="1",
            validation_profile="smoke",
            thresholds={},
            rccl_library_path=None,
        )
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
        cfg = RcclConfig(
            required_nodes=1,
            collectives=["all_reduce_perf"],
            datatype="float",
            num_ranks=8,
            ranks_per_node=8,
            rccl_tests_dir="/opt/rccl-tests/build",
            rocm_path="/opt/rocm",
            mpi_root=None,
            mpirun_path=None,
            env_script=None,
            artifacts_output_dir="/tmp/out",
            artifacts_export_raw=False,
            config_echo={},
            start_size="1024",
            end_size="16g",
            step_factor="2",
            warmups="10",
            iterations="20",
            cycles="1",
            validation_profile="smoke",
            thresholds={},
            rccl_library_path=None,
        )
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
            mpi_root=None,
            profile="thresholds",
            thresholds={"all_reduce_perf": {"bus_bw": {"1024": "not-a-number"}}},
        )
        body["rccl"]["run"].pop("mpi_root", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(body))
            with self.assertRaises(ValueError) as ctx:
                load_rccl_config(str(config_path), cluster_dict)
        self.assertIn("number", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
