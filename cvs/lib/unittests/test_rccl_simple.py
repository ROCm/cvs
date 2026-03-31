import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("pytest", types.SimpleNamespace(fail=lambda message: (_ for _ in ()).throw(AssertionError(message))))

from cvs.lib.rccl_simple import RcclConfig, build_collective_command, load_rccl_config, parse_and_validate_results


class _DummySshHandle:
    def __init__(self):
        self.commands = []

    def exec(self, command, timeout=None):  # noqa: ARG002
        self.commands.append(command)
        return {"node0": ""}


class TestRcclSimple(unittest.TestCase):
    def test_load_rccl_config_minimal_multi_node(self):
        config = {
            "rccl": {
                "mode": "multi_node",
                "rccl_tests_dir": "/opt/rccl-tests/build",
                "mpi_root": "/usr",
                "rocm_path": "/opt/rocm",
                "env_script": "/tmp/env.sh",
                "num_ranks": "16",
                "ranks_per_node": "8",
                "collectives": ["all_reduce_perf"],
                "datatype": "float",
                "output_json": "/tmp/rccl_result.json",
            }
        }
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(config))
            loaded = load_rccl_config(str(config_path), cluster_dict)

        self.assertEqual(loaded.mode, "multi_node")
        self.assertEqual(loaded.num_ranks, 16)
        self.assertEqual(loaded.ranks_per_node, 8)
        self.assertEqual(loaded.collectives, ["all_reduce_perf"])

    def test_load_rccl_config_rejects_multi_dtype_sweep(self):
        config = {
            "rccl": {
                "mode": "multi_node",
                "rccl_tests_dir": "/opt/rccl-tests/build",
                "mpi_root": "/usr",
                "rocm_path": "/opt/rocm",
                "data_type_list": ["float", "bfloat16"],
            }
        }
        cluster_dict = {"username": "tester", "node_dict": {"node0": {}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rccl.json"
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_rccl_config(str(config_path), cluster_dict)

    def test_parse_and_validate_results_single_node(self):
        config = RcclConfig(
            mode="single_node",
            collectives=["all_reduce_perf"],
            datatype="float",
            num_ranks=8,
            ranks_per_node=8,
            rccl_tests_dir="/opt/rccl-tests/build",
            rocm_path="/opt/rocm",
            mpi_root=None,
            mpirun_path=None,
            env_script=None,
            output_json="/tmp/rccl_result.json",
            start_size="1024",
            end_size="16g",
            step_factor="2",
            warmups="10",
            iterations="20",
            cycles="1",
            verify_bus_bw=True,
            verify_bw_dip=True,
            verify_lat_dip=True,
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
        self.assertEqual(summary["bus_bw_check"], "passed")

    def test_build_collective_command_uses_mpirun_for_multi_node(self):
        config = RcclConfig(
            mode="multi_node",
            collectives=["all_reduce_perf"],
            datatype="float",
            num_ranks=16,
            ranks_per_node=8,
            rccl_tests_dir="/opt/rccl-tests/build",
            rocm_path="/opt/rocm",
            mpi_root="/usr",
            mpirun_path="/usr/bin/mpirun",
            env_script="/tmp/env.sh",
            output_json="/tmp/rccl_result.json",
            start_size="1024",
            end_size="16g",
            step_factor="2",
            warmups="10",
            iterations="20",
            cycles="1",
            verify_bus_bw=False,
            verify_bw_dip=False,
            verify_lat_dip=False,
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


if __name__ == "__main__":
    unittest.main()
