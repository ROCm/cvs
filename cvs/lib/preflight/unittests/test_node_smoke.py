"""Unit tests for Primus node_smoke preflight integration."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.node_smoke import (
    _JSON_BEGIN,
    _JSON_END,
    _resolve_dump_path,
    NodeSmokeCheck,
    build_node_smoke_flags,
    build_remote_node_smoke_command,
    parse_node_smoke_output,
)


class TestBuildNodeSmokeFlags(unittest.TestCase):
    def test_default_flags_include_dump_path(self):
        flags = build_node_smoke_flags(dump_path="/tmp/smoke")
        self.assertIn("--dump-path /tmp/smoke", flags)

    def test_empty_dump_path_uses_default(self):
        flags = build_node_smoke_flags(dump_path="")
        self.assertIn("--dump-path output/preflight", flags)

    def test_resolve_dump_path_from_empty_config_value(self):
        cfg = {
            "reporting": {"artifacts_root_dir": "/tmp/{user-id}/preflight"},
            "node_smoke": {"dump_path": ""},
        }
        self.assertEqual(_resolve_dump_path(cfg), "/tmp/{user-id}/preflight/node_smoke")

    def test_rdma_and_host_limits(self):
        flags = build_node_smoke_flags(
            dump_path="/home/testuser/preflight",
            expected_rdma_nics=8,
            ulimit_l_min_gb=64,
            shm_min_gb=16,
            allow_foreign_procs=True,
        )
        self.assertIn("--expected-rdma-nics 8", flags)
        self.assertIn("--ulimit-l-min-gb 64", flags)
        self.assertIn("--shm-min-gb 16", flags)
        self.assertIn("--allow-foreign-procs", flags)

    def test_extra_args_forwarded(self):
        flags = build_node_smoke_flags(
            dump_path="/tmp/smoke",
            extra_args=["--no-clean-dump-path", "--allow-foreign-procs"],
        )
        self.assertIn("--no-clean-dump-path", flags)
        self.assertIn("--allow-foreign-procs", flags)


class TestBuildRemoteCommand(unittest.TestCase):
    def test_includes_distributed_env_and_json_markers(self):
        cmd = build_remote_node_smoke_command(
            primus_dir="/home/testuser/Primus",
            venv_activate="/home/testuser/envs/preflight/.venv/bin/activate",
            node_rank=1,
            nnodes=4,
            master_addr="node0",
            master_port=1234,
            gpus_per_node=8,
            dump_path="/tmp/preflight/node_smoke",
            smoke_flags="--dump-path /tmp/preflight/node_smoke",
            nccl_ib_hca="rdma0,rdma1",
            nccl_ib_gid_index=3,
        )
        self.assertIn("export NODE_RANK=1", cmd)
        self.assertIn("export NNODES=4", cmd)
        self.assertIn("export MASTER_ADDR=node0", cmd)
        self.assertIn("/home/testuser/Primus/runner/primus-cli direct --single -- node_smoke", cmd)
        self.assertIn(_JSON_BEGIN, cmd)
        self.assertIn(_JSON_END, cmd)
        self.assertIn("NCCL_IB_HCA=rdma0,rdma1", cmd)
        self.assertIn("NCCL_IB_GID_INDEX=3", cmd)


class TestParseNodeSmokeOutput(unittest.TestCase):
    def test_parse_status_from_log_line(self):
        output = "some log\nwrote /tmp/smoke/host.json status=PASS duration=12.3s\n"
        parsed = parse_node_smoke_output(output)
        self.assertEqual(parsed["status"], "PASS")

    def test_parse_embedded_json(self):
        payload = '{"host": "node0", "status": "FAIL", "fail_reasons": ["gpu_processes: pid=99"]}'
        output = f"log line\n{_JSON_BEGIN}\n{payload}\n{_JSON_END}\n"
        parsed = parse_node_smoke_output(output)
        self.assertEqual(parsed["status"], "FAIL")
        self.assertEqual(parsed["fail_reasons"], ["gpu_processes: pid=99"])
        self.assertIsNotNone(parsed["node_payload"])

    def test_empty_output_fails(self):
        parsed = parse_node_smoke_output("")
        self.assertEqual(parsed["status"], "FAIL")


class TestNodeSmokeCheckRun(unittest.TestCase):
    def _config(self):
        return {
            "node_smoke": {
                "connectivity_mode": "run",
                "auto_setup": False,
                "primus_dir": "/home/testuser/Primus",
                "venv_activate": "/home/testuser/envs/preflight/.venv/bin/activate",
            }
        }

    def test_exec_cmd_list_aligns_with_reachable_hosts_subset(self):
        """cmd_list[i] must match reachable_hosts[i]; non-target hosts get 'true'."""
        phdl = MagicMock()
        phdl.reachable_hosts = ["node0", "node1", "node2"]
        phdl.exec_cmd_list.return_value = {
            "node0": "wrote /tmp/smoke/a.json status=PASS\n",
            "node1": "skipped",
            "node2": "wrote /tmp/smoke/c.json status=PASS\n",
        }

        checker = NodeSmokeCheck(phdl, ["node0", "node2"], self._config())
        results = checker.run()

        cmd_list = phdl.exec_cmd_list.call_args[0][0]
        self.assertEqual(len(cmd_list), len(phdl.reachable_hosts))
        self.assertEqual(cmd_list[1], "true")
        self.assertIn("primus-cli", cmd_list[0])
        self.assertIn("primus-cli", cmd_list[2])
        self.assertIn("export NODE_RANK=0", cmd_list[0])
        self.assertIn("export NODE_RANK=1", cmd_list[2])
        self.assertEqual(set(results["node_results"]), {"node0", "node2"})
        self.assertEqual(results["node_results"]["node0"]["node_rank"], 0)
        self.assertEqual(results["node_results"]["node2"]["node_rank"], 1)


if __name__ == "__main__":
    unittest.main()

