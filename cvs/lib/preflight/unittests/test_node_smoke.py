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
    resolve_rdma_gid_index,
    resolve_rdma_interfaces,
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
            "reporting": {"artifacts_root_dir": "/home/{user-id}/preflight"},
            "node_smoke": {"dump_path": ""},
        }
        self.assertEqual(_resolve_dump_path(cfg), "/home/{user-id}/preflight/node_smoke")

    def test_resolve_dump_path_without_reporting_section(self):
        cfg = {"node_smoke": {"dump_path": ""}}
        self.assertEqual(_resolve_dump_path(cfg), "/tmp/preflight/node_smoke")

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

    def test_tier2_perf_flags(self):
        flags = build_node_smoke_flags(
            dump_path="/tmp/smoke",
            tier2_perf=True,
            gemm_tflops_min=700,
            hbm_gbs_min=4500,
            rccl_gbs_min=180,
            rccl_size_mb=64,
            rccl_timeout_sec=120,
        )
        self.assertIn("--tier2-perf", flags)
        self.assertIn("--gemm-tflops-min 700", flags)
        self.assertIn("--hbm-gbs-min 4500", flags)
        self.assertIn("--rccl-gbs-min 180", flags)
        self.assertIn("--rccl-size-mb 64", flags)
        self.assertIn("--rccl-timeout-sec 120", flags)

    def test_tier2_perf_off_omits_threshold_flags(self):
        flags = build_node_smoke_flags(
            dump_path="/tmp/smoke",
            tier2_perf=False,
            gemm_tflops_min=700,
        )
        self.assertNotIn("--tier2-perf", flags)
        self.assertNotIn("--gemm-tflops-min", flags)


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


class TestResolveRdmaConfig(unittest.TestCase):
    def test_interfaces_from_connectivity_check_rdma(self):
        cfg = {
            "connectivity_check": {"rdma": {"interfaces": ["rdma0", "rdma1"], "gid_index": "3"}},
        }
        self.assertEqual(resolve_rdma_interfaces(cfg), ["rdma0", "rdma1"])
        self.assertEqual(resolve_rdma_gid_index(cfg), "3")

    def test_legacy_node_check_fallback(self):
        cfg = {
            "node_check": {"rdma_interfaces": ["legacy0"], "gid_index": "7"},
        }
        self.assertEqual(resolve_rdma_interfaces(cfg), ["legacy0"])
        self.assertEqual(resolve_rdma_gid_index(cfg), "7")


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

    def test_tier2_perf_extends_ssh_timeout(self):
        phdl = MagicMock()
        phdl.reachable_hosts = ["node0"]
        phdl.exec_cmd_list.return_value = {"node0": "wrote /tmp/smoke/a.json status=PASS\n"}

        cfg = self._config()
        cfg["node_smoke"]["tier2_perf"] = True
        cfg["node_smoke"]["ssh_timeout"] = 300
        checker = NodeSmokeCheck(phdl, ["node0"], cfg)
        checker.run()

        timeout = phdl.exec_cmd_list.call_args.kwargs.get("timeout") or phdl.exec_cmd_list.call_args[1].get("timeout")
        self.assertEqual(timeout, 600)
        cmd = phdl.exec_cmd_list.call_args[0][0][0]
        self.assertIn("--tier2-perf", cmd)
        self.assertIn("--gemm-tflops-min 600", cmd)
        self.assertIn("--hbm-gbs-min 2000", cmd)
        self.assertIn("--rccl-gbs-min 100", cmd)


class TestPreflightNodeSmokeReporting(unittest.TestCase):
    def test_preflight_check_display_names(self):
        from cvs.lib.preflight.report import preflight_check_display_name

        self.assertEqual(preflight_check_display_name("node_smoke_tier1"), "Node Smoke Tier 1")
        self.assertEqual(preflight_check_display_name("node_smoke_tier2"), "Node Smoke Tier 2")
        self.assertEqual(preflight_check_display_name("node_smoke_tier3"), "Node Smoke Tier 3")
        self.assertEqual(preflight_check_display_name("node_smoke"), "Node Smoke Tier 1")
        self.assertEqual(preflight_check_display_name("tier3_info"), "Node Smoke Tier 3")

    def test_node_smoke_tier_summaries_use_tier_labels(self):
        from cvs.lib.preflight.report import PreflightReportGenerator

        tier1_payload = {
            "tier1": {
                "per_gpu": [{"gpu": i, "status": "PASS"} for i in range(8)],
                "gpu_processes": {"ok": True},
                "nics": {"ok": True},
                "host_limits": {"ok": True},
                "gpu_low_level": {"ok": True},
                "xgmi": {"ok": True},
                "tooling": {"ok": True},
                "gpu_visibility": {"ok": True},
            }
        }
        tier1_results = {
            "tier2_perf": True,
            "gpus_per_node": 8,
            "tier1_tests_run": 39,
            "tier2_tests_run": 17,
            "node_results": {
                "node0": {"status": "PASS", "node_payload": tier1_payload},
                "node1": {"status": "PASS", "node_payload": tier1_payload},
            },
            "failed_nodes": [],
            "unknown_nodes": [],
            "passing_nodes": ["node0", "node1"],
            "total_nodes": 2,
            "tier2_thresholds": {
                "gemm_tflops_min": 600,
                "hbm_gbs_min": 2000,
                "rccl_gbs_min": 100,
            },
        }
        tier3_results = {
            "skipped": False,
            "tier3_tests_run": 27,
            "node_results": {"node0": {"status": "PASS"}, "node1": {"status": "PASS"}},
            "failed_nodes": [],
            "unknown_nodes": [],
            "passing_nodes": ["node0", "node1"],
            "total_nodes": 2,
        }

        generator = PreflightReportGenerator(None, {}, config_dict={})
        tier1_summary = generator._summarize_node_smoke_tier1_results(tier1_results)
        tier2_summary = generator._summarize_node_smoke_tier2_results(tier1_results)
        tier3_summary = generator._summarize_node_smoke_tier3_results(tier3_results)

        self.assertIn(
            "2/2 nodes passed Node Smoke Tier 1; 39 tests run per node",
            tier1_summary["summary"],
        )
        self.assertIn("Node Smoke Tier 2", tier2_summary["summary"])
        self.assertIn("17 tests run per node", tier2_summary["summary"])
        self.assertIn(
            "2/2 nodes passed Node Smoke Tier 3; 27 tests run cluster-wide",
            tier3_summary["summary"],
        )


if __name__ == "__main__":
    unittest.main()
