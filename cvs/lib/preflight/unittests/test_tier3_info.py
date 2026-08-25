"""Unit tests for Primus Tier 3 preflight info integration."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.tier3_info import (
    _REPORT_BEGIN,
    _REPORT_END,
    Tier3InfoCheck,
    _resolve_dump_path,
    build_preflight_info_flags,
    build_remote_preflight_info_command,
    parse_preflight_info_output,
    resolve_tier3_setting,
)


class TestBuildPreflightInfoFlags(unittest.TestCase):
    def test_default_flags_include_host_gpu_network(self):
        flags = build_preflight_info_flags(dump_path="/tmp/tier3")
        self.assertIn("--host", flags)
        self.assertIn("--gpu", flags)
        self.assertIn("--network", flags)
        self.assertIn("--dump-path /tmp/tier3", flags)
        self.assertIn("--disable-pdf", flags)

    def test_resolve_dump_path_without_reporting_section(self):
        self.assertEqual(_resolve_dump_path({"tier3_info": {"dump_path": ""}}), "/tmp/preflight/node_smoke_tier3")

    def test_master_node_includes_report_markers(self):
        cmd = build_remote_preflight_info_command(
            primus_dir="/home/testuser/Primus",
            venv_activate="/home/testuser/envs/preflight/.venv/bin/activate",
            node_rank=0,
            nnodes=4,
            master_addr="node0",
            master_port=1234,
            gpus_per_node=8,
            dump_path="/tmp/preflight/tier3_info",
            preflight_flags="--host --gpu --network",
            report_file_name="tier3_info",
        )
        self.assertIn("preflight --host", cmd)
        self.assertNotIn("--single", cmd)
        self.assertIn(_REPORT_BEGIN, cmd)


class TestParsePreflightInfoOutput(unittest.TestCase):
    def test_parse_status_and_report(self):
        output = (
            "[Primus:Preflight] checks=host,gpu,network host=node0 status=PASS\n"
            f"{_REPORT_BEGIN}\n# Primus Preflight Report\n{_REPORT_END}\n"
        )
        parsed = parse_preflight_info_output(output)
        self.assertEqual(parsed["status"], "PASS")
        self.assertIn("# Primus Preflight Report", parsed["report_markdown"])


class TestResolveTier3Setting(unittest.TestCase):
    def test_falls_back_to_node_smoke_for_primus_paths_only(self):
        cfg = {
            "tier3_info": {"connectivity_mode": "run"},
            "node_smoke": {"primus_dir": "/home/user/Primus"},
        }
        self.assertEqual(resolve_tier3_setting(cfg, "primus_dir"), "/home/user/Primus")

    def test_connectivity_mode_does_not_inherit_node_smoke(self):
        cfg = {
            "node_smoke": {"connectivity_mode": "run", "primus_dir": "/home/user/Primus"},
        }
        self.assertEqual(resolve_tier3_setting(cfg, "connectivity_mode", "skip"), "skip")

    def test_tier3_skipped_when_only_node_smoke_enabled(self):
        phdl = MagicMock()
        phdl.reachable_hosts = ["node0"]
        cfg = {
            "node_smoke": {
                "connectivity_mode": "run",
                "primus_dir": "/home/user/Primus",
                "venv_activate": "/home/user/.venv/bin/activate",
            },
        }
        results = Tier3InfoCheck(phdl, ["node0"], cfg).run()
        self.assertTrue(results.get("skipped"))
        self.assertEqual(results.get("mode"), "skip")
        phdl.exec_cmd_list.assert_not_called()

    def test_nccl_env_derived_from_connectivity_check_rdma(self):
        cfg = {
            "tier3_info": {"connectivity_mode": "run", "auto_setup": False},
            "node_smoke": {
                "primus_dir": "/home/user/Primus",
                "venv_activate": "/home/user/.venv/bin/activate",
            },
            "connectivity_check": {
                "rdma": {"interfaces": ["rdma0", "rdma1"], "gid_index": "3"},
            },
        }
        check = Tier3InfoCheck(MagicMock(), ["node0"], cfg)
        self.assertEqual(check.nccl_ib_hca, "rdma0,rdma1")
        self.assertEqual(check.nccl_ib_gid_index, 3)


class TestTier3InfoCheckRun(unittest.TestCase):
    def _config(self):
        return {
            "tier3_info": {"connectivity_mode": "run", "auto_setup": False},
            "node_smoke": {
                "primus_dir": "/home/testuser/Primus",
                "venv_activate": "/home/testuser/envs/preflight/.venv/bin/activate",
            },
        }

    def test_parallel_ssh_launch(self):
        phdl = MagicMock()
        phdl.reachable_hosts = ["node0", "node1"]
        phdl.exec_cmd_list.return_value = {
            "node0": "[Primus:Preflight] checks=host,gpu,network host=node0 status=PASS\n",
            "node1": "[Primus:Preflight] checks=host,gpu,network host=node1 status=PASS\n",
        }

        results = Tier3InfoCheck(phdl, ["node0", "node1"], self._config()).run()
        self.assertEqual(results["status"], "PASS")
        cmd_list = phdl.exec_cmd_list.call_args[0][0]
        self.assertIn("export NODE_RANK=0", cmd_list[0])
        self.assertIn("export NODE_RANK=1", cmd_list[1])


if __name__ == "__main__":
    unittest.main()
